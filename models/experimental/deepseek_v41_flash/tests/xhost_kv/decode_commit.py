# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 of the fabric push (``PREFILL_D_P_FABRIC_PUSH.md``): the decode model's remote-commit API, written by a
synthetic sender. Two ranks of one ``tt-run`` job, as the servers will run:

* rank 1 (host 41, decode): builds ``DeepSeekV41Model`` (layers 0-20 by default: every kv source), opens two
  sessions, commits seeded random prefill states into session B through ``commit_host_prefill_state`` (the host
  path), reserves session A's pages with ``remote_commit_targets`` and sends the targets to rank 0; on rank 0's
  "done" it runs ``finish_remote_commit`` and compares A with the states and with B, byte for byte, on every chip of
  every stage: the rings (written ones and zeroed ones), the paged compressed entries and index keys
  (``_gather_paged_entries``) and the dense copies.
* rank 0 (host 40, prefill): regenerates the same states from the seed and writes them into session A's pages with
  fabric unicast writes: decode chip ``chip_id`` from mesh 0's chip of the same index, one core a chip, work lists
  (destination buffer, page) streamed from DRAM. It replies "done" after its program has finished, as the prefill
  server will.

Both ranks open their mesh with ``chat_cli.open_mesh_device``, so the router settings match by construction. Launch
from the repo root on host 40 (host 41 has the same checkout at the same path)::

    DEEPSEEK_V4_FABRIC_PAYLOAD=6208 tt-run \\
      --rank-binding tests/tt_metal/distributed/config/dual_bh_galaxy_8x4_hosts_40_41_two_mesh_rank_bindings.yaml \\
      --mpi-args "--host DC16-2-BG-0202-u20-40,DC16-2-BG-0202-u13-41 \\
        --map-by rankfile:file=$PWD/tests/tt_metal/distributed/config/dual_8x4_hosts_40_41.rankfile \\
        --mca btl self,tcp --mca btl_tcp_if_include ens5f0np0.1 --bind-to none --tag-output" \\
      bash -c 'cd $TT_METAL_HOME && PYTHONPATH=$TT_METAL_HOME LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib exec \\
        python_env/bin/python models/experimental/deepseek_v41_flash/tests/xhost_kv/decode_commit.py'

``DEEPSEEK_V4_DECODE_LAYERS`` (default 21) and ``--tokens`` (default 8192) size the run; rank 1 exits non-zero on a
mismatch.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

import torch

import ttnn
from models.experimental.deepseek_v41_flash.tests.xhost_kv.common import (
    MESH_SHAPE,
    PORT,
    RECEIVER_RANK,
    SENDER_RANK,
    connect,
    init_rank,
    log,
    page_bytes_of,
)
from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import recv_msg, send_msg

CORE = ttnn.CoreCoord(0, 0)
K = 32  # work-list entries a table page, and pages read into L1 a NoC barrier
STREAMS = ("kv", "tiles")  # 1 KiB compressed-KV rows; 2 KiB tiles (index keys and rings)

_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#if !defined(FABRIC_2D)
#error "the push needs a 2D fabric: a remote mesh is reached through its exit chip"
#endif
using namespace tt::tt_fabric::mesh::experimental;

constexpr uint32_t NB = get_compile_time_arg_val(0);  // the receiver chip's DRAM banks
constexpr uint32_t K = get_compile_time_arg_val(1);   // work-list entries a table page
constexpr uint32_t PAGE_A = get_compile_time_arg_val(2);
constexpr uint32_t PAGE_B = get_compile_time_arg_val(3);
constexpr uint32_t SLOT = get_compile_time_arg_val(4);  // L1 bytes a staged page (64 B aligned, >= both pages)
constexpr auto src_a_args = TensorAccessorArgs<5>();
constexpr auto tbl_a_args = TensorAccessorArgs<src_a_args.next_compile_time_args_offset()>();
constexpr auto src_b_args = TensorAccessorArgs<tbl_a_args.next_compile_time_args_offset()>();
constexpr auto tbl_b_args = TensorAccessorArgs<src_b_args.next_compile_time_args_offset()>();

struct Banks {
    uint32_t x[NB], y[NB], off[NB];
};

// Source page i to page t[2i + 1] of the receiver's interleaved buffer at t[2i], for i < n (t: the work list).
template <typename Src, typename Tbl, typename Sender>
FORCE_INLINE void send_stream(
    const Src& src, const Tbl& tbl, uint32_t page, uint32_t n, const Banks& banks, Sender& sender,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint8_t dst_dev, uint16_t dst_mesh, uint32_t buf, uint32_t tbuf) {
    for (uint32_t i0 = 0; i0 < n; i0 += K) {
        const uint32_t nb = (n - i0) < K ? (n - i0) : K;
        noc_async_read(tbl.get_noc_addr(i0 / K), tbuf, K * 8);
        for (uint32_t k = 0; k < nb; ++k) {
            noc_async_read(src.get_noc_addr(i0 + k), buf + k * SLOT, page);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbuf);
        for (uint32_t k = 0; k < nb; ++k) {
            const uint32_t base = t[2 * k], p = t[2 * k + 1];
            const uint32_t bank = p % NB;
            const uint64_t dst = get_noc_addr(banks.x[bank], banks.y[bank], banks.off[bank] + base + (p / NB) * page);
            fabric_unicast_noc_unicast_write<Sender, false>(
                &sender, hdr, dst_dev, dst_mesh, buf + k * SLOT, page, tt::tt_fabric::NocUnicastCommandHeader{dst});
            noc_async_writes_flushed();  // the header and the staged page left L1 before either is rewritten
        }
    }
}

void kernel_main() {
    size_t a = 0;
    const uint32_t src_a = get_arg_val<uint32_t>(a++);
    const uint32_t tbl_a = get_arg_val<uint32_t>(a++);
    const uint32_t n_a = get_arg_val<uint32_t>(a++);
    const uint32_t src_b = get_arg_val<uint32_t>(a++);
    const uint32_t tbl_b = get_arg_val<uint32_t>(a++);
    const uint32_t n_b = get_arg_val<uint32_t>(a++);
    Banks banks;
    for (uint32_t b = 0; b < NB; ++b) {
        banks.x[b] = get_arg_val<uint32_t>(a++);
        banks.y[b] = get_arg_val<uint32_t>(a++);
        banks.off[b] = get_arg_val<uint32_t>(a++);
    }

    tt::tt_fabric::RoutingPlaneConnectionManager conn;
    open_connections(conn, 1, a);
    auto& slot = conn.get(0);
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    fabric_set_unicast_route(hdr, slot.dst_dev_id, slot.dst_mesh_id);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t tbuf = buf + K * SLOT;

    send_stream(TensorAccessor(src_a_args, src_a, PAGE_A), TensorAccessor(tbl_a_args, tbl_a, K * 8), PAGE_A, n_a,
                banks, slot.sender, hdr, slot.dst_dev_id, slot.dst_mesh_id, buf, tbuf);
    send_stream(TensorAccessor(src_b_args, src_b, PAGE_B), TensorAccessor(tbl_b_args, tbl_b, K * 8), PAGE_B, n_b,
                banks, slot.sender, hdr, slot.dst_dev_id, slot.dst_mesh_id, buf, tbuf);
    close_connections(conn);
}
"""


def make_states(seed: int, total: int, ring_layers: list[int], sources: list[tuple[int, int]], dims: dict) -> dict:
    """Seeded random prefill states in ``commit_host_prefill_state``'s format (base 0): bf16 ``kv_tail``
    ``[window, head_dim]`` of each ring layer, ``compressed_kv`` / ``idx_keys`` ``[total // ratio, dim]`` of each
    kv source. Both ranks call it, so only addresses cross TCP."""

    def randn(layer: int, tag: int, *shape: int) -> torch.Tensor:
        g = torch.Generator().manual_seed((seed * 1009 + layer) * 31 + tag)
        return torch.randn(*shape, generator=g).to(torch.bfloat16)

    layers: dict[int, dict] = {}
    for li in ring_layers:
        layers.setdefault(li, {})["kv_tail"] = randn(li, 0, dims["window"], dims["head_dim"])
    for src, ratio in sources:
        n = total // ratio
        layers.setdefault(src, {}).update(
            compressed_kv=randn(src, 1, n, dims["head_dim"]), idx_keys=randn(src, 2, n, dims["index_dim"])
        )
    return {"seq_len": total, "base": 0, "layers": layers}


# --------------------------------------------------------------------------------------------------------------- #
# rank 0: the synthetic sender
# --------------------------------------------------------------------------------------------------------------- #
def stage_work(targets: dict, states: dict) -> dict[str, tuple[list[tuple[int, int]], list[torch.Tensor]]]:
    """What one chip of a stage receives, per stream: ``(work list of (buffer, page), page bytes in order)``.
    Every chip of a stage holds the same buffers at the same addresses, so a stage's chips share one list."""
    window, block, slot = targets["window"], targets["block"], targets["slot"]
    layers = states["layers"]
    rows = {r["ratio"]: r["blocks"] for r in targets["page_rows"]}
    sources = {s["layer"]: s for s in targets["sources"]}
    work = {}
    for stage in targets["stages"]:
        kv_work, kv_pages, tile_work, tile_pages = [], [], [], []
        for ring in stage["rings"]:
            tail = layers[ring["layer"]]["kv_tail"]
            pages = page_bytes_of(tail.reshape(1, 1, *tail.shape), ttnn.TILE_LAYOUT)
            first = slot * window * tail.shape[1] // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)
            tile_work += [(ring["addr"], first + j) for j in range(pages.shape[0])]
            tile_pages.append(pages)
        for pool in stage["pools"]:
            src = sources[pool["source"]]
            ckv, keys = layers[src["layer"]]["compressed_kv"], layers[src["layer"]]["idx_keys"]
            row, e0, n = rows[src["ratio"]], src["first_entry"], ckv.shape[0]
            kv_work += [(pool["kv"], row[e // block] * block + e % block) for e in range(e0, n)]
            kv_pages.append(page_bytes_of(ckv[e0:n].reshape(n - e0, 1, 1, -1), ttnn.ROW_MAJOR_LAYOUT))
            b0, b1 = e0 // block, -(-n // block)
            padded = torch.zeros((b1 - b0) * block, keys.shape[1], dtype=keys.dtype)
            padded[: n - b0 * block] = keys[b0 * block : n]
            pages = page_bytes_of(padded.reshape(b1 - b0, 1, block, -1), ttnn.TILE_LAYOUT)
            per_block = pages.shape[0] // (b1 - b0)
            tile_work += [(pool["keys"], row[b] * per_block + t) for b in range(b0, b1) for t in range(per_block)]
            tile_pages.append(pages)
        work[stage["stage"]] = {"kv": (kv_work, kv_pages), "tiles": (tile_work, tile_pages)}
    return work


def upload_streams(mesh, targets: dict, work: dict) -> tuple[dict, dict, dict]:
    """Per stream, the staged pages ``[R, C, n_max, words]`` and work tables ``[R, C, pages, 2 K]`` (uint32,
    ROW_MAJOR, one chip's slice each), and each sending chip's ``{stream: count}``."""
    R, C = MESH_SHAPE
    counts: dict[tuple[int, int], dict[str, int]] = {}
    per_chip: dict[tuple[int, int], dict] = {}
    for stage in targets["stages"]:
        for chip in stage["chips"]:
            coord = divmod(chip["chip_id"], C)
            if coord in per_chip:
                sys.exit(f"sender: mesh 0 chip {coord} would serve two decode chips")
            per_chip[coord] = work[stage["stage"]]
            counts[coord] = {s: len(work[stage["stage"]][s][0]) for s in STREAMS}
    srcs, tables = {}, {}
    for stream in STREAMS:
        page = targets["page_bytes"]["kv" if stream == "kv" else "keys"]
        n_max = max([1] + [c[stream] for c in counts.values()])
        n_tbl = -(-n_max // K)
        data = torch.zeros(R, C, n_max, page // 4, dtype=torch.int32)
        table = torch.zeros(R, C, n_tbl, 2 * K, dtype=torch.int32)
        for (r, c), w in per_chip.items():
            entries, pages = w[stream]
            if not entries:
                continue
            data[r, c, : len(entries)] = torch.cat(pages).view(torch.int32).reshape(len(entries), -1)
            flat = torch.tensor(entries, dtype=torch.int64).reshape(-1)
            flat = torch.where(flat >= 2**31, flat - 2**32, flat).to(torch.int32)
            table[r, c].view(-1)[: flat.numel()] = flat
        for out, host in ((srcs, data), (tables, table)):
            out[stream] = ttnn.from_torch(
                host,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=MESH_SHAPE),
            )
    return srcs, tables, counts


def build_push(mesh, targets: dict, srcs: dict, tables: dict, counts: dict) -> ttnn.MeshProgramDescriptor:
    """One program a sending chip: its core streams both work lists to its decode chip."""
    C = MESH_SHAPE[1]
    pages = [targets["page_bytes"]["kv"], targets["page_bytes"]["keys"]]
    slot = -(-max(pages) // 64) * 64
    ct_tail = []
    for stream in STREAMS:
        ct_tail += list(ttnn.TensorAccessorArgs(srcs[stream]).get_compile_time_args())
        ct_tail += list(ttnn.TensorAccessorArgs(tables[stream]).get_compile_time_args())
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])
    md = ttnn.MeshProgramDescriptor()
    for stage in targets["stages"]:
        for chip in stage["chips"]:
            coord = ttnn.MeshCoordinate(*divmod(chip["chip_id"], C))
            banks = chip["banks"]
            n = counts[divmod(chip["chip_id"], C)]
            own = []
            for stream in STREAMS:
                own += [int(srcs[stream].buffer_address()), int(tables[stream].buffer_address()), n[stream]]
            own += [v for bank in banks for v in bank]
            prog = ttnn.ProgramDescriptor()
            size = K * slot + K * 8 + 64
            prog.cbs = [
                ttnn.CBDescriptor(
                    total_size=size,
                    core_ranges=crs,
                    format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, size)],
                )
            ]
            prog.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=crs,
                    compile_time_args=[len(banks), K, pages[0], pages[1], slot] + ct_tail,
                    runtime_args=ttnn.RuntimeArgs(),
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                )
            ]
            src_node = mesh.get_fabric_node_id(coord)
            dst_node = ttnn.FabricNodeId(ttnn.MeshId(targets["mesh_id"]), chip["chip_id"])
            links = list(ttnn.get_forwarding_link_indices(mesh, src_node, dst_node))
            if not links:
                sys.exit(f"sender: no fabric route from {src_node} to {dst_node}")
            fabric = ttnn.setup_routing_plane_connection(mesh, src_node, [dst_node], [links[0]], prog, 0, CORE)
            prog.kernels[0].runtime_args[CORE.x][CORE.y] = own + list(fabric)
            md[ttnn.MeshCoordinateRange(coord, coord)] = prog
    return md


def run_sender(args) -> int:
    from models.experimental.deepseek_v4_flash.demo.chat_cli import open_mesh_device

    init_rank(SENDER_RANK, "sender")
    with open_mesh_device(None, mesh_shape=MESH_SHAPE) as mesh:
        payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        log("sender", f"mesh {tuple(mesh.shape)} open, packet payload {payload} B")
        with socket.create_server(("0.0.0.0", args.port)) as listener:
            sock, addr = listener.accept()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            req, _ = recv_msg(sock)
            assert req["op"] == "commit", req
            targets = req["targets"]
            if max(targets["page_bytes"].values()) > payload:
                sys.exit(f"sender: pages of {targets['page_bytes']} B exceed the {payload} B fabric payload")
            sources = [(s["layer"], s["ratio"]) for s in targets["sources"]]
            states = make_states(req["seed"], targets["seq_len"], req["ring_layers"], sources, req["dims"])
            work = stage_work(targets, states)
            srcs, tables, counts = upload_streams(mesh, targets, work)
            md = build_push(mesh, targets, srcs, tables, counts)
            t0 = time.perf_counter()
            ttnn.generic_op([*srcs.values(), *tables.values()], md)
            ttnn.synchronize_device(mesh)
            seconds = time.perf_counter() - t0
            page = {"kv": targets["page_bytes"]["kv"], "tiles": targets["page_bytes"]["keys"]}
            nbytes = sum(n[s] * page[s] for n in counts.values() for s in STREAMS)
            send_msg(sock, {"done": True, "seconds": seconds, "bytes": nbytes})
            log(
                "sender",
                f"{len(counts)} decode chips, {nbytes / 2**20:.1f} MiB in {seconds * 1e3:.1f} ms "
                f"({nbytes / seconds / 1e9:.2f} GB/s incl. launch)",
            )
            recv_msg(sock)  # the receiver's bye: keep the job (and this mesh's fabric) up until it has checked
            sock.close()
            for t in (*srcs.values(), *tables.values()):
                ttnn.deallocate(t)
    return 0


# --------------------------------------------------------------------------------------------------------------- #
# rank 1: the decode model
# --------------------------------------------------------------------------------------------------------------- #
def _chip_rows(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    """Every chip's copy of ``tensor`` as int16 ``[rows, width]`` (bf16 bits)."""
    out = []
    for shard in ttnn.get_device_tensors(tensor):
        host = ttnn.to_torch(shard)
        out.append(host.reshape(-1, host.shape[-1]).contiguous().view(torch.int16))
    return out


def check(model, states: dict, sid_a: int, sid_b: int, ring_layers: set[int]) -> int:
    """Rows of session A that differ from the states or from session B, over every chip of every stage."""
    total = states["seq_len"]
    window, topk = model.sliding_window, model.config.index_topk
    slot_a, slot_b = model._slot_of[sid_a], model._slot_of[sid_b]
    layers = states["layers"]
    bad = 0
    for sm in model.submeshes_io:
        groups = {}
        for li in sm["layers"]:
            scache = sm["scaches"][li]
            want = layers[li]["kv_tail"] if li in ring_layers else torch.zeros(window, scache.kv.shape[3])
            want = want.to(torch.bfloat16).view(torch.int16)
            for chip, rows in enumerate(_chip_rows(scache.kv)):
                for sid, slot in ((sid_a, slot_a), (sid_b, slot_b)):
                    n = int((rows[slot * window : (slot + 1) * window] != want).any(dim=-1).sum())
                    if n:
                        log("receiver", f"stage {sm['index']} chip {chip} layer {li} ring, session {sid}: {n} bad rows")
                    bad += n
            src = model._kv_source_of.get(li)
            if src in layers and "compressed_kv" in layers[src] and scache.comp_kv is not None:
                groups.setdefault(src, scache)
        for src, scache in groups.items():
            ratio = int(model.config.compress_ratios[src])
            n, group = total // ratio, model._paged_ratios[ratio].layer_type
            want = {
                "kv": layers[src]["compressed_kv"].view(torch.int16),
                "keys": layers[src]["idx_keys"].view(torch.int16),
            }
            got = {}
            for sid in (sid_a, sid_b):
                kv, keys = model._gather_paged_entries(scache, group, sid, n)
                got[sid] = {"kv": _chip_rows(kv), "keys": _chip_rows(keys)}
                ttnn.deallocate(kv)
                ttnn.deallocate(keys)
            for name in ("kv", "keys"):
                for chip, (a, b) in enumerate(zip(got[sid_a][name], got[sid_b][name])):
                    n_bad = int((a != want[name]).any(dim=-1).sum()) + int((a != b).any(dim=-1).sum())
                    if n_bad:
                        log("receiver", f"stage {sm['index']} chip {chip} source {src} {name}: {n_bad} bad rows")
                    bad += n_bad
            if scache.dense_kv is not None:
                nd = -(-min(n, topk) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                for chip, rows in enumerate(_chip_rows(scache.dense_kv)):
                    a = rows[slot_a * topk : slot_a * topk + nd]
                    b = rows[slot_b * topk : slot_b * topk + nd]
                    n_bad = int((a != b).any(dim=-1).sum())
                    if n_bad:
                        log("receiver", f"stage {sm['index']} chip {chip} source {src} dense: {n_bad} bad rows")
                    bad += n_bad
    return bad


def run_receiver(args) -> int:
    import contextlib

    from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
        _build_rope,
        _construct_model,
        _traced_max_seq,
    )
    from models.experimental.deepseek_v4_flash.demo.chat_cli import open_mesh_device
    from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config

    init_rank(RECEIVER_RANK, "receiver")
    os.environ.setdefault("DEEPSEEK_V4_DECODE_LAYERS", "21")
    mesh_cm = open_mesh_device(None, mesh_shape=MESH_SHAPE, l1_small_size=args.l1_small_size)
    with mesh_cm as mesh, contextlib.ExitStack() as stack:
        system_config = load_system_config(mesh_device=mesh)
        set_active_system_config(system_config)
        t0 = time.perf_counter()
        model, lm_head, loader, config = _construct_model(mesh, stack, system_config=system_config)
        max_seq = _traced_max_seq(config, args.tokens + model.sliding_window, model.num_layers)
        model.prepare_static_decode(
            _build_rope(config, loader.snapshot_dir, max_seq), max_seq, lm_head=lm_head, num_sessions=2
        )
        log("receiver", f"{model.num_layers}-layer decode model ready in {time.perf_counter() - t0:.0f} s")

        ring_layers = [li for li in range(model.num_layers) if li < config.candidate_source_layer_id]
        sources = [
            (src, int(config.compress_ratios[src])) for src in config.kv_source_layer_ids if src < model.num_layers
        ]
        dims = {"window": model.sliding_window, "head_dim": config.head_dim, "index_dim": config.index_head_dim}
        states = make_states(args.seed, args.tokens, ring_layers, sources, dims)
        sid_b = model.open_session()
        sid_a = model.open_session()
        model.commit_host_prefill_state(states, sid_b)
        log("receiver", f"session {sid_b}: {args.tokens} tokens committed through the host path")

        targets = model.remote_commit_targets(sid_a, args.tokens, ring_layers)
        log(
            "receiver",
            f"session {sid_a} (slot {targets['slot']}): {len(targets['stages'])} stage(s) to write, "
            f"mesh {targets['mesh_id']}",
        )
        sock = connect(args.sender, args.port)
        send_msg(
            sock, {"op": "commit", "targets": targets, "seed": args.seed, "ring_layers": ring_layers, "dims": dims}
        )
        done, _ = recv_msg(sock)
        assert done.get("done"), done
        model.finish_remote_commit(sid_a)
        bad = check(model, states, sid_a, sid_b, set(ring_layers))
        send_msg(sock, {"op": "bye"})
        sock.close()
        log(
            "receiver",
            f"sender wrote {done['bytes'] / 2**20:.1f} MiB in {done['seconds'] * 1e3:.1f} ms; "
            + ("PASS: byte-exact" if not bad else f"FAIL: {bad} bad row(s)"),
        )
        return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sender", default="10.82.97.40", help="rank 0's address (host 40)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--tokens", type=int, default=8192, help="prefilled length (a multiple of 128)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--l1-small-size", type=int, default=10240, help="as the decode server's default")
    args = parser.parse_args()
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    return run_sender(args) if rank == SENDER_RANK else run_receiver(args)


if __name__ == "__main__":
    sys.exit(main())
