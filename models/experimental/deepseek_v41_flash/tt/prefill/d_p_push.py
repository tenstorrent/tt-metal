# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 of the fabric push (``PREFILL_D_P_FABRIC_PUSH.md``): metal_2's prefill state written from its caches
straight into a decode session's DRAM over TT-Fabric, no host copy of the compressed entries.

:func:`push` takes the targets a decode model's ``remote_commit_targets`` returned (where the session's rings, entry
pages and key blocks live on every decode chip) and writes, from the 8 x 4 prefill mesh:

* compressed KV entries from the slab of the chip that owns them (one 1 KiB row a page, dedup: one owner per entry),
* index-key blocks (64 entries, eight 2 KiB tiles) from a device tilize of the owner row's slab, by one of that
  row's chips (TP-replicated: any column; they take turns),
* the window rings (the last ``window`` rows of each layer, read on the host from the chips owning those ring blocks
  and tilized there: 128 KiB a layer) from a staging tensor every chip holds, the tiles spread over every chip.

Every sending chip runs one core: a work list in DRAM of (source, source page, destination chip, destination buffer,
destination page) entries, ``K`` at a time, the pages staged in L1 and sent as fabric unicast writes. A remote mesh
is reached through this mesh's exit chip, so one connection a chip carries every destination: the route is rewritten
when the destination changes. A destination page goes to its interleaved bank (``page % banks``) at the bank's
offset from the targets' bank table.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

import ttnn

CORE = ttnn.CoreCoord(0, 0)
K = 32  # work-list entries a table page, and pages read into L1 a NoC barrier
SLOT = 2048  # L1 bytes a staged page (the largest page: a tile)
ENTRY_WORDS = 4  # (source | destination << 8, source page, destination buffer, destination page)

_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#if !defined(FABRIC_2D)
#error "the push needs a 2D fabric: a remote mesh is reached through its exit chip"
#endif
using namespace tt::tt_fabric::mesh::experimental;

constexpr uint32_t NB = get_compile_time_arg_val(0);    // a destination chip's DRAM banks
constexpr uint32_t K = get_compile_time_arg_val(1);     // work-list entries a table page
constexpr uint32_t SLOT = get_compile_time_arg_val(2);  // L1 bytes a staged page
constexpr uint32_t NSRC = get_compile_time_arg_val(3);  // source buffers
constexpr uint32_t DST_BYTES = get_compile_time_arg_val(4);  // the destination table
constexpr uint32_t DST_WORDS = 2 + 3 * NB;  // a destination: fabric chip, mesh, then (x, y, offset) a bank
constexpr auto args = TensorAccessorArgs<5>();  // every buffer read here: DRAM interleaved

void kernel_main() {
    size_t a = 0;
    uint32_t src_addr[NSRC], src_page[NSRC];
    for (uint32_t s = 0; s < NSRC; ++s) {
        src_addr[s] = get_arg_val<uint32_t>(a++);
        src_page[s] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t tbl = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t dsts = get_arg_val<uint32_t>(a++);

    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t tbuf = buf + K * SLOT;
    const uint32_t dbuf = tbuf + K * 16;
    noc_async_read(TensorAccessor(args, dsts, DST_BYTES).get_noc_addr(0), dbuf, DST_BYTES);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbuf);
    volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbuf);
    const auto table = TensorAccessor(args, tbl, K * 16);

    tt::tt_fabric::RoutingPlaneConnectionManager conn;
    open_connections(conn, 1, a);
    auto& slot = conn.get(0);
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    uint32_t cur = 0xffffffffu;

    for (uint32_t i0 = 0; i0 < n; i0 += K) {
        const uint32_t nb = (n - i0) < K ? (n - i0) : K;
        noc_async_read(table.get_noc_addr(i0 / K), tbuf, K * 16);
        noc_async_read_barrier();
        for (uint32_t k = 0; k < nb; ++k) {
            const uint32_t s = t[4 * k] & 0xff;
            const auto src = TensorAccessor(args, src_addr[s], src_page[s]);
            noc_async_read(src.get_noc_addr(t[4 * k + 1]), buf + k * SLOT, src_page[s]);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < nb; ++k) {
            const uint32_t page = src_page[t[4 * k] & 0xff], j = t[4 * k] >> 8;
            const uint32_t base = t[4 * k + 2], p = t[4 * k + 3];
            volatile tt_l1_ptr uint32_t* dst = d + j * DST_WORDS;
            if (j != cur) {
                fabric_set_unicast_route(hdr, dst[0], dst[1]);
                cur = j;
            }
            const uint32_t bank = p % NB;
            const uint64_t addr =
                get_noc_addr(dst[2 + 3 * bank], dst[3 + 3 * bank], dst[4 + 3 * bank] + base + (p / NB) * page);
            fabric_unicast_noc_unicast_write<decltype(slot.sender), false>(
                &slot.sender, hdr, dst[0], dst[1], buf + k * SLOT, page, tt::tt_fabric::NocUnicastCommandHeader{addr});
            noc_async_writes_flushed();  // the header and the staged page left L1 before either is rewritten
        }
    }
    close_connections(conn);
}
"""


def _int32(x: torch.Tensor) -> torch.Tensor:
    """uint32 values held in int64 -> the same bits as int32."""
    return torch.where(x >= 2**31, x - 2**32, x).to(torch.int32)


def _owner_chips(c, owner: torch.Tensor, spread: torch.Tensor, cols: int) -> torch.Tensor:
    """The sending chip (row-major mesh index) of entries whose owner stripe is ``owner`` in metal_2's served cache
    ``c``: a dedup cache's owners are the chips themselves; a TP-replicated one's are mesh rows, any of whose columns
    holds the rows (``spread`` picks one)."""
    if c.tp_axis is not None:
        return owner
    return owner * cols + spread % cols


def _slab_rows(c, b0: int, b1: int) -> ttnn.Tensor:
    """Local rows of owner-stripe blocks ``[b0, b1)`` of metal_2's served cache ``c`` (every chip's own), DRAM
    interleaved ROW_MAJOR."""
    slab = c.slab if c.slab is not None else c.served
    hs = c.stripe // c.ratio
    if b1 * hs > c.crows:
        raise ValueError(f"owner blocks up to {b1} of {hs} rows: past the cache's {c.crows} local rows")
    return slab.slice_rows(b0 * hs, b1 * hs)


def push(d_p, targets: dict, slot: int = 0, times: Optional[dict] = None) -> dict:
    """Write cache slot ``slot`` of ``d_p`` (a :class:`~.d_p_bridge.DPPrefill` whose slot holds a prompt of
    ``targets["seq_len"]`` tokens) into the decode session ``targets`` describes. Returns ``{"bytes", "pages",
    "seconds"}`` (``seconds``: the device program, launch included); ``times``: host seconds per part, added up.
    The writes have left this mesh when it returns; the receiver commits after a reply over TCP."""
    from models.demos.deepseek_v41_flash_d_p.tt.served import dealloc

    from .d_p_bridge import _window_rows

    times = times if times is not None else {}
    t0 = time.perf_counter()
    mesh, cfg = d_p.mesh_device, d_p.cfg
    rows_m, cols = tuple(mesh.shape)
    chips = rows_m * cols
    T, start, block, window = (int(targets[k]) for k in ("seq_len", "start", "block", "window"))
    if window != cfg.window:
        raise ValueError(f"decode's {window}-row window is not this prefill's {cfg.window}")
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    if max(targets["page_bytes"].values()) > payload:
        raise ValueError(f"pages of {targets['page_bytes']} B exceed the {payload} B fabric payload")
    st = d_p.runtime.state(d_p.slots[slot], 0)
    skipped = set(d_p.runtime.model.decoder)
    rows_of = {r["ratio"]: torch.tensor(r["blocks"], dtype=torch.int64) for r in targets["page_rows"]}

    dests, first_dst = [], {}  # every decode chip; stage -> index of its first chip in ``dests``
    for stage in targets["stages"]:
        first_dst[stage["stage"]] = len(dests)
        dests += stage["chips"]
    banks = {len(chip["banks"]) for chip in dests}
    if len(banks) != 1:
        raise ValueError(f"decode chips with {sorted(banks)} DRAM banks")
    nb = banks.pop()

    sources: list[tuple[ttnn.Tensor, int]] = []  # (DRAM interleaved buffer, page bytes); a work entry's source index
    # (sender, source, source page, destination, destination buffer, destination page) columns, int64
    work: list[torch.Tensor] = []
    free = []

    def add(sender: torch.Tensor, dst: int, base: int, src_page: torch.Tensor, page: torch.Tensor) -> None:
        """Pages ``src_page`` of the newest source to pages ``page`` of buffer ``base`` on decode chip ``dst``."""
        full = torch.ones_like(page)
        work.append(torch.stack([sender, full * (len(sources) - 1), src_page, full * dst, full * base, page]))

    for source in targets["sources"]:
        layer, ratio, e0 = int(source["layer"]), int(source["ratio"]), int(source["first_entry"])
        n = T // ratio
        if e0 != start // ratio:
            raise ValueError(f"source {layer}: first entry {e0} for a commit from position {start}")
        if n <= e0:
            continue
        row = rows_of[ratio]
        pools = [
            (first_dst[stage["stage"]] + k, pool)
            for stage in targets["stages"]
            for pool in stage["pools"]
            if int(pool["source"]) == layer
            for k in range(len(stage["chips"]))
        ]
        comp = st.compress[layer]

        c = comp.ckv
        hs = c.stripe // ratio
        period = c.owners * hs
        b0, b1 = e0 // period, -(-n // period)
        sl = _slab_rows(c, b0, b1)
        free.append(sl)
        if int(sl.shape[-1]) * 2 != targets["page_bytes"]["kv"]:
            raise ValueError(f"source {layer}: {sl.shape[-1]}-wide entries, decode pages of {targets['page_bytes']}")
        sources.append((sl, int(sl.shape[-1]) * 2))
        e = torch.arange(e0, n, dtype=torch.int64)
        owner, local = (e % period) // hs, (e // period - b0) * hs + e % hs
        page = row[e // block] * block + e % block
        for dst, pool in pools:
            add(_owner_chips(c, owner, e + dst, cols), dst, int(pool["kv"]), local, page)

        c = comp.index_k
        if c.fp8:
            raise ValueError("index keys in fp8_row192: the push needs DS41_INDEX_K_SERVED=bf16")
        hs = c.stripe // ratio
        period = c.owners * hs
        if hs % block:
            raise ValueError(f"source {layer}: {hs} key rows an owner stripe split {block}-entry blocks")
        b0, b1 = e0 // period, -(-n // period)
        sl = _slab_rows(c, b0, b1)
        keys = ttnn.to_layout(sl, ttnn.TILE_LAYOUT)
        dealloc(sl)
        free.append(keys)
        sources.append((keys, targets["page_bytes"]["keys"]))
        tile_cols = int(keys.shape[-1]) // ttnn.TILE_SIZE
        per_block = block // ttnn.TILE_SIZE * tile_cols
        b = torch.arange(e0 // block, -(-n // block), dtype=torch.int64)
        eb = b * block
        owner, local = (eb % period) // hs, (eb // period - b0) * hs + eb % hs
        tile = torch.arange(per_block, dtype=torch.int64)
        src_page = (((local // ttnn.TILE_SIZE) * tile_cols)[:, None] + tile).reshape(-1)
        page = ((row[b] * per_block)[:, None] + tile).reshape(-1)
        for dst, pool in pools:
            sender = _owner_chips(c, owner, b + dst, cols).repeat_interleave(per_block)
            add(sender, dst, int(pool["keys"]), src_page, page)
    t1 = time.perf_counter()
    times["compressed"] = times.get("compressed", 0.0) + t1 - t0

    ring_layers = sorted({int(r["layer"]) for stage in targets["stages"] for r in stage["rings"]})
    if ring_layers:
        p0 = max(0, T - window)
        tails = torch.zeros(len(ring_layers), window, cfg.head_dim, dtype=torch.bfloat16)
        for k, li in enumerate(ring_layers):
            if li not in skipped:
                tails[k, window - (T - p0) :] = _window_rows(st.window[li], p0, T)
        staged = ttnn.from_torch(
            tails.reshape(1, 1, -1, cfg.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        free.append(staged)
        sources.append((staged, targets["page_bytes"]["ring"]))
        per_layer = window * cfg.head_dim // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)
        first = int(targets["slot"]) * per_layer
        tile = torch.arange(per_layer, dtype=torch.int64)
        index = {li: k for k, li in enumerate(ring_layers)}
        for stage in targets["stages"]:
            for ring in stage["rings"]:
                src_page = index[int(ring["layer"])] * per_layer + tile
                for k in range(len(stage["chips"])):
                    dst = first_dst[stage["stage"]] + k
                    add((src_page + dst) % chips, dst, int(ring["addr"]), src_page, first + tile)
    t2 = time.perf_counter()
    times["window"] = times.get("window", 0.0) + t2 - t1
    if len(sources) > 255 or len(dests) > 2**24:
        raise ValueError(f"{len(sources)} sources / {len(dests)} destinations: past the work entry's fields")
    for t, _ in sources:
        mc = t.memory_config()
        assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED and mc.buffer_type == ttnn.BufferType.DRAM, mc

    sender, src, src_page, dst, base, page = torch.cat(work, dim=1)
    order = torch.argsort(sender, stable=True)
    sender, src, src_page, dst, base, page = (x[order] for x in (sender, src, src_page, dst, base, page))
    counts = torch.bincount(sender, minlength=chips)
    offsets = torch.cumsum(counts, 0) - counts
    rank = torch.arange(sender.numel(), dtype=torch.int64) - offsets[sender]
    n_tbl = max(1, -(-int(counts.max()) // K))
    table = torch.zeros(chips, n_tbl * K, ENTRY_WORDS, dtype=torch.int32)
    table[sender, rank] = _int32(torch.stack([src | (dst << 8), src_page, base, page], dim=1))
    table_t = ttnn.from_torch(
        table.reshape(rows_m, cols, n_tbl, K * ENTRY_WORDS),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows_m, cols)),
    )
    dst_words = 2 + 3 * nb
    dst_table = torch.zeros(-(-len(dests) * dst_words // 16) * 16, dtype=torch.int64)
    for j, chip in enumerate(dests):
        dst_table[j * dst_words : (j + 1) * dst_words] = torch.tensor(
            [chip["chip_id"], targets["mesh_id"]] + [v for bank in chip["banks"] for v in bank], dtype=torch.int64
        )
    dst_t = ttnn.from_torch(
        _int32(dst_table).reshape(1, 1, 1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    free += [table_t, dst_t]

    dst_nodes = [ttnn.FabricNodeId(ttnn.MeshId(int(targets["mesh_id"])), int(chip["chip_id"])) for chip in dests]
    ct = [nb, K, SLOT, len(sources), int(dst_table.numel()) * 4]
    ct += list(ttnn.TensorAccessorArgs(table_t).get_compile_time_args())
    common = [v for t, p in sources for v in (int(t.buffer_address()), p)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])
    size = K * SLOT + K * 16 + int(dst_table.numel()) * 4 + 64
    md = ttnn.MeshProgramDescriptor()
    for linear in range(chips):
        n = int(counts[linear])
        if not n:
            continue
        coord = ttnn.MeshCoordinate(*divmod(linear, cols))
        src_node = mesh.get_fabric_node_id(coord)
        mine = sorted(set(dst[sender == linear].tolist()))
        directions = {ttnn.get_eth_forwarding_direction(mesh, src_node, dst_nodes[j]) for j in mine}
        if len(directions) != 1 or None in directions:
            raise RuntimeError(f"prefill chip {linear}: decode chips {mine} leave by directions {directions}")
        links = list(ttnn.get_forwarding_link_indices(mesh, src_node, dst_nodes[mine[0]]))
        if not links:
            raise RuntimeError(f"no fabric route from {src_node} to {dst_nodes[mine[0]]}")
        prog = ttnn.ProgramDescriptor()
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=size, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, size)]
            )
        ]
        prog.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=ct,
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        ]
        fabric = ttnn.setup_routing_plane_connection(mesh, src_node, [dst_nodes[mine[0]]], [links[0]], prog, 0, CORE)
        own = common + [int(table_t.buffer_address()), n, int(dst_t.buffer_address())]
        prog.kernels[0].runtime_args[CORE.x][CORE.y] = own + list(fabric)
        md[ttnn.MeshCoordinateRange(coord, coord)] = prog
    t3 = time.perf_counter()
    times["plan"] = times.get("plan", 0.0) + t3 - t2

    ttnn.generic_op([*(t for t, _ in sources), table_t, dst_t], md)
    ttnn.synchronize_device(mesh)
    seconds = time.perf_counter() - t3
    page_of = torch.tensor([p for _, p in sources], dtype=torch.int64)
    nbytes = int(page_of[src].sum())
    for t in free:
        dealloc(t)
    return {"bytes": nbytes, "pages": int(sender.numel()), "seconds": seconds}
