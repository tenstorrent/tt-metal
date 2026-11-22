# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Part 2 of the cross-host KV write test (see common.py): the prefill side, rank 0 on host 40.

Listens for the receiver (decode) on TCP, takes its pool addresses and DRAM bank tables, and for every "write"
request copies seeded random blocks from its own DRAM into the receiver's pages: one core per sending chip, chip
(r, c) of mesh 0 writing chip (r, c) of mesh 1, one fabric unicast write a page, the destination NOC address computed
from the receiver's bank table (not this chip's, so a differently harvested peer is still addressed right). Nothing
is waited for on the receiver: "done" goes back as soon as this side's program has finished, which is the moment a
prefill server would send its TCP reply.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

import torch

import ttnn
from models.experimental.deepseek_v41_flash.tests.xhost_kv.common import (
    MESH_SHAPE,
    PORT,
    POOLS,
    SENDER_RANK,
    block_data,
    close_mesh,
    init_rank,
    log,
    open_mesh,
)
from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import recv_msg, send_msg

ROLE = "sender"
CORE = ttnn.CoreCoord(0, 0)
BATCH = 8  # pages read into L1 a NoC barrier

_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#if !defined(FABRIC_2D)
#error "the cross-host KV write needs a 2D fabric: a remote mesh is reached through its exit chip"
#endif
using namespace tt::tt_fabric::mesh::experimental;

constexpr uint32_t NB = get_compile_time_arg_val(0);     // the receiver chip's DRAM banks
constexpr uint32_t BATCH = get_compile_time_arg_val(1);  // pages read a NoC barrier
constexpr uint32_t SLOT = get_compile_time_arg_val(2);   // L1 bytes a staged page (64 B aligned)
constexpr auto kv_args = TensorAccessorArgs<3>();
constexpr auto key_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();

struct Banks {
    uint32_t x[NB], y[NB], off[NB];
};

// Every page of the n_blocks source blocks, in order, to page (blocks[b] * ppb + j) of the receiver's pool at dst_base.
template <typename Src, typename Sender>
FORCE_INLINE void send_pool(
    const Src& src, uint32_t page, uint32_t ppb, uint32_t dst_base, uint32_t n_blocks, uint32_t blocks_arg,
    const Banks& banks, Sender& sender, volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint8_t dst_dev,
    uint16_t dst_mesh, uint32_t buf) {
    const uint32_t total = n_blocks * ppb;
    for (uint32_t i0 = 0; i0 < total; i0 += BATCH) {
        const uint32_t nb = (total - i0) < BATCH ? (total - i0) : BATCH;
        for (uint32_t k = 0; k < nb; ++k) {
            noc_async_read(src.get_noc_addr(i0 + k), buf + k * SLOT, page);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < nb; ++k) {
            const uint32_t i = i0 + k;
            const uint32_t p = get_arg_val<uint32_t>(blocks_arg + i / ppb) * ppb + i % ppb;
            const uint32_t bank = p % NB;
            const uint64_t dst =
                get_noc_addr(banks.x[bank], banks.y[bank], banks.off[bank] + dst_base + (p / NB) * page);
            fabric_unicast_noc_unicast_write<Sender, false>(
                &sender, hdr, dst_dev, dst_mesh, buf + k * SLOT, page, tt::tt_fabric::NocUnicastCommandHeader{dst});
            noc_async_writes_flushed();  // the header and the staged page left L1 before either is rewritten
        }
    }
}

void kernel_main() {
    size_t a = 0;
    const uint32_t kv_src = get_arg_val<uint32_t>(a++);
    const uint32_t kv_dst = get_arg_val<uint32_t>(a++);
    const uint32_t kv_page = get_arg_val<uint32_t>(a++);
    const uint32_t kv_ppb = get_arg_val<uint32_t>(a++);
    const uint32_t key_src = get_arg_val<uint32_t>(a++);
    const uint32_t key_dst = get_arg_val<uint32_t>(a++);
    const uint32_t key_page = get_arg_val<uint32_t>(a++);
    const uint32_t key_ppb = get_arg_val<uint32_t>(a++);
    const uint32_t n_blocks = get_arg_val<uint32_t>(a++);
    Banks banks;
    for (uint32_t b = 0; b < NB; ++b) {
        banks.x[b] = get_arg_val<uint32_t>(a++);
        banks.y[b] = get_arg_val<uint32_t>(a++);
        banks.off[b] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t blocks_arg = a;
    a += n_blocks;

    tt::tt_fabric::RoutingPlaneConnectionManager conn;
    open_connections(conn, 1, a);
    auto& slot = conn.get(0);
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    fabric_set_unicast_route(hdr, slot.dst_dev_id, slot.dst_mesh_id);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;

    send_pool(TensorAccessor(kv_args, kv_src, kv_page), kv_page, kv_ppb, kv_dst, n_blocks, blocks_arg, banks,
              slot.sender, hdr, slot.dst_dev_id, slot.dst_mesh_id, buf);
    send_pool(TensorAccessor(key_args, key_src, key_page), key_page, key_ppb, key_dst, n_blocks, blocks_arg, banks,
              slot.sender, hdr, slot.dst_dev_id, slot.dst_mesh_id, buf);
    close_connections(conn);
}
"""


def sources(mesh, seed: int, round_: int, n_blocks: int) -> dict[str, ttnn.Tensor]:
    """Per pool, this round's blocks in this mesh's DRAM: chip (r, c) holds ``block_data(.., r * C + c, ..)``."""
    R, C = MESH_SHAPE
    out = {}
    for name, (width, layout, _, _) in POOLS.items():
        grid = torch.stack(
            [
                torch.stack([block_data(seed, round_, r * C + c, n_blocks, width)[:, 0] for c in range(C)], dim=1)
                for r in range(R)
            ]
        )  # [R, n, C, 64, w]
        out[name] = ttnn.from_torch(
            grid.reshape(R * n_blocks, C, *grid.shape[3:]),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=MESH_SHAPE),
        )
    return out


def build(mesh, peer: dict, src: dict[str, ttnn.Tensor], blocks: list[int]) -> ttnn.MeshProgramDescriptor:
    """One program a sending chip: its core copies both pools' blocks to the same-coordinate receiver chip."""
    slot = -(-max(p for _, _, p, _ in POOLS.values()) // 64) * 64
    ct_tail = list(ttnn.TensorAccessorArgs(src["kv"]).get_compile_time_args())
    ct_tail += list(ttnn.TensorAccessorArgs(src["keys"]).get_compile_time_args())
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(CORE, CORE)])
    md = ttnn.MeshProgramDescriptor()
    for chip in peer["chips"]:
        coord = ttnn.MeshCoordinate(*chip["coord"])
        banks = chip["banks"]
        own = []
        for name in ("kv", "keys"):
            pool = peer["pools"][name]
            own += [int(src[name].buffer_address()), pool["addr"], pool["page_bytes"], pool["pages_per_block"]]
        own += [len(blocks)] + [v for bank in banks for v in bank] + list(blocks)
        prog = ttnn.ProgramDescriptor()
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=BATCH * slot + 64,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, BATCH * slot + 64)],
            )
        ]
        prog.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=[len(banks), BATCH, slot] + ct_tail,
                runtime_args=ttnn.RuntimeArgs(),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        ]
        src_node = mesh.get_fabric_node_id(coord)
        dst_node = ttnn.FabricNodeId(ttnn.MeshId(peer["mesh_id"]), chip["chip_id"])
        links = list(ttnn.get_forwarding_link_indices(mesh, src_node, dst_node))
        fabric = ttnn.setup_routing_plane_connection(mesh, src_node, [dst_node], [links[0]], prog, 0, CORE)
        prog.kernels[0].runtime_args[CORE.x][CORE.y] = own + list(fabric)
        assert list(prog.kernels[0].runtime_args[CORE.x][CORE.y]) == own + list(fabric), "runtime args not set in place"
        md[ttnn.MeshCoordinateRange(coord, coord)] = prog
    return md


def check_routes(mesh, peer: dict) -> None:
    """Every sending chip has a fabric route to its receiver chip (a RELAXED inter-mesh connection can come up with
    no links at all, and then only a route lookup says so)."""
    by_direction: dict[str, list] = {}
    for chip in peer["chips"]:
        coord = ttnn.MeshCoordinate(*chip["coord"])
        src_node = mesh.get_fabric_node_id(coord)
        dst_node = ttnn.FabricNodeId(ttnn.MeshId(peer["mesh_id"]), chip["chip_id"])
        direction = ttnn.get_eth_forwarding_direction(mesh, src_node, dst_node)
        links = list(ttnn.get_forwarding_link_indices(mesh, src_node, dst_node))
        if direction is None or not links:
            sys.exit(f"{ROLE}: no fabric route from {src_node} to {dst_node} (direction {direction}, links {links})")
        name = ("E", "W", "N", "S", "Z")[direction] if direction < 5 else str(direction)
        by_direction.setdefault(name, []).append(
            f"{tuple(chip['coord'])}=D{src_node.chip_id}/dev{mesh.get_device_id(coord)}/{len(links)}link"
        )
    log(ROLE, f"fabric routes to all {len(peer['chips'])} receiver chips found; first hop by direction:")
    for name, chips in sorted(by_direction.items()):
        log(ROLE, f"  {name}: {' '.join(chips)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--fabric", default="FABRIC_2D_TORUS_XY", help="the same on both ranks")
    parser.add_argument("--payload", type=int, default=None, help="fabric packet payload bytes, the same on both ranks")
    args = parser.parse_args()

    init_rank(SENDER_RANK, ROLE)
    mesh = open_mesh(args.fabric, args.payload)
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    log(ROLE, f"mesh {tuple(mesh.shape)} open, {args.fabric}, packet payload {payload} B")
    listener = socket.create_server(("0.0.0.0", args.port))
    try:
        log(ROLE, f"waiting for the receiver on port {args.port}")
        sock, addr = listener.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        peer, _ = recv_msg(sock)
        assert peer["op"] == "connect", peer
        log(ROLE, f"receiver {addr[0]}: mesh {peer['mesh_id']}, {len(peer['chips'])} chip pair(s)")
        for name, pool in peer["pools"].items():
            if pool["page_bytes"] > payload:
                sys.exit(f"{ROLE}: {name} pages of {pool['page_bytes']} B exceed the {payload} B fabric payload")
        check_routes(mesh, peer)
        while True:
            req, _ = recv_msg(sock)
            if req["op"] == "bye":
                break
            assert req["op"] == "write", req
            round_, blocks = req["round"], req["blocks"]
            src = sources(mesh, peer["seed"], round_, len(blocks))
            md = build(mesh, peer, src, blocks)
            t0 = time.perf_counter()
            ttnn.generic_op([src["kv"], src["keys"]], md)
            ttnn.synchronize_device(mesh)
            seconds = time.perf_counter() - t0
            send_msg(sock, {"done": round_, "seconds": seconds})
            nbytes = len(peer["chips"]) * len(blocks) * sum(p * n for _, _, p, n in POOLS.values())
            log(
                ROLE,
                f"round {round_}: {len(blocks)} blocks x {len(peer['chips'])} chips, {nbytes / 2**20:.1f} MiB in "
                f"{seconds * 1e3:.1f} ms ({nbytes / seconds / 1e9:.2f} GB/s incl. launch)",
            )
            for t in src.values():
                ttnn.deallocate(t)
        sock.close()
        return 0
    finally:
        listener.close()
        close_mesh(mesh)


if __name__ == "__main__":
    sys.exit(main())
