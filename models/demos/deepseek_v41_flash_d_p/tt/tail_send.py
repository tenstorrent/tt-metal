# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DECODER_SKIP=replay128 (P.31): the decoder tail's input, moved over the 2D fabric.

The last chunk's layers 0-19 leave their output on the chips' own rows (OWN_ROWS: chip q = r TP + c holds chunk rows
[q R, (q + 1) R), R = chunk / (SP TP); 160 on 8x4). The decoder tail (tt/model.py TtDS41Model.tail) runs those rows'
last <= 128 positions as a chunk of its own, tail_chunk = 1024 rows on 8x4: chip q holds its rows [q Rt, (q + 1) Rt),
Rt = 32, so the 128 rows sit on mesh row 0 (4 chips, one tile row each) and nothing is padded to 32 rows on 32 chips.

``TailSend(mesh, src_rows, dst_rows, width, dtype)`` moves the tail rows [a, a + n) of a [1, 1, R, width] TILE tensor
(interleaved DRAM, the main model's own rows) into a [1, 1, Rt, width] TILE tensor of the same dtype (the tail's own
rows; rows not written keep their values): piece p = rows a + 32 p .. + 32 (one tile row; a is 32-aligned, R and Rt
multiples of 32, so a piece lies on one source and one destination chip) goes from chip (a + 32 p) // R, tile row
((a + 32 p) % R) / 32, to chip (32 p) // Rt, tile row ((32 p) % Rt) / 32.

One program a (a, pieces) pattern, built once and dispatched prepared (ttnn.bringup.prepare_program / run_prepared,
OV-23): only the two tensors' addresses change between calls. Per chip, one core per first fabric hop its pieces take
(tt-metal's dimension-order routes, Y then X: ttnn.bringup.combine_rf.sparse_combine_l1.route; never two connections
to one router channel, DS41-118), and one core for its local pieces (source = destination chip: NoC copies). A remote
piece is written page by page in packets of whole 64 B lines (the fabric's payload size), its last packet carrying an
atomic increment of the receiver's arrival counter (core (0, 0), an L1_SMALL global semaphore); core (0, 0) of every
receiving chip waits for its pieces' arrivals and takes them back (an atomic add of -count). No ready fence: the
destination buffer is the tail's load-time input, read by the tail's first layer of the same chunk only, a whole
chunk before the next send writes it (as tt/window_send.py).
"""

from __future__ import annotations

import ttnn

from .window_send import _torus, segments

_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;

constexpr uint32_t PAGE = get_compile_time_arg_val(0);    // bytes a page (a tile)
constexpr uint32_t SEG = get_compile_time_arg_val(1);     // bytes a packet (whole 64 B lines)
constexpr uint32_t NSEG = get_compile_time_arg_val(2);    // packets a page
constexpr uint32_t FABRIC = get_compile_time_arg_val(3);  // 1: this core's pieces go over the fabric
constexpr auto i_args = TensorAccessorArgs<4>();
constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();

inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the tail send needs a 2D fabric (agent rule 4)"
#endif
}

void kernel_main() {
    size_t a = 0;
    const uint32_t src = get_arg_val<uint32_t>(a++);
    const uint32_t dst = get_arg_val<uint32_t>(a++);
    const uint32_t sem = get_arg_val<uint32_t>(a++);     // the arrival counter (same address on every chip)
    const uint32_t px = get_arg_val<uint32_t>(a++);      // its core (NoC coordinates, the same on every chip)
    const uint32_t py = get_arg_val<uint32_t>(a++);
    const uint32_t expect = get_arg_val<uint32_t>(a++);  // pieces arriving on this chip (core (0, 0) only)
    const uint32_t nsend = get_arg_val<uint32_t>(a++);
    const size_t sends = a;  // nsend x (first source page, first destination page, pages, mesh id, chip id)
    a += 5 * nsend;
    const auto I = TensorAccessor(i_args, src, PAGE);
    const auto O = TensorAccessor(o_args, dst, PAGE);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;  // 64 B aligned, as the DRAM pages
    if constexpr (FABRIC) {
        auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
        conn.open();
        const uint64_t arrival = get_noc_addr(px, py, sem);
        for (uint32_t j = 0; j < nsend; ++j) {
            const uint32_t s0 = get_arg_val<uint32_t>(sends + 5 * j);
            const uint32_t d0 = get_arg_val<uint32_t>(sends + 5 * j + 1);
            const uint32_t np = get_arg_val<uint32_t>(sends + 5 * j + 2);
            const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(sends + 5 * j + 3));
            const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(sends + 5 * j + 4));
            route_to(hdr, chip_id, mesh_id);
            const uint32_t total = np * NSEG;
            uint32_t k = 0;
            for (uint32_t p = 0; p < np; ++p) {
                for (uint32_t s = 0; s < NSEG; ++s) {
                    const uint32_t off = s * SEG;
                    const uint32_t bytes = (PAGE - off) < SEG ? (PAGE - off) : SEG;
                    noc_async_read(I.get_noc_addr(s0 + p, off), buf, bytes);
                    noc_async_read_barrier();
                    const uint64_t to = O.get_noc_addr(d0 + p, off);  // the receiver's buffer (same address)
                    if (++k == total) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{to, arrival, 1, true}, bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{to}, bytes);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(buf, bytes, reinterpret_cast<uint32_t>(hdr));
                    noc_async_writes_flushed();  // the payload and the header left L1 before they are rewritten
                }
            }
        }
        conn.close();
    } else {
        for (uint32_t j = 0; j < nsend; ++j) {  // local pieces: NoC copies on this chip
            const uint32_t s0 = get_arg_val<uint32_t>(sends + 5 * j);
            const uint32_t d0 = get_arg_val<uint32_t>(sends + 5 * j + 1);
            const uint32_t np = get_arg_val<uint32_t>(sends + 5 * j + 2);
            for (uint32_t p = 0; p < np; ++p) {
                noc_async_read(I.get_noc_addr(s0 + p, 0), buf, PAGE);
                noc_async_read_barrier();
                noc_async_write(buf, O.get_noc_addr(d0 + p, 0), PAGE);
                noc_async_write_barrier();
            }
        }
    }
    if (expect) {
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
        noc_semaphore_wait_min(arrived, expect);
        noc_semaphore_inc(get_noc_addr(sem), 0u - expect);
        noc_async_atomic_barrier();
    }
}
"""

_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}


class TailSend:
    """Rows [a, a + n) (n <= 4 tile rows' worth of chunk rows, a 32-aligned) of a [1, 1, src_rows, width] TILE tensor ->
    rows [0, n) of the tail layout, [1, 1, dst_rows, width] TILE (module docstring). ``__call__(src, dst, a, n)``."""

    def __init__(self, mesh, src_rows: int, dst_rows: int, width: int, dtype):
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan

        self.mesh, self.dtype = mesh, dtype
        assert dtype in _TILE_BYTES, f"tail send: {dtype} pages not supported"
        assert src_rows % 32 == 0 and dst_rows % 32 == 0, (src_rows, dst_rows)
        self.src_rows, self.dst_rows = int(src_rows), int(dst_rows)
        self.wt = -(-int(width) // 32)  # tiles a tile row (the padded width)
        self.page = _TILE_BYTES[dtype]
        self.seg, self.nseg = segments(self.page, int(ttnn.get_tt_fabric_max_payload_size_bytes()))
        self.R, self.C = tuple(mesh.shape)
        self.torus = _torus(mesh)
        self.chips, _, self.wraps = mesh_plan(mesh, 1, self.torus)
        cc = ttnn.CoreCoord(0, 0)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)])
        # (load time: an L1_SMALL counter on core (0, 0), left at 0 after every call)
        self.sem = ttnn.create_global_semaphore(mesh, crs, 0, ttnn.BufferType.L1_SMALL)
        self.sem_addr = int(ttnn.get_global_semaphore_address(self.sem))
        v = mesh.worker_core_from_logical_core(cc)
        self.pxy = (int(v.x), int(v.y))
        self._pp = {}  # (a, pieces) -> (prepared program, patched slots)

    def pieces(self, a: int, n: int) -> list[tuple[tuple[int, int], int, tuple[int, int], int]]:
        """[(source chip, its tile row, destination chip, its tile row)] of rows [a, a + n)."""
        assert a % 32 == 0 and 0 < n, (a, n)
        out = []
        for p in range(-(-n // 32)):
            q, u = divmod(a + 32 * p, self.src_rows)
            qd, ud = divmod(32 * p, self.dst_rows)
            assert q < self.R * self.C and qd < self.R * self.C, (a, n, q, qd)
            out.append((divmod(q, self.C), u // 32, divmod(qd, self.C), ud // 32))
        return out

    def _build(self, src, dst, a: int, n: int):
        from ttnn.bringup.combine_rf.sparse_combine_l1 import route

        node = lambda ch: self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*ch))  # noqa: E731
        work = {}  # chip -> {first hop direction or "local": [(src page, dst page, pages, dst chip)]}
        expect = {}
        for s, tr, d, trd in self.pieces(a, n):
            item = (tr * self.wt, trd * self.wt, self.wt, d)
            if s == d:
                work.setdefault(s, {}).setdefault("local", []).append(item)
                continue
            hops = route(s, d, self.R, self.C, self.wraps[0], self.wraps[1], 0)
            assert hops and hops[0][0] == s, (s, d, hops)
            work.setdefault(s, {}).setdefault(hops[0][1], []).append(item)
            expect[d] = expect.get(d, 0) + 1
        slot = -(-self.seg // 64) * 64
        ct0 = [self.page, self.seg, self.nseg]
        io_ct = list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
        io_ct += list(ttnn.TensorAccessorArgs(dst).get_compile_time_args())
        md = ttnn.MeshProgramDescriptor()
        cores = 0
        for r in range(self.R):
            for c in range(self.C):
                chip = (r, c)
                groups = list(work.get(chip, {}).items())
                if not groups:
                    groups = [("local", [])]  # (core (0, 0): only the wait for arrivals, or nothing)
                prog = ttnn.ProgramDescriptor()
                kernels, cbs = [], []
                for gi, (port_dir, items) in enumerate(groups):
                    cc = ttnn.CoreCoord(gi, 0)
                    crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)])
                    fabric = port_dir != "local"
                    cbs.append(
                        ttnn.CBDescriptor(
                            total_size=(slot if fabric else self.page) + 64,
                            core_ranges=crs,
                            format_descriptors=[
                                ttnn.CBFormatDescriptor(0, ttnn.uint8, (slot if fabric else self.page) + 64)
                            ],
                        )
                    )
                    args = [0, 0, self.sem_addr, self.pxy[0], self.pxy[1], expect.get(chip, 0) if gi == 0 else 0]
                    args.append(len(items))
                    for s0, d0, np_, d in items:
                        dn = node(d)
                        args += [s0, d0, np_, int(dn.mesh_id), int(dn.chip_id)]
                    if fabric:
                        port = self.chips[chip][port_dir]
                        args += list(
                            ttnn.setup_fabric_connection(node(chip), node(port["peer"]), port["links"][0], prog, cc)
                        )
                    rt = ttnn.RuntimeArgs()
                    rt[cc.x][cc.y] = [int(src.buffer_address()), int(dst.buffer_address())] + args[2:]
                    kernels.append(
                        ttnn.KernelDescriptor(
                            kernel_source=_KERNEL,
                            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                            core_ranges=crs,
                            compile_time_args=ct0 + [int(fabric)] + io_ct,
                            runtime_args=rt,
                            config=ttnn.DataMovementConfigDescriptor(
                                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                            ),
                        )
                    )
                    cores += 1
                prog.cbs = cbs
                prog.kernels = kernels
                rng = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))
                md[rng] = prog
        return md, cores

    def __call__(self, src: ttnn.Tensor, dst: ttnn.Tensor, a: int, n: int) -> None:
        assert src.layout == ttnn.TILE_LAYOUT and dst.layout == ttnn.TILE_LAYOUT, (src.layout, dst.layout)
        assert src.dtype == self.dtype and dst.dtype == self.dtype, (src.dtype, dst.dtype, self.dtype)
        assert int(src.padded_shape[-2]) == self.src_rows and int(dst.padded_shape[-2]) == self.dst_rows, (
            src.shape,
            dst.shape,
        )
        key = (int(a), -(-int(n) // 32))
        hit = self._pp.get(key)
        io = [src, dst]
        if hit is None:
            md, cores = self._build(src, dst, int(a), int(n))
            pp = ttnn.bringup.prepare_program(md, io)
            # one source and one destination slot a core: anything else matched a frozen value
            assert pp.patch_count == 2 * cores, f"tail send: {pp.patch_count} patched slots, {cores} cores"
            hit = self._pp[key] = pp
        # (looked up on ttnn.bringup at each call: the bring-up capture wraps run_prepared, F201)
        ttnn.bringup.run_prepared(hit, io)
