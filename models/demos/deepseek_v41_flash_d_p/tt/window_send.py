# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WINDOW_KV_FORMAT=bfp8_tiles (P.18, owner 2026-10-30 A12): the window region's neighbour send.

Every chip holds its own window region (tt/attention.TtWindowAttention.region): [carry 128 | its own q_rows rows] as
bfloat8_b blocks of one tile row (32 rows) a page. Its own rows are its queries' positions; the carry is the 128
positions before them, which (q_rows >= 128) are the last 128 rows of the previous chip in sequence order
(chip-linear q = r TP + c, natural chunk order: tt/attention.WindowGeometry.query_offsets). One program per call:

    chip q < N - 1   its region pages [P0, P0 + 4) (its last 128 rows)  -> chip q + 1's region pages [0, 4)
    chip N - 1       its kept tail [0, 4) (the carry of chip 0: positions [start - 128, start))
                                                                         -> chip 0's region pages [0, 4); then its own
                     region pages [P0, P0 + 4) -> its kept tail (the next chunk's carry of chip 0)
    every chip       waits for the one arrival (the sender's last packet carries an atomic increment, issued after its
                     write landed) and takes it back (an atomic add of -1: a sender one call ahead keeps its increment)

One core per chip (logical (0, 0)), one fabric connection to the route's first hop (tt-metal's dimension-order routes,
torus wraps where the open fabric has them: ttnn.bringup.combine_rf.sparse_combine_l1.route), packets of whole 64 B
lines of a page (the fabric's payload size), one in flight. No ready fence: a sender writes a receiver's region of
the same layer only, and a whole chunk later at the earliest: in the model every layer's MoE exchange orders a chip's
next write after the receiver's attention read of the last one (the region is per layer). Built once per (mesh, page
geometry); dispatched prepared (ttnn.bringup.prepare_program / run_prepared, OV-23): only the region's and the tail's
addresses change between calls (io tensors [region, tail]).

WINDOW_KV_FORMAT=bf16_tiles (P.26): the same send on a bf16 ROW_MAJOR region (one 1 KiB row a page): the carry is
128 pages (``carry_pages``), read B pages (B x 1 KiB <= 16 KiB) a NoC barrier into L1 before their packets go out
(``BATCH``; 1 for the 18496 B bfloat8_b block pages: the P.18 loop as it was).

ATTN_FUSE=on (P.64, bf16_tiles regions only): ``WindowSend.fused`` adds the window write's local copies to the same
program, so the chip's own rows reach the region, the ring and the neighbour in one launch (was: ttnn.bringup.
slice_write into the region, up to 4 x (ttnn.slice + ttnn.bringup.slice_write) into the ring, then the send):

    copy kernel      (the second data-movement RISC of the send's core row, cores (0, 0) .. (NC - 1, 0)): rows of the
                     chip's own untilized rows ``rows`` [1, 1, q_rows, 512] -> region rows [128, 128 + q_rows) and
                     the ring pieces (src row, ring row, rows; per-call values), the concatenated rows split evenly
                     over the NC cores
    send             as above, but the pages sent (and the last chip's kept tail) are read from ``rows``' last 128
                     rows (the same bytes the region gets), so the send does not wait for the copy
"""

from __future__ import annotations

import ttnn

WINDOW_ROW0 = 128  # bf16 regions: the chip's own rows start at region row 128 (after the carry)
PAGES = 4  # 128 rows: the window's carry (bfloat8_b block pages)
BATCH_BYTES = 16384  # bf16_tiles (P.26): L1 bytes of the pages read before their packets go out

_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a page (a 32-row block)
constexpr uint32_t SEG = get_compile_time_arg_val(1);   // bytes a packet (whole 64 B lines)
constexpr uint32_t NSEG = get_compile_time_arg_val(2);  // packets a page
constexpr uint32_t NP = get_compile_time_arg_val(3);    // pages sent
constexpr uint32_t P0 = get_compile_time_arg_val(4);    // the region's first page sent (its last NP pages)
constexpr uint32_t B = get_compile_time_arg_val(5);     // pages read a NoC barrier (BATCH; 1 when NSEG > 1)
constexpr uint32_t SLOT = (SEG + 63) & ~63u;            // L1 bytes a page's packet takes in the buffer
constexpr auto r_args = TensorAccessorArgs<6>();
constexpr auto t_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();

inline void route_to(volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr, uint16_t chip_id, uint16_t mesh_id) {
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the window send needs a 2D fabric (agent rule 4)"
#endif
}

void kernel_main() {
    size_t a = 0;
    const uint32_t region = get_arg_val<uint32_t>(a++);
    const uint32_t tail = get_arg_val<uint32_t>(a++);
    const uint32_t last = get_arg_val<uint32_t>(a++);  // 1: send the kept tail, then keep my own last pages
    const uint32_t sem = get_arg_val<uint32_t>(a++);   // the arrival counter (same address on every chip)
    const uint32_t px = get_arg_val<uint32_t>(a++);    // its core (NoC coordinates, the same on every chip)
    const uint32_t py = get_arg_val<uint32_t>(a++);
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const auto R = TensorAccessor(r_args, region, PAGE);
    const auto T = TensorAccessor(t_args, tail, PAGE);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    cb_reserve_back(0, 1);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;  // 64 B aligned, as the DRAM pages
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    route_to(hdr, chip_id, mesh_id);
    conn.open();
    const uint64_t arrival = get_noc_addr(px, py, sem);
    constexpr uint32_t total = NP * NSEG;
    uint32_t k = 0;
    for (uint32_t p0 = 0; p0 < NP; p0 += B) {
        const uint32_t nb = (NP - p0) < B ? (NP - p0) : B;
        for (uint32_t s = 0; s < NSEG; ++s) {
            const uint32_t off = s * SEG;
            const uint32_t bytes = (PAGE - off) < SEG ? (PAGE - off) : SEG;
            for (uint32_t i = 0; i < nb; ++i) {
                const uint32_t p = p0 + i;
                const uint64_t src = last ? T.get_noc_addr(p, off) : R.get_noc_addr(P0 + p, off);
                noc_async_read(src, buf + i * SLOT, bytes);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < nb; ++i) {
                const uint64_t dst = R.get_noc_addr(p0 + i, off);  // the receiver's region (same buffer address)
                if (++k == total) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{dst, arrival, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(buf + i * SLOT, bytes, reinterpret_cast<uint32_t>(hdr));
                noc_async_writes_flushed();  // the header left L1 before the next packet rewrites it
            }
        }
    }
    conn.close();
    if (last) {  // my last NP pages are chip 0's carry of the next chunk
        for (uint32_t p0 = 0; p0 < NP; p0 += B) {
            const uint32_t nb = (NP - p0) < B ? (NP - p0) : B;
            for (uint32_t s = 0; s < NSEG; ++s) {
                const uint32_t off = s * SEG;
                const uint32_t bytes = (PAGE - off) < SEG ? (PAGE - off) : SEG;
                for (uint32_t i = 0; i < nb; ++i) {
                    noc_async_read(R.get_noc_addr(P0 + p0 + i, off), buf + i * SLOT, bytes);
                }
                noc_async_read_barrier();
                for (uint32_t i = 0; i < nb; ++i) {
                    noc_async_write(buf + i * SLOT, T.get_noc_addr(p0 + i, off), bytes);
                }
                noc_async_write_barrier();
            }
        }
    }
    volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    noc_semaphore_wait_min(arrived, 1);
    noc_semaphore_inc(get_noc_addr(sem), 0u - 1u);
    noc_async_atomic_barrier();
}
"""

# ATTN_FUSE=on (P.64): the send of _KERNEL reading the sent pages (and the last chip's kept pages) from the chip's own
# untilized rows (io tensor ``rows``, its last NP rows from row Q0) instead of the region's last NP pages
_KERNEL_FUSED = (
    _KERNEL.replace(
        "constexpr auto t_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();",
        "constexpr auto t_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();\n"
        "constexpr auto o_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();\n"
        "constexpr uint32_t Q0 = get_compile_time_arg_val(o_args.next_compile_time_args_offset());",
    )
    .replace(
        "    const uint32_t tail = get_arg_val<uint32_t>(a++);\n",
        "    const uint32_t tail = get_arg_val<uint32_t>(a++);\n    const uint32_t rows = get_arg_val<uint32_t>(a++);\n",
    )
    .replace(
        "    const auto T = TensorAccessor(t_args, tail, PAGE);\n",
        "    const auto T = TensorAccessor(t_args, tail, PAGE);\n    const auto O = TensorAccessor(o_args, rows, PAGE);\n",
    )
    .replace(": R.get_noc_addr(P0 + p, off);", ": O.get_noc_addr(Q0 + p, off);")
    .replace(
        "noc_async_read(R.get_noc_addr(P0 + p0 + i, off), buf + i * SLOT, bytes);",
        "noc_async_read(O.get_noc_addr(Q0 + p0 + i, off), buf + i * SLOT, bytes);",
    )
)
assert _KERNEL_FUSED.count("O.get_noc_addr(Q0") == 2, "the fused send kernel's edits did not all apply"

# ATTN_FUSE=on (P.64): the window write's local row copies (one 1 KiB row a page), on the send's core row
_COPY_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a row (a page)
constexpr uint32_t B = get_compile_time_arg_val(1);     // rows read a NoC barrier
constexpr uint32_t NC = get_compile_time_arg_val(2);    // copy cores
constexpr uint32_t NSEG = get_compile_time_arg_val(3);  // segments: the region's, then the ring pieces
constexpr auto o_args = TensorAccessorArgs<4>();
constexpr auto r_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
constexpr auto g_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t region = get_arg_val<uint32_t>(1);
    const uint32_t ring = get_arg_val<uint32_t>(2);
    const uint32_t k = get_arg_val<uint32_t>(3);  // this core's index of NC
    constexpr uint32_t S0 = 4;                     // segment s: (src row, dst row, rows) at S0 + 3 s
    const auto O = TensorAccessor(o_args, rows, PAGE);
    const auto R = TensorAccessor(r_args, region, PAGE);
    const auto G = TensorAccessor(g_args, ring, PAGE);
    uint32_t total = 0;
    for (uint32_t s = 0; s < NSEG; ++s) {
        total += get_arg_val<uint32_t>(S0 + 3 * s + 2);
    }
    const uint32_t lo = total * k / NC;
    const uint32_t hi = total * (k + 1) / NC;
    const uint32_t buf = (get_write_ptr(1) + 63) & ~63u;  // 64 B aligned, as the DRAM pages
    uint32_t g = 0;  // first row of segment s in the concatenated rows
    for (uint32_t s = 0; s < NSEG; ++s) {
        const uint32_t src0 = get_arg_val<uint32_t>(S0 + 3 * s);
        const uint32_t dst0 = get_arg_val<uint32_t>(S0 + 3 * s + 1);
        const uint32_t n = get_arg_val<uint32_t>(S0 + 3 * s + 2);
        const uint32_t b = lo > g ? lo : g;
        const uint32_t e = hi < g + n ? hi : g + n;
        for (uint32_t r = b; r < e; r += B) {
            const uint32_t nb = (e - r) < B ? (e - r) : B;
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read(O.get_noc_addr(src0 + r - g + i), buf + i * PAGE, PAGE);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < nb; ++i) {
                const uint32_t d = dst0 + r - g + i;
                noc_async_write(buf + i * PAGE, s == 0 ? R.get_noc_addr(d) : G.get_noc_addr(d), PAGE);
            }
            noc_async_write_barrier();
        }
        g += n;
    }
}
"""

RING_PIECES = 4  # ATTN_FUSE=on: ring pieces a call (192 rows over stripes of >= 128 rows: <= 3, one split by the wrap)
# ATTN_FUSE=on: copy cores a chip (one row from (0, 0)); P.64 probe on 8x4 (tests/test_window_write_fused.py), 1 / 2 /
# 4 / 8 cores: host 74 / 81 / 96 / 123 us a call, device 54 / 62 / 75 / 54 us (the send dominates)
COPY_CORES = 1
COPY_BATCH = 8  # ATTN_FUSE=on: rows a copy core reads a NoC barrier (8 KiB of L1)


def _torus(mesh) -> tuple[bool, bool]:
    """(wrap X, wrap Y) of the open fabric (as the MoE block's routing stream reads it)."""
    cfg = ttnn.get_fabric_config()
    fc = ttnn.FabricConfig
    tx = {getattr(fc, n) for n in ("FABRIC_2D_TORUS_XY", "FABRIC_2D_TORUS_X") if hasattr(fc, n)}
    ty = {getattr(fc, n) for n in ("FABRIC_2D_TORUS_XY", "FABRIC_2D_TORUS_Y") if hasattr(fc, n)}
    return cfg in tx, cfg in ty


def segments(page_bytes: int, payload: int) -> tuple[int, int]:
    """(bytes a packet, packets a page): whole 64 B lines, at most the fabric's payload."""
    line = 64
    cap = payload // line * line
    assert cap >= line, f"fabric payload {payload} B"
    n = -(-page_bytes // cap)
    seg = -(-(-(-page_bytes // n)) // line) * line
    assert seg <= cap and (n - 1) * seg < page_bytes, (seg, n, page_bytes)
    return seg, n


class WindowSend:
    """The neighbour send of every window layer on one mesh (module docstring). ``__call__(region, tail)``: region
    [1, 1, pages, page_bytes] uint8 ROW_MAJOR DRAM (the layer's per-chip region), tail [1, 1, carry_pages,
    page_bytes] (the slot's kept tail of that layer, read and written on the last chip only). bf16_tiles (P.26): a
    bf16 region [1, 1, rows, 512], one row a page (page_bytes 1024, carry_pages 128): shapes checked in bytes."""

    CORE = (0, 0)

    def __init__(self, mesh, pages: int, page_bytes: int, carry_pages: int = PAGES, copy_cores: int = COPY_CORES):
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan
        from ttnn.bringup.combine_rf.sparse_combine_l1 import route

        self.mesh, self.pages, self.page_bytes = mesh, pages, page_bytes
        R, C = tuple(mesh.shape)
        self.n = R * C
        assert self.n > 1, "the window send needs at least two chips"
        self.carry = carry_pages
        assert pages >= 2 * carry_pages, f"a region of {pages} pages: the carry needs a chip's own rows >= 128"
        self.p0 = pages - carry_pages
        self.seg, self.nseg = segments(page_bytes, int(ttnn.get_tt_fabric_max_payload_size_bytes()))
        slot = -(-self.seg // 64) * 64
        self.batch = max(1, min(carry_pages, BATCH_BYTES // slot)) if self.nseg == 1 else 1
        torus = _torus(mesh)
        chips, _, (wrap_x, wrap_y) = mesh_plan(mesh, 1, torus)
        cc = ttnn.CoreCoord(*self.CORE)
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)])
        # (load time: an L1_SMALL counter, left at 0 after every call)
        self.sem = ttnn.create_global_semaphore(mesh, crs, 0, ttnn.BufferType.L1_SMALL)
        sem_addr = int(ttnn.get_global_semaphore_address(self.sem))
        v = mesh.worker_core_from_logical_core(cc)
        node = lambda ch: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*ch))  # noqa: E731
        self._ct = [self.page_bytes, self.seg, self.nseg, carry_pages, self.p0, self.batch]
        self._chips = {}  # q: (program, runtime args without the io addresses, cores)
        self._links = {}  # q: ((src node, first hop node, link), the args before the connection's, L1 slot bytes)
        self.routes = {}
        for q in range(self.n):
            src, dst = divmod(q, C), divmod((q + 1) % self.n, C)
            hops = route(src, dst, R, C, wrap_x, wrap_y, 0)
            assert hops, (src, dst)
            first = hops[0][1]
            port = chips[src][first]
            self.routes[src] = (dst, [d for _, d in hops])
            prog = ttnn.ProgramDescriptor()
            prog.cbs = [
                ttnn.CBDescriptor(
                    total_size=self.batch * slot + 64,
                    core_ranges=crs,
                    format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, self.batch * slot + 64)],
                )
            ]
            dn = node(dst)
            args = [0, 0, int(q == self.n - 1), sem_addr, v.x, v.y, int(dn.mesh_id), int(dn.chip_id)]
            link = (node(src), node(port["peer"]), port["links"][0])
            args += list(ttnn.setup_fabric_connection(mesh, *link, prog, cc))
            self._chips[q] = (prog, args, crs)
            self._links[q] = (link, args[:8], slot)  # (ATTN_FUSE=on: the fused program sets up its own connection)
        self._pp = None
        self._fpp = None  # ATTN_FUSE=on (P.64): the fused window write, prepared at its first call
        self.copy_cores = int(copy_cores)

    def _build(self, region, tail):
        """The MeshProgramDescriptor with the io addresses in the runtime args (once: prepared after)."""
        ct = self._ct + list(ttnn.TensorAccessorArgs(region).get_compile_time_args())
        ct += list(ttnn.TensorAccessorArgs(tail).get_compile_time_args())
        md = ttnn.MeshProgramDescriptor()
        cc = ttnn.CoreCoord(*self.CORE)
        R, C = tuple(self.mesh.shape)
        for q in range(self.n):
            r, c = divmod(q, C)
            rng = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))
            prog, args, crs = self._chips[q]
            rt = ttnn.RuntimeArgs()
            rt[cc.x][cc.y] = [int(region.buffer_address()), int(tail.buffer_address())] + args[2:]
            prog.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=crs,
                    compile_time_args=ct,
                    runtime_args=rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                )
            ]
            md[rng] = prog
        return md

    @staticmethod
    def _geom(t: ttnn.Tensor) -> list[int]:
        """[pages, bytes a page] of a ROW_MAJOR tensor (one last-dim row a page)."""
        return [int(t.shape[-2]), int(t.shape[-1]) * (2 if t.dtype == ttnn.bfloat16 else 1)]

    def _build_fused(self, region, tail, rows, ring):
        """ATTN_FUSE=on (P.64): the send reading ``rows`` + the copy kernel (module docstring), io [region, tail, rows,
        ring]; the ring pieces' runtime args are value slots (zeros here)."""
        q_rows = int(rows.shape[-2])
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        ct = self._ct + acc(region) + acc(tail) + acc(rows) + [q_rows - self.carry]
        cct = [self.page_bytes, COPY_BATCH, self.copy_cores, 1 + RING_PIECES] + acc(rows) + acc(region) + acc(ring)
        cc = ttnn.CoreCoord(*self.CORE)
        x0, y0 = self.CORE
        ccrs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x0 + self.copy_cores - 1, y0))])
        md = ttnn.MeshProgramDescriptor()
        R, C = tuple(self.mesh.shape)
        a = [int(t.buffer_address()) for t in (region, tail, rows, ring)]
        for q in range(self.n):
            r, c = divmod(q, C)
            rng = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))
            _, _, crs = self._chips[q]
            link, args, slot = self._links[q]
            fp = ttnn.ProgramDescriptor()
            fp.cbs = [
                ttnn.CBDescriptor(
                    total_size=self.batch * slot + 64,
                    core_ranges=crs,
                    format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, self.batch * slot + 64)],
                ),
                ttnn.CBDescriptor(
                    total_size=COPY_BATCH * self.page_bytes + 64,
                    core_ranges=ccrs,
                    format_descriptors=[ttnn.CBFormatDescriptor(1, ttnn.uint8, COPY_BATCH * self.page_bytes + 64)],
                ),
            ]
            args = list(args) + list(ttnn.setup_fabric_connection(self.mesh, *link, fp, cc))
            rt = ttnn.RuntimeArgs()
            rt[cc.x][cc.y] = [a[0], a[1], a[2]] + args[2:]
            crt = ttnn.RuntimeArgs()
            for k in range(self.copy_cores):
                crt[x0 + k][y0] = [a[2], a[0], a[3], k, 0, WINDOW_ROW0, q_rows] + [0] * (3 * RING_PIECES)
            dm = lambda p, n: ttnn.DataMovementConfigDescriptor(processor=p, noc=n)  # noqa: E731
            fp.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_KERNEL_FUSED,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=crs,
                    compile_time_args=ct,
                    runtime_args=rt,
                    config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=_COPY_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ccrs,
                    compile_time_args=cct,
                    runtime_args=crt,
                    config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
                ),
            ]
            md[rng] = fp
        return md

    def fused(self, region: ttnn.Tensor, tail: ttnn.Tensor, rows: ttnn.Tensor, ring: ttnn.Tensor, pieces) -> None:
        """ATTN_FUSE=on (P.64, bf16 regions): ``rows`` [1, 1, q_rows, 512] bf16 ROW_MAJOR (the chip's own rows, pad
        rows zero) -> region rows [128, 128 + q_rows) and the ring pieces [(src row, ring row, rows), ..] (<=
        RING_PIECES; WindowRing.plan), and the neighbour send of rows' last 128 rows: one prepared program."""
        assert self._geom(region) == [self.pages, self.page_bytes], (region.shape, self.pages)
        assert self._geom(tail) == [self.carry, self.page_bytes], tail.shape
        assert self._geom(rows) == [self.pages - WINDOW_ROW0, self.page_bytes], (rows.shape, self.pages)
        assert self._geom(ring)[1] == self.page_bytes and self.carry == WINDOW_ROW0, (ring.shape, self.carry)
        assert len(pieces) <= RING_PIECES, pieces
        io = [region, tail, rows, ring]
        if self._fpp is None:
            slots = [(1, 7 + j) for j in range(3 * RING_PIECES)]
            self._fpp = ttnn.bringup.prepare_program(self._build_fused(region, tail, rows, ring), io, slots)
            # per chip: the send's region / tail / rows, each copy core's rows / region / ring and its value slots
            want = self.n * (3 + 3 * self.copy_cores + 3 * RING_PIECES * self.copy_cores)
            assert (
                self._fpp.patch_count == want
            ), f"window write: {self._fpp.patch_count} patched slots, {want} expected"
        vals = []
        for src, dst, n in pieces:
            vals += [int(src), int(dst), int(n)]
        vals += [0] * (3 * RING_PIECES - len(vals))
        ttnn.bringup.run_prepared(self._fpp, io, vals)

    def __call__(self, region: ttnn.Tensor, tail: ttnn.Tensor) -> None:
        assert self._geom(region) == [self.pages, self.page_bytes], (region.shape, self.pages)
        assert self._geom(tail) == [self.carry, self.page_bytes], tail.shape
        io = [region, tail]
        if self._pp is None:
            self._pp = ttnn.bringup.prepare_program(self._build(region, tail), io)
            # one region slot and one tail slot a chip: anything else matched a frozen value
            assert self._pp.patch_count == 2 * self.n, f"window send: {self._pp.patch_count} patched slots"
        # (looked up on ttnn.bringup at each call: the bring-up capture wraps run_prepared, F201)
        ttnn.bringup.run_prepared(self._pp, io)
