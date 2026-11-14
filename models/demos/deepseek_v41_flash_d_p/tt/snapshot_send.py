# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CHEAP_CCL=on (P.65, OV-45): the served window snapshot of one layer as ONE prepared program a call.

WINDOW_KV_KEEP=snapshot128 (P.42) writes, after each layer's window write, the 128-row served snapshot into every
chip's (slot, layer) slab: snapshot rows [32 b, 32 b + 32) (block b = 0..3) hold the positions of one 32-row ring
block, read from

    zero    no position < cap (cap < 128, or rows the ring does not hold): zeros
    local   a ring block every chip holds (WindowRing.own None): each chip copies its own ring rows
    owned   a ring block only chip o holds (write_own): chip o's ring rows, on every chip

The legacy path (CHEAP_CCL=legacy, tt/attention._snapshot_rows) gathers every chip's whole 192-row ring over the
mesh (two high_bw_all_gathers, TP then SP: ~6 MB into each chip), slices the owner's rows, concatenates them and
writes them with update_padded_kv_cache: 4-6 launches a layer. This program moves only the owner's rows, once per
chip, over a broadcast tree of 1-hop fabric unicasts:

    tree(o)    the parent of chip (r, c) != o = (r0, c0) is its neighbour one step toward o along X (its row) while
               c != c0, else along Y (o's column): o sends down its column and along its row, each chip of o's
               column along its row (torus wraps where the open fabric has them; ties as tt-metal's routing tables)
    home core  (logical (0, 0)): zero / local / own blocks into its slab; for every block it receives, waits for the
               block's arrival counter (a global semaphore, one a block), tells its port cores that relay the block
               (a program semaphore a block), then takes the arrivals back (an atomic add of -1: a sender one call
               ahead keeps its increment)
    port core  (one per direction this chip sends in: logical (1 + d, 0), its own fabric connection to that
               neighbour): per block it sends, the 32 rows (from its ring when it owns the block, else from its slab
               after the home core's signal) into the child's slab rows, packets of whole rows (consecutive slab rows
               in one DRAM page run in one packet), the block's last packet carrying the child's arrival increment
               (fused, after its write)

Programs are built per block pattern (the (kind, owner) of the 4 blocks; one pattern in steady state: the last chip
owns all 4) and dispatched prepared (ttnn.bringup.prepare_program / run_prepared, OV-23): io [ring, slab], per-call
values the slab's first page (its (slot, layer) batch) and the 4 ring rows. Data movement only: the slab's bytes equal
the legacy path's (tests/test_snapshot_send.py)."""

from __future__ import annotations

import ttnn

HOME = (0, 0)  # the home core (logical); port core of direction d at (1 + d, 0)
DIRS = ("Y+", "Y-", "X+", "X-")
BLOCKS = 4  # snapshot blocks of 32 rows (SNAP = 128)
BROWS = 32
ZERO, LOCAL, RECV = 0, 1, 2  # home core's kind of a block (LOCAL: a copy of its own ring rows: local or owned)
NONE, FROM_RING, FROM_SLAB = 0, 1, 2  # port core's role for a block

_HOME_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PAGE = get_compile_time_arg_val(0);   // bytes a row (a page of the ring and of the slab)
constexpr uint32_t NB = 4;                                // blocks
constexpr uint32_t BR = 32;                               // rows a block
constexpr auto g_args = TensorAccessorArgs<1>();
constexpr auto s_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

void kernel_main() {
    // [ring, slab, page0, rr0..rr3 | kind0..3, mask0..3, sem0..3, nports, (px, py) x nports]
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t slab = get_arg_val<uint32_t>(1);
    const uint32_t page0 = get_arg_val<uint32_t>(2);
    const auto G = TensorAccessor(g_args, ring, PAGE);
    const auto S = TensorAccessor(s_args, slab, PAGE);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;  // BR rows, 64 B aligned
    const uint32_t zbuf = buf + BR * PAGE;                // one zero row
    bool zeroed = false;
    for (uint32_t b = 0; b < NB; ++b) {
        const uint32_t kind = get_arg_val<uint32_t>(7 + b);
        if (kind == 0) {
            if (!zeroed) {
                volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zbuf);
                for (uint32_t i = 0; i < PAGE / 4; ++i) {
                    z[i] = 0;
                }
                zeroed = true;
            }
            for (uint32_t i = 0; i < BR; ++i) {
                noc_async_write(zbuf, S.get_noc_addr(page0 + b * BR + i), PAGE);
            }
        } else if (kind == 1) {
            const uint32_t rr = get_arg_val<uint32_t>(3 + b);
            for (uint32_t i = 0; i < BR; ++i) {
                noc_async_read(G.get_noc_addr(rr + i), buf + i * PAGE, PAGE);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < BR; ++i) {
                noc_async_write(buf + i * PAGE, S.get_noc_addr(page0 + b * BR + i), PAGE);
            }
            noc_async_write_barrier();  // (buf is reused by the next block)
        }
    }
    const uint32_t nports = get_arg_val<uint32_t>(19);
    for (uint32_t b = 0; b < NB; ++b) {
        if (get_arg_val<uint32_t>(7 + b) != 2) {
            continue;
        }
        volatile tt_l1_ptr uint32_t* arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(15 + b));
        noc_semaphore_wait_min(arrived, 1);
        const uint32_t mask = get_arg_val<uint32_t>(11 + b);
        for (uint32_t k = 0; k < nports; ++k) {
            if (mask & (1u << k)) {
                const uint32_t px = get_arg_val<uint32_t>(20 + 2 * k);
                const uint32_t py = get_arg_val<uint32_t>(21 + 2 * k);
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(b)), 1);
            }
        }
    }
    noc_async_write_barrier();
    for (uint32_t b = 0; b < NB; ++b) {
        if (get_arg_val<uint32_t>(7 + b) == 2) {
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(15 + b)), 0u - 1u);
        }
    }
    noc_async_atomic_barrier();
}
"""

_PORT_KERNEL = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;

constexpr uint32_t PAGE = get_compile_time_arg_val(0);  // bytes a row
constexpr uint32_t K = get_compile_time_arg_val(1);     // rows a packet when the slab rows are consecutive
constexpr uint32_t NB = 4;
constexpr uint32_t BR = 32;
constexpr auto g_args = TensorAccessorArgs<2>();
constexpr auto s_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

void kernel_main() {
    // [ring, slab, page0, rr0..rr3 | role0..3, sem0..3, hx, hy, mesh_id, chip_id, connection...]
    size_t a = 0;
    const uint32_t ring = get_arg_val<uint32_t>(a++);
    const uint32_t slab = get_arg_val<uint32_t>(a++);
    const uint32_t page0 = get_arg_val<uint32_t>(a++);
    uint32_t rr[NB];
    for (uint32_t b = 0; b < NB; ++b) {
        rr[b] = get_arg_val<uint32_t>(a++);
    }
    uint32_t role[NB], sem[NB];
    for (uint32_t b = 0; b < NB; ++b) {
        role[b] = get_arg_val<uint32_t>(a++);
    }
    for (uint32_t b = 0; b < NB; ++b) {
        sem[b] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t hx = get_arg_val<uint32_t>(a++);
    const uint32_t hy = get_arg_val<uint32_t>(a++);
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const auto G = TensorAccessor(g_args, ring, PAGE);
    const auto S = TensorAccessor(s_args, slab, PAGE);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the snapshot send needs a 2D fabric (agent rule 4)"
#endif
    conn.open();
    for (uint32_t b = 0; b < NB; ++b) {
        if (role[b] == 0) {
            continue;
        }
        if (role[b] == 2) {  // relay: the block has landed in my slab (the home core's signal)
            volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(b));
            noc_semaphore_wait_min(flag, 1);
        }
        const uint32_t p0 = page0 + b * BR;
        for (uint32_t i = 0; i < BR; ++i) {
            noc_async_read(role[b] == 1 ? G.get_noc_addr(rr[b] + i) : S.get_noc_addr(p0 + i), buf + i * PAGE, PAGE);
        }
        noc_async_read_barrier();
        const uint64_t arrival = get_noc_addr(hx, hy, sem[b]);
        for (uint32_t i = 0; i < BR; i += K) {
            const uint64_t dst = S.get_noc_addr(p0 + i);
            const bool run = (K > 1) && (S.get_noc_addr(p0 + i + K - 1) == dst + (K - 1) * PAGE);
            const uint32_t n = run ? K : 1;
            for (uint32_t j = 0; j < K; j += n) {
                const uint64_t d = run ? dst : S.get_noc_addr(p0 + i + j);
                const uint32_t bytes = n * PAGE;
                if (i + j + n == BR) {
                    hdr->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{d, arrival, 1, true}, bytes);
                } else {
                    hdr->to_noc_unicast_write(NocUnicastCommandHeader{d}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(buf + (i + j) * PAGE, bytes, reinterpret_cast<uint32_t>(hdr));
                noc_async_writes_flushed();  // the header left L1 before the next packet rewrites it
            }
        }
    }
    conn.close();
    for (uint32_t b = 0; b < NB; ++b) {
        if (role[b] == 2) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(b)), 0);
        }
    }
}
"""


def _step(n: int, a: int, b: int, wrap: bool) -> int:
    """+1 / -1: the step from a toward b on an axis of n (ring: the shorter way, a tie +1, as the fabric's tables)."""
    if wrap:
        f = (b - a) % n
        return 1 if f * 2 <= n else -1
    return 1 if b > a else -1


def tree(R: int, C: int, owner: tuple, wrap_x: bool, wrap_y: bool) -> dict:
    """{chip: parent} of the broadcast tree from ``owner`` (module docstring): X toward the owner's column first, then
    Y toward its row (so the owner's column is reached first, then every row from it)."""
    r0, c0 = owner
    par = {}
    for r in range(R):
        for c in range(C):
            if (r, c) == owner:
                continue
            if c != c0:
                par[(r, c)] = (r, (c + _step(C, c, c0, wrap_x)) % C)
            else:
                par[(r, c)] = ((r + _step(R, r, r0, wrap_y)) % R, c)
    return par


class SnapshotSend:
    """The window snapshot write of every layer on one mesh (module docstring). ``__call__(ring, slab, page0,
    blocks)``: ring [1, 1, 192, 512] bf16 ROW_MAJOR (the layer's ring; each chip its own), slab the served snapshot
    tensor [batches, 1, 128, 512] bf16 ROW_MAJOR (ND-sharded or interleaved), page0 = batch x 128 (its (slot, layer)
    first row), blocks = 4 x (kind, ring row): kind 'zero', 'local' or the owner's chip-linear index."""

    def __init__(self, mesh):
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan

        from .window_send import _torus

        self.mesh = mesh
        self.R, self.C = tuple(mesh.shape)
        self.n = self.R * self.C
        assert self.n > 1, "the snapshot send needs at least two chips"
        self.chips, self.nb, (self.wrap_x, self.wrap_y) = mesh_plan(mesh, 1, _torus(mesh))
        payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
        self.page = 1024
        k = max(1, min(BROWS, payload // self.page))
        while BROWS % k:
            k -= 1
        self.k = k  # rows a packet (a divisor of 32)
        hc = ttnn.CoreCoord(*HOME)
        hcrs = ttnn.CoreRangeSet([ttnn.CoreRange(hc, hc)])
        # (load time: one arrival counter a block on the home core, left at 0 after every call)
        self.sems = [ttnn.create_global_semaphore(mesh, hcrs, 0, ttnn.BufferType.L1_SMALL) for _ in range(BLOCKS)]
        self.sem_addr = [int(ttnn.get_global_semaphore_address(s)) for s in self.sems]
        v = mesh.worker_core_from_logical_core(hc)
        self.home_xy = (int(v.x), int(v.y))
        self.port_xy = []
        for d in range(len(DIRS)):
            w = mesh.worker_core_from_logical_core(ttnn.CoreCoord(HOME[0] + 1 + d, HOME[1]))
            self.port_xy.append((int(w.x), int(w.y)))
        self._pp = {}  # block pattern -> PreparedProgram
        self.trees = {}

    def _tree(self, o: int) -> dict:
        if o not in self.trees:
            self.trees[o] = tree(self.R, self.C, divmod(o, self.C), self.wrap_x, self.wrap_y)
        return self.trees[o]

    def _dir(self, a: tuple, b: tuple) -> int:
        """Index in DIRS of the port of chip a whose neighbour is b."""
        for d, p in self.chips[a].items():
            if tuple(p["peer"]) == tuple(b):
                return DIRS.index(d)
        raise AssertionError(f"snapshot send: {a} -> {b} is not a fabric neighbour")

    def plan(self, pattern: tuple) -> dict:
        """Per chip (r, c): home kinds [4], {port d: roles [4]}, the relay masks [4] (bit k: the k-th used port)."""
        out = {}
        for r in range(self.R):
            for c in range(self.C):
                out[(r, c)] = dict(kinds=[ZERO] * BLOCKS, ports={}, masks=[0] * BLOCKS)
        for b, kind in enumerate(pattern):
            if kind in ("zero", "local"):
                for ch in out.values():
                    ch["kinds"][b] = ZERO if kind == "zero" else LOCAL
                continue
            o = divmod(int(kind), self.C)
            par = self._tree(int(kind))
            for ch, v in out.items():
                v["kinds"][b] = LOCAL if ch == o else RECV
            for child, parent in par.items():
                d = self._dir(parent, child)
                roles = out[parent]["ports"].setdefault(d, [NONE] * BLOCKS)
                roles[b] = FROM_RING if parent == o else FROM_SLAB
        for v in out.values():
            ds = sorted(v["ports"])
            for b in range(BLOCKS):
                for k, d in enumerate(ds):
                    if v["ports"][d][b] == FROM_SLAB:
                        v["masks"][b] |= 1 << k
        return out

    def _build(self, pattern: tuple, ring: ttnn.Tensor, slab: ttnn.Tensor):
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        hct = [self.page] + acc(ring) + acc(slab)
        pct = [self.page, self.k] + acc(ring) + acc(slab)
        ra, sa = int(ring.buffer_address()), int(slab.buffer_address())
        io_head = [ra, sa, 0, 0, 0, 0, 0]
        plan = self.plan(pattern)
        node = lambda ch: self.mesh.get_fabric_node_id(ttnn.MeshCoordinate(*ch))  # noqa: E731
        dm = ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        )
        md = ttnn.MeshProgramDescriptor()
        hc = ttnn.CoreCoord(*HOME)
        n_ports = 0
        for (r, c), v in plan.items():
            ds = sorted(v["ports"])
            n_ports += len(ds)
            pcs = [ttnn.CoreCoord(HOME[0] + 1 + d, HOME[1]) for d in ds]
            cells = [hc] + pcs
            all_crs = ttnn.CoreRangeSet([ttnn.CoreRange(x, x) for x in cells])
            hcrs = ttnn.CoreRangeSet([ttnn.CoreRange(hc, hc)])
            prog = ttnn.ProgramDescriptor()
            cbs = [
                ttnn.CBDescriptor(
                    total_size=(BROWS + 1) * self.page + 64,
                    core_ranges=hcrs,
                    format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, (BROWS + 1) * self.page + 64)],
                )
            ]
            if pcs:
                pcrs = ttnn.CoreRangeSet([ttnn.CoreRange(x, x) for x in pcs])
                cbs.append(
                    ttnn.CBDescriptor(
                        total_size=BROWS * self.page + 64,
                        core_ranges=pcrs,
                        format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint8, BROWS * self.page + 64)],
                    )
                )
            prog.cbs = cbs
            prog.semaphores = [ttnn.SemaphoreDescriptor(id=b, core_ranges=all_crs, initial_value=0) for b in range(4)]
            hrt = ttnn.RuntimeArgs()
            hargs = io_head + list(v["kinds"]) + list(v["masks"]) + list(self.sem_addr) + [len(ds)]
            for d in ds:
                hargs += list(self.port_xy[d])
            hrt[hc.x][hc.y] = hargs
            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=_HOME_KERNEL,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=hcrs,
                    compile_time_args=hct,
                    runtime_args=hrt,
                    config=dm,
                )
            ]
            if pcs:
                prt = ttnn.RuntimeArgs()
                for d, pc in zip(ds, pcs):
                    port = self.chips[(r, c)][DIRS[d]]
                    peer = tuple(port["peer"])
                    pn = node(peer)
                    args = io_head + list(v["ports"][d]) + list(self.sem_addr)
                    args += [self.home_xy[0], self.home_xy[1], int(pn.mesh_id), int(pn.chip_id)]
                    args += list(ttnn.setup_fabric_connection(node((r, c)), pn, port["links"][0], prog, pc))
                    prt[pc.x][pc.y] = args
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=_PORT_KERNEL,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=pcrs,
                        compile_time_args=pct,
                        runtime_args=prt,
                        config=dm,
                    )
                )
            prog.kernels = kernels
            md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
        return md, n_ports

    def __call__(self, ring: ttnn.Tensor, slab: ttnn.Tensor, page0: int, blocks) -> None:
        assert len(blocks) == BLOCKS, blocks
        assert int(ring.shape[-1]) == 512 and ring.dtype == ttnn.bfloat16, (ring.shape, ring.dtype)
        assert int(slab.shape[-1]) == 512 and int(slab.shape[-2]) == BLOCKS * BROWS, slab.shape
        pattern = tuple(k if k in ("zero", "local") else int(k) for k, _ in blocks)
        io = [ring, slab]
        hit = self._pp.get(pattern)
        if hit is None:
            md, n_ports = self._build(pattern, ring, slab)
            kernels = (0, 1) if n_ports else (0,)
            pp = ttnn.bringup.prepare_program(md, io, [(k, i) for k in kernels for i in range(2, 7)])
            # every core's ring and slab args (io) and its page0 + 4 ring rows (values): a home core a chip and the
            # port cores; nothing else matched an io address
            want = 7 * (self.n + n_ports)
            assert pp.patch_count == want, f"snapshot send: {pp.patch_count} patched slots, {want} expected"
            hit = self._pp[pattern] = (pp, len(kernels))
        pp, nk = hit
        vals = [int(page0)] + [int(rr) for _, rr in blocks]
        ttnn.bringup.run_prepared(pp, io, vals * nk)
