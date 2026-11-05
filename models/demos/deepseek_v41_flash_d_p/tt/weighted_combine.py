# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The routed experts (P.8; the only path since FIX.9): the flat expert and their weighted combine in ONE program per chip, on the
model's layout (tokens replicated over the mesh row, dispatch groups = mesh columns, dispatch and combine along axis 0).

Marko Staletovic's wiring (ttnn/ttnn/bringup/combine_rf/examples/scratch_flat_combine_rf.py) with the combine_rf
column options (combine_planner_prepare / sparse_combine_rf_prepare axis=0): per chip (r, c) and layer

    q(x) rows of the chip's mesh row  -> all_gather over axis 0 -> xg [R S, H] bf16 ROW_MAJOR   (the column's tokens:
                                                                 the flat expert reads its rows by token index)
    top-k ids (uint16) / weights (bf16) of the column's tokens -> all_gather over axis 0 -> one page each
    planner (one core per pass, or     counts / regions [1, Eg] (the column's experts, ids local to the column), the
    COMBINE_PLANNER=multi: up to 32    token index, the y writers' L1 hand-off table, the combine's per-call counts,
    cores per pass, P.11)              the det order table (both planners: bit-identical tables)
    arena (per call, L1 on every core: the flat expert's buffers and the combine's state / rings / accumulators),
    its combine state and the writers' words zeroed, a tiny all_gather over axis 0 (cross-chip barrier: no chip of
    the column writes into another's arena before that one is zeroed)
    per pass p: ttnn.bringup.flat_routed_expert_with_program(flat expert of this layer's experts, x = xg, token
                index, HiFi4 / fp32 DEST, weights_nk) + sparse_combine_rf (axis 0, L1 hand-off, device plan): each
                expert row goes from the y writers straight into the combine's L1, to its token's home chip in the
                column, is multiplied there by its bf16 routing weight (HiFi4, fp32 DEST) and added to the token's fp32
                sum (packer L1 accumulation); rows of tokens [p S_pass, (p + 1) S_pass) of every home
    out [S, H] fp32 (bf16 under CCL_IMPL=fabric_bf16) ROW_MAJOR -> TILE: this column group's partial (the caller adds
    the shared expert's and reduce-scatters over axis 1, tt/ccl.py:sum_cols)

Why passes: the accumulators hold every home token's fp32 row in L1 (5120 x 4 B = 20 KB a token); the free cells
the flat expert leaves (2x2 p300c, 11 x 10 grid: 10 cells, 1 port + 9 accumulators) hold ~558 tokens, a chip's
home tokens are S = chunk / R (1024 at 2048 on 2x2, 4096 on 4x4): P = the fewest passes whose S / P fit
(sparse_combine_rf.accumulator_bytes), each pass one flat call over its tokens' rows (the weights stream once a pass).
One pass of 1024 tokens needs >= 17 accumulators (1322 KB each at ring depth 1), i.e. >= 18 free cells and the
op's A <= 16 cap lifted: an L1 capacity limit, not a hang. The "255-tile" hang was the local ring's grant
batching, fixed in combine_rf (DS41-70); one pass runs up to 558 tokens a chip (310 fp32 tiles an accumulator).
COMBINE_ACC=bf16 (P.9, sparse_combine_rf acc_dtype): the sums in bf16 (10 KB a token: the packer's L1 accumulation
rounds each added row to bf16), so the 1024 home tokens fit the same 9 accumulators in ONE pass (114 tokens x 5 bf16
tiles = 1140 KB an accumulator): the expert weights stream once a layer. TtExperts asserts the single pass for bf16.

Rows are one fabric packet each (10240 B + a 32 B trailer): the mesh opens with a 10880 B fabric payload
(bringup/hooks.py:device_params). Everything that depends only on shapes is built once
per (mesh, chunk); the descriptors that bake the per-call arena's address once per address. No host work per call
beyond program-cache bookkeeping.
"""

from __future__ import annotations

from loguru import logger

import ttnn

from . import ccl
from .ccl import dealloc
from .common import DRAM, SP_AXIS
from .cores import sub_cores, sub_cores_tiled, to_rm
from .settings import settings

TILE = 32
LINKS = 2  # fabric links of the column's port (one port core per direction drives both: ports_per_direction)
FABRIC_PAYLOAD = 10880  # B: one 10240 B expert row + its 32 B trailer per packet (Marko's V4.1 setting)
# the arena may grow past what the flat expert needs, for the accumulators (Marko's FC_ARENA_KB): up to the largest
# arena the flat op itself uses on this grid (bf16 x / h plan: 1356 KB per core, 2x2 p300c, l1_small 24 KB)
ARENA_MAX = 1356 * 1024
# COMBINE_PLANNER=multi (P.11): cores a pass's column multi-core planner asks for (it uses N ceil(S / CT) <= this, CT
# the 32-token chunk of a home's tokens: 2x2, 1024 home tokens: 32 cores of 64 tokens). Marko's K3 default (32).
PLANNER_CORES = 32
_STATES: dict = {}


def _ceil(a: int, b: int) -> int:
    return -(-a // b) * b


def _l1_sharded(mesh, cores, rows, width=16, dtype=ttnn.uint32):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cores])
    return (
        ttnn.Shape([len(cores) * rows, width]),
        dtype,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [rows, width], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )


# The cross-chip barrier after the arena's zeroing (no circular buffer but a 128 B header page: a CCL's CBs sized for the
# 10880 B packets reach past the arena's bottom on a full L1 - stock all_gather 142 KB, the fabric fork 214 KB vs the
# arena at 137 KB, layer 14): one core per chip increments a persistent L1_SMALL semaphore on every other chip of its
# column over the fabric, waits until every peer's increment landed, then takes them back (an atomic add of -peers:
# a peer one call ahead keeps its increment). A peer's increment is sent after its own zeroing (programs run in order).
_BARRIER = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
void kernel_main() {
    size_t a = 0;
    const uint32_t sem = get_arg_val<uint32_t>(a++), n_peer = get_arg_val<uint32_t>(a++);
    const uint16_t mesh = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t mx = get_arg_val<uint32_t>(a++), my = get_arg_val<uint32_t>(a++);
    uint32_t pchip[8], pconn[8];
    for (uint32_t i = 0; i < n_peer; ++i) {
        pchip[i] = get_arg_val<uint32_t>(a++);
        pconn[i] = get_arg_val<uint32_t>(a++);
    }
    cb_reserve_back(0, 1);
    const uint32_t hdr_addr = get_write_ptr(0);
    auto c0 = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
#if __NCONN__ > 1
    auto c1 = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    c1.open();
#endif
    c0.open();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_addr);
    for (uint32_t i = 0; i < n_peer; ++i) {
        hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{get_noc_addr(mx, my, sem), 1, true});
        (void)fabric_set_unicast_route(hdr, static_cast<uint16_t>(pchip[i]), mesh);
#if __NCONN__ > 1
        auto& c = pconn[i] ? c1 : c0;
#else
        auto& c = c0;
        (void)pconn;
#endif
        c.wait_for_empty_write_slot();
        c.send_payload_flush_blocking_from_address(hdr_addr, sizeof(PACKET_HEADER_TYPE));
    }
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    while (*s < n_peer) {
        invalidate_l1_cache();
    }
    noc_semaphore_inc(get_noc_addr(mx, my, sem), static_cast<uint32_t>(0u - n_peer));
    noc_async_atomic_barrier();
    c0.close();
#if __NCONN__ > 1
    c1.close();
#endif
}
"""


def _column_barrier(mesh, core) -> tuple:
    """(MeshProgramDescriptor, semaphore) of the column barrier on logical ``core`` of every chip (built once)."""
    from ttnn.bringup.combine_rf.combine_plan import axis_ports
    from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan
    from ttnn.bringup.combine_rf.sparse_combine_rf import first_hop

    R, C = tuple(mesh.shape)
    cc = ttnn.CoreCoord(*core)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(cc, cc)])
    sem = ttnn.create_global_semaphore(mesh, crs, 0, ttnn.BufferType.L1_SMALL)
    sem_addr = int(ttnn.get_global_semaphore_address(sem))
    chips, _, (wrap_x, wrap_y) = mesh_plan(mesh, 1)
    chips = axis_ports(chips, 0)
    node = lambda ch: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*ch))  # noqa: E731
    v = mesh.worker_core_from_logical_core(cc)
    desc = ttnn.MeshProgramDescriptor()
    for (r, c), ports in chips.items():
        dirs = sorted(ports)  # (the chip's Y directions: a connection each)
        # one connection per direction the chip has: a column end (1 direction on > 2 rows) must not open a second
        # connection to the same router channel (the padded duplicate hung the 8x4 column ends, DS41-118)
        nconn = len(dirs)
        prog = ttnn.ProgramDescriptor()
        prog.cbs = [
            ttnn.CBDescriptor(
                total_size=128, core_ranges=crs, format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, 128)]
            )
        ]
        peers = [(i, c) for i in range(R) if i != r]
        args = [sem_addr, len(peers), int(node((0, 0)).mesh_id), v.x, v.y]
        for pr in peers:
            d = first_hop((r, c), pr, R, C, wrap_x, wrap_y)
            args += [int(node(pr).chip_id), dirs.index(d)]
        for d in dirs:
            args += list(
                ttnn.setup_fabric_connection(node((r, c)), node(ports[d]["peer"]), ports[d]["links"][0], prog, cc)
            )
        rt = ttnn.RuntimeArgs()
        rt[cc.x][cc.y] = args
        prog.kernels = [
            ttnn.KernelDescriptor(
                kernel_source=_BARRIER.replace("__NCONN__", str(nconn)),
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=crs,
                compile_time_args=[],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                ),
            )
        ]
        desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = prog
    return desc, sem


class WeightedCombine:
    """What every layer's weighted experts share on one mesh for one chunk size (tables, planners, buffers, the
    combine's programs). ``plan``: the flat expert's plan (FlatRoutedExpert.plan, the same for every layer)."""

    def __init__(
        self,
        mesh,
        plan,
        *,
        H: int,
        I: int,
        epc: int,
        K: int,
        S: int,
        out_dtype,
        x_bf16: bool,
        h_bf16: bool,
        arena_bytes: int | None = None,
        passes: int | None = None,
        acc_dtype=None,
        planner: str = "one",
        planner_cores: int = PLANNER_CORES,
    ):
        """arena_bytes: the per-core arena (None: the flat expert's, grown for the accumulators up to ARENA_MAX; a
        test override). passes: the pass count (None: the fewest that fit; a test override, >= that). acc_dtype: the
        combine's running sums per home token in L1 (setting COMBINE_ACC, P.9): ttnn.float32 (None: 20 KB a token,
        2x2: 2 passes of 512) or ttnn.bfloat16 (10 KB a token: the 1024 home tokens in one pass). planner (setting
        COMBINE_PLANNER, P.11): "one" = each pass's planner on one core; "multi" = the column multi-core planner on
        up to planner_cores cores per pass (disjoint per pass: the passes' planners run in one program)."""
        from ttnn.bringup.combine_rf.combine_plan import axis_ports, combine_planner_prepare, port_luts
        from ttnn.bringup.combine_rf.sparse_all_gather_2d import mesh_plan, place_ports
        from ttnn.bringup.combine_rf.sparse_combine_rf import L1_MAX_WRITERS, accumulator_bytes

        R, C = tuple(mesh.shape)
        self.mesh, self.R, self.C, self.H, self.I, self.epc, self.K, self.S = mesh, R, C, H, I, epc, K, S
        self.Eg = R * epc  # the column's experts (ExpertMapping column-major: column c holds [c Eg, (c + 1) Eg))
        self.out_dtype, self.x_bf16, self.h_bf16 = out_dtype, x_bf16, h_bf16
        self.acc_dtype = ttnn.float32 if acc_dtype is None else acc_dtype
        acc_kw = dict(out_dtype=out_dtype, acc_dtype=self.acc_dtype)
        acc_name = "fp32" if self.acc_dtype == ttnn.float32 else "bf16"
        self.plan = plan
        grid = mesh.compute_with_storage_grid_size()
        self.grid = grid
        rd_key = "tensix_readers" if "tensix_readers" in plan else "readers"
        taken = {tuple(c) for k in ("arena_cores", "gu", "down", rd_key, "relays", "coords") for c in plan[k]}
        self.cells = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
        self.wcores = [tuple(c) for c in plan["down"]] + ([tuple(c) for c in plan["rdn"]] if plan["rdown"] else [])
        self.writers = len(self.wcores)
        # (the flat plan is capped at the hand-off's writers: experts.py max_y_writers; DS41 east DRAM column 7)
        assert self.writers <= L1_MAX_WRITERS, f"{self.writers} y writers > the L1 hand-off's {L1_MAX_WRITERS}"
        # the column's ports (Y, one core per direction driving both links) and the accumulators the cells leave
        links = LINKS if R > 1 else 1
        chips, _, _ = mesh_plan(mesh, links)
        pc = place_ports(mesh, axis_ports(chips, 0), self.cells)
        port_cells = {(v.x, v.y) for k, v in pc.items() if k[2] == 0}
        self.A = min(16, len([c for c in self.cells if c not in port_cells]))
        assert self.A >= 1, f"no accumulator cell left ({len(self.cells)} free cells, ports {sorted(port_cells)})"
        self.links = links
        # passes: the fewest whose home tokens' sums (fp32 or bf16 rows, acc_dtype) fit the accumulators in an arena
        # of at most ARENA_MAX (or the flat expert's own, if larger)
        flat_bytes = int(plan["arena_tiles"]) * 2048
        cap_bytes = max(flat_bytes, ARENA_MAX)
        P = 1
        while accumulator_bytes(R, self.A, H, -(-S // P), depth=1, **acc_kw) > cap_bytes:
            P += 1
            assert P <= S, "no pass size fits the arena"
        if P > 1:  # (L1 capacity: the accumulators one pass would need)
            a1 = next(
                (a for a in range(self.A, 65) if accumulator_bytes(R, a, H, S, depth=1, **acc_kw) <= cap_bytes),
                None,
            )
            logger.info(
                f"EXPERTS_WEIGHTED S={S}: {P} passes ({acc_name} sums: one pass needs >= {a1} accumulators of <= "
                f"{cap_bytes >> 10} KB, this grid leaves {self.A})"
            )
        if passes is not None:
            assert passes >= P, f"passes={passes}: {-(-S // passes)} home tokens a pass do not fit (fewest {P})"
            P = passes
        self.P = P
        self.S_pass = -(-S // P)
        # (rings: 4 slots per source; the local ring 8 slots, else what the arena leaves: >= 2)
        need = accumulator_bytes(R, self.A, H, self.S_pass, depth=4, local_slots=8, **acc_kw)
        arena_bytes = arena_bytes or max(flat_bytes, min(cap_bytes, _ceil(need, 2048)))
        # the arena: every role's buffers (~all of L1) on every core; per call, at one address
        self.a_rows = arena_bytes // 64
        all_cores = [(x, y) for y in range(grid.y) for x in range(grid.x)]
        self._arena_spec = _l1_sharded(mesh, all_cores, self.a_rows)
        # rows a chip's flat expert may get in a pass: every column token on this chip's experts (<= min(K, epc) each)
        self.rows_cap = _ceil(R * self.S_pass * min(K, epc) + epc * (TILE - 1), TILE)
        self.cap = max(256, _ceil(R * self.S_pass, TILE))  # rows one expert may get in a pass
        logger.info(
            f"EXPERTS_WEIGHTED S={S} passes={P} x {self.S_pass} acc={acc_name} accumulators={self.A} "
            f"port cells={sorted(port_cells)} "
            f"free cells={len(self.cells)} rows_cap={self.rows_cap} arena={arena_bytes >> 10} KB "
            f"flat arena={flat_bytes >> 10} KB acc={need >> 10} KB (ring depth 4, local ring 8) planner={planner}"
        )
        # per-pass planners (one core, column axis), outputs allocated once
        self.luts, self.nps = port_luts(mesh, links, per_direction=True, axis=0)
        Eg = self.Eg
        assert planner in ("one", "multi"), planner
        self.planner = planner
        ncore = grid.x * grid.y
        per = min(int(planner_cores), ncore // P)  # (multi: each pass's planner on its own cores)
        assert planner == "one" or per >= R, f"{P} passes x {R} planner cores do not fit the {ncore}-core grid"
        all_xy = [(i % grid.x, i // grid.x) for i in range(ncore)]
        self.planners = [
            combine_planner_prepare(
                mesh,
                S=self.S_pass,
                K=K,
                EPC=epc,
                A=self.A,
                L=links,
                port_luts=self.luts,
                n_ports=self.nps,
                rows_cap=self.rows_cap,
                axis=0,
                id_offset=lambda r, c: c * Eg,
                tokens=S,
                token_base=p * self.S_pass,
                ids_uint32=True,
                tile_in=True,  # (the gathered top-k tiles as they are: no untilize / reshape to one page)
                core=(p % grid.x, p // grid.x),  # (every pass's planner on its own core: one program, in parallel)
                cores=all_xy[p * per : (p + 1) * per] if planner == "multi" else None,
                # the deterministic combine's order table: every source chip's flat schedule (the plan's schedule
                # parameters, pin 1 and the op's default pin ratio 8), a token's rows summed in a fixed order
                det=dict(
                    rps=plan["sched_rps"], max_e=plan["sched_max_e"], gu_nreg=plan["sched_gu_nreg"], pin=1, pin_ratio=8
                ),
            )
            for p in range(P)
        ]
        if planner == "multi":
            M, CT = self.planners[0].mcc
            self.planner_desc = f"multi: {M} cores x {CT} tokens a pass"
        else:
            self.planner_desc = "one core a pass"
        self._plan_desc = {}  # (ids, wts) addresses -> the passes' planners merged into one program per chip
        # y (the flat expert's row buffer: unused with the L1 hand-off, the op checks its shape) and the combine output
        self.y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([self.rows_cap, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, DRAM
        )
        self.out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([P * self.S_pass, H]), out_dtype, ttnn.ROW_MAJOR_LAYOUT, mesh, DRAM
        )
        self.combines = {}  # (arena, words) addresses -> per pass (run, desc, l1_args), and the zero program
        self.seen = (
            set()
        )  # (EXPERTS_PROGRAM_KEY=addresses) the op's program-cache keys seen (a hit needs no descriptor)
        self.empty = ttnn.MeshProgramDescriptor()
        self.key0 = 9_100_000 + (id(self) % 100_000) * 8
        # the column barrier after the zeroing (built once; its semaphore persistent in L1_SMALL) and its io tensor
        self.tick = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, DRAM
        )
        self.barrier = _column_barrier(mesh, self.cells[-1]) if R > 1 else None

    # ------------------------------------------------------------------ per arena address
    def _combine_for(self, arena, words):
        from ttnn.bringup.combine_rf.l1_zero import l1_zero_program
        from ttnn.bringup.combine_rf.sparse_combine_rf import sparse_combine_rf_prepare

        key = (int(arena.buffer_address()), int(words.buffer_address()))
        hit = self.combines.get(key)
        if hit is not None:
            return hit
        chips = [(r, c) for r in range(self.R) for c in range(self.C)]
        runs = []
        for p, pl in enumerate(self.planners):
            run = sparse_combine_rf_prepare(
                self.mesh,
                self.S_pass,
                self.H,
                {ch: [] for ch in chips},  # (the device plan decides the rows; this only places the cores)
                num_links=self.links,
                n_acc=self.A,
                depth=4,
                cells=self.cells,
                arena=arena,
                writers=self.writers,
                l1=dict(writers=self.wcores, words=key[1]),
                devplan=dict(plan=pl.outs["plan"], table=pl.outs["table"], dord=pl.outs["dord"], top_k=self.K),
                deterministic=True,  # (fp32 sums in a fixed order per token: bit-identical calls)
                ports_per_direction=True,
                piece_max=2 * self.H,  # (one piece a row: the L1 hand-off)
                axis=0,
                out_dtype=self.out_dtype,
                out_row0=p * self.S_pass,
                zero_unseen=True,
                acc_dtype=self.acc_dtype,
                local_keep=2,  # (2 local slots kept: the arena split the pass sizing assumes, as before 856d97d5864)
            )
            assert len(run.cells["accs"]) == self.A, (run.cells, self.A)
            desc, l1_args = run.build_l1(self.y, self.out, arena)
            runs.append([run, desc, l1_args, arena])
        # zero: the combine's state (same cells and layout in every pass's program) and the writers' words
        zr = {c: [(key[0] + o, b, st, n) for o, b, st, n in rs] for c, rs in runs[0][0].zero_runs.items()}
        for c in self.wcores:
            zr.setdefault(c, []).append((key[1], 192 * 16 * 4, 0, 1))
        hit = self.combines[key] = (runs, l1_zero_program(self.mesh, zr), runs[0][0].collector)
        return hit

    # ------------------------------------------------------------------ per call
    def routing(self, idx: ttnn.Tensor, wts: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """idx [1, 1, s, K] (TILE, any integer dtype), wts [1, 1, s, K] fp32 (TILE) of the chip's mesh row -> the
        column's uint32 ids / bf16 weights [1, 1, R s, K] TILE (gathered over axis 0, home row-major; the planners
        read the tiles, tile_in). wts may already be bf16 (ROUTER_IMPL=fused, P.22: no typecast)."""
        w16 = (
            wts
            if wts.dtype == ttnn.bfloat16
            else ttnn.typecast(wts, ttnn.bfloat16, memory_config=DRAM, sub_core_grids=sub_cores_tiled(wts))
        )
        i32 = (
            idx
            if idx.dtype == ttnn.uint32
            else ttnn.typecast(idx, ttnn.uint32, memory_config=DRAM, sub_core_grids=sub_cores_tiled(idx))
        )
        out = []
        for t in (i32, w16):
            out.append(ttnn.all_gather(t, dim=2, cluster_axis=SP_AXIS, memory_config=DRAM) if self.R > 1 else t)
        for t, o in zip((i32, w16), out):
            if t is not o and t is not idx and t is not wts:  # (idx / wts: the caller's)
                dealloc(t)
        return out[0], out[1]

    def run_planners(self, ids: ttnn.Tensor, wts: ttnn.Tensor) -> list[dict]:
        """Every pass's planner in one program per chip (each on its own core); returns each pass's tables."""
        key = (int(ids.buffer_address()), int(wts.buffer_address()))
        desc = self._plan_desc.get(key)
        if desc is None:
            progs = [pl.programs(ids, wts) for pl in self.planners]
            desc = ttnn.MeshProgramDescriptor()
            for ch in progs[0]:
                pd = ttnn.ProgramDescriptor()
                pd.cbs = [cb for pg in progs for cb in pg[ch].cbs]
                pd.kernels = [k for pg in progs for k in pg[ch].kernels]
                pd.semaphores = [sm for pg in progs for sm in pg[ch].semaphores]  # (multi: per pass, disjoint cores)
                desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*ch), ttnn.MeshCoordinate(*ch))] = pd
            self._plan_desc = {key: desc}  # (the latest addresses only)
        ttnn.generic_op([ids, wts] + [t for pl in self.planners for t in pl.tensors], desc)
        return [pl.outs for pl in self.planners]

    def gather_x(self, qx: ttnn.Tensor) -> ttnn.Tensor:
        """q(x) [1, 1, s, H] bf16 TILE of the mesh row -> the column's rows [R s, H] bf16 ROW_MAJOR."""
        xr = to_rm(qx, memory_config=DRAM)
        out = None
        if self.R > 1 and settings.get("EXPERTS_XG_PINNED") == "on":  # one gather buffer for every call (see setting)
            if getattr(self, "_xg_out", None) is None:
                shape = list(xr.shape)
                shape[2] *= self.R
                self._xg_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), xr.dtype, xr.layout, self.mesh, DRAM)
                ccl._PINNED.add(self._xg_out.buffer_address())
            out = self._xg_out
        g = ccl.all_gather(xr, 2, SP_AXIS, out=out) if self.R > 1 else xr
        if g is not xr:
            dealloc(xr)
        return ttnn.reshape(g, [g.shape[-2], self.H], sub_core_grids=sub_cores(g))

    def __call__(self, fe, xg: ttnn.Tensor, ids: ttnn.Tensor, wts: ttnn.Tensor, fidelity: int) -> ttnn.Tensor:
        """One layer: fe = its FlatRoutedExpert (column-local expert ids), xg / ids / wts from gather_x / routing.
        Returns this column group's partial [1, 1, S, H] (out_dtype, TILE)."""
        mesh = self.mesh
        pls = self.run_planners(ids, wts)  # (before the arena: the planners' CBs need L1)
        # (the arena first, at the top of L1 as the flat op's own; the writers' and relays' words below it on their
        # cores: allocated first they push the arena down by their size on every core, into the fused program's static
        # circular buffers when the model holds other L1 tensors - seen on layer 14: CBs end 142464, arena at 140736)
        arena = ttnn.allocate_tensor_on_device(*self._arena_spec)
        words = ttnn.allocate_tensor_on_device(*_l1_sharded(mesh, self.wcores, 192))
        rl = [ttnn.CoreCoord(*c) for c in self.plan["relays"]]
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import crs_rects

        relay_words = ttnn.allocate_tensor_on_device(
            ttnn.Shape([len(rl) * TILE, TILE]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            mesh,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs_rects(rl), (TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR),
            ),
        )
        runs, zero_desc, coll = self._combine_for(arena, words)
        ttnn.generic_op([arena, words], zero_desc)
        if self.R > 1:  # (the barrier: every chip of the column zeroed its arena before any combine row lands)
            ttnn.generic_op([self.tick], self.barrier[0])
        a = lambda t: 0 if t is None else int(t.buffer_address())  # noqa: E731
        patch = settings.get("EXPERTS_PROGRAM_KEY") == "patch"
        for p, (run, desc, l1_args, desc_arena) in enumerate(runs):
            pl = pls[p]
            if patch:
                # (FIX.5: the op keys its program without addresses and patches them on a hit; the combine's own
                # runtime args come from its descriptor, copied in only when extra_addrs differ from the cached
                # program's: the descriptor is passed by reference, no copy; its arena CBs the op binds itself)
                extra = desc
                extra_addrs = [
                    a(arena),
                    a(words),
                    a(self.y),
                    a(self.out),
                    a(pl["plan"]),
                    a(pl["table"]),
                    a(pl.get("dord")),
                ]
            else:
                extra_addrs = []
                key = (
                    id(fe),
                    p,
                    a(xg),
                    a(arena),
                    a(words),
                    a(relay_words),
                    a(pl["counts"]),
                    a(pl["token_index"]),
                    fidelity,
                )
                if key in self.seen:
                    extra = self.empty
                else:
                    if desc_arena is not arena:  # (the CBs bind the arena's buffer: rebind to this call's)
                        desc = run.rebind(arena)
                        runs[p][1], runs[p][3] = desc, arena
                    extra = desc
                    self.seen.add(key)
            ttnn.bringup.flat_routed_expert_with_program(
                xg,
                pl["counts"],
                pl["regions"],
                fe.gidx,
                fe.w_gu,
                fe.w_d,
                reader_down_weights=fe.w_rd,
                done_words=fe.done,
                intermediate=self.I,
                max_tokens_per_expert=fe.m,
                arena=arena,
                y=self.y,
                extra=extra,
                extra_key=self.key0 + p,
                collector=coll["core"],
                counts_addr=coll["counts_addr"],
                writers=self.writers,
                activation=fe.act,
                pin=fe.pin,
                token_index=pl["token_index"],
                down_fp32=True,
                x_bf16=self.x_bf16,
                h_bf16=self.h_bf16,
                l1_args=l1_args,
                relay_words=relay_words,
                math_fidelity=fidelity,
                weights_nk=fe.weights_nk,
                layout=fe.layout,  # (the op re-plans per call: the plan fe's weights were laid out with)
                patch_addresses=patch,
                extra_addrs=extra_addrs,
                max_y_writers=fe.max_y_writers,
            )
        for t in (arena, relay_words, words):
            dealloc(t)
        o = self.out
        if o.shape[-2] != self.S:
            o = ttnn.slice(o, [0, 0], [self.S, self.H], memory_config=DRAM, sub_core_grids=sub_cores(o))
        ot = ttnn.to_layout(o, ttnn.TILE_LAYOUT, memory_config=DRAM, sub_core_grids=sub_cores(o))
        if o is not self.out:
            dealloc(o)
        return ttnn.reshape(ot, [1, 1, self.S, self.H], sub_core_grids=sub_cores(ot))


def weighted_combine(mesh, plan, **kw) -> WeightedCombine:
    """The shared state for (mesh, the flat expert's plan, chunk rows per chip, ...), built once. The plan is part of
    the key: it depends on the weights' format (bfp8 / bfp4 weights give another core layout, P.10), and a state built
    for another plan places the combine's cells on the flat expert's cores."""
    key = (id(mesh), repr(sorted(plan.items())), tuple(sorted((k, str(v)) for k, v in kw.items())))
    hit = _STATES.get(key)
    if hit is None or hit[0] is not mesh:
        hit = _STATES[key] = (mesh, WeightedCombine(mesh, plan, **kw))
    return hit[1]
