# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM read micro-benchmark: should the flat expert's gate/up cores read their own weights instead of 16 readers?

Only the weight movement of one flat_routed_expert call (no compute, no x, no h, nothing consumed), on the cores and
byte volumes of the flat plan for DRB_MODEL (glm-53: H 6144 / I 2048, 8 experts per chip; kimi-k27: H 7168 / I 2048,
12; the overlap test's 8 x 1 scaling), bfp4 (576 B tiles). One program per variant, each timed by the real-time
profiler (median of DRB_ITERS calls):

  down          the down cores stream their down weights (one bank each, [8 x pcd]-tile blocks, 2 per barrier, NOC0;
                the readers' down columns, if any, spread over the down cores: same bytes)
  rd            the 16 readers stream the gate/up weights (2 per bank, 64-tile chunks, NOC0), not forwarded
  rd_fwd        rd + each reader forwards every chunk's 4 blocks to its 4 gate/up cores (NOC1): today's path
  dir_noc0      every gate/up core streams its own blocks (16 tiles, DRB_DBATCH per barrier) from its reader's bank,
                laid out contiguously per core, all on NOC0
  dir_noc1      the same, all on NOC1
  dir_side      the same, NOC0 for banks west of the grid's middle, NOC1 for the east ones (read data takes the short
                way back)
  (DRB_DBATCH: reads per barrier, DRB_DMULT: a core's contiguous blocks per read request)
  dir_strided   dir_side on today's layout (a core's blocks every 64 tiles, interleaved with its 3 neighbours')
  nr_noc0 / nr_noc1 / nr_side   direct reads in a layout without readers (row 0 free for combine): gate/up in the
                4 columns from each DRAM-optimal column, rows 1-8, each cell on its nearest bank; down on the rest
  rd_all / rd_all_fwd   the readers stream the down weights too (gate/up + a 16th of the down bytes each; _fwd
                forwards it all to its gate/up cores as a stand-in for the down cores): run without +down
and each gate/up variant again with the down streams in the same program (+down): the DRAM load of a real call.
Prints us per call, gate/up GB/s, total GB/s and % of 512 GB/s.
  scripts/run_safe_pytest.sh --run-all <this file> -s   (DRB_MODEL=kimi-k27, DRB_ITERS, DRB_DBATCH, DRB_VARIANTS)"""

import os
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.op_unit_tests import test_hybrid_routed_expert_combine as hyb
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants

HERE = os.path.relpath(os.path.dirname(os.path.abspath(__file__)), os.environ.get("TT_METAL_HOME", os.getcwd()))
STREAM, FWD = f"{HERE}/bench_stream.cpp", f"{HERE}/bench_fwd.cpp"
TILE = 576  # bfp4 tile bytes
KBLK = 8
MODEL = os.environ.get("DRB_MODEL", "glm-53")
ITERS = int(os.environ.get("DRB_ITERS", "10"))
DBATCH = int(os.environ.get("DRB_DBATCH", "2"))
DMULT = int(os.environ.get("DRB_DMULT", "1"))  # direct, contiguous layout: blocks per read request
DRAM_GBS = 512.0
GU_VARIANTS = ("rd", "rd_fwd", "rd_all", "rd_all_fwd", "dir_noc0", "dir_noc1", "dir_side", "dir_strided", "nr_noc0", "nr_noc1", "nr_side")


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])


def _cb(index, cores, page_bytes, pages):
    return ttnn.CBDescriptor(
        total_size=page_bytes * pages,
        core_ranges=_crs(cores),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=page_bytes)],
    )


def _kernel(src, cores, ct, rt, risc, noc):
    args = ttnn.RuntimeArgs()
    for (x, y), a in zip(cores, rt):
        args[x][y] = list(a)
    proc = ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1
    return ttnn.KernelDescriptor(
        kernel_source=src,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=_crs(cores),
        compile_time_args=list(ct),
        runtime_args=args,
        config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=ttnn.NOC.NOC_0 if noc == 0 else ttnn.NOC.NOC_1),
    )


class Layout:
    """The flat plan's cores and the bench's DRAM layout: per bank, the 2 readers' gate/up regions, then the down
    cores' regions (down core d in bank d % 8)."""

    def __init__(self, m):
        model = hyb._scaled_model((8, 1), MODEL)
        self.H, self.I = model.EMB_SIZE, model.MOE_INTERMEDIATE_SIZE
        self.E, _, _, max_per_expert = compute_constants(
            hyb._SEQ_LEN_PER_CHIP, model.NUM_ROUTED_EXPERTS, model.NUM_EXPERTS_PER_TOKEN, 8, 8, hyb._CAPACITY_FACTOR
        )
        cv = lambda v: [cv(x) for x in v] if isinstance(v, (list, tuple)) else ((v.x, v.y) if hasattr(v, "x") else v)  # noqa: E731
        p = {k: cv(v) for k, v in dict(
            ttnn._ttnn.operations.bringup.flat_routed_expert_plan(
                m, self.H, self.I, self.E, model.NUM_ROUTED_EXPERTS, max_per_expert
            )
        ).items()}
        self.p = p
        self.grid = m.compute_with_storage_grid_size()
        self.readers, self.gu, self.down = [tuple(c) for c in p["readers"]], [tuple(c) for c in p["gu"]], [
            tuple(c) for c in p["down"]
        ]
        assert p["nsg"] == 1 and len(self.readers) == 16, "one subgrid, 16 readers"
        self.R = p["rg"] * p["g"]  # gate/up cores per reader
        assert len(self.gu) == 16 * self.R, (len(self.gu), self.R)
        self.Ht, self.It = self.H // 32, self.I // 32
        self.nk = self.Ht // KBLK
        self.blk = KBLK * 2 * p["np"] * TILE  # one gate/up core's block of a chunk
        self.chunk = self.R * self.blk  # a reader's chunk
        self.rd_bytes = self.E * self.nk * self.chunk  # one reader's gate/up region
        self.gu_bytes = 16 * self.rd_bytes
        nd = len(self.down)
        self.pcd = [self.Ht // nd + (d < self.Ht % nd) for d in range(nd)]  # all Ht columns over the down cores
        self.dn_bytes = [self.E * self.It * c * TILE for c in self.pcd]
        self.dn_total = sum(self.dn_bytes)
        per_bank = 2 * self.rd_bytes + -(-nd // 8) * max(self.dn_bytes)
        rows = -(-per_bank // (32 * 2 * 32)) * 32  # bf16 rows of a 32-wide shard
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                (rows, 32),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        # row-major, so bank offset o is row o // 64 of the bank's 32 columns; each 64 B row says where it is (bank,
        # row in base 128: exact in bf16) for the delivery check
        self.rows = rows
        r = torch.arange(rows)
        pat = torch.zeros(rows, 8, 32)
        pat[:, :, 0] = torch.arange(8).float()
        pat[:, :, 1], pat[:, :, 2], pat[:, :, 3] = (r // 16384)[:, None], (r // 128 % 128)[:, None], (r % 128)[:, None]
        self.w = ttnn.from_torch(
            pat.reshape(rows, 256).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=m, memory_config=mc
        )
        self.base = self.w.buffer_address()
        self.land_slots = 4
        lmc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_crs(self.gu), (self.land_slots * self.blk // 64, 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.land = ttnn.from_torch(
            torch.full((len(self.gu) * self.land_slots * self.blk // 64, 32), -1.0).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=m,
            memory_config=lmc,
        )
        self.vxy = lambda c: (lambda v: (v.x << 16) | v.y)(m.worker_core_from_logical_core(ttnn.CoreCoord(*c)))
        bank_x = [self.readers[b][0] for b in range(8)]  # bank b's DRAM-optimal reader column
        self.east = [x >= self.grid.x // 2 for x in bank_x]
        # "nr": a layout without readers, combine's row 0 left free: gate/up in 4 columns from each DRAM-optimal
        # column (west 0.., east E..), rows 1-8; each cell reads the nearest bank (8 per bank); every other cell of
        # rows 1-9 except the plan's relays does down work
        e0 = min(x for x in bank_x if x >= self.grid.x // 2)
        self.nr_gu = [(x, y) for y in range(1, 9) for x in list(range(0, 4)) + list(range(e0, min(e0 + 4, self.grid.x)))]
        assert len(self.nr_gu) == 64, len(self.nr_gu)
        pos = [self.readers[b] for b in range(8)]
        pairs = sorted((abs(c[0] - pos[b][0]) + abs(c[1] - pos[b][1]), i, b) for i, c in enumerate(self.nr_gu) for b in range(8))
        self.nr_bank, cap = {}, [8] * 8
        for _, i, b in pairs:
            if i not in self.nr_bank and cap[b]:
                self.nr_bank[i], cap[b] = b, cap[b] - 1
        relays = {tuple(c) for c in p["relays"]}
        self.nr_down = [
            (x, y) for y in range(1, min(self.grid.y, 10)) for x in range(self.grid.x) if (x, y) not in set(self.nr_gu) | relays
        ]

    def down_kernels(self, nr=False):
        # one kernel + CB per distinct block width (pcd differs by one column across the down cores)
        down = self.nr_down if nr else self.down
        nd = len(down)
        pcds = [self.Ht // nd + (d < self.Ht % nd) for d in range(nd)]
        out, cbs = [], []
        for i, pcd in enumerate(sorted(set(pcds))):
            ds = [d for d in range(nd) if pcds[d] == pcd]
            cores, blk = [down[d] for d in ds], KBLK * pcd * TILE
            rt = [[self.base, d % 8, 2 * self.rd_bytes + (d // 8) * self.E * self.It * pcds[0] * TILE] for d in ds]
            out.append(_kernel(STREAM, cores, [2 + i, blk, self.E * self.It // KBLK, 2, 4, blk, 0], rt, 1, 0))
            cbs.append(_cb(2 + i, cores, blk, 4))
        return out, cbs

    def gu_kernels(self, v):
        n_rd = self.E * self.nk
        if v in ("rd", "rd_fwd", "rd_all", "rd_all_fwd"):
            fwd = v.endswith("_fwd")
            if v.startswith("rd_all"):  # the readers also stream a 16th of the down bytes each (no down cores reading)
                per = self.rd_bytes + -(-self.dn_total // 16 // self.chunk) * self.chunk
                n_rd = per // self.chunk
                rt = [[self.base, r % 8, (r // 8) * per] for r in range(16)]
            else:
                rt = [[self.base, r % 8, (r // 8) * self.rd_bytes] for r in range(16)]
            ks = [_kernel(STREAM, self.readers, [0, self.chunk, n_rd, 1, 2, self.chunk, int(fwd)], rt, 0, 0)]
            if fwd:
                land = self.land.buffer_address()
                frt = [[land] + [self.vxy(self.gu[r * self.R + j]) for j in range(self.R)] for r in range(16)]
                ks.append(_kernel(FWD, self.readers, [0, self.R, self.blk, n_rd, self.land_slots], frt, 1, 1))
            return ks, [_cb(0, self.readers, self.chunk, 2)]
        strided = v == "dir_strided"
        groups = {0: [], 1: []}
        if v.startswith("nr_"):  # the reader-free layout: core k of bank b reads quarter k % 8 of the bank's 2 regions
            seen = [0] * 8
            for i, c in enumerate(self.nr_gu):
                b = self.nr_bank[i]
                off, seen[b] = seen[b] * (self.rd_bytes // 4), seen[b] + 1
                groups[{"nr_noc0": 0, "nr_noc1": 1}.get(v, int(self.east[b]))].append((c, [self.base, b, off]))
        for i, c in enumerate(self.gu if not v.startswith("nr_") else []):
            r = i // self.R
            bank, j = r % 8, i % self.R
            off = (r // 8) * self.rd_bytes + (j * self.blk if strided else j * (self.rd_bytes // self.R))
            noc = {"dir_noc0": 0, "dir_noc1": 1}.get(v, int(self.east[bank]))
            groups[noc].append((c, [self.base, bank, off]))
        mult = 1 if strided else DMULT
        assert n_rd % mult == 0, (n_rd, mult)
        req = mult * self.blk
        stride = self.chunk if strided else req
        ks = [
            _kernel(STREAM, [c for c, _ in g], [1, req, n_rd // mult, DBATCH, 2 * DBATCH, stride, 0], [a for _, a in g], 0, noc)
            for noc, g in groups.items()
            if g
        ]
        return ks, [_cb(1, self.nr_gu if v.startswith("nr_") else self.gu, req, 2 * DBATCH)]


def _program(lay, gu_v, with_down):
    ks, cbs = lay.gu_kernels(gu_v) if gu_v else ([], [])
    if with_down:
        dk, dc = lay.down_kernels(nr=bool(gu_v) and gu_v.startswith("nr_"))
        ks, cbs = ks + dk, cbs + dc
    mesh_pd = ttnn.MeshProgramDescriptor()
    coord = ttnn.MeshCoordinate(0, 0)
    mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs)
    return mesh_pd


def _check_delivery(m, lay):
    """rd_fwd once, then every landing slot must be one whole block from the right bank and offset: the last
    land_slots chunks of each reader, block j at receiver j (proves the 36 KB reads and the forwards move all bytes)."""
    ttnn.generic_op([lay.w, lay.land], _program(lay, "rd_fwd", False))
    ttnn.synchronize_device(m)
    land = ttnn.to_torch(lay.land).float().reshape(len(lay.gu), lay.land_slots, lay.blk // 64, 32)
    src = (land[..., 0], land[..., 1] * 16384 + land[..., 2] * 128 + land[..., 3])  # bank, row of every landed row
    got = set()
    for k in range(len(lay.gu)):
        for sl in range(lay.land_slots):
            bank, row = src[0][k, sl], src[1][k, sl]
            assert bool((bank == bank[0]).all()), f"core {k} slot {sl}: rows from several banks"
            assert bool((row - row[0] == torch.arange(len(row))).all()), f"core {k} slot {sl}: rows not contiguous"
            got.add((int(bank[0]), int(row[0])))
    n = lay.E * lay.nk
    want = {
        (r % 8, ((r // 8) * lay.rd_bytes + c * lay.chunk + j * lay.blk) // 64)
        for r in range(16)
        for j in range(lay.R)
        for c in range(n - lay.land_slots, n)
    }
    assert got == want, f"delivered blocks differ: {len(got - want)} unexpected, {len(want - got)} missing"
    print(f"DRB check: {len(got)} landed blocks of {lay.blk} B, each whole and from its bank / offset", flush=True)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dram_read_bench(mesh_device):
    m = mesh_device
    lay = Layout(m)
    p = lay.p
    print(
        f"DRB {MODEL}: H {lay.H} I {lay.I} E {lay.E}, grid {lay.grid.x}x{lay.grid.y}, np {p['np']} g {p['g']}: 16 readers "
        f"x {lay.R} gate/up cores ({len(lay.gu)}), {len(lay.down)} down cores (rdown {p['rdown']}); gate/up "
        f"{lay.gu_bytes / 1e6:.1f} MB (reader chunk {lay.chunk} B, core block {lay.blk} B), down {lay.dn_total / 1e6:.1f} "
        f"MB; east banks {[b for b in range(8) if lay.east[b]]}",
        flush=True,
    )
    rt_on = ttnn.device.IsProgramRealtimeProfilerActive()
    recs = {}

    def cb(batch):
        for r in batch.records:
            recs[r.runtime_id] = (r.end_timestamp - r.start_timestamp) / (r.frequency * 1e3)  # us

    if os.environ.get("DRB_CHECK", "1") == "1":
        _check_delivery(m, lay)
    hdl = ttnn.device.RegisterProgramRealtimeProfilerCallback(cb) if rt_on else None
    sel = os.environ.get("DRB_VARIANTS")
    cases = [(None, True)] + [(v, d) for v in GU_VARIANTS for d in (False, True)]
    if sel:
        keep = sel.split(",")
        cases = [c for c in cases if (c[0] or "down") + ("+down" if c[1] and c[0] else "") in keep]
    rows = []
    try:
        for gu_v, with_down in cases:
            name = (gu_v or "down") + ("+down" if with_down and gu_v else "")
            pd = _program(lay, gu_v, with_down)
            ttnn.generic_op([lay.w, lay.land], pd)  # compile + warm
            ttnn.synchronize_device(m)
            ids, t0 = [], time.perf_counter()
            for _ in range(ITERS):
                i0 = ttnn._ttnn.get_device_operation_id()
                ttnn.generic_op([lay.w, lay.land], pd)
                ttnn.synchronize_device(m)
                ids.append((i0, ttnn._ttnn.get_device_operation_id()))
            wall = (time.perf_counter() - t0) / ITERS * 1e6
            if rt_on:
                time.sleep(0.5)
                per = [sum(recs[i] for i in range(a, b) if i in recs) for a, b in ids]
                per = [x for x in per if x > 0]
            us = statistics.median(per) if rt_on and per else wall
            gu_b = lay.gu_bytes if gu_v else 0
            tot = gu_b + (lay.dn_total if with_down or (gu_v or "").startswith("rd_all") else 0)
            rows.append((name, us, gu_b / us / 1e3, tot / us / 1e3))
            print(
                f"DRB {name:16s} {us:8.1f} us{'' if rt_on else ' (wall)'}  gate/up {gu_b / us / 1e3:6.1f} GB/s  total "
                f"{tot / us / 1e3:6.1f} GB/s ({tot / us / 1e3 / DRAM_GBS:5.1%} of {DRAM_GBS:.0f})",
                flush=True,
            )
    finally:
        if hdl is not None:
            ttnn.device.UnregisterProgramRealtimeProfilerCallback(hdl)
    print("DRB | variant | us | gate/up GB/s | total GB/s | % of 512 |")
    for name, us, g, t in rows:
        print(f"DRB | {name} | {us:.1f} | {g:.1f} | {t:.1f} | {t / DRAM_GBS:.0%} |")
