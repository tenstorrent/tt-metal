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
  dir_rot / dir_rot1   gate/up cores read their own blocks rotating over all 8 banks (as unified_routed_expert_ffn's
                one-tile-row ND shards), NOC by column half / by core parity; DRB_DROT=1: the down cores rotate too
  rd_dfwd       the readers deliver the down weights too: BRISC reads gate/up (for the NCRISC's forwards, NOC1) and
                down chunks, interleaved in proportion, and writes the down chunks on to its 1-2 down cores (NOC0)
  +bg           a real call's other traffic at DRB_M rows per expert (default 128): relays read x and multicast it
                into the gate/up rectangles (NOC0); down cores pass h along a chain (NOC1) and write y rows to DRAM
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
STREAM, FWD, RD2, BG, FWD2 = (
    f"{HERE}/{k}.cpp" for k in ("bench_stream", "bench_fwd", "bench_rd2", "bench_bg", "bench_fwd2")
)
TILE = 576  # bfp4 tile bytes
KBLK = 8
MODEL = os.environ.get("DRB_MODEL", "glm-53")
ITERS = int(os.environ.get("DRB_ITERS", "10"))
DBATCH = int(os.environ.get("DRB_DBATCH", "2"))
DMULT = int(os.environ.get("DRB_DMULT", "1"))  # direct, contiguous layout: blocks per read request
DROT = os.environ.get("DRB_DROT", "0") == "1"  # down cores read their slice rotating over the 8 banks
M_BG = int(os.environ.get("DRB_M", "128"))  # rows per expert for the +bg traffic
DRAM_GBS = 512.0
DFWD = ("rd_dfwd", "rd_dfwd1")
GU_VARIANTS = ("rd", "rd_fwd", "rd_all", "rd_all_fwd", "rd_dfwd", "rd_dfwd1", "dir_rot", "dir_rot1", "dir_noc0", "dir_noc1", "dir_side", "dir_strided", "nr_noc0", "nr_noc1", "nr_side")


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
        # rd_dfwd: down core d is fed by a reader (nearest, at most ceil(nd / 16) each), all down cores the same width
        self.pcd_u = -(-self.Ht // nd)
        self.dn_chunk = KBLK * self.pcd_u * TILE
        self.dn_core = self.E * self.It * self.pcd_u * TILE
        # each reader takes a 16th of the global down-chunk order (down core by down core): segments of
        # (down core, count, first chunk index in that core's sequence); readers r and r + 8 share a bank
        self.dn_per_core = self.E * self.It // KBLK
        total = nd * self.dn_per_core
        self.r_seg = []
        for r in range(16):
            a, b = r * total // 16, (r + 1) * total // 16
            seg, c = [], a
            while c < b:
                d, k = divmod(c, self.dn_per_core)
                n = min(b - c, self.dn_per_core - k)
                seg.append((d, n, k))
                c += n
            self.r_seg.append(seg)
        self.r_dn_chunks = [sum(n for _, n, _ in sg) for sg in self.r_seg]
        dn_area = max(-(-nd // 8) * max(self.dn_bytes), 2 * max(self.r_dn_chunks) * self.dn_chunk)
        # +bg: x rows read by the relays (each rectangle's relays read all of x, bf16), y rows written by the down
        # cores (bf16), M_BG rows per expert
        self.rows_bg = self.E * M_BG
        self.relays = [tuple(c) for c in p["relays"]]
        self.x_rows_relay = self.rows_bg // (len(self.relays) // 2)
        self.x_step_rows = 2
        self.x_read = self.x_step_rows * self.H * 2
        self.x_mcast = self.x_step_rows * self.H * 17 // 16  # bfp8 tiles
        self.h_step = 32 * self.I * 17 // 16  # one row tile of h, bfp8
        self.y_step = 32 * self.pcd_u * 32 * 2  # one row tile of a down core's y columns, bf16
        self.n_bg = self.rows_bg // 32
        x_area = len(self.relays) * -(-self.x_rows_relay // self.x_step_rows // 8) * self.x_read
        y_area = nd * -(-self.n_bg // 8) * self.y_step
        self.dn_off, self.x_off = 2 * self.rd_bytes, 2 * self.rd_bytes + dn_area
        self.y_off = self.x_off + x_area
        per_bank = self.y_off + y_area
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
        self.dn_slots = 4
        dmc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_crs(self.down), ((self.dn_slots * self.dn_chunk + self.h_step) // 64, 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.dn_land_rows = (self.dn_slots * self.dn_chunk + self.h_step) // 64
        self.land_dn = ttnn.from_torch(
            torch.full((nd * self.dn_land_rows, 32), -1.0).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=m,
            memory_config=dmc,
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

    def dn_reader_off(self, r):
        """rd_dfwd: reader r's down region in its bank (its run of down chunks back to back)."""
        return self.dn_off + (r // 8) * max(self.r_dn_chunks) * self.dn_chunk

    def down_kernels(self, nr=False, rot=False):
        # one kernel + CB per distinct block width (pcd differs by one column across the down cores)
        down = self.nr_down if nr else self.down
        nd = len(down)
        pcds = [self.Ht // nd + (d < self.Ht % nd) for d in range(nd)]
        out, cbs = [], []
        for i, pcd in enumerate(sorted(set(pcds))):
            ds = [d for d in range(nd) if pcds[d] == pcd]
            cores, blk = [down[d] for d in ds], KBLK * pcd * TILE
            if rot:  # down core d's slice spread over all 8 banks: its 8th of each bank at d * (slice / 8)
                rt = [[self.base, d % 8, 2 * self.rd_bytes + d * -(-self.E * self.It * pcds[0] * TILE // 8)] for d in ds]
            else:
                rt = [[self.base, d % 8, 2 * self.rd_bytes + (d // 8) * self.E * self.It * pcds[0] * TILE] for d in ds]
            out.append(_kernel(STREAM, cores, [2 + i, blk, self.E * self.It // KBLK, 2, 4, blk if not rot else blk, 0, int(rot)], rt, 1, 0))
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
            ks = [_kernel(STREAM, self.readers, [0, self.chunk, n_rd, 1, 2, self.chunk, int(fwd), 0], rt, 0, 0)]
            if fwd:
                land = self.land.buffer_address()
                frt = [[land] + [self.vxy(self.gu[r * self.R + j]) for j in range(self.R)] for r in range(16)]
                ks.append(_kernel(FWD, self.readers, [0, self.R, self.blk, n_rd, self.land_slots], frt, 1, 1))
            return ks, [_cb(0, self.readers, self.chunk, 2)]
        if v in DFWD:
            one = v == "rd_dfwd1"  # the NCRISC forwards both streams; the BRISC only reads
            page = max(self.chunk, self.dn_chunk) if one else self.chunk
            ks, cbs = [], [_cb(0, self.readers, page, 2), _cb(5, self.readers, self.dn_chunk, 2)]
            land_dn = self.land_dn.buffer_address()
            max_seg = max(len(sg) for sg in self.r_seg)
            segs = [[x for d, n, k in self.r_seg[r] for x in (self.vxy(self.down[d]), n, k)] for r in range(16)]
            rt = [
                [self.base, r % 8, (r // 8) * self.rd_bytes, self.dn_reader_off(r), land_dn, len(self.r_seg[r])] + segs[r]
                for r in range(16)
            ]
            ct = [0, self.chunk, n_rd, 5, self.dn_chunk, self.dn_slots, max_seg, int(one)]
            ks.append(_kernel(RD2, self.readers, ct, rt, 0, 0))
            land = self.land.buffer_address()
            if one:
                frt = [
                    [land] + [self.vxy(self.gu[r * self.R + j]) for j in range(self.R)] + [land_dn, len(self.r_seg[r])] + segs[r]
                    for r in range(16)
                ]
                fct = [0, self.R, self.blk, n_rd, self.land_slots, self.dn_chunk, self.dn_slots, max_seg]
                ks.append(_kernel(FWD2, self.readers, fct, frt, 1, 1))
                return ks, cbs
            land = self.land.buffer_address()
            frt = [[land] + [self.vxy(self.gu[r * self.R + j]) for j in range(self.R)] for r in range(16)]
            ks.append(_kernel(FWD, self.readers, [0, self.R, self.blk, n_rd, self.land_slots], frt, 1, 1))
            return ks, cbs
        strided = v == "dir_strided"
        rot = v in ("dir_rot", "dir_rot1")
        groups = {0: [], 1: []}
        if rot:  # each gate/up core's blocks spread over all 8 banks (its 64th of each bank), NOC0 / NOC1 by column half
            per = self.gu_bytes // 64 // 8
            for i, c in enumerate(self.gu):
                noc = (c[0] >= self.grid.x // 2) if v == "dir_rot" else (i % 2)
                groups[int(noc)].append((c, [self.base, i % 8, i * per]))
        if v.startswith("nr_"):  # the reader-free layout: core k of bank b reads quarter k % 8 of the bank's 2 regions
            seen = [0] * 8
            for i, c in enumerate(self.nr_gu):
                b = self.nr_bank[i]
                off, seen[b] = seen[b] * (self.rd_bytes // 4), seen[b] + 1
                groups[{"nr_noc0": 0, "nr_noc1": 1}.get(v, int(self.east[b]))].append((c, [self.base, b, off]))
        for i, c in enumerate(self.gu if not (v.startswith("nr_") or rot) else []):
            r = i // self.R
            bank, j = r % 8, i % self.R
            off = (r // 8) * self.rd_bytes + (j * self.blk if strided else j * (self.rd_bytes // self.R))
            noc = {"dir_noc0": 0, "dir_noc1": 1}.get(v, int(self.east[bank]))
            groups[noc].append((c, [self.base, bank, off]))
        mult = 1 if strided or rot else DMULT
        assert n_rd % mult == 0, (n_rd, mult)
        req = mult * self.blk
        stride = self.chunk if strided else req  # (rotating: the next chunk in the same bank is one block on)
        ks = [
            _kernel(STREAM, [c for c, _ in g], [1, req, n_rd // mult, DBATCH, 2 * DBATCH, stride, 0, int(rot)], [a for _, a in g], 0, noc)
            for noc, g in groups.items()
            if g
        ]
        return ks, [_cb(1, self.nr_gu if v.startswith("nr_") else self.gu, req, 2 * DBATCH)]


def _bg_kernels(lay):
    """A real call's other traffic at M_BG rows per expert: relays read x (DRAM) and multicast it into their
    rectangle (NOC0); down cores pass h to their chain successor and write y rows to DRAM (NOC1)."""
    e0 = min(lay.readers[b][0] for b in range(8) if lay.east[b])
    rects = [[c for c in lay.gu if (c[0] < e0) == (k == 0)] for k in (0, 1)]
    land = lay.land.buffer_address()
    rt = []
    for i, c in enumerate(lay.relays):
        cells = rects[i % 2]
        lo = m_v(lay, (min(x for x, _ in cells), min(y for _, y in cells)))
        hi = m_v(lay, (max(x for x, _ in cells), max(y for _, y in cells)))
        n_steps = lay.x_rows_relay // lay.x_step_rows
        rt.append([lay.base, lay.x_off + i * -(-n_steps // 8) * lay.x_read, land, lo[0], lo[1], hi[0], hi[1], len(cells)])
    n_x = lay.x_rows_relay // lay.x_step_rows
    ks = [_kernel(BG, lay.relays, [0, n_x, lay.x_read, lay.x_mcast, 6], rt, 0, 0)]
    cbs = [_cb(6, lay.relays, lay.x_read, 1)]
    nd = len(lay.down)
    h_land = lay.land_dn.buffer_address() + lay.dn_slots * lay.dn_chunk
    drt = [
        [lay.base, lay.y_off + d * -(-lay.n_bg // 8) * lay.y_step, h_land, lay.vxy(lay.down[(d + 1) % nd])] for d in range(nd)
    ]
    ks.append(_kernel(BG, lay.down, [1, lay.n_bg, lay.h_step, lay.y_step, 7], drt, 0, 1))
    cbs.append(_cb(7, lay.down, max(lay.h_step, lay.y_step), 1))
    return ks, cbs


def m_v(lay, c):
    v = lay.vxy(c)
    return v >> 16, v & 0xFFFF


def _program(lay, gu_v, with_down, with_bg=False):
    ks, cbs = lay.gu_kernels(gu_v) if gu_v else ([], [])
    if with_down:
        dk, dc = lay.down_kernels(nr=bool(gu_v) and gu_v.startswith("nr_"), rot=DROT)
        ks, cbs = ks + dk, cbs + dc
    if with_bg:
        bk, bc = _bg_kernels(lay)
        ks, cbs = ks + bk, cbs + bc
    mesh_pd = ttnn.MeshProgramDescriptor()
    coord = ttnn.MeshCoordinate(0, 0)
    mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs)
    return mesh_pd


def _check_delivery(m, lay):
    """rd_fwd once, then every landing slot must be one whole block from the right bank and offset: the last
    land_slots chunks of each reader, block j at receiver j (proves the 36 KB reads and the forwards move all bytes)."""
    ttnn.generic_op([lay.w, lay.land, lay.land_dn], _program(lay, "rd_fwd", False))
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
    # rd_dfwd's down forwards: every landing slot of every down core holds one whole chunk of a reader's down run,
    # at the slot its index in the core's sequence maps to. (Which of two readers feeding one core lands last in a
    # slot is a race here: the bench has no credits.)
    for dv in DFWD:
        ttnn.generic_op([lay.w, lay.land, lay.land_dn], _program(lay, dv, False))
        ttnn.synchronize_device(m)
        nd, rows_c = len(lay.down), lay.dn_chunk // 64
        dl = ttnn.to_torch(lay.land_dn).float().reshape(nd, lay.dn_land_rows, 32)[:, : lay.dn_slots * rows_c]
        dl = dl.reshape(nd, lay.dn_slots, rows_c, 32)
        src = (dl[..., 0], dl[..., 1] * 16384 + dl[..., 2] * 128 + dl[..., 3])
        where = {}  # (bank, first row) of each chunk -> its index in its down core's sequence
        for r in range(16):
            c = 0
            for d, n, k in lay.r_seg[r]:
                for i in range(n):
                    where[(r % 8, (lay.dn_reader_off(r) + (c + i) * lay.dn_chunk) // 64)] = k + i
                c += n
        got = set()
        for k in range(nd):
            for sl in range(lay.dn_slots):
                bank, row = src[0][k, sl], src[1][k, sl]
                assert bool((bank == bank[0]).all()), f"down core {k} slot {sl}: rows from several banks"
                assert bool((row - row[0] == torch.arange(len(row))).all()), f"down core {k} slot {sl}: rows not contiguous"
                key = (int(bank[0]), int(row[0]))
                assert key in where and where[key] % lay.dn_slots == sl, f"down core {k} slot {sl}: chunk {key} misplaced"
                got.add(key)
        print(f"DRB check: {len(got)} {dv}: forwarded down chunks of {lay.dn_chunk} B, each whole and from its reader's region", flush=True)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dram_read_bench(mesh_device):
    m = mesh_device
    lay = Layout(m)
    p = lay.p
    print(
        f"DRB {MODEL}: H {lay.H} I {lay.I} E {lay.E}, grid {lay.grid.x}x{lay.grid.y}, np {p['np']} g {p['g']}: 16 readers "
        f"x {lay.R} gate/up cores ({len(lay.gu)}), {len(lay.down)} down cores (rdown {p['rdown']}); gate/up "
        f"{lay.gu_bytes / 1e6:.1f} MB (reader chunk {lay.chunk} B, core block {lay.blk} B), down {lay.dn_total / 1e6:.1f} "
        f"MB; east banks {[b for b in range(8) if lay.east[b]]}; rd_dfwd: down chunks per reader "
        f"{sorted(set(lay.r_dn_chunks))} (segments {sorted(set(len(x) for x in lay.r_seg))}), chunk {lay.dn_chunk} B; +bg at M {M_BG}: x {lay.rows_bg * lay.H * 2 * 2 / 1e6:.1f} MB "
        f"read, h {lay.rows_bg * lay.I * 17 // 16 / 1e6:.1f} MB x {len(lay.down)} hops, y {lay.rows_bg * lay.H * 2 / 1e6:.1f} MB",
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
    nm = lambda v, d, b: (v or ("down" if d else "")) + ("+down" if d and v else "") + ("+bg" if b else "")  # noqa: E731
    cases = [(v, d, b) for b in (False, True) for v in (None,) + GU_VARIANTS for d in (False, True) if nm(v, d, b)]
    cases = [c for c in cases if not (c[0] in ("rd_all", "rd_all_fwd") + DFWD and c[1])]  # down already delivered
    if sel:
        keep = sel.split(",")
        cases = [c for c in cases if nm(*c) in keep]
    rows = []
    try:
        for gu_v, with_down, with_bg in cases:
            name = nm(gu_v, with_down, with_bg)
            pd = _program(lay, gu_v, with_down, with_bg)
            ttnn.generic_op([lay.w, lay.land, lay.land_dn], pd)  # compile + warm
            ttnn.synchronize_device(m)
            ids, t0 = [], time.perf_counter()
            for _ in range(ITERS):
                i0 = ttnn._ttnn.get_device_operation_id()
                ttnn.generic_op([lay.w, lay.land, lay.land_dn], pd)
                ttnn.synchronize_device(m)
                ids.append((i0, ttnn._ttnn.get_device_operation_id()))
            wall = (time.perf_counter() - t0) / ITERS * 1e6
            if rt_on:
                time.sleep(0.5)
                per = [sum(recs[i] for i in range(a, b) if i in recs) for a, b in ids]
                per = [x for x in per if x > 0]
            us = statistics.median(per) if rt_on and per else wall
            gu_b = lay.gu_bytes if gu_v else 0
            dn_b = len(lay.down) * lay.dn_core if gu_v in DFWD else lay.dn_total
            tot = gu_b + (dn_b if with_down or gu_v in ("rd_all", "rd_all_fwd") + DFWD else 0)
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
