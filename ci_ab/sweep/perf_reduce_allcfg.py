# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch all-configuration perf sweep of the generic reduce LLK (not for merge).

Configuration sets (CI_RED_SET selects one or more, comma separated; default all):
  fmt    every format pair of perf_reduce plus Bfp4_b, every DEST width, SUM at LoFi/HiFi2/HiFi3/HiFi4, MAX; 32x32, tile calls
  avg    AVG controls (ELF identical to SUM expected)
  shape  tile shapes 1x32 2x32 4x32 8x32 16x32 16x16 32x16 (bf16, both DEST widths for one pool each)
  sync   DestSync full against half
  block  call form (tile, block of 1/2/4/8) x inputs per output (1, 8 accumulated)
"""
import os
from dataclasses import dataclass

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    PerfRunType,
    ReduceDimension,
    ReducePool,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    REDUCE_POOL_TYPE,
    TILE_COUNT,
    TemplateParameter,
)

MATHOP = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}


@dataclass
class CI_RED(TemplateParameter):
    face_r: int = 16
    nf_r: int = 2
    nf_c: int = 2
    sync_full: int = 0
    call_ct: int = 0
    k_per_out: int = 1

    def convert_to_cpp(self) -> str:
        return "\n".join(
            f"constexpr std::uint32_t CI_{n} = {v};"
            for n, v in (
                ("FACE_R", self.face_r),
                ("NF_R", self.nf_r),
                ("NF_C", self.nf_c),
                ("SYNC_FULL", self.sync_full),
                ("CT", self.call_ct),
                ("K", self.k_per_out),
            )
        )


F = DataFormat
B16, F16, F32, B8, B4 = F.Float16_b, F.Float16, F.Float32, F.Bfp8_b, F.Bfp4_b
DIMS = [ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar]
FID = MathFidelity
SUMFIDS = [FID.LoFi, FID.HiFi2, FID.HiFi3, FID.HiFi4]
NO, YES = DestAccumulation.No, DestAccumulation.Yes


def _cfgs():
    out = []

    def add(s, fi, fo, dacc, dim, pool, fid, ci=None):
        out.append((s, fi, fo, dacc, dim, pool, fid, ci or CI_RED()))

    pairs = [(a, b) for a in (B16, F16, F32, B8) for b in (B16, F16, F32, B8)]
    pairs += [(B4, B16), (B16, B4), (B4, B4)]
    for fi, fo in pairs:
        for dacc in (NO, YES):
            for dim in DIMS:
                add("fmt", fi, fo, dacc, dim, ReducePool.Max, FID.HiFi4)
                for fid in SUMFIDS:
                    add("fmt", fi, fo, dacc, dim, ReducePool.Sum, fid)
    for dim in DIMS:
        for fid in (FID.LoFi, FID.HiFi4):
            add("avg", B16, B16, NO, dim, ReducePool.Average, fid)
    shapes = [(1, 1, 2), (2, 1, 2), (4, 1, 2), (8, 1, 2), (16, 1, 2), (16, 1, 1), (16, 2, 1)]
    for fr, nr, nc in shapes:
        for dacc in (NO, YES):
            for dim in DIMS:
                add("shape", B16, B16, dacc, dim, ReducePool.Max, FID.HiFi4, CI_RED(fr, nr, nc))
                for fid in (FID.LoFi, FID.HiFi4):
                    add("shape", B16, B16, dacc, dim, ReducePool.Sum, fid, CI_RED(fr, nr, nc))
    for fi, fo, dacc in ((B16, B16, NO), (B16, B16, YES), (F32, F32, YES), (B8, B16, YES)):
        for dim in DIMS:
            add("sync", fi, fo, dacc, dim, ReducePool.Max, FID.HiFi4, CI_RED(sync_full=1))
            for fid in (FID.LoFi, FID.HiFi4):
                add("sync", fi, fo, dacc, dim, ReducePool.Sum, fid, CI_RED(sync_full=1))
    for fi, fo, dacc, cts in ((B16, B16, NO, (0, 1, 2, 4, 8)), (B16, B16, YES, (0, 2, 8)), (B8, B8, NO, (0, 2, 8)),
                              (F32, F32, YES, (0, 2, 8)), (B8, B16, YES, (0, 8))):
        for dim in DIMS:
            for k in (1, 8):
                for ct in cts:
                    if (ct, k) == (0, 1):
                        continue  # the fmt set's row
                    ci = CI_RED(call_ct=ct, k_per_out=k)
                    add("block", fi, fo, dacc, dim, ReducePool.Max, FID.HiFi4, ci)
                    for fid in (FID.LoFi, FID.HiFi4):
                        add("block", fi, fo, dacc, dim, ReducePool.Sum, fid, ci)
    want = os.environ.get("CI_RED_SET", "")
    if want:
        keep = set(want.split(","))
        out = [c for c in out if c[0] in keep]
    part = os.environ.get("CI_RED_PART", "")  # "i/n": every n-th configuration from i
    if part:
        i, n = map(int, part.split("/"))
        out = out[i::n]
    return out


def _id(c):
    s, fi, fo, dacc, dim, pool, fid, ci = c
    return (f"{s}-{fi.name}-{fo.name}-{dacc.name}-{dim.name}-{pool.name}-{fid.name}-"
            f"r{ci.face_r}f{ci.nf_r}x{ci.nf_c}-s{ci.sync_full}-ct{ci.call_ct}-k{ci.k_per_out}")


CFGS = _cfgs()


@pytest.mark.perf
@pytest.mark.parametrize("cfg", CFGS, ids=[_id(c) for c in CFGS])
def test_perf_reduce_allcfg(perf_report, cfg):
    _, fi, fo, dacc, dim, pool, fid, ci = cfg
    formats = InputOutputFormat(fi, fo)
    tile_count = 16
    configuration = PerfConfig(
        "sources/reduce_allcfg_perf.cpp",
        formats,
        run_types=[PerfRunType.L1_TO_L1, PerfRunType.UNPACK_ISOLATE, PerfRunType.MATH_ISOLATE, PerfRunType.PACK_ISOLATE],
        templates=[MATH_OP(mathop=MATHOP[dim]), REDUCE_POOL_TYPE(pool), MATH_FIDELITY(fid), ci],
        runtimes=[TILE_COUNT(tile_count), LOOP_FACTOR(64)],
        variant_stimuli=StimuliConfig(
            None, fi, None, fi, fo, tile_count_A=tile_count, tile_count_B=tile_count, tile_count_res=tile_count
        ),
        unpack_to_dest=False,
        dest_acc=dacc,
    )
    configuration.run(perf_report)
