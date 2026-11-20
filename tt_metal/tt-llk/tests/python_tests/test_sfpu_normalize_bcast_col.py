# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# SFPU BCAST_COL normalize: FT = full tile (x - m) * s, TT = two tiles in place, RES = (x + r - m) * s, with m / s =
# column 0 of the mean / inv_std tiles broadcast across the row, against a golden over dest layouts (statistics above /
# below the data tile, second tile below) and the "COL add after normalize without re-running the COL init" sequence
# (layout D_seqadd). rt1 passes the tile indices as runtime values.
# Case string: <var>-<fmt>-<layout>-<stim>-rt<0|1>; fmt F32 = Float32 dest_acc Yes, BF16 = Float16_b dest_acc No.
import zlib
from dataclasses import dataclass

import pytest
import test_eltwise_binary_sfpu as F
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import BroadcastType as LlkBroadcastType
from helpers.llk_params import DestAccumulation, MathOperation, format_dict
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import TemplateParameter

FMTS = {
    "F32": (DataFormat.Float32, DestAccumulation.Yes),
    "BF16": (DataFormat.Float16_b, DestAccumulation.No),
}
VARS = {"FT": 0, "TT": 1, "RES": 2}
# name: (data, second, mean, inv, seq); second = data2 (TT) / residual (RES), unused for FT.
LAYOUTS = {
    "A_in": (0, 1, 2, 3, 0),  # stats above the data tile(s)
    "B_below": (2, 3, 0, 1, 0),  # stats below the data tile (wrapped offsets)
    "C_secbelow": (3, 2, 0, 1, 0),  # stats and second tile below the data tile
    "D_seqadd": (
        0,
        1,
        2,
        3,
        1,
    ),  # then BCAST_COL add out += mean col 0 without re-running the COL init
}


@dataclass
class NORM_COL_LAYOUT(TemplateParameter):
    norm_c_var: str = "FT"
    norm_c_layout: str = "A_in"
    norm_c_runtime: int = 0

    def convert_to_cpp(self) -> str:
        # FT has no second tile; its slot in the layout is simply left unused.
        d, s, m, i, seq = LAYOUTS[self.norm_c_layout]
        na = 1 if self.norm_c_var == "FT" else 2
        nout = 2 if self.norm_c_var == "TT" else 1
        return (
            f"#define NORM_C_VAR {VARS[self.norm_c_var]}\n#define NORM_C_NA {na}\n#define NORM_C_NOUT {nout}\n"
            f"#define NORM_C_IN0 {d}\n#define NORM_C_IN1 {s}\n#define NORM_C_IN2 {m}\n#define NORM_C_IN3 {i}\n#define NORM_C_OUT {d}\n"
            f"#define NORM_C_PACK0 {d}\n#define NORM_C_PACK1 {s}\n#define NORM_C_SEQ {seq}\n#define NORM_C_RUNTIME {self.norm_c_runtime}"
        )


def _stimuli(case, fmt):
    g = torch.Generator().manual_seed(zlib.crc32(case.encode()))
    dt = format_dict[fmt]
    data = torch.randn(32, 32, generator=g) * 7 + 3
    second = torch.randn(32, 32, generator=g) * 5 - 1
    # Statistic tiles: only column 0 is used; the rest is junk, including Inf / NaN that the kernel must ignore.
    mean_t = torch.randn(32, 32, generator=g) * 100
    inv_t = torch.rand(32, 32, generator=g) * 50
    for t in (mean_t, inv_t):
        t[0, 1], t[1, 7], t[2, 8], t[3, 15], t[5, 17], t[20, 31] = (
            float("inf"),
            float("nan"),
            float("-inf"),
            float("inf"),
            float("nan"),
            float("-inf"),
        )
    mean_t[:, 0] = data.mean(dim=1)
    inv_t[:, 0] = 1.0 / data.std(dim=1)
    if "-sp-" in case:
        # A bf16 NaN data value reaches dest as -Inf through the srcA path (also on main), so BF16 gets no data NaN.
        nan = float("nan") if fmt == DataFormat.Float32 else 2.5
        data[0, :6] = torch.tensor([float("inf"), float("-inf"), nan, 0.0, -0.0, 1e30])
        data[17, 20] = -1e30
        data[31, 31] = float(mean_t[31, 0])
        data[9, 3] = float(mean_t[9, 0])
        inv_t[4, 0] = 3.0e19  # large scale: some outputs overflow to +-inf
        mean_t[6, 0] = float("inf")  # whole row becomes NaN / -Inf
        inv_t[7, 0] = 0.0
        second[2, :4] = torch.tensor([float("inf"), -1e30, 0.0, -0.0])
    return data.to(dt), second.to(dt), mean_t.to(dt), inv_t.to(dt)


def _golden(case, data, second, mean_t, inv_t):
    var, fmt_s, layout = case.split("-")[:3]
    f = lambda t: t.to(torch.float32)
    neg_m = -f(mean_t)[:, 0:1]
    inv = f(inv_t)[:, 0:1]
    x = f(data)
    if var == "RES":
        x = x + f(second)
    outs = [(x + neg_m) * inv]
    if var == "TT":
        outs.append((f(second) + neg_m) * inv)
    # |value| rounded by an earlier dest store (0 if none); bounds that rounding's error in _check.
    inter = [torch.zeros_like(o) for o in outs]
    if LAYOUTS[layout][4] == 1:
        # The normalized tile is stored to dest (rounded to its format) before the add reads it.
        inter[0] = outs[0].abs()
        outs[0] = f(outs[0].to(format_dict[FMTS[fmt_s][0]])) + f(mean_t)[:, 0:1]
    return outs, inter


def _check(res, golden, fmt, inter):
    g = golden.to(format_dict[fmt]).to(torch.float32)
    r = res.to(torch.float32)
    nan_g, nan_r = torch.isnan(g), torch.isnan(r)
    assert torch.equal(
        nan_g, nan_r
    ), f"NaN mismatch at {int((nan_g != nan_r).sum())} lanes"
    inf_g = torch.isinf(g)
    assert torch.equal(r[inf_g], g[inf_g]), "Inf mismatch"
    fin = ~(nan_g | inf_g)
    rtol = 2**-20 if fmt == DataFormat.Float32 else 2**-7
    err = (r[fin] - g[fin]).abs()
    tol = rtol * (g[fin].abs() + inter[fin]) + 1e-30
    bad = int((err > tol).sum())
    assert (
        bad == 0
    ), f"{bad}/{int(fin.sum())} finite lanes off (max abs err {float(err.max())})"


def run_norm_col(case, is_perf=False, perf_report=None, run_types=None, loop_factor=1):
    if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR:
        pytest.skip("sfpu_normalize_bcast_col is not supported on Quasar")
    case = case[0] if isinstance(case, tuple) else case
    var, fmt_s, layout, stim, rt = case.split("-")
    fmt, dest_acc = FMTS[fmt_s]
    formats = InputOutputFormat(fmt, fmt)
    dims = [32, 32]
    data, second, mean_t, inv_t = _stimuli(case, fmt)
    tl = lambda t: F.tilize_block(t.flatten(), dims, fmt).flatten()
    a_tiles = [tl(data)] if var == "FT" else [tl(data), tl(second)]
    buf_A = torch.cat(a_tiles)
    buf_B = torch.cat([tl(mean_t), tl(inv_t)])
    nout = 2 if var == "TT" else 1
    num_blocks, num_tiles_in_block = F.get_num_blocks_and_num_tiles_in_block(
        F.DestSync.Half, dest_acc, formats, dims, F.TILE_DIMENSIONS
    )
    kwargs = {
        "test_name": "sources/sfpu_normalize_bcast_col_test.cpp",
        "formats": formats,
        "templates": [
            F.generate_input_dim(dims, dims),
            F.MATH_OP(mathop=MathOperation.SfpuElwsub),
            F.APPROX_MODE(F.ApproximationMode.No),
            F.ITERATIONS(32),
            F.BROADCAST_TYPE(LlkBroadcastType.None_),
            F.SFPU_BCAST_DIM(LlkBroadcastType.Column),
            F.SFPU_DST_ROUNDING_MODE(),
            F.SIGN_MAGNITUDE_FORMAT(False),
            NORM_COL_LAYOUT(var, layout, int(rt[2:])),
        ],
        "runtimes": [
            F.TILE_COUNT(len(a_tiles) + 2),
            F.NUM_BLOCKS(num_blocks),
            F.NUM_TILES_IN_BLOCK(num_tiles_in_block),
            F.LOOP_FACTOR(loop_factor),
            F.NUM_FACES(),
            F.ZERO_POINT(0),
            F.OUT_ZERO_POINT(0),
        ],
        "variant_stimuli": F.StimuliConfig(
            buf_A,
            fmt,
            buf_B,
            fmt,
            fmt,
            tile_count_A=len(a_tiles),
            tile_count_B=2,
            tile_count_res=nout,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": F.unpacks_to_dest(fmt, dest_acc),
        "compile_time_formats": True,
    }
    cfg = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=F._resolve_perf_args(is_perf, perf_report, run_types),
        test_config_kwargs=kwargs,
    )
    if is_perf:
        cfg.run(perf_report)
        return
    res_from_L1 = cfg.run().result
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        return
    res = torch.tensor(res_from_L1, dtype=format_dict[fmt]).flatten()
    outs, inter = _golden(case, data, second, mean_t, inv_t)
    tl32 = lambda ts: torch.cat(
        [F.tilize_block(t.flatten(), dims, DataFormat.Float32).flatten() for t in ts]
    )
    golden = tl32(outs)
    assert len(res) == len(golden)
    _check(res, golden, fmt, tl32(inter))


FUNC_CASES = (
    [
        f"{v}-{f}-{l}-{s}-rt0"
        for v in VARS
        for f in FMTS
        for l in ("A_in", "B_below", "C_secbelow")
        for s in ("rand", "sp")
    ]
    + [f"{v}-F32-{l}-rand-rt1" for v in VARS for l in ("A_in", "C_secbelow")]
    + [f"{v}-{f}-D_seqadd-rand-rt0" for v in VARS for f in FMTS]
)


@parametrize(case=FUNC_CASES)
def test_sfpu_normalize_bcast_col(case):
    run_norm_col(case)
