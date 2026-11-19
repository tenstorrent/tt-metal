# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# _calculate_sfpu_normalize_bcast_scalar_full_tile_: out = (data - mean[0][0]) * inv_std[0][0], against a golden over
# dest layouts (statistics above / below the data tile, out of place, out below the data tile) and call sequences (a
# re-record with another output offset, BCAST_ROW before / after). Case string: <fmt>-<layout>-<stim>-rt<0|1>; fmt
# F32 = Float32 dest_acc Yes, BF16 = Float16_b dest_acc Yes; rt1 passes the tile indices as runtime values.
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
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import TemplateParameter

FMTS = {"F32": DataFormat.Float32, "BF16": DataFormat.Float16_b}
# name: (data, mean, inv_std, out, pack, seq, pre_out); see the .cpp header for seq.
LAYOUTS = {
    "A_in": (0, 1, 2, 0, 0, 0, 0),  # stats above data, in place (compute-API shape)
    "B_below": (2, 0, 1, 2, 2, 0, 0),  # stats below data (wrapped offsets), in place
    "C_oop": (0, 1, 2, 3, 3, 0, 0),  # out of place, out at tile 3
    "D_outbelow": (
        3,
        1,
        0,
        2,
        2,
        0,
        0,
    ),  # stats and out below data (wrapped store offsets)
    "E_pre": (
        0,
        1,
        2,
        0,
        0,
        1,
        3,
    ),  # scalar call to tile 3 first, then in place (re-record, other out offset)
    "F_rowpre": (
        0,
        1,
        2,
        0,
        0,
        2,
        3,
    ),  # BCAST_ROW add first (records the ROW body + ADDR_MOD_6)
    "G_rowpost": (
        1,
        2,
        3,
        0,
        0,
        3,
        0,
    ),  # out below data, then BCAST_ROW add out += data row 0
}


@dataclass
class NORM_SCALAR_LAYOUT(TemplateParameter):
    layout: str = "A_in"
    runtime: int = 0

    def convert_to_cpp(self) -> str:
        d, m, i, o, p, s, q = LAYOUTS[self.layout]
        return (
            f"#define NORM_S_DATA {d}\n#define NORM_S_MEAN {m}\n#define NORM_S_INV {i}\n#define NORM_S_OUT {o}\n"
            f"#define NORM_S_PACK {p}\n#define NORM_S_SEQ {s}\n#define NORM_S_PRE_OUT {q}\n#define NORM_S_RUNTIME {self.runtime}"
        )


def _stimuli(case, fmt):
    g = torch.Generator().manual_seed(zlib.crc32(case.encode()))
    dt = format_dict[fmt]
    data = torch.randn(32, 32, generator=g) * 7 + 3
    # Statistic tiles: only [0][0] is used; the rest is junk, including the Inf / NaN that the kernel must ignore.
    mean_t = torch.randn(32, 32, generator=g) * 100
    inv_t = torch.rand(32, 32, generator=g) * 50
    for t in (mean_t, inv_t):
        t[0, 1], t[1, 0], t[0, 8], t[3, 0], t[5, 17] = (
            float("inf"),
            float("nan"),
            float("-inf"),
            float("inf"),
            float("nan"),
        )
    mean_t[0, 0] = float(data.mean())
    inv_t[0, 0] = 1.0 / float(data.std())
    if "-sp-" in case:
        # A bf16 NaN data value reaches dest as -Inf through the srcA path (also on main), so BF16 gets no data NaN.
        nan = float("nan") if fmt == DataFormat.Float32 else 2.5
        data[0, :6] = torch.tensor([float("inf"), float("-inf"), nan, 0.0, -0.0, 1e30])
        data[17, 20] = -1e30
        data[31, 31] = float(mean_t[0, 0])
        inv_t[0, 0] = 3.0e19  # large scale: some outputs overflow to +-inf
    return data.to(dt), mean_t.to(dt), inv_t.to(dt)


def _golden(case, data, mean_t, inv_t):
    layout = case.split("-")[1]
    x = data.to(torch.float32)
    neg_m = -mean_t.to(torch.float32)[0, 0]
    inv = inv_t.to(torch.float32)[0, 0]
    out = (x + neg_m) * inv
    if LAYOUTS[layout][5] == 3:
        out = out + x[0, :].unsqueeze(0)
    return out


def _check(res, golden, fmt):
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
    tol = rtol * g[fin].abs() + 1e-30
    bad = int((err > tol).sum())
    assert (
        bad == 0
    ), f"{bad}/{int(fin.sum())} finite lanes off (max abs err {float(err.max())})"


def run_norm_scalar(case):
    if TestConfig.CHIP_ARCH == ChipArchitecture.QUASAR:
        pytest.skip("sfpu_normalize_bcast_scalar is not supported on Quasar")
    case = case[0] if isinstance(case, tuple) else case
    fmt_s, layout, stim, rt = case.split("-")
    fmt = FMTS[fmt_s]
    dest_acc = DestAccumulation.Yes
    formats = InputOutputFormat(fmt, fmt)
    dims = [32, 32]
    data, mean_t, inv_t = _stimuli(case, fmt)
    buf_A = F.tilize_block(data.flatten(), dims, fmt).flatten()
    buf_B = torch.cat(
        [
            F.tilize_block(mean_t.flatten(), dims, fmt).flatten(),
            F.tilize_block(inv_t.flatten(), dims, fmt).flatten(),
        ]
    )
    num_blocks, num_tiles_in_block = F.get_num_blocks_and_num_tiles_in_block(
        F.DestSync.Half, dest_acc, formats, dims, F.TILE_DIMENSIONS
    )
    kwargs = {
        "test_name": "sources/sfpu_normalize_bcast_scalar_test.cpp",
        "formats": formats,
        "templates": [
            F.generate_input_dim(dims, dims),
            F.MATH_OP(mathop=MathOperation.SfpuElwsub),
            F.APPROX_MODE(F.ApproximationMode.No),
            F.ITERATIONS(32),
            F.BROADCAST_TYPE(LlkBroadcastType.None_),
            F.SFPU_BCAST_DIM(LlkBroadcastType.Row),
            F.SFPU_DST_ROUNDING_MODE(),
            F.SIGN_MAGNITUDE_FORMAT(False),
            NORM_SCALAR_LAYOUT(layout, int(rt[2:])),
        ],
        "runtimes": [
            F.TILE_COUNT(3),
            F.NUM_BLOCKS(num_blocks),
            F.NUM_TILES_IN_BLOCK(num_tiles_in_block),
            F.LOOP_FACTOR(1),
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
            tile_count_A=1,
            tile_count_B=2,
            tile_count_res=1,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": F.unpacks_to_dest(fmt, dest_acc),
        "compile_time_formats": True,
    }
    cfg = F.create_test_or_perf_config(
        is_perf=False,
        run_types=F._resolve_perf_args(False, None, None),
        test_config_kwargs=kwargs,
    )
    res_from_L1 = cfg.run().result
    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        return
    res = torch.tensor(res_from_L1, dtype=format_dict[fmt]).flatten()
    golden = F.tilize_block(
        _golden(case, data, mean_t, inv_t).flatten(), dims, DataFormat.Float32
    ).flatten()
    assert len(res) == len(golden)
    _check(res, golden, fmt)


FUNC_CASES = [
    f"{f}-{l}-{s}-rt0" for f in FMTS for l in LAYOUTS for s in ("rand", "sp")
] + [f"F32-{l}-rand-rt1" for l in LAYOUTS]


@parametrize(case=FUNC_CASES)
def test_sfpu_normalize_bcast_scalar(case):
    run_norm_scalar(case)
