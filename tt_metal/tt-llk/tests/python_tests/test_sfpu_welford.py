# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional coverage for the Welford SFPU kernel (ckernel_sfpu_welfords.h).

Drives `sources/sfpu_welford_test.cpp`, which follows ttnn's welford_reduce_h call
sequence: every input tile is copied to dst 0 and folded into the running per-column
mean / M2 (tile row = sample, 32 columns in parallel, consecutive tiles continue the
sample sequence), then the state is finalized into dst 1 / dst 2.

The golden is a float64 Welford over the same row sequence, on the values the device
actually sees (the input rounded to the input format, subnormals flushed). Per-column
tolerances follow the forward error of Welford's recurrence in fp32 (~N*eps scaled by
the column's magnitude and spread), plus the output-format rounding.

Axes:
- formats / dest_acc: fp32 in fp32 Dest (unpack-to-dest, ttnn fp32 layernorm), bf16 in
  bf16 Dest, bf16 in fp32 Dest.
- lut_size: 0 = RISC-side 1/(idx+1) fallback (welford_reduce_*), 256 = reciprocal LUT
  (layernorm / groupnorm).
- finalize: "row" = mean/variance in row layout (_store_mean_var_to_dst_row_),
  "raw" = mean/M2 in the raw layout (_store_mean_m2_to_dst_), which exposes M2 itself.
- partial: whether the last tile goes through _calculate_welfords_partial_tile_.
- population (runtime): "uniform" U(-4, 4); "offset" per-column offsets up to 3.3e7
  with O(1) spread (cancellation in x - mean); "specials" rows carrying +-0, +-NaN,
  +-Inf, subnormals and +-FLT_MAX-range values in dedicated columns. Special columns
  are not asserted against the golden (their device outcome is printed and covered
  by result hashing); every other column is.

Set LLK_WELFORD_DUMP=<path> to append the sha256 of the (mean, var/M2) bit patterns per
test id. Running that on two kernels and diffing the files is the bit-exactness check.
"""

import hashlib
import math
import os
import zlib

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import ApproximationMode, DestAccumulation, format_dict
from helpers.param_config import parametrize, runtime
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    TILE_COUNT,
    WELFORD_CONFIG,
)
from helpers.tilize_untilize import tilize_block, untilize_block

WELFORD_LUT_SIZE = 256

# (input/output format, dest_acc)
FORMAT_CASES = [
    (DataFormat.Float32, DestAccumulation.Yes),
    (DataFormat.Float16_b, DestAccumulation.No),
    (DataFormat.Float16_b, DestAccumulation.Yes),
]

# name -> (last_start_row, last_num_rows)
PARTIAL_CASES = {
    "full": (0, 32),
    "rows0to13": (0, 13),
    "rows7to27": (7, 20),
}

FP32_MIN_NORMAL = 2.0**-126
FP32_EPS = 2.0**-24

# Special-value columns of the "specials" population. Each entry fills one column;
# (row, value) pairs are overlaid on an otherwise U(-4, 4) column unless "fill" is given.
SPECIAL_COLUMNS = {
    0: {"fill": 0.0},  # all +0
    1: {"fill": -0.0},  # all -0
    2: {"fill": -0.0, "rows": [(0, 0.0)]},  # +0 first, then -0
    3: {"rows": [(5, math.nan)]},  # +NaN mid-sequence
    4: {"rows": [(17, -math.nan)]},  # -NaN mid-sequence
    5: {"rows": [(3, math.inf)]},  # +Inf
    6: {"rows": [(9, -math.inf)]},  # -Inf
    7: {"rows": [(2, math.inf), (30, -math.inf)]},  # +Inf then -Inf
    8: {"fill": 1.0e-40},  # fp32 subnormal (bf16 keeps a subnormal too)
    9: {"alt": 3.0e38},  # +-3e38 alternating: M2 overflows
    10: {"fill": 1.0e30},  # constant large: variance exactly 0 in fp64
    11: {"rows": [(0, 3.0e38)]},  # one huge sample among small ones
    12: {"fill": 1.0e-38, "rows": [(1, -1.0e-38)]},  # smallest normals
}


def _seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def _make_input(population, rows, gen):
    """float64 [rows, 32] stimulus for the given population."""
    x = torch.empty((rows, TILE_DIM), dtype=torch.float64).uniform_(
        -4.0, 4.0, generator=gen
    )
    if population == "offset":
        offsets = [0.0, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, -1.0e4, 3.3e7]
        for c in range(TILE_DIM):
            spread = 1.0 if c < 16 else 10.0
            noise = torch.empty(rows, dtype=torch.float64).uniform_(
                -spread, spread, generator=gen
            )
            x[:, c] = offsets[c % 8] + noise
    elif population == "specials":
        for c, spec in SPECIAL_COLUMNS.items():
            if "fill" in spec:
                x[:, c] = spec["fill"]
            if "alt" in spec:
                x[:, c] = torch.tensor(
                    [spec["alt"] * (-1) ** r for r in range(rows)], dtype=torch.float64
                )
            for r, v in spec.get("rows", []):
                if r < rows:
                    x[r, c] = v
    return x


def _device_view(x_in: torch.Tensor) -> torch.Tensor:
    """float64 view of the values the SFPU sees: input-format rounding, then the Dst-path
    subnormal flush (sign kept)."""
    v = x_in.to(torch.float64)
    tiny = (v.abs() < FP32_MIN_NORMAL) & (v != 0)
    return torch.where(tiny, torch.copysign(torch.zeros_like(v), v), v)


def _welford_fp64(x: torch.Tensor):
    """float64 Welford over the rows of x ([n, 32]); returns (mean, M2)."""
    mean = torch.zeros(x.shape[1], dtype=torch.float64)
    m2 = torch.zeros(x.shape[1], dtype=torch.float64)
    for n in range(x.shape[0]):
        d = x[n] - mean
        mean = mean + d / (n + 1)
        m2 = m2 + d * (x[n] - mean)
    return mean, m2


def _raw_to_columns(face_rows: torch.Tensor) -> torch.Tensor:
    """Decode the raw LREG layout _store_mean_m2_to_dst_ writes: face 0 row g, even column
    2k holds column (16 if g >= 2 else 0) + 2k + (g & 1)."""
    out = torch.empty(TILE_DIM, dtype=face_rows.dtype)
    for g in range(4):
        for k in range(8):
            out[(16 if g >= 2 else 0) + 2 * k + (g & 1)] = face_rows[g, 2 * k]
    return out


def _bits(t: torch.Tensor) -> bytes:
    return t.to(torch.float32).contiguous().view(torch.int32).numpy().tobytes()


@parametrize(
    format_case=list(range(len(FORMAT_CASES))),
    lut_size=[0, WELFORD_LUT_SIZE],
    finalize=["row", "raw"],
    partial=list(PARTIAL_CASES),
    tile_cnt=runtime([1, 2, 5]),
    population=runtime(["uniform", "offset", "specials"]),
)
def test_sfpu_welford(format_case, lut_size, finalize, partial, tile_cnt, population):
    fmt, dest_acc = FORMAT_CASES[format_case]
    formats = InputOutputFormat(fmt, fmt)
    torch_format = format_dict[fmt]
    last_start_row, last_num_rows = PARTIAL_CASES[partial]

    rows = tile_cnt * TILE_DIM
    gen = torch.Generator().manual_seed(_seed(fmt.name, population, tile_cnt))
    x64 = _make_input(population, rows, gen)
    src_A = x64.to(torch_format).flatten()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    input_dimensions = [rows, TILE_DIM]
    src_A_tilized = tilize_block(src_A, input_dimensions, stimuli_format=fmt).flatten()

    configuration = TestConfig(
        "sources/sfpu_welford_test.cpp",
        formats,
        templates=[
            APPROX_MODE(ApproximationMode.No),
            WELFORD_CONFIG(
                lut_size=lut_size,
                finalize_raw=(finalize == "raw"),
                last_start_row=last_start_row,
                last_num_rows=last_num_rows,
            ),
        ],
        runtimes=[TILE_COUNT(tile_cnt)],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            fmt,
            src_B,
            fmt,
            fmt,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=2,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    res = torch.tensor(res_from_L1, dtype=format_dict[fmt])
    res = untilize_block(res, fmt, [2 * TILE_DIM, TILE_DIM]).to(torch.float64)
    if finalize == "row":
        dev_mean, dev_second = res[0], res[TILE_DIM]
    else:
        dev_mean = _raw_to_columns(res[0:4])
        dev_second = _raw_to_columns(res[TILE_DIM : TILE_DIM + 4])

    # Golden over exactly the rows the kernel consumed.
    xs = _device_view(src_A.to(torch.float64).view(rows, TILE_DIM))
    last_base = (tile_cnt - 1) * TILE_DIM
    used = torch.cat(
        [
            xs[:last_base],
            xs[last_base + last_start_row : last_base + last_start_row + last_num_rows],
        ]
    )
    n = used.shape[0]
    g_mean, g_m2 = _welford_fp64(used)
    g_second = g_m2 / n if finalize == "row" else g_m2

    dump = os.environ.get("LLK_WELFORD_DUMP")
    if dump:
        digest = hashlib.sha256(_bits(dev_mean) + _bits(dev_second)).hexdigest()
        test_id = os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0]
        with open(dump, "a") as fh:
            fh.write(f"{test_id}\t{digest}\n")

    out_rel = 2.0**-7 if fmt == DataFormat.Float16_b else 2.0**-23
    special_cols = set(SPECIAL_COLUMNS) if population == "specials" else set()
    failures = []
    for c in range(TILE_DIM):
        col = used[:, c]
        if c in special_cols:
            print(
                f"SPECIAL col={c:2d} n={n} dev_mean={dev_mean[c].item()!r} "
                f"gold_mean={g_mean[c].item()!r} dev_{finalize}={dev_second[c].item()!r} "
                f"gold={g_second[c].item()!r}"
            )
            continue
        maxabs = col.abs().max().item()
        maxdev = (col - g_mean[c]).abs().max().item()
        tol_mean = 4 * n * FP32_EPS * maxabs + out_rel * abs(g_mean[c].item()) + 1e-30
        m2_err = 8 * n * FP32_EPS * (n * maxabs * maxdev + g_m2[c].item())
        tol_second = (
            (m2_err / n if finalize == "row" else m2_err)
            + out_rel * abs(g_second[c].item())
            + 1e-30
        )
        err_mean = abs(dev_mean[c].item() - g_mean[c].item())
        err_second = abs(dev_second[c].item() - g_second[c].item())
        if not (err_mean <= tol_mean and err_second <= tol_second):
            failures.append(
                f"col {c}: mean dev={dev_mean[c].item()!r} gold={g_mean[c].item()!r} "
                f"err={err_mean:.3e} tol={tol_mean:.3e}; {finalize} dev={dev_second[c].item()!r} "
                f"gold={g_second[c].item()!r} err={err_second:.3e} tol={tol_second:.3e}"
            )
    assert not failures, f"{len(failures)} column(s) outside tolerance:\n" + "\n".join(
        failures
    )
