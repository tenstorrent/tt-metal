# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional coverage for the EMA SFPU kernel.

Two tests, both driving `sources/sfpu_ema_test.cpp` through `_run_ema_on_device`:

- `test_sfpu_ema` -- the default-suite check, one alpha over 1/2/4 time tiles.
- `test_sfpu_ema_alpha_sweep` -- nightly, 1000 alphas against an fp64 reference.

The sweep exists because the value the functional test pins, alpha = 0.25, is the
least informative one available for this kernel: 0.25 is an exact power of two, so
`alpha * EMA_old` is a pure exponent shift with no rounding, and any reassociation
of the recurrence is invisible at that alpha. Every other alpha rounds, and
`ttnn.ema` takes an arbitrary float alpha with no default, so the documented input
range is the whole interval.
"""

import os
import struct

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    EMA_ALPHA_BETA,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# EMA smoothing factor chosen for the functional test (ttnn.ema has no default alpha;
# it is a required argument). beta is derived as 1 - alpha.
EMA_ALPHA = 0.25
EMA_BETA = 1.0 - EMA_ALPHA

# Time tiles per sweep case. Kept small because the sweep is 1000 cases wide;
# 2 tiles is 64 time steps per channel, enough for the recurrence to settle and for
# rounding differences to accumulate through the carry.
SWEEP_NUM_TIME_TILES = 2

# k/1000 for k in 0..999. alpha=0 is a degenerate pass-through (beta=1) and is kept
# deliberately: it is the cheapest check that the carry term is actually multiplied
# in rather than ignored.
_ALPHA_STEPS = 1000


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _ema_golden(input_2d: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Continuous per-column EMA down the rows (time axis), carry starting at 0.

    EMA_new = alpha * EMA_old + beta * input. The device carries EMA_old in LREG4
    at full fp32 precision across tiles, so the recurrence runs in fp32 over the
    whole [rows, cols] input; only the stored output is later rounded to the
    output format.
    """
    rows, cols = input_2d.shape
    out = torch.empty((rows, cols), dtype=torch.float32)
    prev = torch.zeros(cols, dtype=torch.float32)
    x = input_2d.to(torch.float32)
    for t in range(rows):
        cur = alpha * prev + beta * x[t]
        out[t] = cur
        prev = cur
    return out


def _ema_golden_fp64(input_2d: torch.Tensor, alpha_f32: float, beta_f32: float):
    """The same recurrence in fp64, using the fp32-rounded alpha/beta the kernel is
    handed. Only the arithmetic is widened, not the constants, so the residual is the
    kernel's own rounding rather than a constant mismatch."""
    rows, cols = input_2d.shape
    out = torch.empty((rows, cols), dtype=torch.float64)
    prev = torch.zeros(cols, dtype=torch.float64)
    x = input_2d.to(torch.float64)
    a = torch.tensor(alpha_f32, dtype=torch.float64)
    b = torch.tensor(beta_f32, dtype=torch.float64)
    for t in range(rows):
        prev = a * prev + b * x[t]
        out[t] = prev
    return out


def _run_ema_on_device(alpha: float, beta: float, num_time_tiles: int, dest_acc):
    """Run the EMA kernel over a [num_time_tiles*32, 32] input.

    Returns (untilized device result, untilized 2D input view). Row = time step,
    column = parallel channel, for both.
    """
    torch.manual_seed(0)

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    torch_format = format_dict[formats.input_format]

    # [rows, cols] = [num_time_tiles*32 time steps, 32 parallel channels].
    input_dimensions = [num_time_tiles * TILE_DIM, TILE_DIM]
    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE

    stimuli_size = (tile_cnt * ELEMENTS_PER_TILE,)
    src_A = torch.empty(stimuli_size, dtype=torch_format).uniform_(-4.0, 4.0)
    src_B = torch.zeros_like(src_A)

    # Untilized 2D view for the golden (row = time step, col = channel).
    golden_input = src_A.view(input_dimensions[0], input_dimensions[1])

    # Tilize the input the same way the device consumes it (row-major tile order,
    # tile k = rows [32k, 32k+32)).
    src_A_tilized = tilize_block(
        src_A, input_dimensions, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/sfpu_ema_test.cpp",
        formats,
        templates=[
            APPROX_MODE(ApproximationMode.No),
            EMA_ALPHA_BETA(alpha_bits=_f32_bits(alpha), beta_bits=_f32_bits(beta)),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
        ],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, input_dimensions)
    return res_tensor, golden_input


# dest_acc=Yes (32-bit DEST) is intentionally not swept: the EMA kernel is hardcoded for
# 16-bit bf16 DEST (_ema_load/store_current_input_ use SFPLOADI_MOD0_FLOATB at fixed fp16
# offsets with no is_fp32_dest_acc_en branch), so a 32-bit DEST is not supported.
@parametrize(
    dest_acc=[DestAccumulation.No],
    num_time_tiles=[1, 2, 4],
)
def test_sfpu_ema(dest_acc, num_time_tiles):
    res_tensor, golden_input = _run_ema_on_device(
        EMA_ALPHA, EMA_BETA, num_time_tiles, dest_acc
    )
    golden_tensor = _ema_golden(golden_input, EMA_ALPHA, EMA_BETA)

    assert passed_test(
        golden_tensor, res_tensor, DataFormat.Float16_b
    ), "EMA result does not match golden"


@pytest.mark.nightly
@parametrize(
    alpha_index=list(range(_ALPHA_STEPS)),
    dest_acc=[DestAccumulation.No],
)
def test_sfpu_ema_alpha_sweep(alpha_index, dest_acc):
    """Sweep alpha = k/1000 for k in 0..999 against an fp64 reference recurrence.

    Measured against fp64 rather than a same-precision golden so the residual is
    attributable to the kernel rather than to a mix of kernel and golden error.

    Set LLK_EMA_SWEEP_DUMP=<path> to also append the raw output bits per alpha.
    Running that on two kernels and diffing the files is how bit-exactness between
    two implementations is established -- see the exactness note in the EMA perf PR.
    """
    # fp32-round alpha and derive beta the way ema_program_factory.cpp does
    # (beta_bits = bit_cast<uint32_t>(1.0f - alpha)), so this sweep exercises the
    # exact constant pair the ttnn op would produce.
    alpha = float(torch.tensor(alpha_index / _ALPHA_STEPS, dtype=torch.float32))
    beta = float(torch.tensor(1.0 - alpha, dtype=torch.float32))

    res, golden_input = _run_ema_on_device(alpha, beta, SWEEP_NUM_TIME_TILES, dest_acc)
    golden = _ema_golden_fp64(golden_input, alpha, beta)

    got = res.to(torch.float64)
    err = (got - golden).abs()
    rms = float(torch.sqrt(torch.mean(err * err)))
    peak = float(err.max())
    scale = float(torch.sqrt(torch.mean(golden * golden)))
    gpeak = float(golden.abs().max())
    # Peak error has to be judged against the peak magnitude, not the RMS: the
    # bfloat16 step at |v| is 2^-8*|v|, so the largest outputs carry the largest
    # absolute quantisation error. Reported in half-ULPs of the peak value, which
    # is the natural unit -- 1.0 would be perfect round-to-nearest with no
    # accumulation through the carry.
    half_ulp = 2**-9 * max(gpeak, 1e-30)
    peak_half_ulps = peak / half_ulp

    dump = os.environ.get("LLK_EMA_SWEEP_DUMP")
    if dump:
        with open(dump, "a") as fh:
            flat = res.flatten().to(torch.float32).tolist()
            for i, v in enumerate(flat):
                b = struct.unpack(">I", struct.pack(">f", float(v)))[0]
                fh.write("a%04d\t%d\t0x%08x\n" % (alpha_index, i, b))

    print(
        f"SWEEP alpha={alpha:<12.9g} rms={rms:.6e} peak={peak:.6e} "
        f"gpeak={gpeak:.6e} signal_rms={scale:.6e} peak_half_ulps={peak_half_ulps:.3f}"
    )

    # Loose correctness bound only. The output is bfloat16, so a half-ULP of the
    # peak value is the floor; error above that is fp32 accumulation through the
    # carry over 64 time steps. The bound is deliberately generous because the
    # point of this sweep is the printed characterisation, not a tight assertion --
    # it exists to catch a kernel that is wrong, not to police the last ULP.
    assert peak_half_ulps <= 64.0, (
        f"alpha={alpha!r}: peak error {peak:.6e} is {peak_half_ulps:.1f} half-ULPs "
        f"of the peak value {gpeak:.6e} (rms {rms:.6e}, signal rms {scale:.6e})"
    )
