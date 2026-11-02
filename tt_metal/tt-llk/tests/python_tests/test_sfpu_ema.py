# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional coverage for the EMA SFPU kernel.

Three tests, all driving `sources/sfpu_ema_test.cpp` through `_run_sfpu_ema`:

- `test_sfpu_ema` -- the default-suite check, one alpha over 1/2/4 time tiles.
- `test_sfpu_ema_after_binary_init` -- the same with an eltwise binary init before each `ema_tile`.
- `test_sfpu_ema_alpha_sweep` -- nightly, 1000 alphas against an fp64 reference.

`perf_sfpu_ema.py` reuses `_run_sfpu_ema` with `is_perf=True`.

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
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    Transpose,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import PerfRunType, create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    EMA_ALPHA_BETA,
    EMA_INTERLEAVED_INIT,
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    fp32_bits,
)
from helpers.tile_constants import MAX_FACE_R_DIM, MAX_NUM_FACES
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test, result_tensor

# EMA smoothing factor chosen for the functional test (ttnn.ema has no default alpha;
# it is a required argument). beta is derived as 1 - alpha.
EMA_ALPHA = 0.25
EMA_BETA = 1.0 - EMA_ALPHA

_FUNCTIONAL_FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)

SFPU_EMA_PERF_SWEEP = dict(
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float32],
        same=True,
    ),
    dest_acc=get_valid_dest_accumulation_modes,
)

# Time tiles per sweep case. Kept small because the sweep is 1000 cases wide;
# 2 tiles is 64 time steps per channel, enough for the recurrence to settle and for
# rounding differences to accumulate through the carry.
SWEEP_NUM_TIME_TILES = 2

AFTER_INIT_NUM_TIME_TILES = 2

# k/1000 for k in 0..999. alpha=0 is a degenerate pass-through (beta=1) and is kept
# deliberately: it is the cheapest check that the carry term is actually multiplied
# in rather than ignored.
_ALPHA_STEPS = 1000


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


def _run_sfpu_ema(
    alpha: float,
    beta: float,
    dest_acc,
    *,
    formats=_FUNCTIONAL_FORMATS,
    num_time_tiles: int | None = None,
    input_dimensions: list[int] | None = None,
    binary_init_before_ema: bool = False,
    is_perf: bool = False,
    perf_report=None,
    run_types=None,
    loop_factor: int = 1,
):
    """Run the shared functional/perf EMA kernel.

    Exactly one of ``num_time_tiles`` and ``input_dimensions`` selects the input size.
    Functional callers receive the untilized result plus its 2D input view.

    With ``is_perf=True``, ``perf_report`` is required, ``run_types`` selects the
    measured paths, ``loop_factor`` amortizes their tile loops, and the runner returns
    ``None`` without generating host stimuli or a golden. ``binary_init_before_ema``
    is functional-only; it runs an eltwise binary init between datacopy and ``ema_tile``.
    """
    if is_perf and binary_init_before_ema:
        raise ValueError("binary_init_before_ema is not supported when is_perf=True")
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]

    if input_dimensions is None:
        if num_time_tiles is None:
            raise ValueError("num_time_tiles or input_dimensions must be provided")
        input_dimensions = [num_time_tiles * TILE_DIM, TILE_DIM]
    elif num_time_tiles is not None:
        raise ValueError("provide num_time_tiles or input_dimensions, not both")

    tile_cnt, _, faces_to_generate = calculate_tile_and_face_counts(
        input_dimensions,
        input_dimensions,
        face_r_dim=MAX_FACE_R_DIM,
        num_faces=MAX_NUM_FACES,
    )

    src_A_tilized = src_B = golden_input = None
    if not is_perf:
        torch.manual_seed(0)
        torch_format = format_dict[formats.input_format]
        stimuli_size = (tile_cnt * ELEMENTS_PER_TILE,)
        src_A = torch.empty(stimuli_size, dtype=torch_format).uniform_(-4.0, 4.0)
        src_B = torch.zeros_like(src_A)

        # Untilized 2D view for the golden (row = time step, col = channel).
        golden_input = src_A.view(input_dimensions[0], input_dimensions[1])

        # Tilize the input the same way the device consumes it (row-major tile
        # order, tile k = rows [32k, 32k+32)).
        src_A_tilized = tilize_block(
            src_A, input_dimensions, stimuli_format=formats.input_format
        ).flatten()

    test_config_kwargs = {
        "test_name": "sources/sfpu_ema_test.cpp",
        "formats": formats,
        "templates": [
            APPROX_MODE(ApproximationMode.No),
            EMA_ALPHA_BETA(alpha_bits=fp32_bits(alpha), beta_bits=fp32_bits(beta)),
            TILE_COUNT(tile_cnt),
            LOOP_FACTOR(loop_factor),
            NUM_FACES(num_faces=faces_to_generate),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        "runtimes": [],
        "variant_stimuli": StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt if is_perf else 1,
            tile_count_res=tile_cnt,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": formats.input_format.is_32_bit(),
        "compile_time_formats": True,
    }
    if not is_perf:
        test_config_kwargs["templates"].append(
            EMA_INTERLEAVED_INIT(binary_init_before_ema=binary_init_before_ema)
        )
        test_config_kwargs["disable_format_inference"] = True

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    res_tensor = result_tensor(res_from_L1, formats.output_format)
    res_tensor = untilize_block(res_tensor, formats.output_format, input_dimensions)
    return res_tensor, golden_input


# Both DEST widths are swept. The kernel's SFPLOAD/SFPSTORE take the element format from
# the configured DEST rather than fixing it, so each width has to be covered on its own;
# the ttnn ema op runs with a 32-bit DEST, so that leg is the one production uses.
@parametrize(
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    num_time_tiles=[1, 2, 4],
)
def test_sfpu_ema(dest_acc, num_time_tiles):
    res_tensor, golden_input = _run_sfpu_ema(
        EMA_ALPHA, EMA_BETA, dest_acc, num_time_tiles=num_time_tiles
    )
    golden_tensor = _ema_golden(golden_input, EMA_ALPHA, EMA_BETA)

    assert passed_test(
        golden_tensor, res_tensor, DataFormat.Float16_b
    ), "EMA result does not match golden"


# One axis, so pytest's own parametrize: the harness helper hands a single axis over as a tuple.
@pytest.mark.parametrize(
    "dest_acc",
    [DestAccumulation.No, DestAccumulation.Yes],
    ids=["dest_acc:No", "dest_acc:Yes"],
)
def test_sfpu_ema_after_binary_init(dest_acc):
    res_tensor, golden_input = _run_sfpu_ema(
        EMA_ALPHA,
        EMA_BETA,
        dest_acc,
        num_time_tiles=AFTER_INIT_NUM_TIME_TILES,
        binary_init_before_ema=True,
    )
    golden_tensor = _ema_golden(golden_input, EMA_ALPHA, EMA_BETA)

    assert passed_test(
        golden_tensor, res_tensor, DataFormat.Float16_b
    ), "EMA result after an eltwise binary init does not match golden"


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

    res, golden_input = _run_sfpu_ema(
        alpha, beta, dest_acc, num_time_tiles=SWEEP_NUM_TIME_TILES
    )
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
