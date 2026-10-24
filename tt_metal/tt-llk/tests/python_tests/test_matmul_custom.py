# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import BootMode
from helpers.format_config import DataFormat, InputOutputFormat, is_dest_acc_needed
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.matmul_sweep import (
    generate_matmul_dimension_combinations,
    generate_tile_dims,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    MATH_FIDELITY,
    NUM_FACES,
    THROTTLE_LEVEL,
    TILE_COUNT,
    TemplateParameter,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# Throttle levels supported by the no-mop matmul math LLK. Throttle only inserts
# NOPs between MVMULs to cap compute throughput, so the numeric result is IDENTICAL
# for every level -> the golden is the same MatmulGolden as the non-throttled path.
#
# Blackhole implements run_throttled_sequence_no_mop<1..5>, so it sweeps 0-5.
# Wormhole B0's LLK static_asserts THROTTLE_LEVEL == 0 (llk_math_matmul_custom_no_mop.h:
# "Wormhole custom no-mop matmul only supports THROTTLE_LEVEL == 0") -- levels 1-5 have no
# throttle sequences and do not compile there. Any non-Blackhole arch is therefore level 0
# only; throttle 1-5 on WH is unsupported by design, not an untested gap. This gate is
# arch-explicit (not a bare else) so a future arch that gains throttle sequences must opt in.
if get_chip_architecture() == ChipArchitecture.BLACKHOLE:
    THROTTLE_LEVELS = [0, 1, 2, 3, 4, 5]
else:
    THROTTLE_LEVELS = [0]


@dataclass
class MATMUL_NO_MOP_PRESERVE_SRC_ZERO_FLAG(TemplateParameter):
    """Math sets the Src zero flag to keep before the no-mop init, as a preceding copy_tile_init does."""

    no_mop_preserve_src_zero_flag: bool = False

    def convert_to_cpp(self) -> str:
        return f"constexpr bool MATMUL_NO_MOP_PRESERVE_SRC_ZERO_FLAG = {str(self.no_mop_preserve_src_zero_flag).lower()};"


def _dest_bank_max_tiles(formats, dest_acc):
    """Max result-tile count for a (format, dest_acc) pair on DestSync.Half.

    A 32-bit destination register (dest_acc=Yes, or a format the harness forces
    onto 32-bit dest) holds 4 tiles; the 16-bit destination holds 8. Mirrors
    perf_matmul._dest_bank_max_tiles so test and perf coverage share the rule.
    """
    if is_dest_acc_needed(formats) or dest_acc == DestAccumulation.Yes:
        return 4
    return 8


MATMUL_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float32,
        DataFormat.Bfp8_b,
    ]
)
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]


def _run_matmul_custom(
    math_fidelity,
    formats,
    dest_acc,
    input_A_dimensions,
    input_B_dimensions,
    throttle_level: int,
    boot_mode: BootMode,
    preserve_zero_flag: bool = False,
    operands=None,
):
    """Run the no-mop matmul and check it against the golden; return the tilized result.

    ``operands`` optionally supplies ``(A, B)`` row-major tensors instead of the uniform stimuli.
    """
    torch_format = format_dict[formats.output_format]

    sfpu_false_spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=sfpu_false_spec,
        spec_B=sfpu_false_spec,
    )
    if operands is not None:
        torch_a, torch_b = operands
        src_A = torch_a.reshape(src_A.shape).to(src_A.dtype)
        src_B = torch_b.reshape(src_B.shape).to(src_B.dtype)

    # Calculate all matmul dimensions using helper function
    matmul_dims = generate_tile_dims((input_A_dimensions, input_B_dimensions))

    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        # Golden cannot model FPU strided for tilized data computation, so we tilize output after computation
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )

    if formats.input_format != DataFormat.Bfp8_b:
        tilized_A = tilize_block(
            src_A, dimensions=input_A_dimensions, stimuli_format=formats.input_format
        )
        tilized_B = tilize_block(
            src_B, dimensions=input_B_dimensions, stimuli_format=formats.input_format
        )
    else:
        # BFP8 format requires special handling for tilization
        tilized_A = src_A
        tilized_B = src_B

    configuration = TestConfig(
        "sources/matmul_custom_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            THROTTLE_LEVEL(throttle_level),
            MATMUL_NO_MOP_PRESERVE_SRC_ZERO_FLAG(preserve_zero_flag),
        ],
        runtimes=[
            NUM_FACES(),
            TILE_COUNT(matmul_dims.output_tile_cnt),
            CRK_TILE_DIMM(matmul_dims.ct_dim, matmul_dims.rt_dim, matmul_dims.kt_dim),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=matmul_dims.output_tile_cnt,
        ),
        dest_acc=dest_acc,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"

    return res_tensor


@parametrize(
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    formats=MATMUL_FORMATS,
    dest_acc=DEST_ACC_MODES,
    # dimensions depends on (formats, dest_acc): the max result-tile count is set by
    # which dest register the pair lands in (see _dest_bank_max_tiles). Kept as its
    # own axis -- rather than a packed (format, dest_acc, dims) tuple -- so test and
    # perf coverage line up axis-for-axis (cf. perf_matmul.matmul_combos).
    dimensions=lambda formats, dest_acc: generate_matmul_dimension_combinations(
        _dest_bank_max_tiles(formats, dest_acc)
    ),
)
def test_matmul_custom(
    math_fidelity,
    formats,
    dest_acc,
    dimensions,
    boot_mode=BootMode.DEFAULT,
):
    input_A_dimensions, input_B_dimensions = dimensions
    _run_matmul_custom(
        math_fidelity,
        formats,
        dest_acc,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=0,
        boot_mode=boot_mode,
    )


# Representative fidelity x format subset for the throttle sweep. Throttle inserts
# NOPs between MVMULs without changing the numeric result, so it is orthogonal to
# format/fidelity/dims; a small subset per throttle level keeps the sweep tractable
# while still crossing LoFi (single-phase) and HiFi (multi-phase) code paths and
# both 16-bit (Float16_b) and 32-bit (Float32) operands.
THROTTLE_FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])
# Single 32x32 output tile (ct=rt=kt=1). This is the regime the throttled no-mop
# matmul is designed and used in (the SDPA full-tile path); the hand-written
# throttle sequences replay a fixed single-tile MVMUL walk. Multi-tile / multi-K
# accumulation is validated at throttle 0 by test_matmul_custom above.
THROTTLE_DIMS = ([32, 32], [32, 32])


@parametrize(
    throttle_level=THROTTLE_LEVELS,
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    formats=THROTTLE_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_matmul_custom_throttle(
    throttle_level,
    math_fidelity,
    formats,
    dest_acc,
    boot_mode=BootMode.DEFAULT,
):
    # Known limitation of the current LLK: throttle levels 4 and 5 only advance
    # the fidelity-phase counter on the final MVMUL of the sequence (they use
    # ADDR_MOD_4 with fidelity.incr=0 at the phase boundary, unlike levels 1-3
    # which use the fidelity-incrementing ADDR_MOD_5/6). For a high-fidelity
    # (multi-phase) matmul this collapses the extra phases and yields ~half the
    # result, so levels 4/5 are only correct for single-phase (LoFi) fidelity.
    # Cover levels 4/5 with LoFi and levels 0-3 with both fidelities.
    if throttle_level >= 4 and math_fidelity != MathFidelity.LoFi:
        pytest.skip(
            "throttle levels 4/5 do not increment the fidelity phase per LLK; "
            "only correct for LoFi (single-phase)"
        )

    input_A_dimensions, input_B_dimensions = THROTTLE_DIMS
    _run_matmul_custom(
        math_fidelity,
        formats,
        dest_acc,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=throttle_level,
        boot_mode=boot_mode,
    )


# Smallest positive bf16 denormal, 2^-133; bf16 denormals are its multiples 1..127 (7 mantissa bits).
BF16_DENORMAL_MIN = 2.0**-133
BF16_DENORMAL_MANTISSAS = 1 << 7
# B is scaled up so a kept denormal gives a visibly nonzero product; the normal rows of A are scaled down
# by the same amount so their products stay O(1).
OPERAND_SCALE = 2.0**100
# ct=1 (single tile), ct=2 (in1 walks, in0 reused) and rt=2 (in0 walks, in1 reused): every reuse arm of
# the no-mop replay, on a single k-tile so each output row is one MVMUL pass over its in0 row.
DENORMAL_DIMS = [
    ([32, 32], [32, 32]),
    ([32, 32], [32, 64]),
    ([64, 32], [32, 32]),
]


@pytest.mark.parametrize("math_fidelity", [MathFidelity.LoFi, MathFidelity.HiFi4])
@pytest.mark.parametrize("dimensions", DENORMAL_DIMS)
def test_matmul_custom_flushes_denormal_in0_after_keep_flag(math_fidelity, dimensions):
    """The no-mop init must restore the Src zero flag a preceding datacopy left at keep.

    Rows 0-3 of A (in0, loaded to SrcB) are bf16 denormals, which a bf16 MVMUL flushes to zero; scaled by
    B they give nonzero products if kept, so those output rows must come out exactly 0. The golden cannot
    tell flushed rows from kept ones at its tolerance, so only the exact-zero check guards them. The other
    rows are normal and check the rest of the product against the golden.
    """
    input_A_dimensions, input_B_dimensions = dimensions
    M, K = input_A_dimensions
    _, N = input_B_dimensions
    denormal_rows = 4

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.float32) / OPERAND_SCALE
    torch_a[:denormal_rows] = (
        BF16_DENORMAL_MIN
        * torch.randint(1, BF16_DENORMAL_MANTISSAS, (denormal_rows, K)).float()
    )
    torch_b = torch.randn((K, N), dtype=torch.float32) * OPERAND_SCALE

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    res = _run_matmul_custom(
        math_fidelity,
        formats,
        DestAccumulation.No,
        input_A_dimensions,
        input_B_dimensions,
        throttle_level=0,
        boot_mode=BootMode.DEFAULT,
        preserve_zero_flag=True,
        operands=(torch_a, torch_b),
    )
    out = untilize_block(
        res, stimuli_format=formats.output_format, dimensions=[M, N]
    ).reshape(M, N)
    kept = out[:denormal_rows].float()
    assert (
        torch.count_nonzero(kept) == 0
    ), f"denormal in0 rows were not flushed: max |value| {kept.abs().max().item():.3e}"
