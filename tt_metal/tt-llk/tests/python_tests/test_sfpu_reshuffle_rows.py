# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
reshuffle_rows SFPU test. Covers llk_sfpu/ckernel_sfpu_reshuffle_rows.h.

Each (input, accumulator) tile pair gets accumulator[mask[r]] += input[r] for every input
row r; mask entries >= 32 skip the row. The input tile must come back unchanged.
"""

from typing import List

import torch
from conftest import skip_for_quasar
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    ReshuffleRowsGolden,
    TilizeGolden,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import ApproximationMode, PerfRunType, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import BuildMode, TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    LOOP_FACTOR,
    RESHUFFLE_MASK,
    TILE_COUNT,
)
from helpers.tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    MAX_FACE_R_DIM,
)
from helpers.utils import result_tensor

SKIP = ReshuffleRowsGolden.NO_DESTINATION
ROWS = DEFAULT_TILE_R_DIM


def _mixed_mask() -> List[int]:
    """Seeded mix of repeated targets and skipped rows."""
    gen = torch.Generator().manual_seed(1234)
    targets = torch.randint(0, ROWS, (ROWS,), generator=gen)
    skipped = torch.rand(ROWS, generator=gen) < 0.25  # skip rate
    return [
        SKIP if skip else int(target)
        for target, skip in zip(targets.tolist(), skipped.tolist())
    ]


# Many-to-one masks check that a row's store lands before the next row reloads the same accumulator
# rows; the skip masks cover skipped rows inside a 4-row group and fully skipped groups.
MASKS = {
    "identity": list(range(ROWS)),
    "reverse": [ROWS - 1 - i for i in range(ROWS)],
    "all_skip": [SKIP] * ROWS,
    "halves_to_boundary": [
        MAX_FACE_R_DIM if i < MAX_FACE_R_DIM else MAX_FACE_R_DIM - 1
        for i in range(ROWS)
    ],
    "all_to_first": [0] * ROWS,
    "all_to_last": [ROWS - 1] * ROWS,
    "skip_last_row_of_group": [SKIP if i % 4 == 3 else i for i in range(ROWS)],
    "skip_first_row_of_group": [
        SKIP if i % 4 == 0 else (i * 7) % ROWS for i in range(ROWS)
    ],
    "alternate_groups_skipped": [
        SKIP if (i // 4) % 2 else ROWS - 1 - i for i in range(ROWS)
    ],
    "mixed": _mixed_mask(),
    # Every target >= 32 must be skipped, not just 255 (32 would alias row 16, 64 the next tile).
    "out_of_range": [
        [32, 47, 48, 254, 64, 100, 128, 200][(i // 2) % 8] if i % 2 else i
        for i in range(ROWS)
    ],
}

# Two pairs, so the second runs at a non-zero Dest index.
DIMENSIONS = [64, 64]
TILE_CNT = 4


def _mask_buffer(mask: List[int]) -> torch.Tensor:
    """One UInt8 tile whose first 32 bytes are the mask; the rest is the skip sentinel."""
    buffer = torch.full((ROWS * DEFAULT_TILE_C_DIM,), SKIP, dtype=torch.int32)
    buffer[: len(mask)] = torch.tensor(mask, dtype=torch.int32)
    return buffer


# Shared with perf_sfpu_reshuffle_rows.py so the two sweeps stay aligned.
RESHUFFLE_ROWS_SWEEP = dict(
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32], same=True
    ),
    dest_acc=get_valid_dest_accumulation_modes,
    mask_name=list(MASKS),
)


def _run_sfpu_reshuffle_rows(
    formats,
    dest_acc,
    mask_name,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    """Drive one reshuffle_rows variant; perf runs return after the measurement."""
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]
    mask = MASKS[mask_name]

    # Values in [-2, 2] on a grid that keeps every partial sum exact, so the check is bit-exact. Float32
    # uses a 2^-16 grid so a sum rounded to bf16 is caught; 16-bit formats use half-integers.
    step = 2**16 if formats.input_format == DataFormat.Float32 else 2
    src_A = (
        torch.randint(
            -2 * step,
            2 * step + 1,
            tuple(DIMENSIONS),
            generator=torch.Generator().manual_seed(0),
        )
        / step
    ).to(format_dict[formats.input_format])
    expected = get_golden_generator(ReshuffleRowsGolden)(
        src_A, mask, DIMENSIONS, formats.output_format
    )
    device_src_A = get_golden_generator(TilizeGolden)(
        src_A, DIMENSIONS, formats.input_format
    )

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs={
            "test_name": "sources/sfpu_reshuffle_rows_test.cpp",
            "formats": formats,
            "templates": [
                APPROX_MODE(ApproximationMode.No),
                TILE_COUNT(TILE_CNT),
                LOOP_FACTOR(loop_factor),
                RESHUFFLE_MASK(mask_name),
            ],
            "runtimes": [],
            "variant_stimuli": StimuliConfig(
                device_src_A,
                formats.input_format,
                _mask_buffer(mask),
                DataFormat.UInt8,
                formats.output_format,
                tile_count_A=TILE_CNT,
                tile_count_B=1,
                tile_count_res=TILE_CNT,
            ),
            "dest_acc": dest_acc,
            "unpack_to_dest": formats.input_format.is_32_bit(),
        },
    )
    if is_perf:
        # Perf runs write only runtimes to L1; the cost depends on the mask, so write the stimuli too.
        if TestConfig.BUILD_MODE != BuildMode.PRODUCE:
            configuration.variant_stimuli.write(TestConfig.TENSIX_LOCATION)
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result
    res_tensor = result_tensor(res_from_L1, formats.output_format)
    res_tensor = get_golden_generator(UntilizeGolden)(
        res_tensor, formats.output_format, DIMENSIONS
    )

    result = res_tensor.to(torch.float32).reshape(DIMENSIONS)
    expected = expected.to(torch.float32).reshape(DIMENSIONS)
    mismatched_rows = sorted(
        {int(r) for r in (result != expected).nonzero()[:, 0].tolist()}
    )
    assert torch.equal(result, expected), (
        f"reshuffle_rows mismatch for mask {mask_name} {mask}: logical rows "
        f"{mismatched_rows} differ\ngot\n{result}\nexpected\n{expected}"
    )


@skip_for_quasar
@parametrize(**RESHUFFLE_ROWS_SWEEP)
def test_sfpu_reshuffle_rows(formats, dest_acc, mask_name):
    _run_sfpu_reshuffle_rows(formats, dest_acc, mask_name)
