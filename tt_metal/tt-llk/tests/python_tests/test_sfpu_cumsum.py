# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Cumsum vehicle (lane FK cumsum-fresh registration — first vehicle for
this op).

Contract (the production compute-API cumsum_tile at Wt=1): inclusive prefix
sum down the rows (per column), in place, consecutive tiles CONTINUING the
row sequence — a [num_time_tiles*32, 32] input is a cumsum over
num_time_tiles*32 rows for 32 parallel columns.

Arms: CUMSUM_IMPL 0 = production raw-TTI kernel (ckernel_sfpu_cumsum.h,
SFPTRANSP-bracketed replay chains, LREG4-7 running-prefix cross-call ABI);
CUMSUM_IMPL 1 = the first typed semantic body (fresh_cpp/cumsum.h).
"""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    PerfRunType,
    format_dict,
)
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CUMSUM_IMPL,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

_CUMSUM_IMPL_IDS = {0: "production", 1: "fresh"}


def _cumsum_golden(input_2d: torch.Tensor) -> torch.Tensor:
    """Serial per-column prefix in fp32 (the device recurrence runs at Dst
    precision per add; the fp32 golden rides the format tolerance gate)."""
    return torch.cumsum(input_2d.to(torch.float32), dim=0)


@pytest.mark.parametrize("cumsum_impl", [0, 1], ids=lambda i: _CUMSUM_IMPL_IDS[i])
@pytest.mark.parametrize("num_time_tiles", [1, 32], ids=lambda n: f"t{n}")
def test_sfpu_cumsum(num_time_tiles, cumsum_impl):
    torch.manual_seed(0)

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    torch_format = format_dict[formats.input_format]

    input_dimensions = [num_time_tiles * TILE_DIM, TILE_DIM]
    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE

    # Positive, well-conditioned stimuli: a 1024-step bf16 prefix over a
    # sign-mixed input loses relative accuracy to cancellation, which would
    # gate the FORMAT tolerance rather than the kernels under test.
    src_A = torch.empty((tile_cnt * ELEMENTS_PER_TILE,), dtype=torch_format).uniform_(
        0.001, 0.1
    )
    src_B = torch.zeros_like(src_A)

    golden_input = src_A.view(input_dimensions[0], input_dimensions[1])
    golden_tensor = _cumsum_golden(golden_input)

    src_A_tilized = tilize_block(
        src_A, input_dimensions, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/sfpu_cumsum_test.cpp",
        formats,
        templates=[
            APPROX_MODE(ApproximationMode.No),
            CUMSUM_IMPL(cumsum_impl),
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
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, input_dimensions)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), f"[{_CUMSUM_IMPL_IDS[cumsum_impl]}] cumsum result does not match golden"


@pytest.mark.parametrize("cumsum_impl", [0, 1], ids=lambda i: _CUMSUM_IMPL_IDS[i])
@pytest.mark.parametrize("num_time_tiles", [1, 32], ids=lambda n: f"t{n}")
def test_sfpu_cumsum_device_profile(perf_report, num_time_tiles, cumsum_impl):
    """MATH-zone sample of CUMSUM_BODY per arm/tile count (zone inside the
    tile loop: mean(MATH_ISOLATE) is per tile at every count)."""
    torch.manual_seed(0)
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    torch_format = format_dict[formats.input_format]

    tile_cnt = num_time_tiles
    src_A = torch.empty((tile_cnt * ELEMENTS_PER_TILE,), dtype=torch_format).uniform_(
        0.001, 0.1
    )
    src_B = torch.zeros_like(src_A)
    src_A_tilized = tilize_block(
        src_A, [tile_cnt * TILE_DIM, TILE_DIM], stimuli_format=formats.input_format
    ).flatten()

    configuration = PerfConfig(
        "sources/sfpu_cumsum_test.cpp",
        formats,
        run_types=[PerfRunType.MATH_ISOLATE],
        templates=[
            APPROX_MODE(ApproximationMode.No),
            CUMSUM_IMPL(cumsum_impl),
        ],
        runtimes=[TILE_COUNT(tile_cnt)],
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
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    configuration.run(perf_report, run_count=1)
    frame = perf_report.frame()
    rows = frame[frame["marker"] == "CUMSUM_BODY"]
    assert len(rows) == 1, frame.to_string(index=False)
    cycles = float(rows.iloc[0]["mean(MATH_ISOLATE)"])
    assert cycles > 0
    print(
        f"CUMSUM_DEVICE_PROFILE {_CUMSUM_IMPL_IDS[cumsum_impl]} "
        f"num_time_tiles={num_time_tiles} math_cycles_per_tile={int(cycles)}"
    )
