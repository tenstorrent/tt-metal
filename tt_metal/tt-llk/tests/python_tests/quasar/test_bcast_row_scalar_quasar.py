# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import PerfRunType, format_dict
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR

TILE_ROWS = 32
TILE_COLS = 32
FACE_ELEM_CNT = 16 * 16
FACE_ROW_ELEM_CNT = 16

BCAST_ROW_LANES = list(range(FACE_ROW_ELEM_CNT)) + list(
    range(FACE_ELEM_CNT, FACE_ELEM_CNT + FACE_ROW_ELEM_CNT)
)

BCAST_ROW_SOURCES = {
    "scalar": {
        "cpp": "sources/quasar/bcast_experiments/bcast_row_scalar_quasar_test.cpp",
        "requires_vector_ext": False,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
    "rvv": {
        "cpp": "sources/quasar/bcast_experiments/bcast_row_rvv_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
    "vgather": {
        "cpp": "sources/quasar/bcast_experiments/bcast_row_vgather_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.L1_TO_L1],
    },
}


@pytest.mark.quasar
@pytest.mark.parametrize("source", list(BCAST_ROW_SOURCES))
def test_bcast_row_scalar_quasar(
    source,
    *,
    run_types=(PerfRunType.L1_TO_L1,),
    loop_factor=1,
    is_perf=False,
    perf_report=None,
):
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    formats = InputOutputFormat(
        input_format=DataFormat.Float16_b,
        output_format=DataFormat.Float16_b,
    )

    torch.manual_seed(42)
    src_A = torch.randn(TILE_ROWS * TILE_COLS).to(torch.bfloat16)
    # The kernels read the first u32 of buffer A as the scalar, double packed: both
    # 16-bit halves must hold the same value.
    src_A[1] = src_A[0]
    scalar = src_A[0]
    src_B = torch.zeros(TILE_ROWS * TILE_COLS, dtype=torch.bfloat16)  # unused

    test_config_kwargs = {
        "test_name": BCAST_ROW_SOURCES[source]["cpp"],
        "formats": formats,
        "templates": [],
        "runtimes": [LOOP_FACTOR(loop_factor)],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=4,
            face_r_dim=16,
            tile_dimensions=[TILE_ROWS, TILE_COLS],
            use_dense_tile_dimensions=True,
        ),
        "requires_vector_ext": BCAST_ROW_SOURCES[source]["requires_vector_ext"],
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    assert (
        len(res_from_L1) == TILE_ROWS * TILE_COLS
    ), "Result tensor is not a full 32x32 tile"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    mismatches = [
        (i, scalar.item(), res_tensor[i].item())
        for i in BCAST_ROW_LANES
        if res_tensor[i] != scalar
    ]
    assert not mismatches, (
        f"{source} row bcast wrote the wrong value to "
        f"{len(mismatches)} of {len(BCAST_ROW_LANES)} row-0 lanes "
        "(L1 lane, expected, actual): " + str(mismatches)
    )
