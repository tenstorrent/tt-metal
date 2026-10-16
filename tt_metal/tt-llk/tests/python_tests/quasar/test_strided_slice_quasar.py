# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import PerfRunType, format_dict
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR, STRIDED_SLICE

TILE_ROWS = 32
TILE_COLS = 32
SLICE_END = TILE_COLS

# (start, step) pairs; end is the full row.
STRIDED_SLICE_PARAMS = [(0, 2), (1, 2), (0, 3), (0, 4), (0, 8)]
# Perf runs LOOP_FACTOR iterations per case, so it sweeps a reduced set: a small step (most kept
# elements, where FPU compress should lead), a large step (fewest kept, where RVV vlse should lead),
# and the midpoint between them.
PERF_STRIDED_SLICE_PARAMS = [(0, 2), (0, 4), (0, 8)]

STRIDED_SLICE_SOURCES = {
    "llk": {
        "cpp": "sources/quasar/strided_slice/strided_slice_fpu_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.L1_TO_L1],
    },
    "scalar": {
        "cpp": "sources/quasar/strided_slice/strided_slice_scalar_quasar_test.cpp",
        "requires_vector_ext": False,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
    "rvv_vlse": {
        "cpp": "sources/quasar/strided_slice/strided_slice_rvv_vlse_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
}


@pytest.mark.quasar
@pytest.mark.parametrize("slice_start, slice_step", STRIDED_SLICE_PARAMS)
@pytest.mark.parametrize("source", list(STRIDED_SLICE_SOURCES))
def test_strided_slice_quasar(
    source,
    slice_start,
    slice_step,
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

    golden_tensor = src_A.view(TILE_ROWS, TILE_COLS)[
        :, slice_start:SLICE_END:slice_step
    ]
    kept_cnt = golden_tensor.shape[1]

    test_config_kwargs = {
        "test_name": STRIDED_SLICE_SOURCES[source]["cpp"],
        "formats": formats,
        "templates": [],
        "runtimes": [
            LOOP_FACTOR(loop_factor),
            STRIDED_SLICE(slice_start, SLICE_END, slice_step),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            # Scratch only: llk builds its keep mask here on device from STRIDED_SLICE;
            # scalar and vlse do not read buffer B.
            torch.zeros(TILE_COLS, dtype=torch.uint8),
            DataFormat.UInt8,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=4,
            face_r_dim=16,
            tile_dimensions=[TILE_ROWS, TILE_COLS],
            use_dense_tile_dimensions=True,
        ),
        "requires_vector_ext": STRIDED_SLICE_SOURCES[source]["requires_vector_ext"],
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
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(
        res_from_L1, dtype=format_dict[formats.output_format]
    ).view(TILE_ROWS, TILE_COLS)[:, :kept_cnt]

    mismatches = [
        (r, c, golden_tensor[r, c].item(), res_tensor[r, c].item())
        for r in range(TILE_ROWS)
        for c in range(kept_cnt)
        if res_tensor[r, c] != golden_tensor[r, c]
    ]
    assert not mismatches, (
        f"{source} strided slice [{slice_start}:{SLICE_END}:{slice_step}] returned the wrong "
        f"element for {len(mismatches)} of {TILE_ROWS * kept_cnt} kept elements "
        "(row, col, expected, actual), first 32: " + str(mismatches[:32])
    )
