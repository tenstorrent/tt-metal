# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import PerfRunType, format_dict
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR

GATHER_ELEM_CNT = 32
TILE_DIMENSIONS = [1, GATHER_ELEM_CNT]
NUM_FACES = 2
FACE_R_DIM = 1


def generate_indices() -> torch.Tensor:
    """Random gather indices in [0, GATHER_ELEM_CNT), one per output lane (repeats allowed)."""
    return torch.randint(0, GATHER_ELEM_CNT, (GATHER_ELEM_CNT,), dtype=torch.uint8)


VGATHER_SOURCES = {
    "llk": {
        "cpp": "sources/quasar/vgather_row/vgather_row_fpu_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.L1_TO_L1],
    },
    "scalar": {
        "cpp": "sources/quasar/vgather_row/vgather_row_scalar_quasar_test.cpp",
        "requires_vector_ext": False,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
    "rvv": {
        "cpp": "sources/quasar/vgather_row/vgather_row_rvv_lux_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
    "rvv_vrgather": {
        "cpp": "sources/quasar/vgather_row/vgather_row_rvv_vrgather_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
}


@pytest.mark.quasar
@pytest.mark.parametrize("source", list(VGATHER_SOURCES))
def test_vgather_elems_quasar(
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
    src_A = torch.randn(GATHER_ELEM_CNT).to(torch.bfloat16)
    indices = generate_indices()

    golden_tensor = src_A[indices.to(torch.int64)]

    test_config_kwargs = {
        "test_name": VGATHER_SOURCES[source]["cpp"],
        "formats": formats,
        "templates": [],
        "runtimes": [LOOP_FACTOR(loop_factor)],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            indices,
            DataFormat.UInt8,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=NUM_FACES,
            face_r_dim=FACE_R_DIM,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
        ),
        "requires_vector_ext": VGATHER_SOURCES[source]["requires_vector_ext"],
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

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    mismatches = [
        (i, indices[i].item(), golden_tensor[i].item(), res_tensor[i].item())
        for i in range(GATHER_ELEM_CNT)
        if res_tensor[i] != golden_tensor[i]
    ]
    assert not mismatches, (
        f"{source} gather returned the wrong element for "
        f"{len(mismatches)} of {GATHER_ELEM_CNT} lanes "
        "(lane, index, expected, actual): " + str(mismatches)
    )


TILE_ROWS = 32
TILE_COLS = GATHER_ELEM_CNT

VGATHER_TILE_SOURCES = {
    "llk": {
        "cpp": "sources/quasar/vgather_tile/vgather_tile_fpu_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.L1_TO_L1],
    },
    "scalar": {
        "cpp": "sources/quasar/vgather_tile/vgather_tile_scalar_quasar_test.cpp",
        "requires_vector_ext": False,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
    "rvv": {
        "cpp": "sources/quasar/vgather_tile/vgather_tile_rvv_lux_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
    "rvv_vrgather": {
        "cpp": "sources/quasar/vgather_tile/vgather_tile_rvv_vrgather_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
}


@pytest.mark.quasar
@pytest.mark.parametrize("source", list(VGATHER_TILE_SOURCES))
def test_vgather_tile_quasar(
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
    indices = generate_indices()

    golden_tensor = src_A.view(TILE_ROWS, TILE_COLS)[:, indices.to(torch.int64)]

    test_config_kwargs = {
        "test_name": VGATHER_TILE_SOURCES[source]["cpp"],
        "formats": formats,
        "templates": [],
        "runtimes": [LOOP_FACTOR(loop_factor)],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            # Only the 1x32 index vector is written; buffer B keeps a full 32x32 UInt8
            # slot in L1 (tile_dimensions is shared), of which the kernels read the
            # first GATHER_ELEM_CNT bytes.
            indices,
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
        "requires_vector_ext": VGATHER_TILE_SOURCES[source]["requires_vector_ext"],
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
        len(res_from_L1) == golden_tensor.numel()
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(
        res_from_L1, dtype=format_dict[formats.output_format]
    ).view(TILE_ROWS, TILE_COLS)

    mismatches = [
        (r, w, indices[w].item(), golden_tensor[r, w].item(), res_tensor[r, w].item())
        for r in range(TILE_ROWS)
        for w in range(TILE_COLS)
        if res_tensor[r, w] != golden_tensor[r, w]
    ]
    assert not mismatches, (
        f"{source} tile gather returned the wrong element for "
        f"{len(mismatches)} of {TILE_ROWS * TILE_COLS} elements "
        "(row, lane, index, expected, actual), first 32: " + str(mismatches[:32])
    )
