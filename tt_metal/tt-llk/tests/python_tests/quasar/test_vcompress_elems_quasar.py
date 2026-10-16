# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import PerfRunType, format_dict
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import LOOP_FACTOR

COMPRESS_ELEM_CNT = 32
TILE_DIMENSIONS = [1, COMPRESS_ELEM_CNT]
NUM_FACES = 2
FACE_R_DIM = 1


def generate_mask() -> torch.Tensor:
    return torch.randint(0, 2, (COMPRESS_ELEM_CNT,), dtype=torch.uint8)


def pack_mask(mask: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([1 << b for b in range(8)], dtype=torch.int32)
    packed = (mask.view(-1, 8).to(torch.int32) * weights).sum(dim=1).to(torch.uint8)
    padded = torch.zeros(COMPRESS_ELEM_CNT, dtype=torch.uint8)
    padded[: packed.numel()] = packed
    return padded


def compress_golden(src: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    keep = mask.to(torch.bool)
    kept = src[..., keep]
    golden = torch.zeros_like(src)
    golden[..., : kept.shape[-1]] = kept
    return golden


VCOMPRESS_SOURCES = {
    "llk": {
        "cpp": "sources/quasar/vcompress_row/vcompress_row_fpu_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.L1_TO_L1],
    },
    "scalar": {
        "cpp": "sources/quasar/vcompress_row/vcompress_row_scalar_quasar_test.cpp",
        "requires_vector_ext": False,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
    "rvv_vcompress": {
        "cpp": "sources/quasar/vcompress_row/vcompress_row_rvv_vcompress_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
}


@pytest.mark.quasar
@pytest.mark.parametrize("source", list(VCOMPRESS_SOURCES))
def test_vcompress_elems_quasar(
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
    src_A = torch.randn(COMPRESS_ELEM_CNT).to(torch.bfloat16)
    mask = generate_mask()

    golden_tensor = compress_golden(src_A, mask)
    kept_cnt = int(mask.sum().item())

    test_config_kwargs = {
        "test_name": VCOMPRESS_SOURCES[source]["cpp"],
        "formats": formats,
        "templates": [],
        "runtimes": [LOOP_FACTOR(loop_factor)],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            pack_mask(mask),
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
        "requires_vector_ext": VCOMPRESS_SOURCES[source]["requires_vector_ext"],
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
        (i, golden_tensor[i].item(), res_tensor[i].item())
        for i in range(kept_cnt)
        if res_tensor[i] != golden_tensor[i]
    ]
    assert not mismatches, (
        f"{source} compress returned the wrong element for "
        f"{len(mismatches)} of {kept_cnt} kept lanes "
        "(lane, expected, actual): " + str(mismatches)
    )


TILE_ROWS = 32
TILE_COLS = COMPRESS_ELEM_CNT

VCOMPRESS_TILE_SOURCES = {
    "llk": {
        "cpp": "sources/quasar/vcompress_tile/vcompress_tile_fpu_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.L1_TO_L1],
    },
    "scalar": {
        "cpp": "sources/quasar/vcompress_tile/vcompress_tile_scalar_quasar_test.cpp",
        "requires_vector_ext": False,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
    "rvv_vcompress": {
        "cpp": "sources/quasar/vcompress_tile/vcompress_tile_rvv_vcompress_quasar_test.cpp",
        "requires_vector_ext": True,
        "perf_run_types": [PerfRunType.UNPACK_ISOLATE],
    },
}


@pytest.mark.quasar
@pytest.mark.parametrize("source", list(VCOMPRESS_TILE_SOURCES))
def test_vcompress_tile_quasar(
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
    mask = generate_mask()

    golden_tensor = compress_golden(src_A.view(TILE_ROWS, TILE_COLS), mask)
    kept_cnt = int(mask.sum().item())

    test_config_kwargs = {
        "test_name": VCOMPRESS_TILE_SOURCES[source]["cpp"],
        "formats": formats,
        "templates": [],
        "runtimes": [LOOP_FACTOR(loop_factor)],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            # Only the packed 32-bit mask is written; buffer B keeps a full 32x32 UInt8
            # slot in L1 (tile_dimensions is shared), of which the kernels read the
            # first 4 bytes.
            pack_mask(mask),
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
        "requires_vector_ext": VCOMPRESS_TILE_SOURCES[source]["requires_vector_ext"],
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
        (r, w, golden_tensor[r, w].item(), res_tensor[r, w].item())
        for r in range(TILE_ROWS)
        for w in range(kept_cnt)
        if res_tensor[r, w] != golden_tensor[r, w]
    ]
    assert not mismatches, (
        f"{source} tile compress returned the wrong element for "
        f"{len(mismatches)} of {TILE_ROWS * kept_cnt} kept elements "
        "(row, lane, expected, actual), first 32: " + str(mismatches[:32])
    )
