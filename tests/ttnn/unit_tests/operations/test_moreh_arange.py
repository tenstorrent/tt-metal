# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_allclose_and_pcc
from tests.ttnn.unit_tests.operations.test_utils import get_lib_dtype


def run_moreh_arange(start_end_step, optional_output, dtype, tilized, device):
    """Run a comparison of arange results between torch and ttnn."""
    # Prepare inputs
    start, end, step = start_end_step
    if (dtype == "int32") and (start != int(start) or end != int(end) or step != int(step)):
        pytest.skip(f"start, end, and step must be integers when using int32 dtype")
    if dtype == "bfloat8_b" and not tilized:
        pytest.skip(f"bfloat8_b requires TILE_LAYOUT")

    # TODO @mrshaw01: Support bfloat8_b in kernel
    if dtype == "bfloat8_b":
        pytest.skip(f"bfloat8_b is not supported in the kernel")

    # Compute using torch
    torch_dtype = get_lib_dtype(torch, dtype)
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    expected_output = torch.arange(start=start, end=end, step=step).to(torch_dtype)

    # Compute using ttnn
    ttnn_dtype = get_lib_dtype(ttnn, dtype)
    if ttnn_dtype is None:
        ttnn_dtype = ttnn.bfloat16
    any_cpu = torch.randn([32, 32])
    any_npu = ttnn.from_torch(
        any_cpu,
        device=device,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT if tilized else ttnn.ROW_MAJOR_LAYOUT,
    )
    L = expected_output.shape[0]
    output_npu = (
        ttnn.from_torch(
            torch.empty([1, L]),
            device=device,
            dtype=ttnn_dtype,
            layout=ttnn.TILE_LAYOUT if tilized else ttnn.ROW_MAJOR_LAYOUT,
        )
        if optional_output
        else None
    )

    output_npu = ttnn.operations.moreh.arange(
        start,
        end,
        step,
        any_npu,
        output=output_npu,
        untilize_out=not tilized,
        dtype=get_lib_dtype(ttnn, dtype),
    )
    actual_output = ttnn.to_torch(output_npu).reshape([L])

    # Assert shape and value comparison
    assert actual_output.shape == expected_output.shape
    passing, out = comp_allclose_and_pcc(expected_output, actual_output, rtol=0.1, atol=0.1)
    logger.info(out)
    assert passing


@pytest.mark.parametrize(
    "start_end_step",
    [
        [0, 32, 1],
        [2.3, 15.3, 0.5],
        [10.9, -13, -0.3],
        [-100, 32 * 10, 1],
        [0, 32000, 1],
        [2300.3, 15300.3, 0.5392],
        [10900.9, -13000, -0.3111],
        [-10000, 32 * 10000, 1],
    ],
)
@pytest.mark.parametrize(
    "optional_output",
    [True, False],
)
@pytest.mark.parametrize(
    "dtype",
    [None, "bfloat8_b", "bfloat16", "int32", "float32"],
)
@pytest.mark.parametrize(
    "tilized",
    [True, False],
)
def test_arange(start_end_step, optional_output, dtype, tilized, device):
    """Test arange functionality with different parameters."""
    run_moreh_arange(start_end_step, optional_output, dtype, tilized, device)


@pytest.mark.parametrize(
    "start_end_step",
    [
        [10900.9, -13000, -0.3111],
    ],
)
@pytest.mark.parametrize(
    "optional_output",
    [True, False],
)
@pytest.mark.parametrize(
    "dtype",
    [None, "bfloat8_b", "bfloat16", "int32", "float32"],
)
def test_arange_callback(start_end_step, optional_output, dtype, device):
    """Test arange functionality with callback and program cache validation."""
    torch.manual_seed(2024)
    num_program_cache_entries_list = []
    for i in range(2):
        run_moreh_arange(start_end_step, optional_output, dtype, True, device)
        torch_dummy = torch.randn([32, 32])
        tt_dummy = ttnn.from_torch(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
