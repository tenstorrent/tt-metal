# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc
from models.utility_functions import is_grayskull


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "mem_configs",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
)
@pytest.mark.parametrize("out_dtype", (ttnn.uint32, ttnn.uint16))
def test_binary_eq(input_shapes, out_dtype, mem_configs, device):
    if is_grayskull():
        pytest.skip("GS does not support fp32/uint32/uint16 data types")

    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -90, 100, device, True)

    cq_id = 0
    mem_cfg = mem_configs

    tt_output_tensor_on_device = ttnn.eq(
        input_tensor, other_tensor, memory_config=mem_cfg, dtype=out_dtype, queue_id=cq_id
    )

    golden_tensor = torch.eq(in_data, other_data)
    comp_pass = compare_pcc([tt_output_tensor_on_device], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "mem_configs",
    (
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    ),
)
@pytest.mark.parametrize("out_dtype", (ttnn.uint32, ttnn.uint16))
def test_bw_binary_eq_opt_output(input_shapes, device, mem_configs, out_dtype):
    if is_grayskull():
        pytest.skip("GS does not support fp32/uint32/uint16 data types")

    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -90, 100, device, True)
    _, out_tensor = data_gen_with_range(input_shapes, -70, 60, device, True)

    cq_id = 0
    mem_cfg = mem_configs

    ttnn.typecast(out_tensor, out_dtype, memory_config=mem_cfg)

    ttnn.eq(input_tensor, other_tensor, memory_config=mem_cfg, output_tensor=out_tensor, queue_id=cq_id)

    golden_tensor = torch.eq(in_data, other_data)
    comp_pass = compare_pcc([out_tensor], [golden_tensor])
    assert comp_pass
