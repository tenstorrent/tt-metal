# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import tt_lib as ttl


from tests.tt_eager.python_api_testing.sweep_tests import pytorch_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.tt_eager.python_api_testing.sweep_tests.tt_lib_ops import eltwise_logical_andi as tt_eltwise_logical_andi


def run_eltwise_logical_andi_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    immediate,
    data_seed,
    device,
):
    random.seed(0)
    torch.manual_seed(data_seed)

    if in_mem_config == "SYSTEM_MEMORY":
        in_mem_config = None

    x = torch.Tensor(size=input_shape).uniform_(-100, 100).to(torch.bfloat16)
    x_ref = x.detach().clone()

    # get referent value
    ref_value = pytorch_ops.logical_andi(x_ref, immediate=immediate)

    # calculate tt output
    logger.info("Running eltwise_andi test")
    tt_result = tt_eltwise_logical_andi(
        x=x,
        immediate=immediate,
        device=device,
        dtype=[dtype],
        layout=[dlayout],
        input_mem_config=[in_mem_config],
        output_mem_config=out_mem_config,
    )

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)

    assert success


# eltwise-logical_andi,"[[6, 9, 192, 128]]","{'dtype': [<DataType.BFLOAT8_B: 3>], 'layout': [<Layout.TILE: 1>], 'input_mem_config': [None], 'output_mem_config': tt::tt_metal::MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt), 'immediate': 0}",19790443,(),error,"TT_FATAL @ /home/ubuntu/tt-metal/tt_eager/tt_dnn/op_library/bcast/bcast_op.cpp:94: input_tensor_a.get_dtype() == input_tensor_b.get_dtype()

test_sweep_args = [
    (
        (6, 9, 192, 128),
        ttl.tensor.DataType.BFLOAT8_B,
        ttl.tensor.Layout.TILE,
        "SYSTEM_MEMORY",
        ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.INTERLEAVED, ttl.tensor.BufferType.L1),
        0,
        19790443,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, immediate, data_seed",
    (test_sweep_args),
)
def test_eltwise_logical_andi_test(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    out_mem_config,
    immediate,
    data_seed,
    device,
):
    run_eltwise_logical_andi_tests(
        input_shape,
        dtype,
        dlayout,
        in_mem_config,
        out_mem_config,
        immediate,
        data_seed,
        device,
    )
