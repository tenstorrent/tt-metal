# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import torch

import tt_lib as ttl

from tt_lib.utils import _nearest_32
from models.utility_functions import comp_pcc
import ttnn

TILE_HEIGHT = TILE_WIDTH = 32


def shape_padded(shape):
    return [shape[0], shape[1], _nearest_32(shape[2]), _nearest_32(shape[3])]


@pytest.mark.parametrize(
    "act_shape",
    (
        pytest.param([1, 7, 7, 2048]),
        ([1, 1, 32, 64]),
    ),
    ids=["resnet50_unpadded", "tile_divisible"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttl.tensor.DataType.BFLOAT16,),
    ids=[
        "BFLOAT16",
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 11264}], indirect=True)
def test_run_average_pool(act_shape, dtype, device, use_program_cache, enable_async):
    device.enable_async(enable_async)

    batch_size, _, _, channels = act_shape

    torch.manual_seed(0)

    interleaved_mem_config_L1 = ttl.tensor.MemoryConfig(
        memory_layout=ttl.tensor.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttl.tensor.BufferType.L1,
    )

    trace_loops = 10

    out_shape = [1] * len(act_shape)
    out_shape[-1] = act_shape[-1]
    out_shape_padded = shape_padded(out_shape)

    act = torch.randn(act_shape, dtype=torch.bfloat16).float()
    ttact = ttl.tensor.Tensor(act, ttl.tensor.DataType.BFLOAT16)
    act_shape_padded = shape_padded(act_shape)
    if act_shape != act_shape_padded:
        ttact = ttact.pad_to_tile(0.0)

    ttact_res = ttact.to(device)

    def run_ops(ttact_res):
        return ttnn.average_pool_2d(ttact_res)

    # Compile
    run_ops(ttact_res)
    # Trace
    logger.info("Start Trace capture")
    tid = ttl.device.BeginTraceCapture(device, 0)
    out_res = run_ops(ttact_res)
    ttl.device.EndTraceCapture(device, 0, tid)
    logger.info("Trace captured")

    for iter in range(trace_loops):
        act = torch.randn(act_shape, dtype=torch.bfloat16).float()
        ttact_updated = ttl.tensor.Tensor(act, ttl.tensor.DataType.BFLOAT16)
        act_shape_padded = shape_padded(act_shape)
        if act_shape != act_shape_padded:
            ttact_updated = ttact_updated.pad_to_tile(0.0)
        ttl.tensor.write_tensor(ttact_updated, ttact_res)

        logger.info(f"Running iteration {iter}")
        ttl.device.ReplayTrace(device, 0, tid, True)

        out = out_res.cpu().to(ttl.tensor.Layout.ROW_MAJOR)
        out_shape = [batch_size, 1, 1, channels]
        out_shape_padded = shape_padded(out_shape)
        if out_shape != out_shape_padded:
            out = out.unpad_from_tile(out_shape)

        out_pytorch = out.to_torch()
        out = out.pad_to_tile(0)  # Undo, so next loop unpad_from_tile works again.

        ## reference
        act_channels_first = torch.permute(act, (0, 3, 1, 2))  # Torch operates on channels-first tensors
        golden_pytorch = torch.nn.AdaptiveAvgPool2d((1, 1))(act_channels_first)

        ## test for equivalance
        passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch)
        logger.debug(f"Passing PCC = {passing_pcc}")
        logger.debug(f"Output PCC = {output_pcc}")

        assert passing_pcc

    # Done with the trace, can deallocate the buffers now.
    ttl.device.ReleaseTrace(device, tid)
    device.enable_async(False)
