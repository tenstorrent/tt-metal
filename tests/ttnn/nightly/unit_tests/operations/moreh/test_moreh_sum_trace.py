# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32768}], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32], ids=["bf16", "int32"])
def test_moreh_sum_empty_dim_trace(device, dtype):
    # The zero fill must run on device: trace capture rejects a host upload.
    torch_dtype = torch.int32 if dtype == ttnn.int32 else torch.bfloat16
    tt_input = ttnn.from_torch(
        torch.ones(1, 1, 32, 0, dtype=torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_output = ttnn.from_torch(
        torch.full((1, 1, 32, 1), 7, dtype=torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn.operations.moreh.sum(tt_input, dim=3, keepdim=True, output=tt_output)
    tt_output = ttnn.from_torch(
        torch.full((1, 1, 32, 1), 7, dtype=torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    ttnn.operations.moreh.sum(tt_input, dim=3, keepdim=True, output=tt_output)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.release_trace(device, trace_id)
    assert torch.equal(ttnn.to_torch(tt_output).to(torch_dtype), torch.zeros(1, 1, 32, 1, dtype=torch_dtype))
