# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-side perf test for ``ttnn.softmax`` as a candidate front-end for the generalized MoE gate.

Softmax (over the experts dim) would replace the in-gate sigmoid when ``enable_sigmoid`` is off and
softmax routing is done outside the gate. This captures a trace of repeated ``ttnn.softmax`` calls,
bracketed by signposts, so the perf harness (``perf_softmax.py``) can isolate the device kernel time.

Parametrized over the number of experts {64, 128, 256, 512} (the last/reduce dim). dtype/layout match
the generalized_moe_gate convention (bfloat16, TILE_LAYOUT, L1). 32 rows = one tile height of users.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

SOFTMAX_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG
USERS_PER_ROW = 1  # one tile height


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 7000000}],
    indirect=True,
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 512])
@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
def test_softmax_perf(device, dtype, num_experts, warmup_iters, num_iters, device_params):
    """Capture and execute a trace of ``ttnn.softmax`` over the experts dim for perf measurement."""
    shape = [1, 1, USERS_PER_ROW, num_experts]

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_ref = torch.softmax(torch_input.float(), dim=-1)

    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=SOFTMAX_MEMORY_CONFIG,
    )

    def run_softmax():
        return ttnn.softmax(tt_input, dim=-1, memory_config=SOFTMAX_MEMORY_CONFIG)

    # Compile the op
    run_softmax()
    ttnn.synchronize_device(device)

    # Capture warmup trace
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(warmup_iters):
        run_softmax()
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace ({num_experts} experts, {num_iters} iters)")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(num_iters):
        run_softmax()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)

    # Execute main trace, bracketed by signposts for the perf harness
    logger.info("Executing main trace")
    signpost("start")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    signpost("stop")

    # One more eager call to validate correctness
    tt_out = run_softmax()
    ttnn.synchronize_device(device)
    tt_out_torch = ttnn.to_torch(tt_out).float()
    assert_with_pcc(torch_ref, tt_out_torch, 0.99)

    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
