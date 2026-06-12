# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-side perf test for the fused ``ttnn.experimental.topk_router_gpt`` op (gpt-oss MoE gate).

The op fuses matmul + bias + top-k + softmax for a 128-expert router: it takes hidden states
[B=32, hidden] and a router weight [hidden, 128] (+ bias [B, 128]) and returns the top-k expert
indices and softmax-normalized weights. This test captures a trace of repeated calls, bracketed by
signposts, so the harness (``perf_gpt_topk_router.py``) can isolate the device-kernel time — directly
comparable to TTMoEGate's two-op (linear + generalized_moe_gate) path on the gpt_oss config. Tensor setup
mirrors the unit test
``tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_router_gpt.py``.

gpt-oss-120b production shape: B=32 (decode), hidden=2880, num_experts=128, top_k=4.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import skip_for_blackhole


@skip_for_blackhole("topk_router_gpt requires 12 DRAM-aligned cores; Blackhole only has 8")
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 7000000, "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW}],
    indirect=True,
)
@pytest.mark.parametrize("B, hidden, num_experts, top_k", [(32, 2880, 128, 4)])
@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
def test_topk_router_gpt_perf(device, B, hidden, num_experts, top_k, warmup_iters, num_iters, device_params):
    """Capture and execute a trace of the fused gpt-oss router op for perf measurement."""
    torch.manual_seed(42)
    torch_input = torch.randn(B, hidden, dtype=torch.bfloat16)
    torch_weight = torch.randn(hidden, num_experts, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, num_experts, dtype=torch.bfloat16).expand(B, num_experts).contiguous()

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    def run_router():
        return ttnn.experimental.topk_router_gpt(
            tt_input,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            k=top_k,
            num_experts=num_experts,
        )

    # Compile the op
    run_router()
    ttnn.synchronize_device(device)

    # Capture warmup trace
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(warmup_iters):
        run_router()
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace (gpt-oss router: B={B}, hidden={hidden}, N={num_experts}, top_k={top_k})")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(num_iters):
        run_router()
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
    ttnn.synchronize_device(device)

    # One more eager call + light sanity. The fused op returns (indices, weights) — indices is FIRST.
    res_idx, _ = run_router()
    ttnn.synchronize_device(device)
    idx = ttnn.to_torch(res_idx)[:B, :top_k].to(torch.int32)
    assert int(idx.min()) >= 0 and int(idx.max()) < num_experts, f"indices out of range: {idx}"

    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
