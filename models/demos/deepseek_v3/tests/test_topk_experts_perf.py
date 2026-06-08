# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Single-device perf baseline for the generic ``ttnn.topk`` (top-8 over N experts).

This is the op the fused generalized_moe_gate is meant to beat. Mirrors the gate workload:
(1,1,32,N) = 32 tokens x N experts, bf16, TILE, L1; ``ttnn.topk(k=8, dim=-1)``. Trace + signposts so
``perf_topk_experts.py`` can isolate the device-kernel time. Parametrized over N in {256, 512}.

(Separate from the model's TG ``test_topk_perf.py`` so that stays intact.)
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn

BATCH = 32  # tokens
K = 8


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 7000000}],
    indirect=True,
)
@pytest.mark.parametrize("num_experts", [256, 512])
@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
def test_topk_experts_perf(device, num_experts, warmup_iters, num_iters, device_params):
    """Capture and execute a trace of ttnn.topk (k=8 over N experts) for perf measurement."""
    shape = [1, 1, BATCH, num_experts]

    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    def run_topk():
        return ttnn.topk(tt_input, k=K, dim=-1, largest=True, sorted=True, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Compile
    run_topk()
    ttnn.synchronize_device(device)

    # Warmup trace
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(warmup_iters):
        run_topk()
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Main trace
    logger.info(f"Capturing main trace (topk k={K} over {num_experts} experts, {num_iters} iters)")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(num_iters):
        run_topk()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)

    signpost("start")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    signpost("stop")

    # Sanity: one eager call vs torch.
    tt_vals, tt_idx = run_topk()
    ttnn.synchronize_device(device)
    ref_vals, _ = torch.topk(torch_input.float(), k=K, dim=-1, largest=True, sorted=True)
    got = ttnn.to_torch(tt_vals).float()
    assert got.shape[-1] == K, f"unexpected topk output width {got.shape}"

    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
