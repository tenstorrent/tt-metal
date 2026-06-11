# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-side perf test for the common ``TTMoEGate`` on the GPT-OSS gate config.

Runs ``TTMoEGate.forward`` (router projection + gate op + slice/view) in an eager loop bracketed by
signposts, so the harness (``perf_tt_moe_gate.py``) can isolate the device-kernel time. GPT-OSS gate =
linear(Wx + b) → top-4 → softmax-over-selected (128 experts), which TTMoEGate runs as ttnn.linear (router
LINEAR bias) + generalized_moe_gate (top-4 + output-softmax). Measurement only (no perf-target gating).

This is the SAME (weights, indices) routing GPT-OSS's fused ``ttnn.experimental.topk_router_gpt`` produces;
here we measure TTMoEGate's two-op realization end-to-end.
"""

from pathlib import Path

import pytest
import torch
import yaml
from loguru import logger
from tracy import signpost

import ttnn
from models.common.modules.moe.tt_moe_gate import TTMoEGate

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "modules/moe/configs"


@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
def test_tt_moe_gate_perf(device, warmup_iters, num_iters):
    """Eager-loop device-perf for TTMoEGate on gpt_oss (128 experts, top-4, softmax + router linear bias)."""
    raw = yaml.safe_load((CONFIGS_DIR / "gpt_oss.yaml").read_text())
    num_experts = raw["num_routed_experts"]  # 128
    k = raw["select_experts_k"]  # 4
    hidden = raw["hidden_size"]  # 2880
    batch = 32  # decode: one token per core

    torch.manual_seed(42)
    gate_weight = ((2 * torch.rand((hidden, num_experts), dtype=torch.bfloat16)) - 1) * 0.1
    # gpt_oss: router LINEAR bias (router_bias) → torch_gate_proj_bias; no score-correction bias.
    proj_bias = (2 * torch.rand((num_experts,)) - 1) if raw.get("router_bias") else None
    correction_bias = (2 * torch.rand((num_experts,)) - 1) if raw.get("score_correction_bias") else None

    gate = TTMoEGate(
        device,
        num_experts=num_experts,
        select_experts_k=k,
        hidden_size=hidden,
        torch_gate_weight=gate_weight,
        torch_gate_bias=correction_bias,
        torch_gate_proj_bias=proj_bias,
        n_group=raw.get("n_group", 1),
        score_func=raw.get("score_func", "softmax"),
        scaling_factor=raw.get("routed_scaling_factor", 1.0),
        matmul_compute_config=raw.get("gate_matmul_compute"),
    )
    tt_x = ttnn.from_torch(
        ((2 * torch.rand((batch, hidden), dtype=torch.bfloat16)) - 1).reshape(1, 1, batch, hidden),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def run():
        return gate.forward(tt_x)

    # Compile + warmup (outside the measured window).
    run()
    ttnn.synchronize_device(device)
    for _ in range(warmup_iters):
        run()
    ttnn.synchronize_device(device)

    # Measured window, bracketed by signposts for the perf harness.
    logger.info(f"[tt_moe_gate perf] gpt_oss: N={num_experts} k={k} hidden={hidden} batch={batch}, {num_iters} iters")
    signpost("start")
    for _ in range(num_iters):
        run()
    ttnn.synchronize_device(device)
    signpost("stop")

    # One eager call + light sanity (indices in range).
    _, idx = run()
    ttnn.synchronize_device(device)
    di = ttnn.to_torch(idx).reshape(batch, k).to(torch.int32)
    assert int(di.min()) >= 0 and int(di.max()) < num_experts, f"indices out of range: {di}"


if __name__ == "__main__":
    pytest.main([__file__])
