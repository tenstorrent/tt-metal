# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Qwen3NextTopKRouter`, column-parallel over the EXPERT axis.

Reference: `transformers/models/qwen3_next/modeling_qwen3_next.py::Qwen3NextTopKRouter`:

    logits  = hidden_states @ weight.T             # (tokens, num_experts)
    probs   = softmax(logits, -1)                  # in fp32
    top, ix = topk(probs, top_k, -1)
    scores  = top / top.sum(-1, keepdim=True)      # when norm_topk_prob
    return logits, scores, ix

Tensor-parallel scheme -- the one place the MoE's "router stays replicated" rule needs care:

  * The router `weight` is (num_experts, hidden_dim) and num_experts (512) is by far its larger
    axis, so it IS worth splitting: this is a plain COLUMN-parallel projection whose output
    features are the experts. Each chip owns a disjoint contiguous block of expert columns.
  * But the very next op is a softmax over ALL experts, and after that a global top-k -- both need
    the whole expert axis. So the collective is an `all_gather` on the feature axis IMMEDIATELY
    after the matmul, before any reduction. Contiguous sharding on dim=-1 makes the gathered
    concatenation land in the golden expert order (chip 0: [0, E/2), chip 1: [E/2, E)).
  * Everything downstream of the gather (softmax / top-k / renormalise) then runs REPLICATED, so
    every chip reaches the identical routing decision -- which is exactly what the expert-parallel
    bank in `_stubs/experts.py` relies on to slice out its own experts consistently.
  * `norm_topk_prob` divides by the sum of the SELECTED probabilities only, so the denominator is
    formed after the top-k, never from the full softmax.
"""
from __future__ import annotations

import ttnn

from models.demos.qwen3_coder_next._stubs.gated_delta_net import (
    matmul_weight,
    num_devices,
    replicate_mapper,
    shard_mapper,
    to_device,
)


class TtQwen3NextTopKRouter:
    """Native ttnn Qwen3-Next MoE router, expert-axis column-parallel + all_gather."""

    def __init__(self, device, *, weight, hidden_size, num_experts, top_k, norm_topk_prob) -> None:
        self.device = device
        self.weight = weight
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.num_devices = num_devices(device)
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("top_k_router stub needs the torch reference module for its weights")

        w = torch_module.weight.detach().float()  # (num_experts, hidden_dim)
        num_experts = int(getattr(torch_module, "num_experts", w.shape[0]))
        hidden = int(getattr(torch_module, "hidden_dim", w.shape[1]))

        n = num_devices(device)
        # Split the expert axis only when it divides evenly; otherwise fall back to replication,
        # which is still numerically exact (it just does the same work on every chip).
        tp = n if num_experts % n == 0 else 1
        mapper = shard_mapper(device, tp, -1) if tp > 1 else replicate_mapper(device, n)

        return cls(
            device,
            weight=to_device(matmul_weight(w), device, mesh_mapper=mapper),
            hidden_size=hidden,
            num_experts=num_experts,
            top_k=int(torch_module.top_k),
            norm_topk_prob=bool(torch_module.norm_topk_prob),
        )

    def __call__(self, hidden_states, *args, **kwargs):
        seq = int(hidden_states.shape[-2])
        x = ttnn.reshape(hidden_states, (1, 1, seq, self.hidden_size))

        logits = ttnn.linear(x, self.weight, compute_kernel_config=self.compute_config)
        if int(logits.shape[-1]) != self.num_experts:
            # Column-parallel: reassemble the full expert axis before the softmax needs it.
            logits = ttnn.all_gather(logits, dim=-1)

        # SELECT ON THE FULL-WIDTH SOFTMAX, NORMALISE IN FLOAT32.
        # The reference is `softmax(router_logits, dtype=torch.float)` -> `topk` -> divide by the
        # selected sum. `ttnn.topk` only accepts bfloat16, so the ranking runs on the bf16
        # softmax; that is safe because the softmax spans all 512 experts, which is tile-aligned,
        # so no padding lane can enter the reduction. The VALUE path is then lifted to float32
        # before the renormalising divide -- the reference does that divide in float32 too, and
        # the sum is padding-safe because the top_k lanes beyond k are zeros.
        probs = ttnn.softmax(logits, dim=-1)
        top_values, indices = ttnn.topk(probs, self.top_k, dim=-1)
        scores = top_values
        if self.norm_topk_prob:
            top32 = ttnn.typecast(top_values, ttnn.float32)
            scores = ttnn.multiply(top32, ttnn.reciprocal(ttnn.sum(top32, dim=-1, keepdim=True)))
        # The reference casts the weights back to the logits' dtype before they scale the experts.
        scores = ttnn.typecast(scores, logits.dtype)

        logits = ttnn.reshape(logits, (1, seq, self.num_experts))
        scores = ttnn.reshape(scores, (1, seq, self.top_k))
        indices = ttnn.reshape(indices, (1, seq, self.top_k))
        return logits, scores, indices


def build(device, torch_module=None):
    return TtQwen3NextTopKRouter.build(device, torch_module)


def top_k_router(device, torch_module=None):
    return TtQwen3NextTopKRouter.build(device, torch_module)
