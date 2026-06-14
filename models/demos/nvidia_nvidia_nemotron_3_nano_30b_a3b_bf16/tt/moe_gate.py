# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEGate (NemotronHTopkRouter) — TP=4 on QB 4-chip Blackhole.

Fully on-device, trace-compatible.  All routing ops (sigmoid, topk,
gather, normalize, scatter) run on device so no D2H/H2D transfers happen
inside the forward.  Returns dense [1, 1, 1, 128] routing weights on device
for sparse_matmul (zeros for inactive experts, normalised+scaled for active).

n_group=1, topk_group=1: group selection is trivial (all experts in one group),
so the group-mask step from the reference model is a no-op here.
"""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _upload

N_ROUTED_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 6
ROUTED_SCALING_FACTOR = 2.5

# Bias tensors need reshaping to [1, 1, 1, 128] before upload but the reshaped
# tensor is a new Python object (new id()) on every call, defeating _upload's
# cache.  Cache them separately keyed by (id(cpu_bias), id(mesh)).
_BIAS_DEVICE_CACHE: dict = {}  # (id(cpu_bias), id(mesh)) → device_tensor


def _get_bias_tt(bias_cpu: torch.Tensor, mesh_device: MeshDevice) -> ttnn.Tensor:
    key = (id(bias_cpu), id(mesh_device))
    if key in _BIAS_DEVICE_CACHE:
        return _BIAS_DEVICE_CACHE[key]
    bias_4d = bias_cpu.bfloat16().reshape(1, 1, 1, -1).contiguous()
    bias_tt = _upload(bias_4d, mesh_device, shard_dim=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    _BIAS_DEVICE_CACHE[key] = bias_tt
    return bias_tt


def moe_gate_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [tokens, 2688] bf16 on device
    weight: torch.Tensor,  # [128, 2688] float32 CPU (held by WeightCache, id stable)
    e_score_correction_bias: torch.Tensor,  # [128] float32 CPU (held by WeightCache)
    n_routed_experts: int = N_ROUTED_EXPERTS,
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOK,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
) -> ttnn.Tensor:
    """Returns dense routing-weights [1, 1, 1, 128] bfloat16 on device.

    Fully on-device, trace-compatible (no D2H or H2D inside the forward).
    Non-active expert positions are zero; active positions hold the normalised,
    scaled routing weights.  Shape is suitable for sparse_matmul sparsity.
    """
    # Pass `weight` directly (not weight.bfloat16()) so _upload sees the stable
    # id(weight) from WeightCache.  _upload converts to bfloat16 internally.
    w_tt = _upload(weight, mesh_device, shard_dim=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    bias_tt = _get_bias_tt(e_score_correction_bias, mesh_device)

    # Gate logits: [tokens, hidden] × [hidden, experts] → [tokens, 128]
    logits = ttnn.linear(hidden_states, w_tt, transpose_b=True)

    # Work in 4D for topk/scatter op compatibility
    logits_4d = ttnn.unsqueeze_to_4D(logits)  # [1, 1, tokens, 128]

    scores = ttnn.sigmoid(logits_4d)
    logits_4d.deallocate(True)

    # Add correction bias for expert selection only (not for final weights)
    scores_biased = ttnn.add(scores, bias_tt)

    # TopK: select the top-6 experts by biased score
    top_k_vals_biased, top_k_indices = ttnn.topk(
        scores_biased, k=num_experts_per_tok, dim=-1, largest=True, sorted=False
    )
    top_k_vals_biased.deallocate(True)
    scores_biased.deallocate(True)

    # Gather unbiased sigmoid weights at selected expert positions
    top_k_vals = ttnn.gather(scores, dim=-1, index=top_k_indices)

    # Normalize: divide by sum then apply routing scale factor
    if norm_topk_prob:
        top_k_sum = ttnn.sum(top_k_vals, dim=-1, keepdim=True)
        top_k_vals = ttnn.div(top_k_vals, top_k_sum)
        top_k_sum.deallocate(True)
    top_k_vals = ttnn.mul(top_k_vals, routed_scaling_factor)

    # Scatter into dense [1, 1, tokens, 128] — inactive experts remain zero
    dense_routing = ttnn.scatter(
        ttnn.zeros_like(scores),
        dim=-1,
        index=top_k_indices,
        src=top_k_vals,
    )
    scores.deallocate(True)
    top_k_vals.deallocate(True)
    top_k_indices.deallocate(True)

    return dense_routing  # [1, 1, tokens, 128] bf16 on device, replicated
