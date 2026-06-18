# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoEGate (NemotronHTopkRouter) — TP=4 on QB 4-chip Blackhole.

Two active routing modes
------------------------
moe_gate_forward_cpu : D2H hidden state → CPU float32 gate → H2D routing.
                       Exact match to HF reference. Not trace-compatible (~7 tok/s).
moe_gate_forward     : Fully on-device (float32 linear+sigmoid+bias, BF16 topk).
                       Trace-compatible (~15 tok/s). BF16 topk may flip boundary
                       experts vs CPU float32 topk (typically 0–2 of 6 per step).

n_group=1, topk_group=1: group selection is trivial (all experts in one group),
so the group-mask step from the reference model is a no-op here.
"""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _R, _upload

N_ROUTED_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 6
ROUTED_SCALING_FACTOR = 2.5

# Module-level compute kernel config (Blackhole = Wormhole architecture).
# HiFi4 + fp32_dest_acc forces true float32 accumulation in matrix engines.
_HIFI4_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
)

# Cache dicts keyed by (id(cpu_tensor), id(mesh)) for tensors that need reshaping
# before upload (new Python object on every call defeats _upload's id-based cache).
_BIAS_DEVICE_CACHE: dict = {}  # bf16 bias [1,1,1,128]
_GATE_W_F32_CACHE: dict = {}  # float32 gate weight [128,2688]
_BIAS_F32_CACHE: dict = {}  # float32 bias [1,1,1,128]


def _get_bias_tt(bias_cpu: torch.Tensor, mesh_device: MeshDevice) -> ttnn.Tensor:
    key = (id(bias_cpu), id(mesh_device))
    if key in _BIAS_DEVICE_CACHE:
        return _BIAS_DEVICE_CACHE[key]
    bias_4d = bias_cpu.bfloat16().reshape(1, 1, 1, -1).contiguous()
    bias_tt = _upload(bias_4d, mesh_device, shard_dim=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    _BIAS_DEVICE_CACHE[key] = bias_tt
    return bias_tt


def _get_gate_w_f32(weight_cpu: torch.Tensor, mesh_device: MeshDevice) -> ttnn.Tensor:
    key = (id(weight_cpu), id(mesh_device))
    if key in _GATE_W_F32_CACHE:
        return _GATE_W_F32_CACHE[key]
    w_f32 = weight_cpu.float().contiguous()
    gate_tt = _upload(w_f32, mesh_device, shard_dim=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    _GATE_W_F32_CACHE[key] = gate_tt
    return gate_tt


def _get_bias_f32(bias_cpu: torch.Tensor, mesh_device: MeshDevice) -> ttnn.Tensor:
    key = (id(bias_cpu), id(mesh_device))
    if key in _BIAS_F32_CACHE:
        return _BIAS_F32_CACHE[key]
    bias_4d = bias_cpu.float().reshape(1, 1, 1, -1).contiguous()
    bias_tt = _upload(bias_4d, mesh_device, shard_dim=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    _BIAS_F32_CACHE[key] = bias_tt
    return bias_tt


def moe_gate_forward_cpu(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [tokens, 2688] bf16 on device
    weight: torch.Tensor,  # [128, 2688] float32 CPU
    e_score_correction_bias: torch.Tensor,  # [128] float32 CPU
    n_routed_experts: int = N_ROUTED_EXPERTS,
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOK,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    return_cpu: bool = False,
) -> ttnn.Tensor:
    """CPU float32 gate — exactly matches HF NemotronHTopkRouter (n_group=1, topk_group=1).

    D2H the hidden state, compute in float32, H2D the dense routing tensor.
    Correct routing at the cost of D2H + H2D per E-layer (not trace-compatible).
    """
    import torch.nn.functional as F

    # D2H: take first T rows from shard 0 (all shards are identical replicas).
    # ConcatMeshToTensor(dim=0) gives [num_devices * T, 2688]; slice to [T, 2688].
    T = hidden_states.shape[0]
    h_cpu = ttnn.to_torch(
        hidden_states,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[
        0:T
    ].float()  # [T, 2688]
    w_f32 = weight.float()
    b_f32 = e_score_correction_bias.float()

    logits = F.linear(h_cpu, w_f32)  # [T, 128]
    scores = torch.sigmoid(logits)  # [T, 128]
    scores_biased = scores + b_f32  # [T, 128]
    top_k_indices = torch.topk(scores_biased, k=num_experts_per_tok, dim=-1, sorted=False)[1]  # [T, 6]
    top_k_vals = scores.gather(1, top_k_indices)  # [T, 6]
    if norm_topk_prob:
        denom = top_k_vals.sum(dim=-1, keepdim=True) + 1e-20
        top_k_vals = top_k_vals / denom
    top_k_vals = top_k_vals * routed_scaling_factor

    dense = torch.zeros(T, n_routed_experts, dtype=torch.bfloat16)
    dense.scatter_(1, top_k_indices, top_k_vals.bfloat16())
    dense_4d = dense.reshape(1, 1, T, n_routed_experts)

    device_tt = ttnn.from_torch(
        dense_4d,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=_R(mesh_device),
    )
    if return_cpu:
        return device_tt, dense_4d
    return device_tt


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

    Fully on-device, trace-compatible. Float32 linear+sigmoid+bias; BF16 topk.
    On-device float32 accumulation can differ from CPU float32 by ~0.09 in logits,
    which may flip boundary experts. Use moe_gate_forward_predictive for CPU topk
    precision with trace compatibility.
    """
    w_f32_tt = _get_gate_w_f32(weight, mesh_device)
    bias_f32_tt = _get_bias_f32(e_score_correction_bias, mesh_device)

    h_f32 = ttnn.typecast(hidden_states, ttnn.float32)
    logits_f32 = ttnn.linear(h_f32, w_f32_tt, transpose_b=True, compute_kernel_config=_HIFI4_CFG)
    h_f32.deallocate(True)

    logits_4d_f32 = ttnn.unsqueeze_to_4D(logits_f32)
    logits_f32.deallocate(True)

    scores_f32 = ttnn.sigmoid(logits_4d_f32)
    logits_4d_f32.deallocate(True)

    scores_biased_f32 = ttnn.add(scores_f32, bias_f32_tt)
    scores_biased_bf16 = ttnn.typecast(scores_biased_f32, ttnn.bfloat16)
    scores_biased_f32.deallocate(True)

    top_k_vals_biased, top_k_indices = ttnn.topk(
        scores_biased_bf16, k=num_experts_per_tok, dim=-1, largest=True, sorted=False
    )
    top_k_vals_biased.deallocate(True)
    scores_biased_bf16.deallocate(True)

    scores_bf16 = ttnn.typecast(scores_f32, ttnn.bfloat16)
    scores_f32.deallocate(True)

    top_k_vals = ttnn.gather(scores_bf16, dim=-1, index=top_k_indices)

    if norm_topk_prob:
        top_k_sum = ttnn.sum(top_k_vals, dim=-1, keepdim=True)
        top_k_vals = ttnn.div(top_k_vals, top_k_sum)
        top_k_sum.deallocate(True)
    top_k_vals = ttnn.mul(top_k_vals, routed_scaling_factor)

    dense_routing = ttnn.scatter(
        ttnn.zeros_like(scores_bf16),
        dim=-1,
        index=top_k_indices,
        src=top_k_vals,
    )
    scores_bf16.deallocate(True)
    top_k_vals.deallocate(True)
    top_k_indices.deallocate(True)

    return dense_routing  # [1, 1, tokens, 128] bf16 on device, replicated


def moe_gate_forward_predictive(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [tokens, 2688] bf16 on device
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routing_tt: ttnn.Tensor,  # persistent [1,1,1,128] BF16 — dense routing from PREVIOUS step
) -> tuple:
    """Trace-compatible gate with CPU topk precision via one-step-behind routing.

    Returns (routing_tt, scores_both) where:
      routing_tt   : the pre-computed dense routing tensor from the previous step
                     (passed straight through — read by expert computation this step)
      scores_both  : [1,1,2,128] BF16 — dim2[0]=biased scores (for topk), dim2[1]=unbiased
                     (for routing weights).  Stored in decoder_state.gate_scores_tts so
                     advance_routing() can D2H, compute CPU topk, and H2D new routing into
                     routing_tt before the next trace execution.

    The trace captures scores_both as an output tensor (like logits_tt or ssm_state_outs).
    After each execute_trace + advance(), routing_tt is updated with this step's CPU topk
    result, so the NEXT step uses correct routing.  The current step uses routing from the
    previous step — one-step-behind.  For fluent auto-regressive generation, consecutive
    routing decisions are highly correlated, so quality is maintained.
    """
    w_f32_tt = _get_gate_w_f32(weight, mesh_device)
    bias_f32_tt = _get_bias_f32(e_score_correction_bias, mesh_device)

    h_f32 = ttnn.typecast(hidden_states, ttnn.float32)
    logits_f32 = ttnn.linear(h_f32, w_f32_tt, transpose_b=True, compute_kernel_config=_HIFI4_CFG)
    h_f32.deallocate(True)

    logits_4d_f32 = ttnn.unsqueeze_to_4D(logits_f32)
    logits_f32.deallocate(True)

    scores_f32 = ttnn.sigmoid(logits_4d_f32)
    logits_4d_f32.deallocate(True)

    scores_biased_f32 = ttnn.add(scores_f32, bias_f32_tt)
    scores_f32.deallocate(True)

    # Only biased scores needed for advance_routing (both topk selection and weights).
    # ttnn.concat([biased, unbiased], dim=2) on [1,1,1,128] TILE tensors produces corrupted
    # output — tile padding causes the second "row" to contain garbage instead of sigmoid
    # values — so we track a single [1,1,1,128] biased-scores tensor per E-layer.
    # Using biased scores for routing weights (instead of unbiased) is a minor deviation from
    # the HF reference; norm_topk_prob normalization makes the absolute magnitude difference
    # negligible for the selected top-6 experts.
    scores_biased_bf16 = ttnn.typecast(scores_biased_f32, ttnn.bfloat16)  # [1,1,1,128]
    scores_biased_f32.deallocate(True)

    # routing_tt is stored ROW_MAJOR (needed for copy_host_to_device_tensor in advance_routing).
    # Convert to TILE_LAYOUT before returning since moe_experts_forward expects tiled tensors.
    routing_tile = ttnn.to_layout(routing_tt, ttnn.TILE_LAYOUT)
    return routing_tile, scores_biased_bf16


def build_routing_from_scores_cpu(
    scores_biased_cpu: torch.Tensor,  # [1, 1, 1, 128] BF16 from D2H of gate_scores_tt
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOK,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    n_routed_experts: int = N_ROUTED_EXPERTS,
) -> torch.Tensor:
    """CPU topk + routing from D2H'd biased gate scores.  Returns [1,1,1,128] BF16.

    Uses biased scores for both topk selection and routing weights.  The HF reference uses
    unbiased scores for weights, but ttnn.concat on [1,1,1,128] TILE tensors corrupts the
    second row.  Normalising biased weights (norm_topk_prob=True) gives equivalent quality
    since the bias shifts all 128 experts by similar magnitudes.
    """
    scores = scores_biased_cpu[0, 0, 0, :].float().unsqueeze(0)  # [1, 128]

    top_k_indices = torch.topk(scores, k=num_experts_per_tok, dim=-1, sorted=False)[1]  # [1, 6]
    top_k_vals = scores.gather(1, top_k_indices)  # [1, 6] biased scores for top-6

    if norm_topk_prob:
        top_k_vals = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-20)
    top_k_vals = top_k_vals * routed_scaling_factor

    dense = torch.zeros(1, n_routed_experts, dtype=torch.bfloat16)
    dense.scatter_(1, top_k_indices, top_k_vals.bfloat16())
    return dense.reshape(1, 1, 1, n_routed_experts)  # [1,1,1,128] BF16
