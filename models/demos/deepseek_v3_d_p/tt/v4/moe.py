# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash MoE for the pure-ttnn prefill: a thin builder over the shared ``TtMoe`` (dispatch / fused routed
experts / combine / shared expert are dimension-agnostic).

V4 specifics: 256 routed experts, top-6, one shared expert of the same 2048 intermediate, a single expert group (the
grouped top-k collapses to plain top-k), ``sqrtsoftplus`` scoring, route scale 1.5, and layers 0..2 hash-routed
(``tid2eid[input_ids]`` picks the experts; the gate still scores them) -> ``GateComputeMode.HASH_DEVICE`` with the
token ids passed per chunk. The routed experts run the fused ``Silu`` kernel; the reference's ``swiglu_limit`` clamp
(gate <= 10, |up| <= 10) is NOT applied yet (plan M8 ``SiluClamped``; the deviation is measured by the MoE test).

Weights come in the reference module's names (``tt/v4/weights/hf_names.py``): ``mlp.gate.weight``,
``mlp.gate.e_score_correction_bias`` (learned layers) / ``mlp.gate.tid2eid`` (hash layers), per-expert
``gate_proj/up_proj/down_proj`` in HF ``[out, in]`` orientation, ``mlp.shared_experts.*``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode


def is_hash_layer(cfg, layer_idx: int) -> bool:
    return cfg.mlp_layer_types[layer_idx] == "hash_moe"


def build_v4_moe(
    mesh_device,
    cfg,
    layer_idx: int,
    *,
    seq_len_per_chip: int,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    tid2eid: Optional[torch.Tensor],
    routed_expert_weights: Optional[list],
    shared_expert_weights: Optional[dict],
    num_links=2,
    topology=ttnn.Topology.Linear,
    dispatch_buffer_capacity_factor: int = 2,
    routed_expert_weights_dtype=ttnn.bfloat8_b,
    weight_cache_path: Optional[Path] = None,
    routing_use_l1_small_for_semaphores: bool = True,
    overlap_shared_expert_with_dispatch: bool = True,
    num_routed_experts: Optional[int] = None,
) -> TtMoe:
    """``routed_expert_weights`` is the list of THIS chip's experts... no: TtMoe takes the full list (all experts, HF
    orientation) and shards it by its dispatch table; pass None when the .tensorbin cache is complete.
    ``num_routed_experts`` overrides ``cfg.n_routed_experts`` for reduced-expert tests."""
    mesh_config = extract_mesh_config(mesh_device)
    n_experts = int(num_routed_experts or cfg.n_routed_experts)
    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        n_experts,
        cfg.num_experts_per_tok,
        mesh_device.get_num_devices(),
        mesh_config.dispatch_group_size,
        dispatch_buffer_capacity_factor,
    )
    hash_layer = is_hash_layer(cfg, layer_idx)
    if hash_layer:
        assert tid2eid is not None, f"layer {layer_idx} is hash-routed: tid2eid required"
        gate_bias = torch.zeros(n_experts) if gate_bias is None else gate_bias  # unused by the hash gate
    else:
        assert gate_bias is not None, f"layer {layer_idx} is a learned-gate layer: e_score_correction_bias required"
    return TtMoe(
        mesh_device=mesh_device,
        dispatch_group_size=mesh_config.dispatch_group_size,
        num_dispatch_groups=mesh_config.num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_routed_experts=n_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=cfg.hidden_size,
        hidden_dim=cfg.moe_intermediate_size,
        n_expert_groups=1,
        n_limited_groups=1,
        route_scale=float(cfg.routed_scaling_factor),
        num_links=num_links,
        topology=topology,
        routed_expert_weights=routed_expert_weights,
        shared_expert_weights=shared_expert_weights,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=routed_expert_weights_dtype,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        gate_weights={"weight": gate_weight, "e_score_correction_bias": gate_bias},
        gate_fallback_mode=GateComputeMode.HASH_DEVICE if hash_layer else GateComputeMode.DEVICE_FP32,
        weight_cache_path=weight_cache_path,
        layer_idx=layer_idx,
        overlap_shared_expert_with_dispatch=overlap_shared_expert_with_dispatch,
        routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
        rms_norm_eps=cfg.rms_norm_eps,
        score_func=cfg.scoring_func,
        hash_table=tid2eid,
    )


def reference_moe_weights(ref) -> dict:
    """Split a DeepseekV4SparseMoeBlock's fused expert stack into TtMoe's per-expert HF-orientation dicts."""
    E, two_i, _ = ref.experts.gate_up_proj.shape
    inter = two_i // 2
    routed = []
    with torch.no_grad():
        for e in range(E):
            gu = ref.experts.gate_up_proj[e]  # [2I, D]: gate rows then up rows (chunk(2) in the reference)
            routed.append(
                {
                    "gate_proj": gu[:inter].clone(),
                    "up_proj": gu[inter:].clone(),
                    "down_proj": ref.experts.down_proj[e].clone(),
                }
            )
        shared = {
            "gate_proj": ref.shared_experts.gate_proj.weight.clone(),
            "up_proj": ref.shared_experts.up_proj.weight.clone(),
            "down_proj": ref.shared_experts.down_proj.weight.clone(),
        }
        gate_weight = ref.gate.weight.clone()
        gate_bias = getattr(ref.gate, "e_score_correction_bias", None)
        tid2eid = getattr(ref.gate, "tid2eid", None)
    return {
        "gate_weight": gate_weight,
        "gate_bias": None if gate_bias is None else gate_bias.clone(),
        "tid2eid": None if tid2eid is None else tid2eid.clone(),
        "routed_expert_weights": routed,
        "shared_expert_weights": shared,
    }
