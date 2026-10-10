# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash MoE for the prefill: the shared ``TtMoe`` with V4.1's routing.

``model.py`` ``Gate`` / ``Expert`` / ``MoE``: 384 routed experts, top-6, ``sqrtsoftplus`` scores, the correction bias
(``ffn.gate.bias``) picks the experts but the weights come from the raw scores, normalised (``norm_topk_prob``) and
scaled by ``route_scale`` 1.5; one shared expert of the same 2304 intermediate; SwiGLU with ``swiglu_limit`` 10 (up clamped
both sides, gate from above) on both. No hash-routed layers. That is V4-Flash's gate with other counts, so the V4-Flash
kernels serve it (``GateComputeMode.DEVICE_FP32``, the fused ``SiluClamped`` routed experts).

The routed experts are FP4 in the checkpoint (e2m1 + an e8m0 scale per 32 along K): ``TtMoe`` converts a list of
per-expert HF-orientation dicts, so ``LazyExperts`` dequantises each expert only when the conversion asks for it (all 384
experts of a layer in fp32 would be 54 GB of host RAM). With ``weight_cache_path`` the converted tensors are cached as
.tensorbin and later builds pass ``routed_expert_weights=None``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants, extract_mesh_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode


class LazyExperts(Sequence):
    """``[expert e] -> {"gate_proj": w1, "up_proj": w3, "down_proj": w2}`` (bf16, ``[out, in]``), read from the checkpoint
    on access."""

    def __init__(self, ck, layer: int, n_experts: int):
        self.ck, self.layer, self.n = ck, int(layer), int(n_experts)
        self._last = (None, None)  # the gather reads [e]["gate_proj"], [e]["up_proj"], [e]["down_proj"] back to back

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, e):
        if isinstance(e, slice):
            return [self[i] for i in range(*e.indices(self.n))]
        e = int(e)
        if self._last[0] != e:
            w = self.ck.expert(self.layer, e)
            self._last = (
                e,
                {"gate_proj": w["w1"].bfloat16(), "up_proj": w["w3"].bfloat16(), "down_proj": w["w2"].bfloat16()},
            )
        return self._last[1]


def build_v41_moe(
    mesh_device,
    cfg,
    layer: int,
    w: dict,
    *,
    seq_len_per_chip: int,
    ck=None,
    num_links=2,
    topology=ttnn.Topology.Linear,
    dispatch_buffer_capacity_factor: int = 2,
    routed_expert_weights_dtype=None,
    weight_cache_path: Optional[Path] = None,
    load_routed_from_cache: bool = False,
) -> TtMoe:
    """``w``: the layer's ``V41Checkpoint.layer(L)`` dict (gate / shared expert); the routed experts are read through
    ``ck`` (``LazyExperts``) unless ``load_routed_from_cache`` (a complete .tensorbin cache under ``weight_cache_path``).
    Routed weights default to BFP8 (``V41_PREFILL_EXPERT_DTYPE=bfp4`` for BFP4): a BFP block shares one exponent over 16
    OUTPUT channels of ``W.T`` whose e8m0 scales differ, so BFP4's 3 mantissa bits cannot hold the FP4 values (tt-blaze
    DS41F-0019: the decode ring's BFP4 experts cost 0.009-0.016 MoE PCC); at 384 x 2304 BFP8 is ~9 GB per chip for the 20
    encoder layers on 8 x 4."""
    if routed_expert_weights_dtype is None:
        routed_expert_weights_dtype = (
            ttnn.bfloat4_b if os.environ.get("V41_PREFILL_EXPERT_DTYPE", "bfp8") == "bfp4" else ttnn.bfloat8_b
        )
    mesh_config = extract_mesh_config(mesh_device)
    n_experts = int(cfg.n_routed_experts)
    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        n_experts,
        cfg.n_activated_experts,
        mesh_device.get_num_devices(),
        mesh_config.dispatch_group_size,
        dispatch_buffer_capacity_factor,
    )
    routed = None if load_routed_from_cache else LazyExperts(ck, layer, n_experts)
    shared = {
        "gate_proj": w["ffn.shared_experts.w1.weight"].float(),
        "up_proj": w["ffn.shared_experts.w3.weight"].float(),
        "down_proj": w["ffn.shared_experts.w2.weight"].float(),
    }
    return TtMoe(
        mesh_device=mesh_device,
        dispatch_group_size=mesh_config.dispatch_group_size,
        num_dispatch_groups=mesh_config.num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_routed_experts=n_experts,
        num_experts_per_tok=cfg.n_activated_experts,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=cfg.dim,
        hidden_dim=cfg.moe_inter_dim,
        n_expert_groups=1,
        n_limited_groups=1,
        route_scale=float(cfg.route_scale),
        num_links=num_links,
        topology=topology,
        routed_expert_weights=routed,
        shared_expert_weights=shared,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=routed_expert_weights_dtype,
        shared_expert_activations_dtype=ttnn.bfloat16,
        activation=getattr(ttnn.RoutedExpertActivation, "SiluClamped", ttnn.RoutedExpertActivation.Silu),
        shared_expert_swiglu_limit=float(cfg.swiglu_limit),
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        gate_weights={"weight": w["ffn.gate.weight"].float(), "e_score_correction_bias": w["ffn.gate.bias"].float()},
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=weight_cache_path,
        layer_idx=layer,
        overlap_shared_expert_with_dispatch=True,
        routing_use_l1_small_for_semaphores=True,
        rms_norm_eps=cfg.norm_eps,
        score_func=cfg.score_func,
    )
