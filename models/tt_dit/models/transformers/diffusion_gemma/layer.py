# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Encoder/decoder text layer for DiffusionGemma.

Mirrors ``DiffusionGemmaEncoderTextLayer`` / ``DiffusionGemmaDecoderTextLayer`` from
``transformers.models.diffusion_gemma.modeling_diffusion_gemma``. The two HF classes
are 99% identical — they differ only in which attention class is used
(``DiffusionGemmaEncoderTextAttention`` vs ``DiffusionGemmaDecoderTextAttention``).
We collapse them into one class: ``encoder_kv`` argument in forward triggers the
decoder behavior (read-only cross-attention to the encoder cache).

Layer body (Gemma-3-style sandwich norms):

    h = input_layernorm(x)
    attn_out, k_local, v_local = self_attn(h, cos, sin, mask, encoder_kv)
    h = residual + post_attention_layernorm(attn_out)

    residual = h
    h_dense = post_feedforward_layernorm_1( mlp( pre_feedforward_layernorm(h) ) )
    h_moe   = post_feedforward_layernorm_2(
                 moe( router_input=residual,                                 # RAW
                      expert_input=pre_feedforward_layernorm_2(residual) )   # NORMED
              )
    h = residual + post_feedforward_layernorm(h_dense + h_moe)
    h = h * layer_scalar
"""

from __future__ import annotations

import torch

import ttnn

from ....encoders.gemma4.attention import Gemma4Attention
from ....layers.feedforward import GatedMLP
from ....layers.module import Module, Parameter
from ....layers.normalization import RMSNorm
from ....parallel.config import DiTParallelConfig
from .moe import DiffusionGemmaMoE


class DiffusionGemmaLayer(Module):
    """One DiffusionGemma transformer layer (encoder or decoder mode)."""

    def __init__(
        self,
        *,
        is_sliding: bool,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
        num_experts: int,
        top_k_experts: int,
        moe_intermediate_size: int,
        rms_norm_eps: float,
        moe_state_dict: dict | None,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
        parallel_config: DiTParallelConfig,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        expert_dtype: ttnn.DataType = ttnn.bfloat16,
        router_dtype: ttnn.DataType = ttnn.bfloat16,
        tensor_cache_path: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # Self-attention (single class, sliding-or-full layer-type).
        self.self_attn = Gemma4Attention(
            is_sliding=is_sliding,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            sliding_window=sliding_window,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        # Dense MLP (one of two parallel paths post-attention).
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        # MoE (the second of two parallel paths). Built via demos/gemma4 wrapper,
        # which loads weights at __init__ time — caller passes the substate.
        self.experts_and_router = DiffusionGemmaMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            moe_intermediate_size=moe_intermediate_size,
            rms_norm_eps=rms_norm_eps,
            state_dict=moe_state_dict,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            num_links=num_links,
            topology=topology,
            expert_dtype=expert_dtype,
            router_dtype=router_dtype,
            tensor_cache_path=tensor_cache_path,
        )

        # Seven Gemma-3-sandwich norms. All have learned scale, no bias.
        norm_kwargs = dict(
            embedding_dim=hidden_size,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.input_layernorm = RMSNorm(**norm_kwargs)
        self.post_attention_layernorm = RMSNorm(**norm_kwargs)
        self.pre_feedforward_layernorm = RMSNorm(**norm_kwargs)
        self.post_feedforward_layernorm = RMSNorm(**norm_kwargs)
        self.post_feedforward_layernorm_1 = RMSNorm(**norm_kwargs)
        self.post_feedforward_layernorm_2 = RMSNorm(**norm_kwargs)
        self.pre_feedforward_layernorm_2 = RMSNorm(**norm_kwargs)

        # Per-layer scalar — registered as a buffer in HF (init 1.0), multiplied
        # against the layer output. We allocate a tile-aligned [1, 1] tensor.
        self.layer_scalar = Parameter(total_shape=[1, 1], device=mesh_device, dtype=ttnn.bfloat16)

        # Precision knobs shared across the 7 layer-norms and layer-level ops. HiFi4 + fp32
        # dest accumulator + no packer L1 accumulation matches the attention/vision-attention
        # config. Without this the norms use device default (HiFi2 + bf16 dest) — every
        # normalization loses precision, cascading into MoE routing near-tie flips.
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Strip the router/experts subtree (handled at __init__ time) and reshape layer_scalar."""
        # DiffusionGemmaMoE consumes its weights at construction; drop MoE-related state keys
        # here so the standard loader doesn't descend into them.
        if "experts_and_router" in state and isinstance(state["experts_and_router"], dict):
            state.pop("experts_and_router", None)
        for key in list(state.keys()):
            if key.startswith("router.") or key.startswith("experts."):
                state.pop(key, None)

        # HF layer_scalar is shape (1,); ttnn wants [1, 1].
        if "layer_scalar" in state and state["layer_scalar"].ndim == 1:
            state["layer_scalar"] = state["layer_scalar"].reshape(1, 1)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        encoder_kv: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """One layer forward. Returns (hidden, k_local, v_local); K/V are captured for the
        encoder cache the decoder will consume."""
        cc = self.compute_config
        # Self-attention block.
        residual = hidden_states
        h = self.input_layernorm(hidden_states, compute_kernel_config=cc)
        attn_out, k_local, v_local = self.self_attn(h, cos, sin, attention_mask=attention_mask, encoder_kv=encoder_kv)
        ttnn.deallocate(h)
        attn_out = self.post_attention_layernorm(attn_out, compute_kernel_config=cc)
        hidden_states = ttnn.add(residual, attn_out)
        ttnn.deallocate(attn_out)

        # Feed-forward block: dense MLP + MoE in parallel, summed.
        residual = hidden_states

        h_dense = self.pre_feedforward_layernorm(hidden_states, compute_kernel_config=cc)
        h_dense = self.mlp(h_dense)
        # Gather GatedMLP's TP-fractured output back to replicated.
        if self.parallel_config.tensor_parallel.factor > 1:
            h_dense = self.ccl_manager.all_gather_persistent_buffer(
                h_dense, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        h_dense = self.post_feedforward_layernorm_1(h_dense, compute_kernel_config=cc)

        # Router sees raw residual; experts see pre_feedforward_layernorm_2(residual).
        expert_input = self.pre_feedforward_layernorm_2(residual, compute_kernel_config=cc)
        h_moe = self.experts_and_router(router_input=residual, expert_input=expert_input)
        ttnn.deallocate(expert_input)
        h_moe = self.post_feedforward_layernorm_2(h_moe, compute_kernel_config=cc)

        combined = ttnn.add(h_dense, h_moe)
        ttnn.deallocate(h_dense)
        ttnn.deallocate(h_moe)
        combined = self.post_feedforward_layernorm(combined, compute_kernel_config=cc)
        hidden_states = ttnn.add(residual, combined)
        ttnn.deallocate(combined)

        hidden_states = ttnn.multiply(hidden_states, self.layer_scalar.data)
        return hidden_states, k_local, v_local
