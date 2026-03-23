# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 Audio-Video Transformer Block for tt_dit.

Extends the video-only LTXTransformerBlock with:
- Audio self-attention, cross-attention, and feedforward
- Bidirectional audio-video cross-attention
- Separate AdaLN modulation for audio and cross-modal paths

Reference: LTX-2 transformer.py BasicAVTransformerBlock
"""

from __future__ import annotations

import torch

import ttnn

from ....layers.embeddings import LTXAdaLayerNormSingle
from ....layers.feedforward import ParallelFeedForward
from ....layers.linear import ColParallelLinear, Linear
from ....layers.module import Module, ModuleList, Parameter
from ....layers.normalization import DistributedLayerNorm, DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.substate import pop_substate, rename_substate
from .attention_ltx import LTXAttention


class LTXAudioVideoTransformerBlock(Module):
    """
    LTX-2 AudioVideo DiT transformer block.

    Processes both video and audio modalities with bidirectional cross-attention.

    Structure per modality (video and audio independently):
    1. RMSNorm + AdaLN → self-attention + gate residual
    2. RMSNorm → text cross-attention + residual
    3. Bidirectional A↔V cross-attention (when both modalities active)
    4. RMSNorm + AdaLN → feedforward + gate residual
    """

    def __init__(
        self,
        *,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        video_ffn_dim: int = 16384,
        audio_ffn_dim: int = 8192,
        video_num_heads: int = 32,
        audio_num_heads: int = 32,
        video_cross_attention_dim: int = 4096,
        audio_cross_attention_dim: int = 2048,
        eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        rms_norm_kwargs = {
            "norm_eps": eps,
            "norm_elementwise_affine": False,
            "bias": False,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
            "mesh_device": mesh_device,
            "ccl_manager": ccl_manager,
        }

        attn_kwargs = {
            "eps": eps,
            "mesh_device": mesh_device,
            "ccl_manager": ccl_manager,
            "parallel_config": parallel_config,
            "is_fsdp": is_fsdp,
        }

        # === VIDEO PATH ===
        self.norm1 = DistributedRMSNorm(embedding_dim=video_dim, **rms_norm_kwargs)
        self.attn1 = LTXAttention(dim=video_dim, num_heads=video_num_heads, is_self=True, **attn_kwargs)
        self.norm2 = DistributedRMSNorm(embedding_dim=video_dim, **rms_norm_kwargs)
        self.attn2 = LTXAttention(
            dim=video_dim,
            num_heads=video_num_heads,
            is_self=False,
            context_dim=video_cross_attention_dim,
            **attn_kwargs,
        )
        self.norm3 = DistributedRMSNorm(embedding_dim=video_dim, **rms_norm_kwargs)
        self.ffn = ParallelFeedForward(
            video_dim,
            inner_dim=video_ffn_dim,
            activation_fn="gelu_tanh",
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        # 9 params: shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff, shift_ca, scale_ca, gate_ca
        self.scale_shift_table = Parameter(
            total_shape=[1, 1, 9, video_dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )
        # 2 params for prompt context (K/V) modulation: shift_kv, scale_kv
        # NOT TP-sharded — modulates full-dim prompt context
        self.prompt_scale_shift_table = Parameter(
            total_shape=[1, 1, 2, video_dim],
            mesh_axes=[None, None, None, None],
            device=mesh_device,
            dtype=ttnn.float32,
        )

        # === AUDIO PATH ===
        self.audio_norm1 = DistributedRMSNorm(embedding_dim=audio_dim, **rms_norm_kwargs)
        self.audio_attn1 = LTXAttention(dim=audio_dim, num_heads=audio_num_heads, is_self=True, **attn_kwargs)
        self.audio_norm2 = DistributedRMSNorm(embedding_dim=audio_dim, **rms_norm_kwargs)
        self.audio_attn2 = LTXAttention(
            dim=audio_dim,
            num_heads=audio_num_heads,
            is_self=False,
            context_dim=audio_cross_attention_dim,
            **attn_kwargs,
        )
        self.audio_norm3 = DistributedRMSNorm(embedding_dim=audio_dim, **rms_norm_kwargs)
        self.audio_ff = ParallelFeedForward(
            audio_dim,
            inner_dim=audio_ffn_dim,
            activation_fn="gelu_tanh",
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.audio_scale_shift_table = Parameter(
            total_shape=[1, 1, 9, audio_dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )
        self.audio_prompt_scale_shift_table = Parameter(
            total_shape=[1, 1, 2, audio_dim],
            mesh_axes=[None, None, None, None],
            device=mesh_device,
            dtype=ttnn.float32,
        )

        # === BIDIRECTIONAL CROSS-ATTENTION ===
        # A→V: video queries (projected from video_dim → audio_dim), audio keys/values
        # Attention operates in audio_dim space, output projects back to video_dim
        self.audio_to_video_attn = LTXAttention(
            dim=audio_dim,
            num_heads=audio_num_heads,
            is_self=False,
            context_dim=audio_dim,
            query_input_dim=video_dim,
            output_dim=video_dim,
            **attn_kwargs,
        )
        # V→A: audio queries, video keys/values (projected from video_dim → audio_dim)
        self.video_to_audio_attn = LTXAttention(
            dim=audio_dim,
            num_heads=audio_num_heads,
            is_self=False,
            context_dim=video_dim,
            **attn_kwargs,
        )

        # Cross-attention AdaLN tables
        self.scale_shift_table_a2v_ca_audio = Parameter(
            total_shape=[1, 1, 5, audio_dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )
        self.scale_shift_table_a2v_ca_video = Parameter(
            total_shape=[1, 1, 5, video_dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )

        self.ff_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Rename ff keys
        rename_substate(state, "ff.net.0.proj", "ffn.ff1")
        rename_substate(state, "ff.net.2", "ffn.ff2")
        rename_substate(state, "audio_ff.net.0.proj", "audio_ff.ff1")
        rename_substate(state, "audio_ff.net.2", "audio_ff.ff2")

        # Unsqueeze all scale_shift_tables (no truncation — use full 9 params)
        for key in [
            "scale_shift_table",
            "audio_scale_shift_table",
            "prompt_scale_shift_table",
            "audio_prompt_scale_shift_table",
            "scale_shift_table_a2v_ca_audio",
            "scale_shift_table_a2v_ca_video",
        ]:
            if key in state:
                state[key] = state[key].unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        video_1BND: ttnn.Tensor,
        audio_1BND: ttnn.Tensor,
        video_prompt: ttnn.Tensor,
        audio_prompt: ttnn.Tensor,
        video_temb: ttnn.Tensor,  # (1, B, 9, video_dim) from adaln_single
        audio_temb: ttnn.Tensor,  # (1, B, 9, audio_dim) from audio_adaln_single
        av_ca_temb: ttnn.Tensor,  # Cross-attention video-side temb (1, B, 5, video_dim)
        video_N: int,
        audio_N: int,
        video_rope_cos: ttnn.Tensor,
        video_rope_sin: ttnn.Tensor,
        audio_rope_cos: ttnn.Tensor,
        audio_rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        av_ca_audio_temb: ttnn.Tensor | None = None,
        video_prompt_temb: ttnn.Tensor | None = None,  # (1, B, 2, video_dim) from prompt_adaln_single
        audio_prompt_temb: ttnn.Tensor | None = None,  # (1, B, 2, audio_dim)
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Process both video and audio through one transformer block.

        Uses all 9 modulation params from scale_shift_table:
        - indices 0-2: self-attention (shift, scale, gate)
        - indices 3-5: feedforward (shift, scale, gate)
        - indices 6-8: text cross-attention (shift_q, scale_q, gate)
        """

        # === UNPACK VIDEO MODULATION (9 params) ===
        shifted_v = self.scale_shift_table.data + video_temb
        (
            v_shift_sa,
            v_scale_sa,
            v_gate_sa,
            v_shift_ff,
            v_scale_ff,
            v_gate_ff,
            v_shift_ca,
            v_scale_ca,
            v_gate_ca,
        ) = ttnn.chunk(shifted_v, 9, dim=2)
        v_gate_sa = ttnn.typecast(v_gate_sa, dtype=ttnn.bfloat16)
        v_gate_ff = ttnn.typecast(v_gate_ff, dtype=ttnn.bfloat16)
        v_gate_ca = ttnn.typecast(v_gate_ca, dtype=ttnn.bfloat16)

        # === VIDEO SELF-ATTENTION ===
        video_normed = self.norm1(video_1BND)
        video_normed = video_normed * (1.0 + v_scale_sa) + v_shift_sa
        video_1BND = self.attn1(
            spatial_1BND=video_normed,
            N=video_N,
            rope_cos=video_rope_cos,
            rope_sin=video_rope_sin,
            trans_mat=trans_mat,
            addcmul_residual=video_1BND,
            addcmul_gate=v_gate_sa,
        )

        # === VIDEO TEXT CROSS-ATTENTION (with AdaLN) ===
        # Query modulation from scale_shift_table indices 6-8
        video_ca_input = self.norm2(video_1BND) * (1.0 + v_scale_ca) + v_shift_ca
        # Context (K/V) modulation from prompt_scale_shift_table + prompt_temb
        if video_prompt_temb is not None:
            shifted_prompt_v = self.prompt_scale_shift_table.data + video_prompt_temb
            v_kv_shift, v_kv_scale = ttnn.chunk(shifted_prompt_v, 2, dim=2)
            video_prompt_mod = video_prompt * (1.0 + v_kv_scale) + v_kv_shift
        else:
            video_prompt_mod = video_prompt
        video_ca_out = self.attn2(spatial_1BND=video_ca_input, N=video_N, prompt_1BLP=video_prompt_mod)
        video_1BND = video_1BND + video_ca_out * v_gate_ca

        # === UNPACK AUDIO MODULATION (9 params) ===
        shifted_a = self.audio_scale_shift_table.data + audio_temb
        (
            a_shift_sa,
            a_scale_sa,
            a_gate_sa,
            a_shift_ff,
            a_scale_ff,
            a_gate_ff,
            a_shift_ca,
            a_scale_ca,
            a_gate_ca,
        ) = ttnn.chunk(shifted_a, 9, dim=2)
        a_gate_sa = ttnn.typecast(a_gate_sa, dtype=ttnn.bfloat16)
        a_gate_ff = ttnn.typecast(a_gate_ff, dtype=ttnn.bfloat16)
        a_gate_ca = ttnn.typecast(a_gate_ca, dtype=ttnn.bfloat16)

        # === AUDIO SELF-ATTENTION ===
        audio_normed = self.audio_norm1(audio_1BND)
        audio_normed = audio_normed * (1.0 + a_scale_sa) + a_shift_sa
        audio_1BND = self.audio_attn1(
            spatial_1BND=audio_normed,
            N=audio_N,
            rope_cos=audio_rope_cos,
            rope_sin=audio_rope_sin,
            trans_mat=trans_mat,
            addcmul_residual=audio_1BND,
            addcmul_gate=a_gate_sa,
        )

        # === AUDIO TEXT CROSS-ATTENTION (with AdaLN) ===
        audio_ca_input = self.audio_norm2(audio_1BND) * (1.0 + a_scale_ca) + a_shift_ca
        if audio_prompt_temb is not None:
            shifted_prompt_a = self.audio_prompt_scale_shift_table.data + audio_prompt_temb
            a_kv_shift, a_kv_scale = ttnn.chunk(shifted_prompt_a, 2, dim=2)
            audio_prompt_mod = audio_prompt * (1.0 + a_kv_scale) + a_kv_shift
        else:
            audio_prompt_mod = audio_prompt
        audio_ca_out = self.audio_attn2(spatial_1BND=audio_ca_input, N=audio_N, prompt_1BLP=audio_prompt_mod)
        audio_1BND = audio_1BND + audio_ca_out * a_gate_ca

        # === BIDIRECTIONAL A↔V CROSS-ATTENTION ===
        # Video-side cross-attention modulation
        shifted_av = self.scale_shift_table_a2v_ca_video.data + av_ca_temb
        v_ca_shift, v_ca_scale, a_ca_shift_v, a_ca_scale_v, v_ca_gate = ttnn.chunk(shifted_av, 5, dim=2)
        v_ca_gate = ttnn.typecast(v_ca_gate, dtype=ttnn.bfloat16)

        # Gather cross-attention contexts (TP-sharded → full dim for to_kv)
        if self.parallel_config.tensor_parallel.factor > 1:
            audio_context = self.ccl_manager.all_gather_persistent_buffer(
                audio_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
            video_context = self.ccl_manager.all_gather_persistent_buffer(
                video_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        else:
            audio_context = audio_1BND
            video_context = video_1BND

        # A→V: audio provides context for video
        video_scaled = self.norm1(video_1BND) * (1.0 + v_ca_scale) + v_ca_shift
        a2v_output = self.audio_to_video_attn(spatial_1BND=video_scaled, N=video_N, prompt_1BLP=audio_context)
        video_1BND = video_1BND + a2v_output * v_ca_gate

        # Audio-side cross-attention modulation
        _av_ca_audio_temb = av_ca_audio_temb if av_ca_audio_temb is not None else av_ca_temb[:, :, :5, : self.audio_dim]
        shifted_av_a = self.scale_shift_table_a2v_ca_audio.data + _av_ca_audio_temb
        a_ca_shift, a_ca_scale, v_ca_shift_a, v_ca_scale_a, a_ca_gate = ttnn.chunk(shifted_av_a, 5, dim=2)
        a_ca_gate = ttnn.typecast(a_ca_gate, dtype=ttnn.bfloat16)

        # V→A: video provides context for audio
        audio_scaled = self.audio_norm1(audio_1BND) * (1.0 + a_ca_scale) + a_ca_shift
        v2a_output = self.video_to_audio_attn(spatial_1BND=audio_scaled, N=audio_N, prompt_1BLP=video_context)
        audio_1BND = audio_1BND + v2a_output * a_ca_gate

        # === VIDEO FEEDFORWARD ===
        video_normed = self.norm3(video_1BND)
        video_normed = video_normed * (1.0 + v_scale_ff) + v_shift_ff
        if self.parallel_config.tensor_parallel.factor > 1:
            video_normed = self.ccl_manager.all_gather_persistent_buffer(
                video_normed, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        video_ff = self.ffn(video_normed, compute_kernel_config=self.ff_compute_kernel_config)
        video_1BND = ttnn.addcmul(video_1BND, video_ff, v_gate_ff)

        # === AUDIO FEEDFORWARD ===
        audio_normed = self.audio_norm3(audio_1BND)
        audio_normed = audio_normed * (1.0 + a_scale_ff) + a_shift_ff
        if self.parallel_config.tensor_parallel.factor > 1:
            audio_normed = self.ccl_manager.all_gather_persistent_buffer(
                audio_normed, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        audio_ff = self.audio_ff(audio_normed, compute_kernel_config=self.ff_compute_kernel_config)
        audio_1BND = ttnn.addcmul(audio_1BND, audio_ff, a_gate_ff)

        return video_1BND, audio_1BND


class LTXAudioVideoTransformerModel(Module):
    """
    Full LTX-2 AudioVideo DiT transformer model.

    Wraps N x LTXAudioVideoTransformerBlock with model-level components:
    - Separate patchify_proj, adaln_single, norm_out, proj_out for video and audio
    - Cross-attention timestep conditioning (av_ca adaln modules)

    Reference: LTX-2 model.py LTXModel(model_type=AudioVideo)
    """

    def __init__(
        self,
        *,
        # Video config
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        # Audio config
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        # Common
        norm_eps: float = 1e-6,
        ffn_mult: int = 4,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        self.inner_dim = num_attention_heads * attention_head_dim  # 4096
        self.audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim  # 2048
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # === VIDEO MODEL-LEVEL ===
        self.patchify_proj = ColParallelLinear(
            in_channels,
            self.inner_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.adaln_single = LTXAdaLayerNormSingle(
            embedding_dim=self.inner_dim,
            embedding_coefficient=9,
            mesh_device=mesh_device,
        )
        # Prompt timestep conditioning (coefficient=2: shift_kv, scale_kv for context modulation)
        self.prompt_adaln_single = LTXAdaLayerNormSingle(
            embedding_dim=self.inner_dim,
            embedding_coefficient=2,
            mesh_device=mesh_device,
        )
        self.norm_out = DistributedLayerNorm(
            self.inner_dim,
            norm_eps=norm_eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.scale_shift_table = Parameter(
            total_shape=[1, 1, 2, self.inner_dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )
        self.proj_out = Linear(self.inner_dim, out_channels, bias=True, mesh_device=mesh_device)

        # === AUDIO MODEL-LEVEL ===
        self.audio_patchify_proj = ColParallelLinear(
            audio_in_channels,
            self.audio_inner_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.audio_adaln_single = LTXAdaLayerNormSingle(
            embedding_dim=self.audio_inner_dim,
            embedding_coefficient=9,
            mesh_device=mesh_device,
        )
        self.audio_prompt_adaln_single = LTXAdaLayerNormSingle(
            embedding_dim=self.audio_inner_dim,
            embedding_coefficient=2,
            mesh_device=mesh_device,
        )
        self.audio_norm_out = DistributedLayerNorm(
            self.audio_inner_dim,
            norm_eps=norm_eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.audio_scale_shift_table = Parameter(
            total_shape=[1, 1, 2, self.audio_inner_dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )
        self.audio_proj_out = Linear(self.audio_inner_dim, audio_out_channels, bias=True, mesh_device=mesh_device)

        # === CROSS-ATTENTION TIMESTEP CONDITIONING ===
        # These produce per-step modulation params for bidirectional A↔V cross-attention
        self.av_ca_video_scale_shift_adaln_single = LTXAdaLayerNormSingle(
            embedding_dim=self.inner_dim,
            embedding_coefficient=4,
            mesh_device=mesh_device,
        )
        self.av_ca_audio_scale_shift_adaln_single = LTXAdaLayerNormSingle(
            embedding_dim=self.audio_inner_dim,
            embedding_coefficient=4,
            mesh_device=mesh_device,
        )
        self.av_ca_a2v_gate_adaln_single = LTXAdaLayerNormSingle(
            embedding_dim=self.inner_dim,
            embedding_coefficient=1,
            mesh_device=mesh_device,
        )
        self.av_ca_v2a_gate_adaln_single = LTXAdaLayerNormSingle(
            embedding_dim=self.audio_inner_dim,
            embedding_coefficient=1,
            mesh_device=mesh_device,
        )

        # === TRANSFORMER BLOCKS ===
        video_ffn_dim = self.inner_dim * ffn_mult
        audio_ffn_dim = self.audio_inner_dim * ffn_mult
        self.transformer_blocks = ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                LTXAudioVideoTransformerBlock(
                    video_dim=self.inner_dim,
                    audio_dim=self.audio_inner_dim,
                    video_ffn_dim=video_ffn_dim,
                    audio_ffn_dim=audio_ffn_dim,
                    video_num_heads=num_attention_heads,
                    audio_num_heads=audio_num_attention_heads,
                    video_cross_attention_dim=cross_attention_dim,
                    audio_cross_attention_dim=audio_cross_attention_dim,
                    eps=norm_eps,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    is_fsdp=is_fsdp,
                )
            )

        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Prepare 22B checkpoint state dict for loading."""
        # Unsqueeze output scale_shift_tables
        for key in ["scale_shift_table", "audio_scale_shift_table"]:
            if key in state:
                state[key] = state[key].unsqueeze(0).unsqueeze(0)

        # No truncation needed — adaln_single uses coefficient=9 matching the 22B checkpoint

        # Remove 22B-specific keys not yet implemented
        pop_substate(state, "caption_projection")
        pop_substate(state, "audio_caption_projection")
        pop_substate(state, "video_embeddings_connector")
        pop_substate(state, "audio_embeddings_connector")

    def forward(self, **kwargs):
        """Forward pass — delegates to inner_step for denoising."""
        return self.inner_step(**kwargs)

    def inner_step(
        self,
        # Video
        video_1BNI_torch: torch.Tensor,
        video_prompt_1BLP: ttnn.Tensor,
        video_rope_cos: ttnn.Tensor,
        video_rope_sin: ttnn.Tensor,
        video_N: int,
        # Audio
        audio_1BNI_torch: torch.Tensor,
        audio_prompt_1BLP: ttnn.Tensor,
        audio_rope_cos: ttnn.Tensor,
        audio_rope_sin: ttnn.Tensor,
        audio_N: int,
        # Shared
        trans_mat: ttnn.Tensor,
        timestep_torch: torch.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run one denoising step for both video and audio on device."""
        from ....utils.tensor import bf16_tensor, float32_tensor

        # Push spatial inputs to device (SP-sharded)
        video_1BNI = bf16_tensor(
            video_1BNI_torch,
            device=self.mesh_device,
            mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
            shard_dim=-2,
        )
        audio_1BNI = bf16_tensor(
            audio_1BNI_torch,
            device=self.mesh_device,
            mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
            shard_dim=-2,
        )

        # Timestep embedding
        B_size = video_1BNI_torch.shape[1]
        timestep = float32_tensor(
            timestep_torch.reshape(1, 1, B_size, 1) * 1000.0,
            device=self.mesh_device,
        )

        # Video modulation (9 params: 3 self-attn + 3 FF + 3 cross-attn)
        video_modulation, video_emb_ts = self.adaln_single(timestep)
        B = video_modulation.shape[2]
        video_mod_1B9D = ttnn.reshape(video_modulation, (1, B, 9, self.inner_dim))
        if self.parallel_config.tensor_parallel.factor > 1:
            video_mod_1B9D = ttnn.mesh_partition(
                video_mod_1B9D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        # Video prompt timestep uses SAME scaled timestep as main adaln
        video_prompt_mod, _ = self.prompt_adaln_single(timestep)
        video_prompt_1B2D = ttnn.reshape(video_prompt_mod, (1, B, 2, self.inner_dim))

        # Audio modulation (9 params)
        audio_modulation, audio_emb_ts = self.audio_adaln_single(timestep)
        audio_mod_1B9D = ttnn.reshape(audio_modulation, (1, B, 9, self.audio_inner_dim))
        if self.parallel_config.tensor_parallel.factor > 1:
            audio_mod_1B9D = ttnn.mesh_partition(
                audio_mod_1B9D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        # Audio prompt timestep uses SAME scaled timestep
        audio_prompt_mod, _ = self.audio_prompt_adaln_single(timestep)
        audio_prompt_1B2D = ttnn.reshape(audio_prompt_mod, (1, B, 2, self.audio_inner_dim))

        # Cross-attention timestep conditioning
        # scale_shift produces 4 params per modality (2 for Q side, 2 for KV side)
        # gate produces 1 param per direction
        av_ca_video_ss, _ = self.av_ca_video_scale_shift_adaln_single(timestep)
        av_ca_video_ss = ttnn.reshape(av_ca_video_ss, (1, B, 4, self.inner_dim))
        av_ca_a2v_gate, _ = self.av_ca_a2v_gate_adaln_single(timestep)
        av_ca_a2v_gate = ttnn.reshape(av_ca_a2v_gate, (1, B, 1, self.inner_dim))
        # Concat scale_shift (4) + gate (1) = 5 params for video side
        av_ca_video_temb = ttnn.concat([av_ca_video_ss, av_ca_a2v_gate], dim=2)

        av_ca_audio_ss, _ = self.av_ca_audio_scale_shift_adaln_single(timestep)
        av_ca_audio_ss = ttnn.reshape(av_ca_audio_ss, (1, B, 4, self.audio_inner_dim))
        av_ca_v2a_gate, _ = self.av_ca_v2a_gate_adaln_single(timestep)
        av_ca_v2a_gate = ttnn.reshape(av_ca_v2a_gate, (1, B, 1, self.audio_inner_dim))
        av_ca_audio_temb = ttnn.concat([av_ca_audio_ss, av_ca_v2a_gate], dim=2)

        if self.parallel_config.tensor_parallel.factor > 1:
            av_ca_video_temb = ttnn.mesh_partition(
                av_ca_video_temb, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
            av_ca_audio_temb = ttnn.mesh_partition(
                av_ca_audio_temb, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        # Patchify
        video_1BND = self.patchify_proj(video_1BNI)
        audio_1BND = self.audio_patchify_proj(audio_1BNI)

        # Transformer blocks
        for block in self.transformer_blocks:
            video_1BND, audio_1BND = block(
                video_1BND=video_1BND,
                audio_1BND=audio_1BND,
                video_prompt=video_prompt_1BLP,
                audio_prompt=audio_prompt_1BLP,
                video_temb=video_mod_1B9D,
                audio_temb=audio_mod_1B9D,
                av_ca_temb=av_ca_video_temb,
                video_N=video_N,
                audio_N=audio_N,
                video_rope_cos=video_rope_cos,
                video_rope_sin=video_rope_sin,
                audio_rope_cos=audio_rope_cos,
                audio_rope_sin=audio_rope_sin,
                trans_mat=trans_mat,
                av_ca_audio_temb=av_ca_audio_temb,
                video_prompt_temb=video_prompt_1B2D,
                audio_prompt_temb=audio_prompt_1B2D,
            )

        # Output projection — video
        v_inner_local = video_emb_ts.shape[-1]
        v_emb_1B1D = ttnn.reshape(video_emb_ts, (1, B, 1, v_inner_local))
        if self.parallel_config.tensor_parallel.factor > 1:
            v_emb_1B1D = ttnn.mesh_partition(
                v_emb_1B1D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        shifted_v = self.scale_shift_table.data + v_emb_1B1D
        v_shift_out, v_scale_out = ttnn.chunk(shifted_v, 2, dim=2)
        video_normed = self.norm_out(video_1BND)
        video_1BND = video_normed * (1.0 + v_scale_out) + v_shift_out
        if self.parallel_config.tensor_parallel.factor > 1:
            video_1BND = self.ccl_manager.all_gather_persistent_buffer(
                video_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        video_out = self.proj_out(
            video_1BND, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=ttnn.float32
        )

        # Output projection — audio
        a_inner_local = audio_emb_ts.shape[-1]
        a_emb_1B1D = ttnn.reshape(audio_emb_ts, (1, B, 1, a_inner_local))
        if self.parallel_config.tensor_parallel.factor > 1:
            a_emb_1B1D = ttnn.mesh_partition(
                a_emb_1B1D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        shifted_a = self.audio_scale_shift_table.data + a_emb_1B1D
        a_shift_out, a_scale_out = ttnn.chunk(shifted_a, 2, dim=2)
        audio_normed = self.audio_norm_out(audio_1BND)
        audio_1BND = audio_normed * (1.0 + a_scale_out) + a_shift_out
        if self.parallel_config.tensor_parallel.factor > 1:
            audio_1BND = self.ccl_manager.all_gather_persistent_buffer(
                audio_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        audio_out = self.audio_proj_out(
            audio_1BND, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=ttnn.float32
        )

        # SP gather
        if self.parallel_config.sequence_parallel.factor > 1:
            video_out = self.ccl_manager.all_gather_persistent_buffer(
                video_out, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
            )
            audio_out = self.ccl_manager.all_gather_persistent_buffer(
                audio_out, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
            )

        return video_out, audio_out

    @staticmethod
    def device_to_host(tt_tensor: ttnn.Tensor) -> torch.Tensor:
        """Move a ttnn device tensor to a torch host tensor."""
        return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0])
