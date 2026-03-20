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

from ....layers.feedforward import ParallelFeedForward
from ....layers.module import Module, Parameter
from ....layers.normalization import DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.substate import rename_substate
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
        self.scale_shift_table = Parameter(
            total_shape=[1, 1, 6, video_dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
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
            total_shape=[1, 1, 6, audio_dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )

        # === BIDIRECTIONAL CROSS-ATTENTION ===
        # A→V: video queries, audio keys/values
        self.audio_to_video_attn = LTXAttention(
            dim=video_dim, num_heads=audio_num_heads, is_self=False, context_dim=audio_dim, **attn_kwargs
        )
        # V→A: audio queries, video keys/values
        self.video_to_audio_attn = LTXAttention(
            dim=audio_dim, num_heads=audio_num_heads, is_self=False, context_dim=video_dim, **attn_kwargs
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

        # Rename audio norm keys (LTX uses norm1_a/norm1_b naming for video)
        # For audio: audio_attn1/audio_attn2 are already correct in state dict

        # Unsqueeze scale_shift_tables
        for key in [
            "scale_shift_table",
            "audio_scale_shift_table",
            "scale_shift_table_a2v_ca_audio",
            "scale_shift_table_a2v_ca_video",
        ]:
            if key in state:
                state[key] = state[key].unsqueeze(0).unsqueeze(0)

        # Remove optional prompt AdaLN if present
        keys_to_remove = [k for k in state if "prompt_scale_shift" in k or "audio_prompt_scale_shift" in k]
        for k in keys_to_remove:
            del state[k]

    def forward(
        self,
        video_1BND: ttnn.Tensor,
        audio_1BND: ttnn.Tensor,
        video_prompt: ttnn.Tensor,
        audio_prompt: ttnn.Tensor,
        video_temb: ttnn.Tensor,
        audio_temb: ttnn.Tensor,
        av_ca_temb: ttnn.Tensor,  # Cross-attention timestep embedding
        video_N: int,
        audio_N: int,
        video_rope_cos: ttnn.Tensor,
        video_rope_sin: ttnn.Tensor,
        audio_rope_cos: ttnn.Tensor,
        audio_rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Process both video and audio through one transformer block."""

        # === VIDEO SELF-ATTENTION ===
        shifted_v = self.scale_shift_table.data + video_temb
        v_shift, v_scale, v_gate, v_c_shift, v_c_scale, v_c_gate = ttnn.chunk(shifted_v, 6, dim=2)
        v_gate = ttnn.typecast(v_gate, dtype=ttnn.bfloat16)
        v_c_gate = ttnn.typecast(v_c_gate, dtype=ttnn.bfloat16)

        video_normed = self.norm1(video_1BND)
        video_normed = video_normed * (1.0 + v_scale) + v_shift
        video_1BND = self.attn1(
            spatial_1BND=video_normed,
            N=video_N,
            rope_cos=video_rope_cos,
            rope_sin=video_rope_sin,
            trans_mat=trans_mat,
            addcmul_residual=video_1BND,
            addcmul_gate=v_gate,
        )

        # Video text cross-attention
        video_normed = self.norm2(video_1BND)
        video_1BND = video_1BND + self.attn2(spatial_1BND=video_normed, N=video_N, prompt_1BLP=video_prompt)

        # === AUDIO SELF-ATTENTION ===
        shifted_a = self.audio_scale_shift_table.data + audio_temb
        a_shift, a_scale, a_gate, a_c_shift, a_c_scale, a_c_gate = ttnn.chunk(shifted_a, 6, dim=2)
        a_gate = ttnn.typecast(a_gate, dtype=ttnn.bfloat16)
        a_c_gate = ttnn.typecast(a_c_gate, dtype=ttnn.bfloat16)

        audio_normed = self.audio_norm1(audio_1BND)
        audio_normed = audio_normed * (1.0 + a_scale) + a_shift
        audio_1BND = self.audio_attn1(
            spatial_1BND=audio_normed,
            N=audio_N,
            rope_cos=audio_rope_cos,
            rope_sin=audio_rope_sin,
            trans_mat=trans_mat,
            addcmul_residual=audio_1BND,
            addcmul_gate=a_gate,
        )

        # Audio text cross-attention
        audio_normed = self.audio_norm2(audio_1BND)
        audio_1BND = audio_1BND + self.audio_attn2(spatial_1BND=audio_normed, N=audio_N, prompt_1BLP=audio_prompt)

        # === BIDIRECTIONAL A↔V CROSS-ATTENTION ===
        # Use av_ca_temb for cross-attention modulation
        shifted_av = self.scale_shift_table_a2v_ca_video.data + av_ca_temb[:, :, :5, : self.video_dim]
        v_ca_shift, v_ca_scale, a_ca_shift_v, a_ca_scale_v, v_ca_gate = ttnn.chunk(shifted_av, 5, dim=2)
        v_ca_gate = ttnn.typecast(v_ca_gate, dtype=ttnn.bfloat16)

        # A→V: audio provides context for video
        video_scaled = self.norm1(video_1BND) * (1.0 + v_ca_scale) + v_ca_shift
        # Simplified: skip audio scaling for now, just pass through
        a2v_output = self.audio_to_video_attn(spatial_1BND=video_scaled, N=video_N, prompt_1BLP=audio_1BND)
        video_1BND = video_1BND + a2v_output * v_ca_gate

        # V→A: video provides context for audio
        shifted_av_a = self.scale_shift_table_a2v_ca_audio.data + av_ca_temb[:, :, :5, : self.audio_dim]
        a_ca_shift, a_ca_scale, v_ca_shift_a, v_ca_scale_a, a_ca_gate = ttnn.chunk(shifted_av_a, 5, dim=2)
        a_ca_gate = ttnn.typecast(a_ca_gate, dtype=ttnn.bfloat16)

        audio_scaled = self.audio_norm1(audio_1BND) * (1.0 + a_ca_scale) + a_ca_shift
        v2a_output = self.video_to_audio_attn(spatial_1BND=audio_scaled, N=audio_N, prompt_1BLP=video_1BND)
        audio_1BND = audio_1BND + v2a_output * a_ca_gate

        # === FEEDFORWARD ===
        # Video FF
        video_normed = self.norm3(video_1BND)
        video_normed = video_normed * (1.0 + v_c_scale) + v_c_shift
        if self.parallel_config.tensor_parallel.factor > 1:
            video_normed = self.ccl_manager.all_gather_persistent_buffer(
                video_normed, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        video_ff = self.ffn(video_normed, compute_kernel_config=self.ff_compute_kernel_config)
        video_1BND = ttnn.addcmul(video_1BND, video_ff, v_c_gate)

        # Audio FF
        audio_normed = self.audio_norm3(audio_1BND)
        audio_normed = audio_normed * (1.0 + a_c_scale) + a_c_shift
        if self.parallel_config.tensor_parallel.factor > 1:
            audio_normed = self.ccl_manager.all_gather_persistent_buffer(
                audio_normed, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        audio_ff = self.audio_ff(audio_normed, compute_kernel_config=self.ff_compute_kernel_config)
        audio_1BND = ttnn.addcmul(audio_1BND, audio_ff, a_c_gate)

        return video_1BND, audio_1BND
