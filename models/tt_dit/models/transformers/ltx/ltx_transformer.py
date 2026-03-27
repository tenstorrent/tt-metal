# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unified LTX-2 Transformer for both video-only and audio-video modes.

When has_audio=False, this is equivalent to the original LTXTransformerBlock/Model.
When has_audio=True, it adds:
- Audio self-attention, cross-attention, and feedforward
- Bidirectional audio-video cross-attention
- Separate AdaLN modulation for audio and cross-modal paths

Reference: LTX-2 transformer.py BasicAVTransformerBlock
"""

from __future__ import annotations

import torch
from loguru import logger

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


class LTXTransformerBlock(Module):
    """
    Unified LTX-2 DiT transformer block for video-only and audio-video modes.

    When has_audio=True:
    - Processes both video and audio modalities with bidirectional cross-attention.
    - Returns tuple (video_1BND, audio_1BND).

    When has_audio=False:
    - Processes video only (equivalent to the original LTXTransformerBlock).
    - Returns single video_1BND tensor.
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
        has_audio: bool = False,
        apply_gated_attention: bool = False,
        cross_attention_adaln: bool = True,
    ) -> None:
        super().__init__()

        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.has_audio = has_audio
        self.cross_attention_adaln = cross_attention_adaln
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
            "apply_gated_attention": apply_gated_attention,
        }

        # Video-only uses fsdp_mesh_axis for FFN; AV does not.
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if (is_fsdp and not has_audio) else None

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
            fsdp_mesh_axis=fsdp_mesh_axis,
        )
        self.adaln_coeff = 9 if cross_attention_adaln else 6
        self.scale_shift_table = Parameter(
            total_shape=[1, 1, self.adaln_coeff, video_dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )
        if cross_attention_adaln:
            self.prompt_scale_shift_table = Parameter(
                total_shape=[1, 1, 2, video_dim],
                mesh_axes=[None, None, None, None],
                device=mesh_device,
                dtype=ttnn.float32,
            )

        # === AUDIO PATH (only when has_audio=True) ===
        if has_audio:
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
                total_shape=[1, 1, self.adaln_coeff, audio_dim],
                mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
                device=mesh_device,
                dtype=ttnn.float32,
            )
            if cross_attention_adaln:
                self.audio_prompt_scale_shift_table = Parameter(
                    total_shape=[1, 1, 2, audio_dim],
                    mesh_axes=[None, None, None, None],
                    device=mesh_device,
                    dtype=ttnn.float32,
                )

            # === BIDIRECTIONAL CROSS-ATTENTION ===
            self.audio_to_video_attn = LTXAttention(
                dim=audio_dim,
                num_heads=audio_num_heads,
                is_self=False,
                context_dim=audio_dim,
                query_input_dim=video_dim,
                output_dim=video_dim,
                **attn_kwargs,
            )
            self.video_to_audio_attn = LTXAttention(
                dim=audio_dim,
                num_heads=audio_num_heads,
                is_self=False,
                context_dim=video_dim,
                **attn_kwargs,
            )
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
        rename_substate(state, "ff.net.0.proj", "ffn.ff1")
        rename_substate(state, "ff.net.2", "ffn.ff2")

        if self.has_audio:
            rename_substate(state, "audio_ff.net.0.proj", "audio_ff.ff1")
            rename_substate(state, "audio_ff.net.2", "audio_ff.ff2")

        tables = ["scale_shift_table"]
        if self.cross_attention_adaln:
            tables.append("prompt_scale_shift_table")
        else:
            # 6-output mode has no prompt tables — remove if present in checkpoint
            pop_substate(state, "prompt_scale_shift_table")
        if self.has_audio:
            tables.append("audio_scale_shift_table")
            if self.cross_attention_adaln:
                tables.append("audio_prompt_scale_shift_table")
            else:
                pop_substate(state, "audio_prompt_scale_shift_table")
            tables += [
                "scale_shift_table_a2v_ca_audio",
                "scale_shift_table_a2v_ca_video",
            ]
        for key in tables:
            if key in state:
                state[key] = state[key].unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        video_1BND: ttnn.Tensor,
        video_prompt: ttnn.Tensor,
        video_temb: ttnn.Tensor,  # (1, B, 9, video_dim)
        video_N: int,
        video_rope_cos: ttnn.Tensor,
        video_rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor | None = None,
        video_prompt_temb: ttnn.Tensor | None = None,
        # Audio args (only used when has_audio=True)
        audio_1BND: ttnn.Tensor | None = None,
        audio_prompt: ttnn.Tensor | None = None,
        audio_temb: ttnn.Tensor | None = None,
        av_ca_temb: ttnn.Tensor | None = None,
        audio_N: int = 0,
        audio_rope_cos: ttnn.Tensor | None = None,
        audio_rope_sin: ttnn.Tensor | None = None,
        av_ca_audio_temb: ttnn.Tensor | None = None,
        audio_prompt_temb: ttnn.Tensor | None = None,
        video_cross_pe_cos: ttnn.Tensor | None = None,
        video_cross_pe_sin: ttnn.Tensor | None = None,
        audio_cross_pe_cos: ttnn.Tensor | None = None,
        audio_cross_pe_sin: ttnn.Tensor | None = None,
        video_cross_pe_cos_full: ttnn.Tensor | None = None,
        video_cross_pe_sin_full: ttnn.Tensor | None = None,
        audio_cross_pe_cos_full: ttnn.Tensor | None = None,
        audio_cross_pe_sin_full: ttnn.Tensor | None = None,
        skip_cross_attn: bool = False,
        skip_self_attn: bool = False,
        audio_attn_mask: ttnn.Tensor | None = None,
        audio_padding_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        # === VIDEO MODULATION ===
        shifted_v = self.scale_shift_table.data + video_temb
        chunks = ttnn.chunk(shifted_v, self.adaln_coeff, dim=2)
        v_shift_sa, v_scale_sa, v_gate_sa = chunks[0], chunks[1], chunks[2]
        v_shift_ff, v_scale_ff, v_gate_ff = chunks[3], chunks[4], chunks[5]
        v_gate_sa = ttnn.typecast(v_gate_sa, dtype=ttnn.bfloat16)
        v_gate_ff = ttnn.typecast(v_gate_ff, dtype=ttnn.bfloat16)
        if self.cross_attention_adaln:
            v_shift_ca, v_scale_ca, v_gate_ca = chunks[6], chunks[7], chunks[8]
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
            skip_qk=skip_self_attn,
        )

        # === VIDEO TEXT CROSS-ATTENTION ===
        if self.cross_attention_adaln:
            video_ca_input = self.norm2(video_1BND) * (1.0 + v_scale_ca) + v_shift_ca
            if video_prompt_temb is not None:
                shifted_prompt_v = self.prompt_scale_shift_table.data + video_prompt_temb
                v_kv_shift, v_kv_scale = ttnn.chunk(shifted_prompt_v, 2, dim=2)
                video_prompt_mod = video_prompt * (1.0 + v_kv_scale) + v_kv_shift
            else:
                video_prompt_mod = video_prompt
            video_ca_out = self.attn2(spatial_1BND=video_ca_input, N=video_N, prompt_1BLP=video_prompt_mod)
            video_1BND = video_1BND + video_ca_out * v_gate_ca
        else:
            # 6-output mode: cross-attention with fixed norm (no AdaLN shift/scale/gate)
            video_ca_input = self.norm2(video_1BND)
            video_ca_out = self.attn2(spatial_1BND=video_ca_input, N=video_N, prompt_1BLP=video_prompt)
            video_1BND = video_1BND + video_ca_out

        if not self.has_audio:
            # === VIDEO-ONLY FEEDFORWARD ===
            video_normed = self.norm3(video_1BND)
            video_normed = video_normed * (1.0 + v_scale_ff) + v_shift_ff
            if self.parallel_config.tensor_parallel.factor > 1:
                video_normed = self.ccl_manager.all_gather_persistent_buffer(
                    video_normed, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            video_ff = self.ffn(video_normed, compute_kernel_config=self.ff_compute_kernel_config)
            video_1BND = ttnn.addcmul(video_1BND, video_ff, v_gate_ff)
            return video_1BND

        # === AUDIO PATH (has_audio=True from here) ===
        shifted_a = self.audio_scale_shift_table.data + audio_temb
        a_chunks = ttnn.chunk(shifted_a, self.adaln_coeff, dim=2)
        a_shift_sa, a_scale_sa, a_gate_sa = a_chunks[0], a_chunks[1], a_chunks[2]
        a_shift_ff, a_scale_ff, a_gate_ff = a_chunks[3], a_chunks[4], a_chunks[5]
        a_gate_sa = ttnn.typecast(a_gate_sa, dtype=ttnn.bfloat16)
        a_gate_ff = ttnn.typecast(a_gate_ff, dtype=ttnn.bfloat16)
        if self.cross_attention_adaln:
            a_shift_ca, a_scale_ca, a_gate_ca = a_chunks[6], a_chunks[7], a_chunks[8]
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
            skip_qk=skip_self_attn,
            attn_mask=audio_attn_mask,
        )

        # === AUDIO TEXT CROSS-ATTENTION ===
        if self.cross_attention_adaln:
            audio_ca_input = self.audio_norm2(audio_1BND) * (1.0 + a_scale_ca) + a_shift_ca
            if audio_prompt_temb is not None:
                shifted_prompt_a = self.audio_prompt_scale_shift_table.data + audio_prompt_temb
                a_kv_shift, a_kv_scale = ttnn.chunk(shifted_prompt_a, 2, dim=2)
                audio_prompt_mod = audio_prompt * (1.0 + a_kv_scale) + a_kv_shift
            else:
                audio_prompt_mod = audio_prompt
            audio_ca_out = self.audio_attn2(spatial_1BND=audio_ca_input, N=audio_N, prompt_1BLP=audio_prompt_mod)
            audio_1BND = audio_1BND + audio_ca_out * a_gate_ca
        else:
            audio_ca_input = self.audio_norm2(audio_1BND)
            audio_ca_out = self.audio_attn2(spatial_1BND=audio_ca_input, N=audio_N, prompt_1BLP=audio_prompt)
            audio_1BND = audio_1BND + audio_ca_out

        # === BIDIRECTIONAL A↔V CROSS-ATTENTION ===
        if not skip_cross_attn:
            shifted_av = self.scale_shift_table_a2v_ca_video.data + av_ca_temb
            v_ca_scale, v_ca_shift, a_ca_scale_v, a_ca_shift_v, v_ca_gate = ttnn.chunk(shifted_av, 5, dim=2)
            v_ca_gate = ttnn.typecast(v_ca_gate, dtype=ttnn.bfloat16)

            _av_ca_audio_temb = (
                av_ca_audio_temb if av_ca_audio_temb is not None else av_ca_temb[:, :, :5, : self.audio_dim]
            )
            shifted_av_a = self.scale_shift_table_a2v_ca_audio.data + _av_ca_audio_temb
            a_scale_a2v, a_shift_a2v, a_scale_v2a, a_shift_v2a, a_ca_gate = ttnn.chunk(shifted_av_a, 5, dim=2)
            a_ca_gate = ttnn.typecast(a_ca_gate, dtype=ttnn.bfloat16)

            video_normed_xattn = self.norm1(video_1BND)
            audio_normed_xattn = self.audio_norm1(audio_1BND)

            # A→V: audio provides context for video
            video_q_a2v = video_normed_xattn * (1.0 + v_ca_scale) + v_ca_shift
            audio_kv_a2v = audio_normed_xattn * (1.0 + a_scale_a2v) + a_shift_a2v
            if self.parallel_config.sequence_parallel.factor > 1:
                audio_kv_a2v = self.ccl_manager.all_gather_persistent_buffer(
                    audio_kv_a2v, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
                )
            # Zero out padded audio tokens so they contribute nothing to A→V cross-attention.
            if audio_padding_mask is not None:
                audio_kv_a2v = ttnn.multiply(audio_kv_a2v, audio_padding_mask)
            if self.parallel_config.tensor_parallel.factor > 1:
                audio_kv_a2v = self.ccl_manager.all_gather_persistent_buffer(
                    audio_kv_a2v, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            a2v_output = self.audio_to_video_attn(
                spatial_1BND=video_q_a2v,
                N=video_N,
                prompt_1BLP=audio_kv_a2v,
                rope_cos=video_cross_pe_cos,
                rope_sin=video_cross_pe_sin,
                k_rope_cos=audio_cross_pe_cos_full,
                k_rope_sin=audio_cross_pe_sin_full,
            )
            video_1BND = video_1BND + a2v_output * v_ca_gate

            # V→A: video provides context for audio
            audio_q_v2a = audio_normed_xattn * (1.0 + a_scale_v2a) + a_shift_v2a
            video_kv_v2a = video_normed_xattn * (1.0 + a_ca_scale_v) + a_ca_shift_v
            if self.parallel_config.sequence_parallel.factor > 1:
                video_kv_v2a = self.ccl_manager.all_gather_persistent_buffer(
                    video_kv_v2a, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
                )
            if self.parallel_config.tensor_parallel.factor > 1:
                video_kv_v2a = self.ccl_manager.all_gather_persistent_buffer(
                    video_kv_v2a, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            v2a_output = self.video_to_audio_attn(
                spatial_1BND=audio_q_v2a,
                N=audio_N,
                prompt_1BLP=video_kv_v2a,
                rope_cos=audio_cross_pe_cos,
                rope_sin=audio_cross_pe_sin,
                k_rope_cos=video_cross_pe_cos_full,
                k_rope_sin=video_cross_pe_sin_full,
            )
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


class LTXTransformerModel(Module):
    """
    Unified LTX-2 DiT Transformer model for video-only and audio-video modes.

    When has_audio=False: equivalent to the original LTXTransformerModel.
    When has_audio=True: equivalent to the original LTXAudioVideoTransformerModel.

    Architecture:
    - patchify_proj: Linear(in_channels, dim)
    - adaln_single: LTXAdaLayerNormSingle for timestep conditioning
    - transformer_blocks: N x LTXTransformerBlock
    - norm_out + proj_out: output projection with AdaLN
    - (has_audio) audio_patchify_proj, audio_adaln_single, etc.
    - (has_audio) av_ca_* adaln modules for cross-attention timestep conditioning
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
        # Audio config (only used when has_audio=True)
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
        has_audio: bool = False,
        apply_gated_attention: bool = False,
        cross_attention_adaln: bool = True,
    ) -> None:
        super().__init__()

        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.has_audio = has_audio
        self.cross_attention_adaln = cross_attention_adaln
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        if has_audio:
            self.audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim

        # === VIDEO MODEL-LEVEL ===
        self.patchify_proj = ColParallelLinear(
            in_channels,
            self.inner_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        adaln_coeff = 9 if cross_attention_adaln else 6
        self.adaln_single = LTXAdaLayerNormSingle(
            embedding_dim=self.inner_dim,
            embedding_coefficient=adaln_coeff,
            mesh_device=mesh_device,
        )
        if cross_attention_adaln:
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

        # === AUDIO MODEL-LEVEL (only when has_audio=True) ===
        if has_audio:
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
                embedding_coefficient=adaln_coeff,
                mesh_device=mesh_device,
            )
            if cross_attention_adaln:
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

            # Cross-attention timestep conditioning
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
        audio_ffn_dim = self.audio_inner_dim * ffn_mult if has_audio else 0
        self.transformer_blocks = ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                LTXTransformerBlock(
                    video_dim=self.inner_dim,
                    audio_dim=self.audio_inner_dim if has_audio else 0,
                    video_ffn_dim=video_ffn_dim,
                    audio_ffn_dim=audio_ffn_dim,
                    video_num_heads=num_attention_heads,
                    audio_num_heads=audio_num_attention_heads if has_audio else 0,
                    video_cross_attention_dim=cross_attention_dim,
                    audio_cross_attention_dim=audio_cross_attention_dim if has_audio else 0,
                    eps=norm_eps,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    is_fsdp=is_fsdp,
                    has_audio=has_audio,
                    apply_gated_attention=apply_gated_attention,
                    cross_attention_adaln=cross_attention_adaln,
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
        tables = ["scale_shift_table"]
        if self.has_audio:
            tables.append("audio_scale_shift_table")
        for key in tables:
            if key in state:
                state[key] = state[key].unsqueeze(0).unsqueeze(0)

        # Remove keys not implemented in TTNN
        pop_substate(state, "video_embeddings_connector")
        pop_substate(state, "audio_embeddings_connector")
        # Caption projection: 22B doesn't need it (connector outputs at cross_attn_dim),
        # 19B needs it (connector outputs at 3840, projection maps to 4096/2048).
        # For now, pop it — projection runs on host in inner_step if needed.
        if "caption_projection.linear_1.weight" in state:
            self._caption_proj_state = pop_substate(state, "caption_projection")
            logger.info(f"Loaded caption_projection: {list(self._caption_proj_state.keys())}")
        else:
            self._caption_proj_state = {}
            pop_substate(state, "caption_projection")
        if self.has_audio:
            if "audio_caption_projection.linear_1.weight" in state:
                self._audio_caption_proj_state = pop_substate(state, "audio_caption_projection")
                logger.info(f"Loaded audio_caption_projection: {list(self._audio_caption_proj_state.keys())}")
            else:
                self._audio_caption_proj_state = {}
                pop_substate(state, "audio_caption_projection")
        # 6-output mode: no prompt_adaln modules
        if not self.cross_attention_adaln:
            pop_substate(state, "prompt_adaln_single")
            if self.has_audio:
                pop_substate(state, "audio_prompt_adaln_single")

    def inner_step(
        self,
        # Video
        video_1BNI_torch: torch.Tensor,
        video_prompt_1BLP: ttnn.Tensor,
        video_rope_cos: ttnn.Tensor,
        video_rope_sin: ttnn.Tensor,
        video_N: int,
        # Shared
        trans_mat: ttnn.Tensor | None,
        timestep_torch: torch.Tensor,
        # Audio (only used when has_audio=True)
        audio_1BNI_torch: torch.Tensor | None = None,
        audio_prompt_1BLP: ttnn.Tensor | None = None,
        audio_rope_cos: ttnn.Tensor | None = None,
        audio_rope_sin: ttnn.Tensor | None = None,
        audio_N: int = 0,
        # Cross-modal positional embeddings (optional)
        video_cross_pe_cos: ttnn.Tensor | None = None,
        video_cross_pe_sin: ttnn.Tensor | None = None,
        audio_cross_pe_cos: ttnn.Tensor | None = None,
        audio_cross_pe_sin: ttnn.Tensor | None = None,
        video_cross_pe_cos_full: ttnn.Tensor | None = None,
        video_cross_pe_sin_full: ttnn.Tensor | None = None,
        audio_cross_pe_cos_full: ttnn.Tensor | None = None,
        audio_cross_pe_sin_full: ttnn.Tensor | None = None,
        skip_cross_attn: bool = False,
        skip_self_attn_blocks: list[int] | None = None,
        audio_attn_mask: ttnn.Tensor | None = None,
        audio_padding_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run one denoising step on device. Returns video output, or (video, audio) tuple."""
        from ....utils.tensor import bf16_tensor, float32_tensor

        # Push video to device (SP-sharded)
        video_1BNI = bf16_tensor(
            video_1BNI_torch,
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

        # Video modulation (6 or 9 params depending on cross_attention_adaln)
        adaln_coeff = 9 if self.cross_attention_adaln else 6
        video_modulation, video_emb_ts = self.adaln_single(timestep)
        B = video_modulation.shape[2]
        video_mod_1BCD = ttnn.reshape(video_modulation, (1, B, adaln_coeff, self.inner_dim))
        if self.parallel_config.tensor_parallel.factor > 1:
            video_mod_1BCD = ttnn.mesh_partition(
                video_mod_1BCD, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        # Video prompt modulation (2 params, only for 9-output mode)
        video_prompt_1B2D = None
        if self.cross_attention_adaln:
            video_prompt_mod, _ = self.prompt_adaln_single(timestep)
            video_prompt_1B2D = ttnn.reshape(video_prompt_mod, (1, B, 2, self.inner_dim))

        # Audio modulation (only when has_audio)
        audio_mod_1BCD = None
        audio_prompt_1B2D = None
        av_ca_video_temb = None
        av_ca_audio_temb = None
        audio_emb_ts = None

        if self.has_audio:
            audio_1BNI = bf16_tensor(
                audio_1BNI_torch,
                device=self.mesh_device,
                mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
                shard_dim=-2,
            )

            audio_modulation, audio_emb_ts = self.audio_adaln_single(timestep)
            audio_mod_1BCD = ttnn.reshape(audio_modulation, (1, B, adaln_coeff, self.audio_inner_dim))
            if self.parallel_config.tensor_parallel.factor > 1:
                audio_mod_1BCD = ttnn.mesh_partition(
                    audio_mod_1BCD, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
                )

            if self.cross_attention_adaln:
                audio_prompt_mod, _ = self.audio_prompt_adaln_single(timestep)
                audio_prompt_1B2D = ttnn.reshape(audio_prompt_mod, (1, B, 2, self.audio_inner_dim))

            # Cross-attention timestep conditioning
            av_ca_video_ss, _ = self.av_ca_video_scale_shift_adaln_single(timestep)
            av_ca_video_ss = ttnn.reshape(av_ca_video_ss, (1, B, 4, self.inner_dim))
            av_ca_a2v_gate, _ = self.av_ca_a2v_gate_adaln_single(timestep)
            av_ca_a2v_gate = ttnn.reshape(av_ca_a2v_gate, (1, B, 1, self.inner_dim))
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

        # Caption projection: map connector output (3840) → cross_attention_dim (4096/2048).
        # Only needed when connector output dim != cross_attention_dim (e.g., 19B distilled).
        # Runs on host since it's a small 2-layer MLP applied once per step.
        if hasattr(self, "_caption_proj_state") and self._caption_proj_state:
            prompt_host = ttnn.to_torch(ttnn.get_device_tensors(video_prompt_1BLP)[0]).float()
            w1 = self._caption_proj_state["linear_1.weight"].float()
            b1 = self._caption_proj_state["linear_1.bias"].float()
            w2 = self._caption_proj_state["linear_2.weight"].float()
            b2 = self._caption_proj_state["linear_2.bias"].float()
            prompt_host = torch.nn.functional.gelu(torch.nn.functional.linear(prompt_host, w1, b1))
            prompt_host = torch.nn.functional.linear(prompt_host, w2, b2)
            video_prompt_1BLP = ttnn.from_torch(
                prompt_host.bfloat16(),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )

        if (
            hasattr(self, "_audio_caption_proj_state")
            and self._audio_caption_proj_state
            and audio_prompt_1BLP is not None
        ):
            a_prompt_host = ttnn.to_torch(ttnn.get_device_tensors(audio_prompt_1BLP)[0]).float()
            aw1 = self._audio_caption_proj_state["linear_1.weight"].float()
            ab1 = self._audio_caption_proj_state["linear_1.bias"].float()
            aw2 = self._audio_caption_proj_state["linear_2.weight"].float()
            ab2 = self._audio_caption_proj_state["linear_2.bias"].float()
            a_prompt_host = torch.nn.functional.gelu(torch.nn.functional.linear(a_prompt_host, aw1, ab1))
            a_prompt_host = torch.nn.functional.linear(a_prompt_host, aw2, ab2)
            audio_prompt_1BLP = ttnn.from_torch(
                a_prompt_host.bfloat16(),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )

        # Patchify
        video_1BND = self.patchify_proj(video_1BNI)
        audio_1BND = self.audio_patchify_proj(audio_1BNI) if self.has_audio else None

        # Transformer blocks
        for block_idx, block in enumerate(self.transformer_blocks):
            result = block(
                video_1BND=video_1BND,
                video_prompt=video_prompt_1BLP,
                video_temb=video_mod_1BCD,
                video_N=video_N,
                video_rope_cos=video_rope_cos,
                video_rope_sin=video_rope_sin,
                trans_mat=trans_mat,
                video_prompt_temb=video_prompt_1B2D,
                audio_1BND=audio_1BND,
                audio_prompt=audio_prompt_1BLP,
                audio_temb=audio_mod_1BCD,
                av_ca_temb=av_ca_video_temb,
                audio_N=audio_N,
                audio_rope_cos=audio_rope_cos,
                audio_rope_sin=audio_rope_sin,
                av_ca_audio_temb=av_ca_audio_temb,
                audio_prompt_temb=audio_prompt_1B2D,
                video_cross_pe_cos=video_cross_pe_cos,
                video_cross_pe_sin=video_cross_pe_sin,
                audio_cross_pe_cos=audio_cross_pe_cos,
                audio_cross_pe_sin=audio_cross_pe_sin,
                video_cross_pe_cos_full=video_cross_pe_cos_full,
                video_cross_pe_sin_full=video_cross_pe_sin_full,
                audio_cross_pe_cos_full=audio_cross_pe_cos_full,
                audio_cross_pe_sin_full=audio_cross_pe_sin_full,
                skip_cross_attn=skip_cross_attn,
                skip_self_attn=skip_self_attn_blocks is not None and block_idx in skip_self_attn_blocks,
                audio_attn_mask=audio_attn_mask,
                audio_padding_mask=audio_padding_mask,
            )
            if self.has_audio:
                video_1BND, audio_1BND = result
            else:
                video_1BND = result

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
        if self.has_audio:
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
            if self.has_audio:
                audio_out = self.ccl_manager.all_gather_persistent_buffer(
                    audio_out, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
                )

        if self.has_audio:
            return video_out, audio_out
        return video_out

    def forward(self, **kwargs):
        """Forward pass — delegates to inner_step."""
        return self.inner_step(**kwargs)

    @staticmethod
    def device_to_host(tt_tensor: ttnn.Tensor) -> torch.Tensor:
        """Move a ttnn device tensor to a torch host tensor."""
        return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0])
