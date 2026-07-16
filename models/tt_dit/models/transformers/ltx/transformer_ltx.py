# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file

import ttnn

from ....layers.embeddings import LTXAdaLayerNormSingle
from ....layers.feedforward import ParallelFeedForward
from ....layers.linear import ColParallelLinear, Linear
from ....layers.module import Module, ModuleList, Parameter
from ....layers.normalization import DistributedLayerNorm, DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils import cache as cache_module
from ....utils.fuse_loras import LoraSpec, fuse_loras_into
from ....utils.substate import pop_substate, rename_substate
from ....utils.tensor import bf16_tensor
from ....utils.tracing import traced_function
from .attention_ltx import LTXAttention

# gen#0 self-warms the DiT via a prep_run=True capture instead of the warmup denoise. Off by default:
# it is only safe when the inner_step @traced_function captures with prep_run under this flag, which it
# does not, so enabling it skips warmup with no self-warm and compiles the DiT cold in the reserved window.
LTX_DIT_PREP_RUN = os.environ.get("LTX_DIT_PREP_RUN", "0") in ("1", "true", "True")

# Fold the three still-unfused gated residuals (audio cross-attn, A->V, V->A) into their to_out matmul
# epilogue, via the primitive the self-attentions already use. Math-identical, and it removes three
# programs per block. The traced step carries a large work-independent floor, so a program removed is
# worth more than the FLOPs it carried. Set to 0 to restore the standalone addcmul for an A/B.
LTX_FOLD_GATED_RESIDUAL = os.environ.get("LTX_FOLD_GATED_RESIDUAL", "1") in ("1", "true", "True")

# PROBE ONLY (not a shipping path): replace the three gated-residual addcmul(t,t1,t2) calls with the
# algebraically identical add(t, multiply(t1,t2)). Same math, different bf16 rounding — a control that
# measures how far the sampler amplifies a rounding-scale perturbation, independent of the fold.
LTX_PROBE_ADDCMUL_SPLIT = os.environ.get("LTX_PROBE_ADDCMUL_SPLIT", "0") in ("1", "true", "True")


def _gated_residual(t: ttnn.Tensor, t1: ttnn.Tensor, t2: ttnn.Tensor) -> ttnn.Tensor:
    if LTX_PROBE_ADDCMUL_SPLIT:
        return ttnn.add(t, ttnn.multiply(t1, t2))
    return ttnn.addcmul(t, t1, t2)


def _tile_preserving_chunk0(x: ttnn.Tensor, n: int) -> list[ttnn.Tensor]:
    """Split ``x`` into ``n`` size-1 slices along dim 0 WITHOUT leaving TILE layout.

    ``ttnn.chunk`` is a host fallback that untilizes to ROW_MAJOR (and, for the
    ``(coeff,B,1,D)`` AdaLN modulation, the ``(1,B,1,D)`` slices never re-tile), forcing
    every downstream ``addcmul``/matmul epilogue to re-tilize the shift/scale/gate vectors
    (the dominant BF16->BF16 tilize cost per block). Slicing dim 0 -- a non-tile outer dim --
    on the already-TILE ``shifted`` tensor keeps every slice in TILE, so consumers take the
    fused LLK path instead of the composite (multiply+add w/ per-input tilize) fallback.
    Output is bit-identical to ``ttnn.chunk`` (pure layout, no value change).

    ``x`` is already TILE here (``TILE scale_shift_table.data`` + temb; ttnn add promotes to
    TILE), so no explicit re-tilize is needed -- ``ttnn.slice`` is a trace-safe device op that
    preserves the input layout, whereas ``ttnn.chunk``'s host untilize is what forced RM."""
    shape = list(x.shape)
    out = []
    for i in range(n):
        starts = [0] * len(shape)
        ends = list(shape)
        starts[0] = i
        ends[0] = i + 1
        out.append(ttnn.slice(x, starts, ends))
    return out


def build_audio_masks(
    audio_N: int, audio_N_real: int, *, mesh_device: ttnn.MeshDevice, sp_axis: int
) -> tuple[ttnn.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None]:
    """SDPA attn mask + padding masks for SP-sharded vs gathered audio tokens.

    Returns ``(attn_mask, pad_mask_sp, pad_mask_full)`` — ``pad_mask_sp`` is sharded on the
    sequence dim for multiply with local audio activations; ``pad_mask_full`` is replicated
    for multiply after the all_gather on A→V keys. All ``None`` when no padding is needed.
    """
    if audio_N <= audio_N_real:
        return None, None, None

    # Column mask only: real/padded queries are barred from attending TO padded keys.
    # Do NOT mask padded-query rows to -inf — that makes all attention scores in those
    # rows -inf → softmax NaN → NaN propagates via padded-token outputs (which we then
    # multiply by 0; IEEE 0*NaN = NaN, not 0). The pad mask already zeros the padded-query
    # outputs after attention, so column-only masking is sufficient and numerically safer
    # at high σ where activations have largest magnitude.
    mask = torch.zeros(1, 1, audio_N, audio_N)
    mask[:, :, :, audio_N_real:] = float("-inf")
    tt_attn_mask = bf16_tensor(mask.to(torch.bfloat16), device=mesh_device, mesh_axis=sp_axis, shard_dim=2)

    pad_mask = torch.ones(1, 1, audio_N, 1, dtype=torch.bfloat16)
    pad_mask[:, :, audio_N_real:, :] = 0.0
    tt_pad_mask_sp = bf16_tensor(pad_mask, device=mesh_device, mesh_axis=sp_axis, shard_dim=2)
    tt_pad_mask_full = bf16_tensor(pad_mask, device=mesh_device)
    return tt_attn_mask, tt_pad_mask_sp, tt_pad_mask_full


def build_video_pad_mask(
    video_N: int, video_N_real: int, *, mesh_device: ttnn.MeshDevice, sp_axis: int
) -> ttnn.Tensor | None:
    """SP-sharded video padding mask ``(1, 1, video_N, 1)``; ``None`` when no padding is needed.

    Multiply the local (sharded) video activations by this to zero padded slots before they
    propagate downstream (self-attn residual / cross-attn K / FF). No SDPA attn_mask is needed
    (unlike audio) — video self-attention uses ring SDPA, which masks padded keys via its
    ``logical_n=video_N_real`` arg.
    """
    if video_N <= video_N_real:
        return None
    pad_mask = torch.ones(1, 1, video_N, 1, dtype=torch.bfloat16)
    pad_mask[:, :, video_N_real:, :] = 0.0
    return bf16_tensor(pad_mask, device=mesh_device, mesh_axis=sp_axis, shard_dim=2)


class LTXTransformerBlock(Module):
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

        # FSDP fractures FFN weights across the SP axis (on top of the TP fracture);
        # without it the FFN is only TP-sharded and replicated across every SP device.
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

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
        # Outer-param layout (coeff, 1, 1, D): keeps each modulation parameter on the
        # non-tiled dim 0 so per-block ttnn.chunk(dim=0) is a free tile-aligned slice
        # (no untilize/slice/tilize). D stays on dim 3 for TP sharding.
        self.scale_shift_table = Parameter(
            total_shape=[self.adaln_coeff, 1, 1, video_dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )
        if cross_attention_adaln:
            self.prompt_scale_shift_table = Parameter(
                total_shape=[2, 1, 1, video_dim],
                mesh_axes=[None, None, None, None],
                device=mesh_device,
                dtype=ttnn.bfloat16,
            )

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
                fsdp_mesh_axis=fsdp_mesh_axis,
            )
            self.audio_scale_shift_table = Parameter(
                total_shape=[self.adaln_coeff, 1, 1, audio_dim],
                mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
                device=mesh_device,
                dtype=ttnn.bfloat16,
            )
            if cross_attention_adaln:
                self.audio_prompt_scale_shift_table = Parameter(
                    total_shape=[2, 1, 1, audio_dim],
                    mesh_axes=[None, None, None, None],
                    device=mesh_device,
                    dtype=ttnn.bfloat16,
                )

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
                total_shape=[5, 1, 1, audio_dim],
                mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
                device=mesh_device,
                dtype=ttnn.bfloat16,
            )
            self.scale_shift_table_a2v_ca_video = Parameter(
                total_shape=[5, 1, 1, video_dim],
                mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
                device=mesh_device,
                dtype=ttnn.bfloat16,
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

        # Scale slots pre-incremented (+1) and cast to bf16 so each runtime
        # modulation fuses to one ttnn.addcmul(shift, x, scale_p1).
        tables_scale_idxs: dict[str, list[int]] = {}
        tables_scale_idxs["scale_shift_table"] = [1, 4, 7] if self.cross_attention_adaln else [1, 4]
        if self.cross_attention_adaln:
            tables_scale_idxs["prompt_scale_shift_table"] = [1]
        else:
            pop_substate(state, "prompt_scale_shift_table")
        if self.has_audio:
            tables_scale_idxs["audio_scale_shift_table"] = [1, 4, 7] if self.cross_attention_adaln else [1, 4]
            if self.cross_attention_adaln:
                tables_scale_idxs["audio_prompt_scale_shift_table"] = [1]
            else:
                pop_substate(state, "audio_prompt_scale_shift_table")
            tables_scale_idxs["scale_shift_table_a2v_ca_audio"] = [0, 2]
            tables_scale_idxs["scale_shift_table_a2v_ca_video"] = [0, 2]
        for key, scale_idxs in tables_scale_idxs.items():
            if key in state:
                # Reference table is [coeff, D]; reshape to outer-param [coeff, 1, 1, D]
                # so dim 0 indexes the parameters (matches the (coeff,1,1,D) Parameter).
                t = state[key].unsqueeze(1).unsqueeze(1).clone()
                t[scale_idxs, :, :, :] += 1.0
                state[key] = t.to(dtype=torch.bfloat16)

    def _modulated_ffn(self, ffn, norm, x_1BND, shift_ff, scale_ff_p1, gate_ff):
        """norm -> AdaLN (shift + x * scale_p1) -> gated FFN residual.

        Ring fuses ff1(AG) + ff2 + RS + addcmul; Linear needs explicit AG + plain ffn().
        """
        normed = norm(x_1BND)
        normed = ttnn.addcmul(shift_ff, normed, scale_ff_p1)
        if self.ccl_manager.topology == ttnn.Topology.Ring:
            return ffn.forward_fused_addcmul(
                normed,
                x_1BND,
                gate_ff,
                scalar=1.0,
                compute_kernel_config=self.ff_compute_kernel_config,
                parallel_config=self.parallel_config,
            )
        else:
            if self.parallel_config.tensor_parallel.factor > 1:
                normed = self.ccl_manager.all_gather_persistent_buffer(
                    normed, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            ff_out = ffn(normed, compute_kernel_config=self.ff_compute_kernel_config)
            return ttnn.addcmul(x_1BND, ff_out, gate_ff)

    def forward(
        self,
        video_1BND: ttnn.Tensor,
        video_prompt: ttnn.Tensor,
        video_temb: ttnn.Tensor,  # (adaln_coeff, B, 1, video_dim)
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
        audio_cross_pe_cos_full: ttnn.Tensor | None = None,
        audio_cross_pe_sin_full: ttnn.Tensor | None = None,
        skip_cross_attn: bool = False,
        skip_self_attn: bool = False,
        audio_attn_mask: ttnn.Tensor | None = None,
        audio_padding_mask: ttnn.Tensor | None = None,
        audio_padding_mask_full: ttnn.Tensor | None = None,
        video_padding_mask: ttnn.Tensor | None = None,
        video_kv_logical_n: int | None = None,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        # Video modulation; `_p1` chunks carry +1 baked into the scale slot (see _prepare_torch_state).
        shifted_v = self.scale_shift_table.data + video_temb
        chunks = _tile_preserving_chunk0(shifted_v, self.adaln_coeff)
        v_shift_sa, v_scale_sa_p1, v_gate_sa = chunks[0], chunks[1], chunks[2]
        v_shift_ff, v_scale_ff_p1, v_gate_ff = chunks[3], chunks[4], chunks[5]
        if self.cross_attention_adaln:
            v_shift_ca, v_scale_ca_p1, v_gate_ca = chunks[6], chunks[7], chunks[8]

        # Video self-attention
        video_normed = self.norm1(video_1BND)
        video_normed = ttnn.addcmul(v_shift_sa, video_normed, v_scale_sa_p1)
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

        # Video text cross-attention
        if self.cross_attention_adaln:
            video_ca_input = ttnn.addcmul(v_shift_ca, self.norm2(video_1BND), v_scale_ca_p1)
            if video_prompt_temb is not None:
                shifted_prompt_v = self.prompt_scale_shift_table.data + video_prompt_temb
                v_kv_shift, v_kv_scale_p1 = _tile_preserving_chunk0(shifted_prompt_v, 2)
                video_prompt_mod = ttnn.addcmul(v_kv_shift, video_prompt, v_kv_scale_p1)
            else:
                video_prompt_mod = video_prompt
            # Fold the gated residual (video_1BND + video_ca_out * v_gate_ca) into to_out's epilogue,
            # mirroring attn1's fused path — avoids a separate full-sequence addcmul pass.
            video_1BND = self.attn2(
                spatial_1BND=video_ca_input,
                N=video_N,
                prompt_1BLP=video_prompt_mod,
                kv_replicated=True,
                addcmul_residual=video_1BND,
                addcmul_gate=v_gate_ca,
            )
        else:
            # 6-output mode: cross-attention with fixed norm (no AdaLN shift/scale/gate)
            video_ca_input = self.norm2(video_1BND)
            video_ca_out = self.attn2(
                spatial_1BND=video_ca_input, N=video_N, prompt_1BLP=video_prompt, kv_replicated=True
            )
            video_1BND = video_1BND + video_ca_out

        if not self.has_audio:
            # Video-only feed forward
            return self._modulated_ffn(self.ffn, self.norm3, video_1BND, v_shift_ff, v_scale_ff_p1, v_gate_ff)

        # Audio path (has_audio=True from here)
        shifted_a = self.audio_scale_shift_table.data + audio_temb
        a_chunks = ttnn.chunk(shifted_a, self.adaln_coeff, dim=0)
        a_shift_sa, a_scale_sa_p1, a_gate_sa = a_chunks[0], a_chunks[1], a_chunks[2]
        a_shift_ff, a_scale_ff_p1, a_gate_ff = a_chunks[3], a_chunks[4], a_chunks[5]
        if self.cross_attention_adaln:
            a_shift_ca, a_scale_ca_p1, a_gate_ca = a_chunks[6], a_chunks[7], a_chunks[8]

        # Audio self-attention
        audio_normed = self.audio_norm1(audio_1BND)
        audio_normed = ttnn.addcmul(a_shift_sa, audio_normed, a_scale_sa_p1)
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

        # Audio text cross-attention
        if self.cross_attention_adaln:
            audio_ca_input = ttnn.addcmul(a_shift_ca, self.audio_norm2(audio_1BND), a_scale_ca_p1)
            if audio_prompt_temb is not None:
                shifted_prompt_a = self.audio_prompt_scale_shift_table.data + audio_prompt_temb
                a_kv_shift, a_kv_scale_p1 = ttnn.chunk(shifted_prompt_a, 2, dim=0)
                audio_prompt_mod = ttnn.addcmul(a_kv_shift, audio_prompt, a_kv_scale_p1)
            else:
                audio_prompt_mod = audio_prompt
            if LTX_FOLD_GATED_RESIDUAL:
                audio_1BND = self.audio_attn2(
                    spatial_1BND=audio_ca_input,
                    N=audio_N,
                    prompt_1BLP=audio_prompt_mod,
                    kv_replicated=True,
                    addcmul_residual=audio_1BND,
                    addcmul_gate=a_gate_ca,
                )
            else:
                audio_ca_out = self.audio_attn2(
                    spatial_1BND=audio_ca_input, N=audio_N, prompt_1BLP=audio_prompt_mod, kv_replicated=True
                )
                audio_1BND = _gated_residual(audio_1BND, audio_ca_out, a_gate_ca)
        else:
            audio_ca_input = self.audio_norm2(audio_1BND)
            audio_ca_out = self.audio_attn2(
                spatial_1BND=audio_ca_input, N=audio_N, prompt_1BLP=audio_prompt, kv_replicated=True
            )
            audio_1BND = audio_1BND + audio_ca_out

        # Bidirectional A<->V cross-attention
        if not skip_cross_attn:
            # Chunk layout [scale, shift, scale, shift, gate]: scale slots (idx 0, 2) have +1 baked in.
            shifted_av = self.scale_shift_table_a2v_ca_video.data + av_ca_temb
            v_ca_scale_p1, v_ca_shift, a_ca_scale_v_p1, a_ca_shift_v, v_ca_gate = ttnn.chunk(shifted_av, 5, dim=0)

            shifted_av_a = self.scale_shift_table_a2v_ca_audio.data + av_ca_audio_temb
            a_scale_a2v_p1, a_shift_a2v, a_scale_v2a_p1, a_shift_v2a, a_ca_gate = ttnn.chunk(shifted_av_a, 5, dim=0)

            video_normed_xattn = self.norm3(video_1BND)
            audio_normed_xattn = self.audio_norm3(audio_1BND)

            # A→V: audio provides context for video
            video_q_a2v = ttnn.addcmul(v_ca_shift, video_normed_xattn, v_ca_scale_p1)
            audio_kv_a2v = ttnn.addcmul(a_shift_a2v, audio_normed_xattn, a_scale_a2v_p1)
            if self.parallel_config.sequence_parallel.factor > 1:
                audio_kv_a2v = self.ccl_manager.all_gather_persistent_buffer(
                    audio_kv_a2v, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
                )
            # Zero padded audio tokens before to_kv so they don't contribute to A->V cross-attention.
            pad_mask_a2v = audio_padding_mask_full if audio_padding_mask_full is not None else audio_padding_mask
            if pad_mask_a2v is not None:
                audio_kv_a2v = ttnn.multiply(audio_kv_a2v, pad_mask_a2v)
            a2v_output = self.audio_to_video_attn(
                spatial_1BND=video_q_a2v,
                N=video_N,
                prompt_1BLP=audio_kv_a2v,
                rope_cos=video_cross_pe_cos,
                rope_sin=video_cross_pe_sin,
                k_rope_cos=audio_cross_pe_cos_full,
                k_rope_sin=audio_cross_pe_sin_full,
                trans_mat=trans_mat,
                addcmul_residual=video_1BND if LTX_FOLD_GATED_RESIDUAL else None,
                addcmul_gate=v_ca_gate if LTX_FOLD_GATED_RESIDUAL else None,
            )
            video_1BND = a2v_output if LTX_FOLD_GATED_RESIDUAL else _gated_residual(video_1BND, a2v_output, v_ca_gate)

            # V→A: video provides context for audio
            audio_q_v2a = ttnn.addcmul(a_shift_v2a, audio_normed_xattn, a_scale_v2a_p1)
            video_kv_v2a = ttnn.addcmul(a_ca_shift_v, video_normed_xattn, a_ca_scale_v_p1)
            # Zero padded video tokens (on the SP-local shard) after the affine, before to_kv.
            if video_padding_mask is not None:
                video_kv_v2a = ttnn.multiply(video_kv_v2a, video_padding_mask)
            v2a_output = self.video_to_audio_attn(
                spatial_1BND=audio_q_v2a,
                N=audio_N,
                prompt_1BLP=video_kv_v2a,
                rope_cos=audio_cross_pe_cos,
                rope_sin=audio_cross_pe_sin,
                # Ring cross keeps video K/V SP-sharded: pass the sharded K-rope and kv_logical_n so
                # the ring SDPA gathers internally instead of a separate K/V all-gather.
                k_rope_cos=video_cross_pe_cos,
                k_rope_sin=video_cross_pe_sin,
                # Audio syncs to the decoded (stripped) grid, so exclude any appended anchor tokens
                # from the video context it attends to. Defaults to video_N when unset.
                kv_logical_n=video_kv_logical_n if video_kv_logical_n is not None else video_N,
                trans_mat=trans_mat,
                addcmul_residual=audio_1BND if LTX_FOLD_GATED_RESIDUAL else None,
                addcmul_gate=a_ca_gate if LTX_FOLD_GATED_RESIDUAL else None,
            )
            audio_1BND = v2a_output if LTX_FOLD_GATED_RESIDUAL else _gated_residual(audio_1BND, v2a_output, a_ca_gate)

        # Video feed forward
        video_1BND = self._modulated_ffn(self.ffn, self.norm3, video_1BND, v_shift_ff, v_scale_ff_p1, v_gate_ff)

        # Audio feed forward
        audio_1BND = self._modulated_ffn(
            self.audio_ff, self.audio_norm3, audio_1BND, a_shift_ff, a_scale_ff_p1, a_gate_ff
        )

        return video_1BND, audio_1BND


class LTXTransformerModel(Module):
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
        image_conditioning: bool = False,
    ) -> None:
        super().__init__()

        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.has_audio = has_audio
        self.cross_attention_adaln = cross_attention_adaln
        # I2V: video AdaLN modulation is per-token (denoise_mask * sigma) instead of batch-scalar.
        # Audio / prompt / A<->V cross AdaLN stay batch-scalar regardless.
        self.image_conditioning = image_conditioning
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        if has_audio:
            self.audio_inner_dim = audio_num_attention_heads * audio_attention_head_dim

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
            dtype=ttnn.bfloat16,
        )
        if cross_attention_adaln:
            self.prompt_adaln_single = LTXAdaLayerNormSingle(
                embedding_dim=self.inner_dim,
                embedding_coefficient=2,
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
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
            dtype=ttnn.bfloat16,
        )
        self.proj_out = Linear(self.inner_dim, out_channels, bias=True, mesh_device=mesh_device)

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
                dtype=ttnn.bfloat16,
            )
            if cross_attention_adaln:
                self.audio_prompt_adaln_single = LTXAdaLayerNormSingle(
                    embedding_dim=self.audio_inner_dim,
                    embedding_coefficient=2,
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat16,
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
                dtype=ttnn.bfloat16,
            )
            self.audio_proj_out = Linear(self.audio_inner_dim, audio_out_channels, bias=True, mesh_device=mesh_device)

            # Cross-attention timestep conditioning
            self.av_ca_video_scale_shift_adaln_single = LTXAdaLayerNormSingle(
                embedding_dim=self.inner_dim,
                embedding_coefficient=4,
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
            )
            self.av_ca_audio_scale_shift_adaln_single = LTXAdaLayerNormSingle(
                embedding_dim=self.audio_inner_dim,
                embedding_coefficient=4,
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
            )
            self.av_ca_a2v_gate_adaln_single = LTXAdaLayerNormSingle(
                embedding_dim=self.inner_dim,
                embedding_coefficient=1,
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
            )
            self.av_ca_v2a_gate_adaln_single = LTXAdaLayerNormSingle(
                embedding_dim=self.audio_inner_dim,
                embedding_coefficient=1,
                mesh_device=mesh_device,
                dtype=ttnn.bfloat16,
            )

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

        # Output-proj config matches WanTransformer's hifi4 config (packer_l1_acc=True).
        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Output-proj table is [shift, scale]; bake +1 into the scale slot and cast to bf16.
        tables = ["scale_shift_table"]
        if self.has_audio:
            tables.append("audio_scale_shift_table")
        for key in tables:
            if key in state:
                t = state[key].unsqueeze(0).unsqueeze(0).clone()
                t[:, :, 1, :] += 1.0
                state[key] = t.to(dtype=torch.bfloat16)

        # Remove keys not implemented in TTNN
        pop_substate(state, "video_embeddings_connector")
        pop_substate(state, "audio_embeddings_connector")
        # Caption projection is a 19B-distilled-only artifact, not implemented in TTNN.
        pop_substate(state, "caption_projection")
        if self.has_audio:
            pop_substate(state, "audio_caption_projection")
        # 6-output mode: no prompt_adaln modules
        if not self.cross_attention_adaln:
            pop_substate(state, "prompt_adaln_single")
            if self.has_audio:
                pop_substate(state, "audio_prompt_adaln_single")

    def forward(
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
        # I2V per-token video timestep (B, N); required when image_conditioning=True.
        video_timestep_torch: torch.Tensor | None = None,
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
        audio_cross_pe_cos_full: ttnn.Tensor | None = None,
        audio_cross_pe_sin_full: ttnn.Tensor | None = None,
        skip_cross_attn: bool = False,
        skip_self_attn_blocks: list[int] | None = None,
        audio_attn_mask: ttnn.Tensor | None = None,
        audio_padding_mask: ttnn.Tensor | None = None,
        audio_padding_mask_full: ttnn.Tensor | None = None,
        video_padding_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        """Host entry: upload torch latents/timestep, then run the device-only inner_step."""
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        video_1BNI = bf16_tensor(video_1BNI_torch, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=-2)
        B_size = video_1BNI_torch.shape[1]
        # Scalar timestep (1,1,B,1) drives audio / prompt / A<->V cross modulation.
        timestep = bf16_tensor(timestep_torch.reshape(1, 1, B_size, 1) * 1000.0, device=self.mesh_device)
        # Per-token video timestep (1,1,B*N,1), SP-sharded on dim 2 to match video_1BNI (shard_dim=-2).
        video_timestep = None
        if self.image_conditioning:
            assert video_timestep_torch is not None, "video_timestep_torch required when image_conditioning=True"
            video_timestep = bf16_tensor(
                video_timestep_torch.reshape(1, 1, -1, 1) * 1000.0,
                device=self.mesh_device,
                mesh_axis=sp_axis,
                shard_dim=-2,
            )
        audio_1BNI = (
            bf16_tensor(audio_1BNI_torch, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=-2)
            if self.has_audio
            else None
        )
        return self.inner_step(
            video_1BNI=video_1BNI,
            timestep=timestep,
            video_timestep=video_timestep,
            audio_1BNI=audio_1BNI,
            video_prompt_1BLP=video_prompt_1BLP,
            video_rope_cos=video_rope_cos,
            video_rope_sin=video_rope_sin,
            video_N=video_N,
            trans_mat=trans_mat,
            audio_prompt_1BLP=audio_prompt_1BLP,
            audio_rope_cos=audio_rope_cos,
            audio_rope_sin=audio_rope_sin,
            audio_N=audio_N,
            video_cross_pe_cos=video_cross_pe_cos,
            video_cross_pe_sin=video_cross_pe_sin,
            audio_cross_pe_cos=audio_cross_pe_cos,
            audio_cross_pe_sin=audio_cross_pe_sin,
            audio_cross_pe_cos_full=audio_cross_pe_cos_full,
            audio_cross_pe_sin_full=audio_cross_pe_sin_full,
            skip_cross_attn=skip_cross_attn,
            skip_self_attn_blocks=skip_self_attn_blocks,
            audio_attn_mask=audio_attn_mask,
            audio_padding_mask=audio_padding_mask,
            audio_padding_mask_full=audio_padding_mask_full,
            video_padding_mask=video_padding_mask,
        )

    @traced_function(device=lambda self: self.mesh_device, clone_prep_inputs=False, prep_run=False)
    def inner_step(
        self,
        *,
        video_1BNI: ttnn.Tensor,
        timestep: ttnn.Tensor,
        video_timestep: ttnn.Tensor | None = None,
        video_ts_pair: ttnn.Tensor | None = None,
        video_pin_mask: ttnn.Tensor | None = None,
        audio_1BNI: ttnn.Tensor | None = None,
        video_prompt_1BLP: ttnn.Tensor = None,
        video_rope_cos: ttnn.Tensor = None,
        video_rope_sin: ttnn.Tensor = None,
        video_N: int = 0,
        trans_mat: ttnn.Tensor | None = None,
        audio_prompt_1BLP: ttnn.Tensor | None = None,
        audio_rope_cos: ttnn.Tensor | None = None,
        audio_rope_sin: ttnn.Tensor | None = None,
        audio_N: int = 0,
        video_cross_pe_cos: ttnn.Tensor | None = None,
        video_cross_pe_sin: ttnn.Tensor | None = None,
        audio_cross_pe_cos: ttnn.Tensor | None = None,
        audio_cross_pe_sin: ttnn.Tensor | None = None,
        audio_cross_pe_cos_full: ttnn.Tensor | None = None,
        audio_cross_pe_sin_full: ttnn.Tensor | None = None,
        skip_cross_attn: bool = False,
        skip_self_attn_blocks: list[int] | None = None,
        audio_attn_mask: ttnn.Tensor | None = None,
        audio_padding_mask: ttnn.Tensor | None = None,
        audio_padding_mask_full: ttnn.Tensor | None = None,
        video_padding_mask: ttnn.Tensor | None = None,
        video_kv_logical_n: int | None = None,
        gather_output: bool = True,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        """Device-only, trace-capturable denoising step. All tensor args are ttnn (no torch).
        ``video_1BNI``/``audio_1BNI``/``timestep`` are the per-step inputs; the rest are
        constant across a denoise stage.

        ``gather_output`` SP-gathers the velocity outputs to full sequence length (default,
        for the host-Euler path). Pass ``False`` to keep them SP-sharded so an on-device
        solver can step the latent without leaving the device (the traced WAN pattern)."""
        # Video modulation (6 or 9 params depending on cross_attention_adaln).
        adaln_coeff = 9 if self.cross_attention_adaln else 6
        # I2V compact path: per-token timestep has 2 values (pinned frame-0 vs. sigma), passed as a
        # (1,1,2,1) pair + {0,1} pin mask. Blend per token to avoid the dense (1,1,N,coeff*D) modulation.
        compact_i2v = self.image_conditioning and video_ts_pair is not None and video_pin_mask is not None
        if compact_i2v:
            N = video_pin_mask.shape[2]
            mod_pair, emb_pair = self.adaln_single(video_ts_pair)  # (1,1,2,coeff*D), (1,1,2,D)
            mod_pair = ttnn.reshape(mod_pair, (1, 2, adaln_coeff, self.inner_dim))
            if self.parallel_config.tensor_parallel.factor > 1:
                mod_pair = ttnn.mesh_partition(
                    mod_pair, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            mod_pair = ttnn.permute(mod_pair, (2, 0, 1, 3))  # (coeff,1,2,Dloc)
            mod_pin, mod_base = ttnn.chunk(mod_pair, 2, dim=2)  # each (coeff,1,1,Dloc)
            ttnn.deallocate(mod_pair)
            # video_mod = base + (pin - base) * mask, materialized per token via broadcast-safe ops.
            mod_delta = ttnn.sub(mod_pin, mod_base)
            mod_delta = ttnn.repeat(mod_delta, ttnn.Shape([1, 1, N, 1]))  # (coeff,1,N,Dloc)
            mod_delta = ttnn.mul(mod_delta, video_pin_mask)  # mask broadcasts over coeff/Dloc
            video_mod_CB1D = ttnn.add(mod_base, mod_delta)  # base broadcasts over N
            ttnn.deallocate(mod_delta)
            ttnn.deallocate(mod_pin)
            ttnn.deallocate(mod_base)
            # Embedded timestep (for norm_out), same per-token blend, kept full-D.
            emb_pin, emb_base = ttnn.chunk(emb_pair, 2, dim=2)  # each (1,1,1,D)
            ttnn.deallocate(emb_pair)
            emb_delta = ttnn.sub(emb_pin, emb_base)
            emb_delta = ttnn.repeat(emb_delta, ttnn.Shape([1, 1, N, 1]))  # (1,1,N,D)
            emb_delta = ttnn.mul(emb_delta, video_pin_mask)
            video_emb_ts = ttnn.add(emb_base, emb_delta)
            ttnn.deallocate(emb_delta)
            ttnn.deallocate(emb_pin)
            ttnn.deallocate(emb_base)
            B = 1
        else:
            # I2V (dense): feed the per-token timestep so each token gets its own AdaLN modulation.
            video_ts = video_timestep if self.image_conditioning else timestep
            video_modulation, video_emb_ts = self.adaln_single(video_ts)
            # dim 2 is B (scalar) or B*N (per-token, B=1 -> N_local on the SP shard).
            X = video_modulation.shape[2]
            video_mod_CB1D = ttnn.reshape(video_modulation, (1, X, adaln_coeff, self.inner_dim))
            if self.parallel_config.tensor_parallel.factor > 1:
                video_mod_CB1D = ttnn.mesh_partition(
                    video_mod_CB1D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            # Move the coeff axis to dim 0 once per step so each block's chunk(dim=0) is free.
            # Scalar: (1,B,coeff,D) -> (coeff,B,1,D). Per-token: (1,N,coeff,D) -> (coeff,1,N,D).
            if self.image_conditioning:
                video_mod_CB1D = ttnn.permute(video_mod_CB1D, (2, 0, 1, 3))
            else:
                video_mod_CB1D = ttnn.permute(video_mod_CB1D, (2, 1, 0, 3))
            B = 1 if self.image_conditioning else X

        # Video prompt modulation (2 params, only for 9-output mode)
        video_prompt_2B1D = None
        if self.cross_attention_adaln:
            video_prompt_mod, _ = self.prompt_adaln_single(timestep)
            video_prompt_2B1D = ttnn.reshape(video_prompt_mod, (1, B, 2, self.inner_dim))
            video_prompt_2B1D = ttnn.permute(video_prompt_2B1D, (2, 1, 0, 3))

        # Audio modulation (only when has_audio)
        audio_mod_CB1D = None
        audio_prompt_2B1D = None
        av_ca_video_temb = None
        av_ca_audio_temb = None
        audio_emb_ts = None

        if self.has_audio:
            audio_modulation, audio_emb_ts = self.audio_adaln_single(timestep)
            audio_mod_CB1D = ttnn.reshape(audio_modulation, (1, B, adaln_coeff, self.audio_inner_dim))
            if self.parallel_config.tensor_parallel.factor > 1:
                audio_mod_CB1D = ttnn.mesh_partition(
                    audio_mod_CB1D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            audio_mod_CB1D = ttnn.permute(audio_mod_CB1D, (2, 1, 0, 3))

            if self.cross_attention_adaln:
                audio_prompt_mod, _ = self.audio_prompt_adaln_single(timestep)
                audio_prompt_2B1D = ttnn.reshape(audio_prompt_mod, (1, B, 2, self.audio_inner_dim))
                audio_prompt_2B1D = ttnn.permute(audio_prompt_2B1D, (2, 1, 0, 3))

            # Cross-attention timestep conditioning; concat on the param axis (dim 0).
            av_ca_video_ss, _ = self.av_ca_video_scale_shift_adaln_single(timestep)
            av_ca_video_ss = ttnn.permute(ttnn.reshape(av_ca_video_ss, (1, B, 4, self.inner_dim)), (2, 1, 0, 3))
            av_ca_a2v_gate, _ = self.av_ca_a2v_gate_adaln_single(timestep)
            av_ca_a2v_gate = ttnn.permute(ttnn.reshape(av_ca_a2v_gate, (1, B, 1, self.inner_dim)), (2, 1, 0, 3))
            av_ca_video_temb = ttnn.concat([av_ca_video_ss, av_ca_a2v_gate], dim=0)

            av_ca_audio_ss, _ = self.av_ca_audio_scale_shift_adaln_single(timestep)
            av_ca_audio_ss = ttnn.permute(ttnn.reshape(av_ca_audio_ss, (1, B, 4, self.audio_inner_dim)), (2, 1, 0, 3))
            av_ca_v2a_gate, _ = self.av_ca_v2a_gate_adaln_single(timestep)
            av_ca_v2a_gate = ttnn.permute(ttnn.reshape(av_ca_v2a_gate, (1, B, 1, self.audio_inner_dim)), (2, 1, 0, 3))
            av_ca_audio_temb = ttnn.concat([av_ca_audio_ss, av_ca_v2a_gate], dim=0)

            if self.parallel_config.tensor_parallel.factor > 1:
                av_ca_video_temb = ttnn.mesh_partition(
                    av_ca_video_temb, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
                av_ca_audio_temb = ttnn.mesh_partition(
                    av_ca_audio_temb, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
                )

        # Patchify
        video_1BND = self.patchify_proj(video_1BNI)
        audio_1BND = self.audio_patchify_proj(audio_1BNI) if self.has_audio else None

        skip_self_attn_set = frozenset(skip_self_attn_blocks) if skip_self_attn_blocks else frozenset()

        # LTX_SKIP_BLOCKS="a,b,..": identity-skip whole blocks (layer-prune experiment; residual passes
        # through unchanged). Baked into the trace, so it must be constant across capture+replay.
        _prune = {int(x) for x in os.environ.get("LTX_SKIP_BLOCKS", "").split(",") if x.strip().isdigit()}

        # Transformer blocks
        for block_idx, block in enumerate(self.transformer_blocks):
            if block_idx in _prune:
                continue
            result = block(
                video_1BND=video_1BND,
                video_prompt=video_prompt_1BLP,
                video_temb=video_mod_CB1D,
                video_N=video_N,
                video_rope_cos=video_rope_cos,
                video_rope_sin=video_rope_sin,
                trans_mat=trans_mat,
                video_prompt_temb=video_prompt_2B1D,
                audio_1BND=audio_1BND,
                audio_prompt=audio_prompt_1BLP,
                audio_temb=audio_mod_CB1D,
                av_ca_temb=av_ca_video_temb,
                audio_N=audio_N,
                audio_rope_cos=audio_rope_cos,
                audio_rope_sin=audio_rope_sin,
                av_ca_audio_temb=av_ca_audio_temb,
                audio_prompt_temb=audio_prompt_2B1D,
                video_cross_pe_cos=video_cross_pe_cos,
                video_cross_pe_sin=video_cross_pe_sin,
                audio_cross_pe_cos=audio_cross_pe_cos,
                audio_cross_pe_sin=audio_cross_pe_sin,
                audio_cross_pe_cos_full=audio_cross_pe_cos_full,
                audio_cross_pe_sin_full=audio_cross_pe_sin_full,
                skip_cross_attn=skip_cross_attn,
                skip_self_attn=block_idx in skip_self_attn_set,
                audio_attn_mask=audio_attn_mask,
                audio_padding_mask=audio_padding_mask,
                audio_padding_mask_full=audio_padding_mask_full,
                video_padding_mask=video_padding_mask,
                video_kv_logical_n=video_kv_logical_n,
            )
            if self.has_audio:
                video_1BND, audio_1BND = result
            else:
                video_1BND = result
            # Profiler drain every 16 blocks (LTX_PROFILE_FLUSH): 16 blocks × ~35 ops stays under the
            # 12k-marker DRAM buffer, so markers are never dropped, while a per-BLOCK drain (a host
            # readback of all 32 devices each block) is far too slow. Profiling only; no effect traced.
            if os.environ.get("LTX_PROFILE_FLUSH") and (
                block_idx % 16 == 15 or block_idx == len(self.transformer_blocks) - 1
            ):
                ttnn.ReadDeviceProfiler(self.mesh_device)

        v_inner_local = video_emb_ts.shape[-1]
        if self.image_conditioning:
            # Per-token (video_emb_ts is (1,1,N,D)): split the (shift, scale) table, broadcast-add per token.
            v_emb_1B1D = video_emb_ts
            if self.parallel_config.tensor_parallel.factor > 1:
                v_emb_1B1D = ttnn.mesh_partition(
                    v_emb_1B1D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            sst_shift, sst_scale_p1 = ttnn.chunk(self.scale_shift_table.data, 2, dim=2)  # each (1,1,1,Dloc)
            v_shift_out = ttnn.add(sst_shift, v_emb_1B1D)  # (1,1,N,Dloc)
            v_scale_out_p1 = ttnn.add(sst_scale_p1, v_emb_1B1D)
        else:
            v_emb_1B1D = ttnn.reshape(video_emb_ts, (1, B, 1, v_inner_local))
            if self.parallel_config.tensor_parallel.factor > 1:
                v_emb_1B1D = ttnn.mesh_partition(
                    v_emb_1B1D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            shifted_v = self.scale_shift_table.data + v_emb_1B1D
            v_shift_out, v_scale_out_p1 = ttnn.chunk(shifted_v, 2, dim=2)
        if self.image_conditioning:
            # Per-token: fused norm_out gamma/beta can't carry per-token scale/shift, so plain
            # layernorm then manual shift + normed * scale_p1, in fp32 to match the T2V branch.
            video_1BND = self.norm_out(video_1BND, dtype=ttnn.float32)
            v_shift_out = ttnn.typecast(v_shift_out, ttnn.float32)
            v_scale_out_p1 = ttnn.typecast(v_scale_out_p1, ttnn.float32)
            video_1BND = ttnn.addcmul(v_shift_out, video_1BND, v_scale_out_p1)
        else:
            # Fuse the AdaLN (1 + scale) * normed + shift modulation into norm_out (WAN pattern).
            video_1BND = self.norm_out(
                video_1BND, dynamic_weight=v_scale_out_p1, dynamic_bias=v_shift_out, dtype=ttnn.float32
            )
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
            a_shift_out, a_scale_out_p1 = ttnn.chunk(shifted_a, 2, dim=2)
            # Fuse the AdaLN (1 + scale) * normed + shift modulation into audio_norm_out (WAN pattern).
            audio_1BND = self.audio_norm_out(
                audio_1BND, dynamic_weight=a_scale_out_p1, dynamic_bias=a_shift_out, dtype=ttnn.float32
            )
            if self.parallel_config.tensor_parallel.factor > 1:
                audio_1BND = self.ccl_manager.all_gather_persistent_buffer(
                    audio_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
            audio_out = self.audio_proj_out(
                audio_1BND, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=ttnn.float32
            )

        # SP gather (skipped when an on-device solver steps the still-sharded latent)
        if gather_output and self.parallel_config.sequence_parallel.factor > 1:
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

    @staticmethod
    def device_to_host(
        tt_tensor: ttnn.Tensor,
        *,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig | None = None,
        sp_already_gathered: bool = False,
        tp_already_gathered: bool = False,
    ) -> torch.Tensor:
        """Assemble SP/TP shards on host (do not read device 0 alone on multi-device meshes).

        ``inner_step`` already AllGathers velocities on dim=2 before return; pass
        ``sp_already_gathered=True`` from the denoise loop to avoid concatenating the
        full sequence twice (audible doubling / ghost vocals).
        """
        if ccl_manager is not None and parallel_config is not None:
            mesh_dims: list[int | None] = [None, None]
            if not tp_already_gathered and parallel_config.tensor_parallel.factor > 1:
                mesh_dims[parallel_config.tensor_parallel.mesh_axis] = 3
            if not sp_already_gathered and parallel_config.sequence_parallel.factor > 1:
                mesh_dims[parallel_config.sequence_parallel.mesh_axis] = 2
            return ccl_manager.device_to_host(tt_tensor, mesh_dims).float().clone()
        return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0]).float()


class LTXTransformerCheckpoint:
    """An LTX transformer checkpoint: parses variant config, builds transformers, and loads weights.

    Mirrors ``WanCheckpoint`` but for the single-file LTX safetensors layout. The pipeline-level
    flags that vary per instance (``has_audio`` / ``image_conditioning``) are passed into ``build``
    explicitly — the checkpoint never reads pipeline state. LoRA specs vary per variant, so they
    are threaded through ``state_dict`` / ``cache_name`` / ``load`` rather than stored.
    """

    def __init__(self, checkpoint_path: str, *, inner_dim: int) -> None:
        self._checkpoint_path = checkpoint_path
        # Transformer + connector detection (key scan + one tensor-shape read). No tensor loads.
        with safe_open(checkpoint_path, framework="pt") as f:
            keys = list(f.keys())
            adaln_key = "model.diffusion_model.adaln_single.linear.weight"
            if adaln_key in keys:
                self.cross_attention_adaln = f.get_tensor(adaln_key).shape[0] > 6 * inner_dim
            else:
                self.cross_attention_adaln = True
            self.has_gate = any("to_gate_logits" in k for k in keys)
        logger.info(f"Detected: has_gate={self.has_gate}, cross_attention_adaln={self.cross_attention_adaln}")

    def state_dict(self, lora_specs: list[LoraSpec]) -> dict[str, torch.Tensor]:
        """Load + LoRA-fuse the transformer state dict from safetensors. Only
        invoked on cache miss by ``cache_module.load_model``."""
        logger.info(f"Transformer cache miss — loading safetensors: {self._checkpoint_path}")
        raw = load_file(self._checkpoint_path)
        prefix = "model.diffusion_model."
        sd = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
        if lora_specs:
            sd = fuse_loras_into(sd, lora_specs)
        return sd

    def cache_name(self, lora_specs: list[LoraSpec]) -> str:
        """Cache key for ``cache_module.load_model``. LoRA-tagged so fused and
        base weights don't alias in ``TT_DIT_CACHE_DIR``."""
        base = os.path.basename(self._checkpoint_path).removesuffix(".safetensors")
        if not lora_specs:
            return base
        tag = "+".join(f"{os.path.basename(s.path).removesuffix('.safetensors')}@{s.strength}" for s in lora_specs)
        return f"{base}.lora-{tag}"

    def build(
        self,
        *,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        cross_attention_dim: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool,
        has_audio: bool,
        image_conditioning: bool,
    ) -> LTXTransformerModel:
        """Construct an ``LTXTransformerModel`` for this checkpoint (weights NOT loaded).

        Loading is deferred so the caller can manage the lifecycle (deallocate / reload).
        """
        return LTXTransformerModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            cross_attention_dim=cross_attention_dim,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            has_audio=has_audio,
            apply_gated_attention=self.has_gate,
            cross_attention_adaln=self.cross_attention_adaln,
            image_conditioning=image_conditioning,
        )

    def load(
        self,
        model: LTXTransformerModel,
        *,
        parallel_config: DiTParallelConfig,
        mesh_shape: tuple[int, ...],
        is_fsdp: bool,
        lora_specs: list[LoraSpec],
    ) -> None:
        """Load (or reload) weights for a previously-built transformer."""
        cache_module.load_model(
            model,
            model_name=self.cache_name(lora_specs),
            subfolder="transformer",
            parallel_config=parallel_config,
            mesh_shape=mesh_shape,
            mesh_device=model.mesh_device,
            is_fsdp=is_fsdp,
            get_torch_state_dict=lambda: self.state_dict(lora_specs),
        )
