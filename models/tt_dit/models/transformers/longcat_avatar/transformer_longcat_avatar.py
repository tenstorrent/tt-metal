# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""LongCat-Video-Avatar DiT (1.5 / 1.6 architecture) for tt_dit.

The residual stream is ``[1, 1, N/sp, D/tp]``: sequence-parallel on ``sp_axis``, hidden fractured on ``tp_axis``.

Per-step conditioning that upstream computes in fp32 (timestep embedding and every adaLN projection) runs on host:
it is a few small matmuls per step, and keeping it there matches the reference exactly. The host uploads one
modulation row per distinct timestep (conditioning frames use t = 0), which the device expands to per-token tensors
only when conditioning frames are present.

Conditioning-frame semantics: tokens of the ``num_cond_latents`` leading latent frames attend only within their own
group (the reference frame, then the conditioning frames), receive no text or audio cross-attention, and are
modulated with the t = 0 rows. Noise tokens attend to everything.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange

import ttnn

from ....layers.feedforward import ParallelFeedForward
from ....layers.linear import ColParallelLinear, Linear
from ....layers.module import Module, ModuleList
from ....layers.normalization import DistributedLayerNorm, DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....reference.longcat_avatar import modeling_longcat_avatar as ref
from ....utils.mochi import get_rot_transformation_mat
from ....utils.substate import pop_substate
from ....utils.tensor import bf16_tensor, from_torch

SDPA_K_CHUNK = 256


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _interleave_heads(tensors: list[torch.Tensor], n_dev: int, num_heads: int, head_dim: int) -> torch.Tensor:
    """Interleaves ``[out, in]`` (or ``[out, 1]``) projections on the head axis so column-parallel fracturing hands
    each device the same heads from every tensor."""
    n_local = num_heads // n_dev
    tensors = [t.T.reshape(t.shape[1], n_dev, n_local, head_dim) for t in tensors]
    merged = torch.cat(tensors, dim=2)
    return merged.reshape(merged.shape[0], -1).T


def _interleave_chunks(t: torch.Tensor, n_chunks: int, n_dev: int) -> torch.Tensor:
    """Reorders the output dim of ``[n_chunks * D, ...]`` so each device's contiguous slice holds its shard of every
    chunk."""
    d = t.shape[0] // n_chunks
    rest = t.shape[1:]
    return t.reshape(n_chunks, n_dev, d // n_dev, *rest).transpose(0, 1).reshape(n_chunks * d, *rest)


@dataclass
class LongCatAvatarContext:
    """Step-invariant device inputs for one latent geometry and conditioning layout."""

    N: int
    latent_shape: tuple[int, int, int]
    num_cond_latents: int
    rope_cos: ttnn.Tensor
    rope_sin: ttnn.Tensor
    trans_mat: ttnn.Tensor
    # [1, 1, N/sp, 1]: 1 on conditioning tokens, else None when there are none
    cond_mask: ttnn.Tensor | None
    cond_mask_f32: ttnn.Tensor | None
    # [1, 1, N/sp, 1]: 1 - cond_mask
    noise_mask: ttnn.Tensor | None
    # restricted self-attention over the conditioning prefix: additive mask [1, 1, N/sp, P_pad] and P_pad
    prefix_bias: ttnn.Tensor | None
    prefix_len: int
    # per-frame audio cross-attention: additive mask [1, 1, N/sp, A_pad]
    audio_bias: ttnn.Tensor


class LongCatAvatarSelfAttention(Module):
    sdpa_chunk_size_map = {
        (8, 4): (288, 512),
        (4, 8): (256, 512),
        (2, 4): (256, 512),
    }

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        self.n_local_heads = num_heads // parallel_config.tensor_parallel.factor

        col = dict(mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl_manager)
        self.qkv = ColParallelLinear(dim, 3 * dim, bias=True, chunks=3, **col)
        self.proj = ColParallelLinear(dim, dim, bias=True, **col)
        norm = dict(
            embedding_dim=dim,
            norm_eps=1e-6,
            norm_elementwise_affine=True,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.q_norm = DistributedRMSNorm(**norm)
        self.k_norm = DistributedRMSNorm(**norm)

        full_grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=full_grid, q_chunk_size=256, k_chunk_size=SDPA_K_CHUNK, exp_approx_mode=False
        )
        self.sdpa_worker_grid = (full_grid.x - 1, full_grid.y)
        q_chunk, k_chunk = self.sdpa_chunk_size_map.get(
            (parallel_config.sequence_parallel.factor, parallel_config.tensor_parallel.factor), (256, 256)
        )
        self.ring_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
        )
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.dummy_joint_input = bf16_tensor(torch.zeros((1, self.n_local_heads, 0, self.head_dim)), device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        n_dev = self.parallel_config.tensor_parallel.factor
        if "qkv.weight" in state:
            q, k, v = state.pop("qkv.weight").chunk(3, dim=0)
            state["qkv.weight"] = _interleave_heads([q, k, v], n_dev, self.num_heads, self.head_dim)
        if "qkv.bias" in state:
            q, k, v = state.pop("qkv.bias").unsqueeze(-1).chunk(3, dim=0)
            state["qkv.bias"] = _interleave_heads([q, k, v], n_dev, self.num_heads, self.head_dim).squeeze(-1)
        for name in ("q_norm.weight", "k_norm.weight"):
            if name in state:
                state[name] = state[name].repeat(self.num_heads)

    def _sdpa_sp(self, q, k, v, N):
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            k,
            v,
            self.dummy_joint_input,
            self.dummy_joint_input,
            self.dummy_joint_input,
            persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(k.shape, 2, sp_axis, dtype=k.dtype),
            persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(v.shape, 2, sp_axis, dtype=v.dtype),
            joint_strategy="rear",
            logical_n=N,
            program_config=self.ring_sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(sp_axis),
            num_links=self.ccl_manager.num_links,
            cluster_axis=sp_axis,
            mesh_device=self.mesh_device,
            topology=self.ccl_manager.topology,
            subdevice_id=self.ccl_manager.ccl_sub_device_id,
            ccl_core_grid_offset=(self.sdpa_worker_grid[0], 0),
            use_column_major_ccl=True,
        )
        return out

    def _gather_prefix(self, t: ttnn.Tensor, prefix_len: int) -> ttnn.Tensor:
        """The first ``prefix_len`` tokens of the full (sequence-parallel) ``t``, replicated on every device."""
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        n_local = t.shape[2]
        if self.parallel_config.sequence_parallel.factor > 1:
            take = min(prefix_len, n_local)
            local = ttnn.slice(t, [0, 0, 0, 0], [t.shape[0], t.shape[1], take, t.shape[3]])
            t = self.ccl_manager.all_gather(local, dim=2, mesh_axis=sp_axis, use_hyperparams=True)
        return ttnn.slice(t, [0, 0, 0, 0], [t.shape[0], t.shape[1], prefix_len, t.shape[3]])

    def forward(self, x_1BND: ttnn.Tensor, ctx: LongCatAvatarContext) -> ttnn.Tensor:
        q, k, v = self.qkv(
            x_1BND, compute_kernel_config=self.mm_compute_kernel_config, parallel_config=self.parallel_config
        )
        norm_kw = dict(
            num_heads_per_device=self.n_local_heads,
            per_head_norm=True,
            rope_cos=ctx.rope_cos,
            rope_sin=ctx.rope_sin,
            trans_mat=ctx.trans_mat,
        )
        q = self.q_norm(q, **norm_kw)
        k = self.k_norm(k, **norm_kw)
        v, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            v, num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
        )

        if self.parallel_config.sequence_parallel.factor > 1:
            out = self._sdpa_sp(q, k, v, ctx.N)
        else:
            out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )
        out = ttnn.transformer.concatenate_heads(out)

        if ctx.prefix_bias is not None:
            k_pre = self._gather_prefix(k, ctx.prefix_len)
            v_pre = self._gather_prefix(v, ctx.prefix_len)
            restricted = ttnn.transformer.scaled_dot_product_attention(
                q,
                k_pre,
                v_pre,
                is_causal=False,
                attn_mask=ctx.prefix_bias,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )
            restricted = ttnn.unsqueeze(ttnn.transformer.concatenate_heads(restricted), 0)
            out = ttnn.unsqueeze(out, 0)
            out = ttnn.add(out, ttnn.multiply(ttnn.subtract(restricted, out), ctx.cond_mask))
        else:
            out = ttnn.unsqueeze(out, 0)

        return self.proj(out, compute_kernel_config=self.mm_compute_kernel_config, parallel_config=self.parallel_config)


class LongCatAvatarCrossAttention(Module):
    """Text cross-attention (``kv_dim`` = hidden) and per-frame audio cross-attention (``kv_dim`` = 768)."""

    def __init__(
        self,
        *,
        dim: int,
        kv_dim: int,
        num_heads: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.parallel_config = parallel_config
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        self.n_local_heads = num_heads // parallel_config.tensor_parallel.factor

        col = dict(mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl_manager)
        self.q_linear = ColParallelLinear(dim, dim, bias=True, **col)
        self.kv_linear = ColParallelLinear(kv_dim, 2 * dim, bias=True, chunks=2, **col)
        self.proj = ColParallelLinear(dim, dim, bias=True, **col)
        norm = dict(
            embedding_dim=dim,
            norm_eps=1e-6,
            norm_elementwise_affine=True,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.q_norm = DistributedRMSNorm(**norm)
        self.k_norm = DistributedRMSNorm(**norm)

        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=256,
            k_chunk_size=SDPA_K_CHUNK,
            exp_approx_mode=False,
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
        )
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        n_dev = self.parallel_config.tensor_parallel.factor
        if "kv_linear.weight" in state:
            k, v = state.pop("kv_linear.weight").chunk(2, dim=0)
            state["kv_linear.weight"] = _interleave_heads([k, v], n_dev, self.num_heads, self.head_dim)
        if "kv_linear.bias" in state:
            k, v = state.pop("kv_linear.bias").unsqueeze(-1).chunk(2, dim=0)
            state["kv_linear.bias"] = _interleave_heads([k, v], n_dev, self.num_heads, self.head_dim).squeeze(-1)
        for name in ("q_norm.weight", "k_norm.weight"):
            if name in state:
                state[name] = state[name].repeat(self.num_heads)

    def forward(self, x_1BND: ttnn.Tensor, kv_1BLC: ttnn.Tensor, attn_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """x_1BND: SP x TP fractured; kv_1BLC: replicated."""
        q = self.q_linear(
            x_1BND, compute_kernel_config=self.mm_compute_kernel_config, parallel_config=self.parallel_config
        )
        k, v = self.kv_linear(kv_1BLC, compute_kernel_config=self.mm_compute_kernel_config)
        q = self.q_norm(q, num_heads_per_device=self.n_local_heads, per_head_norm=True)
        k = self.k_norm(k, num_heads_per_device=self.n_local_heads, per_head_norm=True)
        v, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            v, num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
        )
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=attn_mask,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        out = ttnn.unsqueeze(ttnn.transformer.concatenate_heads(out), 0)
        return self.proj(out, compute_kernel_config=self.mm_compute_kernel_config, parallel_config=self.parallel_config)


class LongCatAvatarBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        audio_dim: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()
        self.parallel_config = parallel_config
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        common = dict(mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config)

        def layer_norm(affine: bool) -> DistributedLayerNorm:
            return DistributedLayerNorm(
                dim,
                norm_eps=1e-6,
                norm_elementwise_affine=affine,
                bias=affine,
                mesh_axis=tp_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )

        self.mod_norm_attn = layer_norm(False)
        self.mod_norm_ffn = layer_norm(False)
        self.pre_crs_attn_norm = layer_norm(True)
        self.pre_video_crs_attn_norm = layer_norm(True)

        self.attn = LongCatAvatarSelfAttention(dim=dim, num_heads=num_heads, **common)
        self.cross_attn = LongCatAvatarCrossAttention(dim=dim, kv_dim=dim, num_heads=num_heads, **common)
        self.audio_cross_attn = LongCatAvatarCrossAttention(dim=dim, kv_dim=audio_dim, num_heads=num_heads, **common)
        self.ffn = ParallelFeedForward(
            dim,
            inner_dim=ffn_dim,
            activation_fn="swiglu",
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )
        self.ff_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        ffn = pop_substate(state, "ffn")
        if "w1.weight" in ffn:
            # ff1 packs [up | gate] and computes up * silu(gate); upstream is w2(silu(w1(x)) * w3(x))
            state["ffn.ff1.weight"] = torch.cat([ffn["w3.weight"], ffn["w1.weight"]], dim=0)
        if "w2.weight" in ffn:
            state["ffn.ff2.weight"] = ffn["w2.weight"]

    @staticmethod
    def select(rows: ttnn.Tensor, ctx: LongCatAvatarContext) -> ttnn.Tensor:
        """[1, 1, 2, D/tp] (cond row, noise row) -> per-token [1, 1, N/sp, D/tp], or the noise row when no token is
        conditioned."""
        width = rows.shape[3]
        noise = ttnn.slice(rows, [0, 0, 1, 0], [1, 1, 2, width])
        if ctx.cond_mask is None:
            return noise
        cond = ttnn.slice(rows, [0, 0, 0, 0], [1, 1, 1, width])
        return ttnn.add(noise, ttnn.multiply(ctx.cond_mask_f32, ttnn.subtract(cond, noise)))

    @staticmethod
    def modulate(norm: DistributedLayerNorm, x: ttnn.Tensor, shift: ttnn.Tensor, scale1p: ttnn.Tensor) -> ttnn.Tensor:
        """``LN(x) * (1 + scale) + shift`` in fp32, cast back to bf16. Per-row (broadcast) or per-token modulation."""
        if shift.shape[2] == 1:
            return norm(x, dynamic_weight=scale1p, dynamic_bias=shift)
        x = norm(x, dtype=ttnn.float32)
        x = ttnn.add(ttnn.multiply(x, scale1p), shift)
        return ttnn.typecast(x, ttnn.bfloat16)

    def forward(
        self,
        x: ttnn.Tensor,
        prompt: ttnn.Tensor,
        audio: ttnn.Tensor,
        mod: dict[str, ttnn.Tensor],
        ctx: LongCatAvatarContext,
    ) -> ttnn.Tensor:
        """
        x: [1, 1, N/sp, D/tp] bf16.
        prompt: [1, 1, L, D] replicated (embedded caption).
        audio: [1, 1, A_pad, 768] replicated (per-frame audio tokens, frame-major).
        mod: fp32 rows [1, 1, 2, D/tp] (cond, noise) for the msa / mlp shift, 1 + scale and gate; audio rows
            [1, 1, 1, D/tp] (noise only).
        """
        x_m = self.modulate(
            self.mod_norm_attn, x, self.select(mod["shift_msa"], ctx), self.select(mod["scale1p_msa"], ctx)
        )
        attn = self.attn(x_m, ctx)
        x = ttnn.addcmul(x, attn, ttnn.typecast(self.select(mod["gate_msa"], ctx), ttnn.bfloat16))

        text = self.cross_attn(self.pre_crs_attn_norm(x), prompt)
        if ctx.noise_mask is not None:
            text = ttnn.multiply(text, ctx.noise_mask)
        x = ttnn.add(x, text)

        a = self.audio_cross_attn(self.pre_video_crs_attn_norm(x), audio, attn_mask=ctx.audio_bias)
        a = self.mod_norm_attn(a, dynamic_weight=mod["audio_scale1p"], dynamic_bias=mod["audio_shift"])
        a = ttnn.multiply(a, ttnn.typecast(mod["audio_gate"], ttnn.bfloat16))
        if ctx.noise_mask is not None:
            a = ttnn.multiply(a, ctx.noise_mask)
        x = ttnn.add(x, a)

        x_m = self.modulate(
            self.mod_norm_ffn, x, self.select(mod["shift_mlp"], ctx), self.select(mod["scale1p_mlp"], ctx)
        )
        ff = self.ffn(x_m, compute_kernel_config=self.ff_compute_kernel_config, parallel_config=self.parallel_config)
        return ttnn.addcmul(x, ff, ttnn.typecast(self.select(mod["gate_mlp"], ctx), ttnn.bfloat16))


class _HostConditioning(torch.nn.Module):
    """The fp32 conditioning upstream computes under autocast: timestep embedding, adaLN projections, caption and
    audio embedders. Runs on host."""

    def __init__(self, cfg: dict, num_layers: int) -> None:
        super().__init__()
        hidden, tdim = cfg["hidden_size"], cfg["adaln_tembed_dim"]
        self.t_embedder = ref.TimestepEmbedder(tdim, cfg["frequency_embedding_size"])
        self.y_embedder = ref.CaptionEmbedder(cfg["caption_channels"], hidden)
        self.audio_proj = ref.AudioProjModel(
            seq_len=cfg["audio_window"],
            seq_len_vf=cfg["audio_window"] + cfg["vae_scale"] - 1,
            blocks=cfg["audio_block"],
            channels=cfg["audio_channel"],
            intermediate_dim=cfg["intermediate_dim"],
            output_dim=cfg["output_dim"],
            context_tokens=cfg["context_tokens"],
        )
        self.adaln = torch.nn.ModuleList(torch.nn.Linear(tdim, 6 * hidden) for _ in range(num_layers))
        self.audio_adaln = torch.nn.ModuleList(torch.nn.Linear(tdim, 3 * hidden) for _ in range(num_layers))
        self.final_adaln = torch.nn.Linear(tdim, 2 * hidden)


class LongCatAvatarTransformer(Module):
    def __init__(
        self,
        config: dict,
        *,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        num_layers: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.num_layers = num_layers if num_layers is not None else config["depth"]
        self.dim = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.head_dim = self.dim // self.num_heads
        self.patch_size = tuple(config["patch_size"])
        self.in_channels = config["in_channels"]
        self.out_channels = config["out_channels"]
        self.vae_scale = config["vae_scale"]
        self.audio_window = config["audio_window"]
        hidden_dim = int(2 * self.dim * config["mlp_ratio"] / 3)
        self.ffn_dim = 256 * ((hidden_dim + 255) // 256)
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        patch_dim = self.in_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.x_embedder = ColParallelLinear(
            patch_dim, self.dim, bias=True, mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl_manager
        )
        self.blocks = ModuleList(
            LongCatAvatarBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                ffn_dim=self.ffn_dim,
                audio_dim=config["output_dim"],
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
            )
            for _ in range(self.num_layers)
        )
        self.norm_final = DistributedLayerNorm(
            self.dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.final_linear = Linear(self.dim, patch_dim // self.in_channels * self.out_channels, mesh_device=mesh_device)
        self.host = _HostConditioning(config, self.num_layers)
        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._contexts: dict[tuple, LongCatAvatarContext] = {}

    # -----------------------------------------------------------------------------------------------------------------
    # weights
    # -----------------------------------------------------------------------------------------------------------------

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        host = {}
        for prefix in ("t_embedder.", "y_embedder.", "audio_proj."):
            for key in [k for k in state if k.startswith(prefix)]:
                host[key] = state.pop(key)
        for i in range(self.num_layers):
            for src, dst in (
                (f"blocks.{i}.adaLN_modulation.1", f"adaln.{i}"),
                (f"blocks.{i}.audio_adaLN_modulation.1", f"audio_adaln.{i}"),
            ):
                for p in ("weight", "bias"):
                    if f"{src}.{p}" in state:
                        host[f"{dst}.{p}"] = state.pop(f"{src}.{p}")
        for p in ("weight", "bias"):
            if f"final_layer.adaLN_modulation.1.{p}" in state:
                host[f"final_adaln.{p}"] = state.pop(f"final_layer.adaLN_modulation.1.{p}")
        if host:
            self.host.load_state_dict(host, strict=True)
            # upstream runs the timestep embedder and adaLN projections under fp32 autocast, the embedders in bf16
            self.host.float()
            self.host.y_embedder.to(torch.bfloat16)
            self.host.audio_proj.to(torch.bfloat16)

        if "x_embedder.proj.weight" in state:
            w = state.pop("x_embedder.proj.weight")  # [D, C, pF, pH, pW] -> patches flattened as (pF pH pW C)
            state["x_embedder.weight"] = w.permute(0, 2, 3, 4, 1).reshape(w.shape[0], -1)
        if "x_embedder.proj.bias" in state:
            state["x_embedder.bias"] = state.pop("x_embedder.proj.bias")
        for p in ("weight", "bias"):
            if f"final_layer.linear.{p}" in state:
                state[f"final_linear.{p}"] = state.pop(f"final_layer.linear.{p}")

    # -----------------------------------------------------------------------------------------------------------------
    # step-invariant inputs
    # -----------------------------------------------------------------------------------------------------------------

    def _sp_tensor(self, t: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        """Uploads ``[1, 1, N_pad, C]`` sharded on the sequence over the SP axis."""
        return from_torch(
            t,
            device=self.mesh_device,
            dtype=dtype,
            mesh_axes=[..., self.parallel_config.sequence_parallel.mesh_axis, None],
        )

    def padded_length(self, N: int) -> int:
        return _round_up(N, ttnn.TILE_SIZE * self.parallel_config.sequence_parallel.factor)

    def get_context(
        self,
        latent_shape: tuple[int, int, int],
        num_cond_latents: int = 0,
        num_ref_latents: int | None = None,
        ref_img_index: int | None = None,
        mask_frame_range: int | None = None,
    ) -> LongCatAvatarContext:
        """``latent_shape``: (T, H, W) of the latent video."""
        assert not mask_frame_range, "mask_frame_range > 0 is not supported yet"
        key = (tuple(latent_shape), num_cond_latents, num_ref_latents, ref_img_index)
        if key in self._contexts:
            return self._contexts[key]

        T, H, W = latent_shape
        N_t, N_h, N_w = T // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]
        hw = N_h * N_w
        N = N_t * hw
        N_pad = self.padded_length(N)
        neg = -1e9

        freqs = ref.rope_3d_freqs(self.head_dim, (N_t, N_h, N_w), ref_img_index, num_ref_latents)
        freqs = torch.nn.functional.pad(freqs, (0, 0, 0, N_pad - N))[None, None]
        rope_cos = self._sp_tensor(freqs.cos(), ttnn.float32)
        rope_sin = self._sp_tensor(freqs.sin(), ttnn.float32)
        trans_mat = bf16_tensor(get_rot_transformation_mat(), device=self.mesh_device)

        token_frame = torch.arange(N_pad) // hw  # padded tokens fall past the last frame
        cond_mask = cond_mask_f32 = noise_mask = prefix_bias = None
        prefix_len = 0
        if num_cond_latents > 0:
            is_cond = (token_frame < num_cond_latents).float()
            cond_mask = self._sp_tensor(is_cond.reshape(1, 1, -1, 1))
            cond_mask_f32 = self._sp_tensor(is_cond.reshape(1, 1, -1, 1), ttnn.float32)
            noise_mask = self._sp_tensor((1 - is_cond).reshape(1, 1, -1, 1))
            # restricted groups: ref frame alone, then the remaining conditioning frames (or one group in i2v)
            P = num_cond_latents * hw
            prefix_len = _round_up(P, ttnn.TILE_SIZE)
            group_end = torch.full((N_pad,), P)
            group_start = torch.zeros(N_pad, dtype=torch.long)
            if num_cond_latents > 1:
                is_ref = token_frame < 1
                group_end = torch.where(is_ref, torch.tensor(hw), group_end)
                group_start = torch.where(is_ref, torch.tensor(0), torch.tensor(hw))
            keys = torch.arange(prefix_len)
            allowed = (keys[None] >= group_start[:, None]) & (keys[None] < group_end[:, None])
            allowed |= (token_frame >= num_cond_latents)[:, None] & (keys[None] < P)  # unused rows, keep finite
            prefix_bias = self._sp_tensor(torch.where(allowed, 0.0, neg).reshape(1, 1, N_pad, prefix_len))

        ctx_tokens = self.config["context_tokens"]
        a_len = _round_up(N_t * ctx_tokens, ttnn.TILE_SIZE)
        audio_frame = torch.arange(a_len) // ctx_tokens
        frame = token_frame.clamp(max=N_t - 1)
        audio_allowed = audio_frame[None] == frame[:, None]
        audio_bias = self._sp_tensor(torch.where(audio_allowed, 0.0, neg).reshape(1, 1, N_pad, a_len))

        ctx = LongCatAvatarContext(
            N=N,
            latent_shape=(N_t, N_h, N_w),
            num_cond_latents=num_cond_latents,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
            cond_mask=cond_mask,
            cond_mask_f32=cond_mask_f32,
            noise_mask=noise_mask,
            prefix_bias=prefix_bias,
            prefix_len=prefix_len,
            audio_bias=audio_bias,
        )
        self._contexts[key] = ctx
        return ctx

    # -----------------------------------------------------------------------------------------------------------------
    # host conditioning
    # -----------------------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def prepare_text(
        self, encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor | None
    ) -> ttnn.Tensor:
        """[1, 1, L, C_text] -> embedded caption [1, 1, L, D] replicated on device (padding rows zeroed)."""
        y = self.host.y_embedder(encoder_hidden_states.to(torch.bfloat16))
        if self.config.get("text_tokens_zero_pad", True) and encoder_attention_mask is not None:
            y = y * encoder_attention_mask[:, None, :, None]
        elif encoder_attention_mask is not None:
            y = y[:, :, encoder_attention_mask[0].bool()]
        return bf16_tensor(y.reshape(1, 1, -1, self.dim), device=self.mesh_device)

    @torch.no_grad()
    def prepare_audio(self, audio_embs: torch.Tensor, N_t: int, num_ref_latents: int | None = None) -> ttnn.Tensor:
        """[1, num_frames, window, blocks, channels] -> per-frame audio tokens [1, 1, A_pad, 768] replicated."""
        tokens = ref.audio_tokens(
            self.host.audio_proj, audio_embs.to(torch.bfloat16), N_t, self.vae_scale, self.audio_window, num_ref_latents
        )
        tokens = tokens.reshape(1, 1, -1, tokens.shape[-1])
        a_len = _round_up(tokens.shape[2], ttnn.TILE_SIZE)
        return bf16_tensor(torch.nn.functional.pad(tokens, (0, 0, 0, a_len - tokens.shape[2])), device=self.mesh_device)

    @torch.no_grad()
    def prepare_modulation(self, timestep: float) -> tuple[list[dict], dict]:
        """Per-block fp32 modulation rows for one step: (cond t = 0, noise t) for msa / mlp, noise only for audio."""
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        ts = torch.tensor([0.0, float(timestep)]).to(torch.bfloat16)
        t_silu = torch.nn.functional.silu(self.host.t_embedder(ts.float(), dtype=torch.float32))

        def upload(rows: torch.Tensor) -> ttnn.Tensor:
            return from_torch(
                rows.reshape(1, 1, rows.shape[-2], -1),
                device=self.mesh_device,
                dtype=ttnn.float32,
                mesh_axes=[..., tp_axis],
            )

        mods = []
        for i in range(self.num_layers):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.host.adaln[i](t_silu).chunk(6, -1)
            a_shift, a_scale, a_gate = self.host.audio_adaln[i](t_silu[1:]).chunk(3, dim=-1)
            mods.append(
                {
                    "shift_msa": upload(shift_msa),
                    "scale1p_msa": upload(1 + scale_msa),
                    "gate_msa": upload(gate_msa),
                    "shift_mlp": upload(shift_mlp),
                    "scale1p_mlp": upload(1 + scale_mlp),
                    "gate_mlp": upload(gate_mlp),
                    "audio_shift": upload(a_shift),
                    "audio_scale1p": upload(1 + a_scale),
                    "audio_gate": upload(a_gate),
                }
            )
        f_shift, f_scale = self.host.final_adaln(t_silu).chunk(2, dim=-1)
        return mods, {"shift": upload(f_shift), "scale1p": upload(1 + f_scale)}

    # -----------------------------------------------------------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------------------------------------------------------

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """[1, C, T, H, W] -> [1, 1, N_pad, pF*pH*pW*C]."""
        pF, pH, pW = self.patch_size
        x = rearrange(latents, "b c (t pf) (h ph) (w pw) -> b (t h w) (pf ph pw c)", pf=pF, ph=pH, pw=pW)
        x = x[None]
        return torch.nn.functional.pad(x, (0, 0, 0, self.padded_length(x.shape[2]) - x.shape[2]))

    def unpatchify(self, x: torch.Tensor, ctx: LongCatAvatarContext) -> torch.Tensor:
        N_t, N_h, N_w = ctx.latent_shape
        pF, pH, pW = self.patch_size
        x = x.reshape(-1, x.shape[-1])[: ctx.N]
        return rearrange(
            x[None],
            "b (t h w) (pf ph pw c) -> b c (t pf) (h ph) (w pw)",
            t=N_t,
            h=N_h,
            w=N_w,
            pf=pF,
            ph=pH,
            pw=pW,
            c=self.out_channels,
        )

    def forward_device(
        self,
        x_1BNI: ttnn.Tensor,
        prompt: ttnn.Tensor,
        audio: ttnn.Tensor,
        mods: list[dict],
        final: dict,
        ctx: LongCatAvatarContext,
    ) -> ttnn.Tensor:
        """x_1BNI: patchified latents [1, 1, N/sp, 64] bf16. Returns [1, 1, N, 64] fp32 gathered on SP."""
        x = self.x_embedder(x_1BNI)
        for block, mod in zip(self.blocks, mods):
            x = block(x, prompt, audio, mod, ctx)

        shift = LongCatAvatarBlock.select(final["shift"], ctx)
        scale1p = LongCatAvatarBlock.select(final["scale1p"], ctx)
        x = LongCatAvatarBlock.modulate(self.norm_final, x, shift, scale1p)
        if self.parallel_config.tensor_parallel.factor > 1:
            x = self.ccl_manager.all_gather_persistent_buffer(
                x, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        x = self.final_linear(x, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=ttnn.float32)
        if self.parallel_config.sequence_parallel.factor > 1:
            x = self.ccl_manager.all_gather_persistent_buffer(
                x, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
            )
        return x

    def forward(
        self,
        latents: torch.Tensor,
        timestep: float,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        audio_embs: torch.Tensor,
        num_cond_latents: int = 0,
        num_ref_latents: int | None = None,
        ref_img_index: int | None = None,
        mask_frame_range: int | None = None,
    ) -> torch.Tensor:
        """Host-tensor convenience wrapper matching the reference forward (scalar noise timestep)."""
        _, _, T, H, W = latents.shape
        ctx = self.get_context((T, H, W), num_cond_latents, num_ref_latents, ref_img_index, mask_frame_range)
        prompt = self.prepare_text(encoder_hidden_states, encoder_attention_mask)
        audio = self.prepare_audio(audio_embs, ctx.latent_shape[0], num_ref_latents)
        mods, final = self.prepare_modulation(timestep)
        x = self._sp_tensor(self.patchify(latents.to(torch.bfloat16)))
        out = self.forward_device(x, prompt, audio, mods, final, ctx)
        out = ttnn.to_torch(ttnn.get_device_tensors(out)[0])
        return self.unpatchify(out, ctx).float()
