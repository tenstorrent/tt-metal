# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN WanTransformer3DModel for Lingbot-VA.

Reuses WanPatchEmbed, WanTimeTextImageEmbedding, and utilities from models.tt_dit.
Defines WanTransformerBlock using native TT WanAttention
Adds Lingbot-VA-specific: in_channels=48, action_embedder, condition_embedder_action,
action_proj_out, and dual forward paths (video + action).
"""

from __future__ import annotations

import os
import torch
from loguru import logger

import ttnn

from models.tt_dit.layers.embeddings import WanPatchEmbed, WanTimeTextImageEmbedding
from models.tt_dit.layers.feedforward import ParallelFeedForward
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList, Parameter
from models.tt_dit.layers.normalization import DistributedLayerNorm
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.padding import get_padded_vision_seq_len, pad_vision_seq_parallel
from models.tt_dit.utils.substate import pop_substate, rename_substate
from models.tt_dit.utils.tensor import bf16_tensor, float32_tensor, from_torch, unflatten

from .attention_wan import WanAttention
from .wan_rotary_pos_embed import WanRotaryPosEmbed


def _align_modulation_to_spatial_layout(spatial_1BND: ttnn.Tensor, t: ttnn.Tensor) -> ttnn.Tensor:
    """Match spatial [1,1,L,C]: drop stray [...,1,...] dims; permute [1,L,1,C] -> [1,1,L,C]."""
    if len(t.shape) == 5 and int(t.shape[3]) == 1:
        a, b, c, _, e = (int(t.shape[i]) for i in range(5))
        t = ttnn.reshape(t, (a, b, c, e))
    if len(spatial_1BND.shape) != 4 or len(t.shape) != 4:
        return t
    b0, b1, seq, c = (int(spatial_1BND.shape[i]) for i in range(4))
    if int(t.shape[0]) == b0 and int(t.shape[1]) == seq and int(t.shape[2]) == 1 and int(t.shape[3]) == c and b1 == 1:
        return ttnn.permute(t, (0, 2, 1, 3))
    return t


# Lingbot-VA config (from reference model.py)
DIM = 3072  # num_attention_heads * attention_head_dim
FFN_DIM = 14336
NUM_HEADS = 24
HEAD_DIM = 128
IN_CHANNELS = 48
OUT_CHANNELS = 48
ACTION_DIM = 30
TEXT_DIM = 4096
FREQ_DIM = 256
NUM_LAYERS = 30
PATCH_SIZE = (1, 2, 2)
CROSS_ATTN_NORM = True
EPS = 1e-6
ROPE_MAX_SEQ_LEN = 1024


class WanTransformerBlock(Module):
    """Transformer block for Lingbot-VA (same structure as tt_dit wan2_2, no wan2_2 dependency)."""

    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attention_norm: bool = True,
        eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.cross_attention_norm = cross_attention_norm
        self.eps = eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.norm1 = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.attn1 = WanAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_self=True,
        )

        self.attn2 = WanAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_self=False,
        )

        self.norm2 = (
            DistributedLayerNorm(
                dim,
                norm_eps=eps,
                norm_elementwise_affine=True,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )
            if cross_attention_norm
            else None
        )

        self.ffn = ParallelFeedForward(
            dim,
            inner_dim=ffn_dim,
            activation_fn="gelu_tanh",
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        self.norm3 = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.scale_shift_table = Parameter(
            total_shape=[1, 1, 6, dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )

        # HiFi4 + fp32 dest acc for FFN (matches attention SDPA/to_out quality targets for PCC).
        self.ff_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "ffn.net.0.proj", "ffn.ff1")
        rename_substate(state, "ffn.net.2", "ffn.ff2")

        if "scale_shift_table" in state:
            state["scale_shift_table"] = state["scale_shift_table"].unsqueeze(0)

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        temb_1BTD: ttnn.Tensor,
        N: int,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        update_cache: int = 0,
        cache_name: str = "pos",
        cached_k: ttnn.Tensor | None = None,
        cached_v: ttnn.Tensor | None = None,
        return_kv: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        spatial_1BND: fractured N on SP, fractured D on TP
        prompt_1BLP: replicated on SP, replicated D on TP
        temb_1BTD: [1, B, 6, D] global conditioning OR [1, B, N_padded, 6*D] per-token conditioning
        N: logical sequence length of the spatial input
        rope_cos_BANH: fractured N on SP, A (num_heads) on TP
        rope_sin_BANH: fractured N on SP, A (num_heads) on TP
        trans_mat: replicated on SP, replicated D on TP

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        """

        # Reference block always uses timestep_proj shaped like [B, L, 6, C] (or flattened [B, L, 6*C] for robotwin).
        sst = self.scale_shift_table.data
        td = temb_1BTD
        if len(td.shape) == 5:
            # [1, B, L, 6, C] after per-position time embed (matches reference unflatten(2, (6, -1))).
            shifted_temb = sst + td
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = ttnn.chunk(shifted_temb, 6, dim=3)
        elif len(td.shape) == 4 and td.shape[2] == 6:
            # Legacy broadcast: [1, B, 6, C]
            shifted_temb = sst + td
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = ttnn.chunk(shifted_temb, 6, dim=2)
        else:
            # Flattened per-token path [1, B, N, 6 * C_local] (e.g. robotwin / prepare_per_token_conditioning).
            sst_flat = ttnn.reshape(sst, (1, 1, 1, sst.shape[-2] * sst.shape[-1]))
            shifted_temb = td + sst_flat
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = ttnn.chunk(shifted_temb, 6, dim=-1)

        shift_msa = _align_modulation_to_spatial_layout(spatial_1BND, shift_msa)
        scale_msa = _align_modulation_to_spatial_layout(spatial_1BND, scale_msa)
        gate_msa = _align_modulation_to_spatial_layout(spatial_1BND, gate_msa)
        c_shift_msa = _align_modulation_to_spatial_layout(spatial_1BND, c_shift_msa)
        c_scale_msa = _align_modulation_to_spatial_layout(spatial_1BND, c_scale_msa)
        c_gate_msa = _align_modulation_to_spatial_layout(spatial_1BND, c_gate_msa)

        gate_msa = ttnn.typecast(gate_msa, dtype=ttnn.bfloat16)
        c_gate_msa = ttnn.typecast(c_gate_msa, dtype=ttnn.bfloat16)

        # Match reference WanTransformerBlock: FP32 LayerNorm then explicit (1+scale)*x+shift, then cast to activations dtype.
        x_dtype = spatial_1BND.dtype
        spatial_normed_1BND = self.norm1(spatial_1BND, dtype=ttnn.float32)
        spatial_normed_1BND = spatial_normed_1BND * ttnn.typecast(1.0 + scale_msa, ttnn.float32) + ttnn.typecast(
            shift_msa, ttnn.float32
        )
        spatial_normed_1BND = ttnn.typecast(spatial_normed_1BND, x_dtype)

        attn1_common = dict(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
            addcmul_residual=spatial_1BND,
            addcmul_gate=gate_msa,
            cached_k=cached_k,
            cached_v=cached_v,
        )
        k_cur = v_cur = None
        if return_kv:
            spatial_1BND, k_cur, v_cur = self.attn1(**attn1_common, return_kv=True)
        else:
            spatial_1BND = self.attn1(**attn1_common, return_kv=False)

        if self.norm2 is not None:
            spatial_normed_1BND = ttnn.typecast(self.norm2(spatial_1BND, dtype=ttnn.float32), x_dtype)
        else:
            spatial_normed_1BND = spatial_1BND
        attn_output_1BND = self.attn2(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            prompt_1BLP=prompt_1BLP,
        )
        spatial_1BND = spatial_1BND + attn_output_1BND

        spatial_normed_1BND = self.norm3(spatial_1BND, dtype=ttnn.float32)
        spatial_normed_1BND = spatial_normed_1BND * ttnn.typecast(1.0 + c_scale_msa, ttnn.float32) + ttnn.typecast(
            c_shift_msa, ttnn.float32
        )
        spatial_normed_1BND = ttnn.typecast(spatial_normed_1BND, x_dtype)

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_normed_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_normed_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        spatial_ff_1BND = self.ffn(spatial_normed_1BND, compute_kernel_config=self.ff_compute_kernel_config)

        # Reference: hidden_states.float() + ff_output.float() * c_gate (fp32 residual, then cast).
        spatial_1BND = ttnn.typecast(
            ttnn.typecast(spatial_1BND, ttnn.float32)
            + ttnn.typecast(spatial_ff_1BND, ttnn.float32) * ttnn.typecast(c_gate_msa, ttnn.float32),
            x_dtype,
        )

        if return_kv:
            return (spatial_1BND, k_cur, v_cur)
        return spatial_1BND


class WanTransformer3DModel(Module):
    """
    TTNN WanTransformer3DModel for Lingbot-VA.

    Supports:
    - Video path: spatial (B, in_channels, F, H, W) -> patch_embed -> blocks -> proj_out -> (B, out_channels, F, H, W)
    - Action path: spatial (B, action_dim, F, H, W) -> action_embedder -> blocks -> action_proj_out -> (B, N, action_dim)
    """

    def __init__(
        self,
        *,
        patch_size: tuple[int, ...] = PATCH_SIZE,
        num_heads: int = NUM_HEADS,
        dim: int = DIM,
        in_channels: int = IN_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        action_dim: int = ACTION_DIM,
        text_dim: int = TEXT_DIM,
        freq_dim: int = FREQ_DIM,
        ffn_dim: int = FFN_DIM,
        num_layers: int = NUM_LAYERS,
        cross_attn_norm: bool = CROSS_ATTN_NORM,
        eps: float = EPS,
        rope_max_seq_len: int = ROPE_MAX_SEQ_LEN,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = True,
    ) -> None:
        super().__init__()

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.is_fsdp = is_fsdp
        self.fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None
        self.cached_rope_features = {}

        self.patch_size = patch_size
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.action_dim = action_dim

        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.rope = WanRotaryPosEmbed(
            mesh_device=mesh_device,
            attention_head_dim=head_dim,
            patch_size=patch_size,
            max_seq_len=rope_max_seq_len,
        )

        self.patch_embedding = WanPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            mesh_device=mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.action_embedder = Linear(
            action_dim,
            dim,
            bias=True,
            mesh_device=mesh_device,
        )

        # Text and timestep for video path; text always from condition_embedder.
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            time_freq_dim=freq_dim,
            time_proj_dim=dim * 6,
            text_embed_dim=text_dim,
            mesh_device=self.mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        # Timestep-only for action path (action_mode=True).
        self.condition_embedder_action = WanTimeTextImageEmbedding(
            dim=dim,
            time_freq_dim=freq_dim,
            time_proj_dim=dim * 6,
            text_embed_dim=text_dim,
            mesh_device=self.mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.blocks = ModuleList(
            WanTransformerBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                cross_attention_norm=cross_attn_norm,
                eps=eps,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
            )
            for _ in range(num_layers)
        )

        self.norm_out = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.proj_out = Linear(
            dim,
            patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
            bias=True,
            mesh_device=mesh_device,
        )

        self.action_proj_out = Linear(
            dim,
            action_dim,
            bias=True,
            mesh_device=mesh_device,
        )

        self.scale_shift_table = Parameter(
            total_shape=[1, 2, dim],
            device=mesh_device,
            mesh_axes=[None, None, parallel_config.tensor_parallel.mesh_axis],
            dtype=ttnn.float32,
        )

        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._attn_caches: dict[str, list[dict | None]] = {}

    def clear_cache(self, cache_name: str) -> None:
        for block in self.blocks:
            if hasattr(block.attn1, "clear_cache"):
                block.attn1.clear_cache(cache_name)
        if cache_name in self._attn_caches:
            self._attn_caches[cache_name] = [None] * len(self.blocks)

    def clear_pred_cache(self, cache_name: str) -> None:
        for block in self.blocks:
            if hasattr(block.attn1, "clear_pred_cache"):
                block.attn1.clear_pred_cache(cache_name)
        if cache_name not in self._attn_caches:
            return
        for cache in self._attn_caches[cache_name]:
            if cache is not None and cache["is_pred"].any():
                cache["mask"][cache["is_pred"]] = False

    def create_empty_cache(
        self,
        cache_name: str,
        attn_window: int,
        latent_token_per_chunk: int,
        action_token_per_chunk: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 1,
    ) -> None:
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (attn_window // 2) * action_token_per_chunk
        num_heads = self.num_heads
        head_dim = self.dim // num_heads
        for block in self.blocks:
            if hasattr(block.attn1, "init_kv_cache"):
                block.attn1.init_kv_cache(
                    cache_name,
                    total_tolen,
                    num_heads,
                    head_dim,
                    device,
                    torch.float32,
                    batch_size,
                )
        cache_device = device if device.type == "cpu" else torch.device("cpu")
        self._attn_caches[cache_name] = []
        for _ in range(len(self.blocks)):
            self._attn_caches[cache_name].append(
                {
                    "k": torch.empty([batch_size, total_tolen, num_heads, head_dim], device=cache_device, dtype=dtype),
                    "v": torch.empty([batch_size, total_tolen, num_heads, head_dim], device=cache_device, dtype=dtype),
                    "id": torch.full((total_tolen,), -1, device=cache_device),
                    "mask": torch.zeros((total_tolen,), dtype=torch.bool, device=cache_device),
                    "is_pred": torch.zeros((total_tolen,), dtype=torch.bool, device=cache_device),
                }
            )

    def _cache_allocate_slots(self, cache: dict, key_size: int) -> torch.Tensor:
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)
        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)
            order = torch.argsort(ids[used])
            need = key_size - free.numel()
            to_free = used[order[:need]]
            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)
        assert free.numel() >= key_size
        return free[:key_size]

    def _cache_next_id(self, cache: dict) -> int:
        ids, mask = cache["id"], cache["mask"]
        if mask.any():
            return int(ids[mask].max().item()) + 1
        return 0

    def _cache_update(self, cache: dict, key: torch.Tensor, value: torch.Tensor, is_pred: bool):
        key_size = key.shape[1]
        slots = self._cache_allocate_slots(cache, key_size)
        new_id = self._cache_next_id(cache)
        cache["k"][:, slots] = key
        cache["v"][:, slots] = value
        cache["mask"][slots] = True
        cache["id"][slots] = new_id
        cache["is_pred"][slots] = is_pred
        return slots

    def _cache_restore(self, cache: dict, slots: torch.Tensor) -> None:
        cache["mask"][slots] = False

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pop_substate(state, "rope")

        # Map reference patch_embedding_mlp (Linear) -> patch_embedding. Ref uses input order (C, p0, p1, p2);
        # WanPatchEmbed.forward does input @ proj_weight, so proj_weight must be (in_dim, embed_dim) with rows (C, p0, p1, p2) = ref_weight.T.
        rename_substate(state, "patch_embedding_mlp", "patch_embedding")
        patch_weight = state.get("patch_embedding.weight")
        if patch_weight is not None:
            p0, p1, p2 = self.patch_size
            if patch_weight.ndim == 2:
                embed_dim, flat = patch_weight.shape
                assert flat == self.in_channels * p0 * p1 * p2
                # (embed_dim, in_dim) -> (in_dim, embed_dim) with rows (C, p0, p1, p2) so TT matches ref
                state["patch_embedding.proj_weight"] = patch_weight.T.contiguous()
                del state["patch_embedding.weight"]
            elif patch_weight.ndim == 5:
                embed_dim = patch_weight.shape[0]
                state["patch_embedding.proj_weight"] = patch_weight.reshape(embed_dim, -1).T.contiguous()
                del state["patch_embedding.weight"]
            else:
                raise ValueError(
                    f"patch_embedding.weight must be 2D (Linear) or 5D (conv), got ndim={patch_weight.ndim}"
                )
        patch_bias = state.get("patch_embedding.bias")
        if patch_bias is not None:
            # WanPatchEmbed expects proj_bias (see embeddings.WanPatchEmbed._prepare_torch_state).
            state["patch_embedding.proj_bias"] = patch_bias.reshape(1, -1)
            del state["patch_embedding.bias"]

    def get_rope_features(self, grid_id: torch.Tensor):
        """Build RoPE cos/sin and transformation matrix from grid_id (B, 3, L).

        Cache key must include grid *values*, not only shape: multi-chunk generate passes the same
        token layout with different ``frame_st_id`` (temporal offset in ``get_mesh_id``). Shape-only
        keys incorrectly reuse chunk-0 RoPE for all chunks → static-looking decoded video while
        single-chunk PCC vs torch stays high.
        """
        gid_cpu = grid_id.detach().contiguous().cpu()
        cache_key = (tuple(grid_id.shape), gid_cpu.numpy().tobytes())
        if cache_key not in self.cached_rope_features:
            rope_features = self.prepare_rope_features(grid_id)
            self.cached_rope_features[cache_key] = rope_features
        return self.cached_rope_features[cache_key]

    def prepare_rope_features(self, grid_id: torch.Tensor):
        """Build RoPE cos/sin and transformation matrix from grid_id (B, 3, L). Pads for sequence parallel."""
        logger.debug("Preparing rope features for shape {}", grid_id.shape)
        grid_id_tt = float32_tensor(grid_id.float(), device=self.mesh_device)
        rope_cos_tt, rope_sin_tt = self.rope(grid_id_tt)
        # WanRotaryPosEmbed returns [B, L, D] with B = grid_id batch (e.g. 2 for classifier-free guidance).
        sh_cos = tuple(rope_cos_tt.shape)
        if len(sh_cos) == 4 and int(sh_cos[0]) == 1:
            rope_cos_1BLD = rope_cos_tt
            rope_sin_1BLD = rope_sin_tt
            B_rope, L, D = int(sh_cos[1]), int(sh_cos[2]), int(sh_cos[3])
        elif len(sh_cos) == 3:
            B_rope, L, D = int(sh_cos[0]), int(sh_cos[1]), int(sh_cos[2])
            if B_rope != int(grid_id.shape[0]):
                logger.warning(
                    "rope batch {} != grid_id batch {}; using rope layout for reshape",
                    B_rope,
                    int(grid_id.shape[0]),
                )
            rope_cos_1BLD = ttnn.reshape(rope_cos_tt, (1, B_rope, L, D))
            rope_sin_1BLD = ttnn.reshape(rope_sin_tt, (1, B_rope, L, D))
        else:
            raise RuntimeError(f"rope expected [B,L,D] or [1,B,L,D], got shape {sh_cos}")
        num_devices = self.parallel_config.sequence_parallel.factor
        padded_len = get_padded_vision_seq_len(L, num_devices)
        if padded_len > L:
            pad_len = padded_len - L
            zeros = ttnn.zeros(
                (1, B_rope, pad_len, D), device=self.mesh_device, dtype=rope_cos_1BLD.dtype, layout=ttnn.TILE_LAYOUT
            )
            rope_cos_1BLD = ttnn.concat([rope_cos_1BLD, zeros], dim=2)
            rope_sin_1BLD = ttnn.concat([rope_sin_1BLD, zeros], dim=2)
        if num_devices == 1:
            tt_rope_cos_1HND = ttnn.typecast(rope_cos_1BLD, ttnn.float32)
            tt_rope_sin_1HND = ttnn.typecast(rope_sin_1BLD, ttnn.float32)
        else:
            # Multi-device SP: shard via from_torch (no in-place shard of device tensor)
            sp_axis = self.parallel_config.sequence_parallel.mesh_axis
            rope_cos_torch = ttnn.to_torch(ttnn.get_device_tensors(rope_cos_1BLD)[0])
            rope_sin_torch = ttnn.to_torch(ttnn.get_device_tensors(rope_sin_1BLD)[0])
            tt_rope_cos_1HND = from_torch(
                rope_cos_torch, device=self.mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None]
            )
            tt_rope_sin_1HND = from_torch(
                rope_sin_torch, device=self.mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None]
            )
        tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=self.mesh_device)
        return tt_rope_cos_1HND, tt_rope_sin_1HND, tt_trans_mat

    def _reference_latent_time_steps_cpu(
        self,
        timestep: torch.Tensor,
        B: int,
        F: int,
        H: int,
        W: int,
        action_mode: bool,
    ) -> torch.Tensor:
        """Match reference `_time_embed` / `repeat_interleave` layout: one scalar timestep per latent token [B, L]."""
        p0, p1, p2 = self.patch_size
        if action_mode:
            mult = H * W
            patch_F = F
        else:
            mult = (H // p1) * (W // p2)
            patch_F = F // p0
        if timestep.ndim == 2:
            assert (
                timestep.shape[0] == B and timestep.shape[1] == patch_F
            ), f"timestep 2D expected ({B}, {patch_F}), got {tuple(timestep.shape)}"
            ts_2d = timestep.float()
        else:
            assert timestep.ndim == 1 and timestep.shape[0] == B
            ts_2d = timestep.unsqueeze(1).expand(B, patch_F).float()
        return torch.repeat_interleave(ts_2d, mult, dim=1)

    def _pad_latent_time_steps_cpu(self, latent_ts: torch.Tensor) -> torch.Tensor:
        """Right-pad timesteps to match SP-padded vision length (replicate last token)."""
        B, L = latent_ts.shape
        L_pad = get_padded_vision_seq_len(L, self.parallel_config.sequence_parallel.factor)
        if L_pad <= L:
            return latent_ts
        tail = latent_ts[:, -1:].expand(B, L_pad - L)
        return torch.cat([latent_ts, tail], dim=1)

    def prepare_text_conditioning(
        self,
        encoder_hidden_states: torch.Tensor | ttnn.Tensor,
        action_mode: bool = False,
    ):
        embedder = self.condition_embedder
        if isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = from_torch(
                encoder_hidden_states,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
            )
        return embedder.forward_text(encoder_hidden_states)

    def prepare_timestep_conditioning(self, latent_time_steps: torch.Tensor, action_mode: bool = False):
        """Match reference `WanTimeTextImageEmbedding.forward`: [B, L] -> temb [1,B,L,D], proj [1,B,L,6,C]."""
        assert latent_time_steps.ndim == 2
        latent_time_steps = latent_time_steps.to(torch.float32)
        B, L = latent_time_steps.shape
        tt_ts = float32_tensor(latent_time_steps.reshape(1, B, L, 1), device=self.mesh_device)
        embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        tt_temb, tt_timestep_proj = embedder.forward_timestep(tt_ts, timestep_seq_len=None)
        tt_timestep_proj = unflatten(tt_timestep_proj, -1, (6, -1))
        return tt_temb, tt_timestep_proj

    def prepare_per_token_conditioning(
        self,
        timestep_per_frame: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        action_mode: bool = False,
        patches_per_frame: int = 1,
        N_padded: int | None = None,
    ):
        """Build per-token timestep conditioning from per-frame timesteps.

        Args:
            timestep_per_frame: [B, F] per-frame timestep values.
            encoder_hidden_states: text prompt tensor.
            action_mode: use action condition embedder.
            patches_per_frame: H_patches * W_patches (spatial patches per frame).
            N_padded: padded sequence length (for SP); defaults to F * patches_per_frame.

        Returns:
            temb_1BND: per-token temb [1, B, N_padded, D] for final norm.
            timestep_proj_1BN6D: per-token conditioning [1, B, N_padded, 6*D_frac] for blocks.
            prompt_1BLP: text conditioning.
        """
        B, F_frames = timestep_per_frame.shape
        N = F_frames * patches_per_frame
        if N_padded is None:
            N_padded = N

        unique_ts = torch.unique(timestep_per_frame)
        ts_to_proj: dict[float, tuple] = {}
        for t_val in unique_ts:
            lt = torch.full((1, 1), float(t_val.item()), dtype=torch.float32)
            temb_i, proj_i = self.prepare_timestep_conditioning(lt, action_mode)
            temb_host = ttnn.to_torch(ttnn.get_device_tensors(temb_i)[0])
            proj_host = ttnn.to_torch(ttnn.get_device_tensors(proj_i)[0])
            ts_to_proj[t_val.item()] = (temb_host, proj_host)

        if not ts_to_proj:
            raise RuntimeError(
                "prepare_per_token_conditioning: empty timestep set "
                f"(timestep_per_frame shape {tuple(timestep_per_frame.shape)}, unique_ts numel={unique_ts.numel()})."
            )
        temb_sample = next(iter(ts_to_proj.values()))
        D_temb = temb_sample[0].shape[-1]
        D_proj6 = temb_sample[1].numel()

        per_token_temb = torch.zeros(1, B, N_padded, D_temb, dtype=torch.float32)
        per_token_proj = torch.zeros(1, B, N_padded, D_proj6, dtype=torch.float32)

        for b_idx in range(B):
            patch_offset = 0
            for f_idx in range(F_frames):
                t_val = timestep_per_frame[b_idx, f_idx].item()
                temb_host, proj_host = ts_to_proj[t_val]
                temb_vec = temb_host.squeeze()[:D_temb]
                proj_vec = proj_host.reshape(-1)[:D_proj6]
                start = patch_offset
                end = patch_offset + patches_per_frame
                per_token_temb[0, b_idx, start:end, :] = temb_vec.unsqueeze(0).expand(patches_per_frame, -1)
                per_token_proj[0, b_idx, start:end, :] = proj_vec.unsqueeze(0).expand(patches_per_frame, -1)
                patch_offset += patches_per_frame

        tt_per_token_temb = float32_tensor(per_token_temb, device=self.mesh_device)
        tt_per_token_proj = float32_tensor(per_token_proj, device=self.mesh_device)
        tt_prompt_1BLP = self.prepare_text_conditioning(encoder_hidden_states, action_mode=action_mode)
        return tt_per_token_temb, tt_per_token_proj, tt_prompt_1BLP

    def prepare_conditioning(
        self,
        latent_time_steps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        action_mode: bool = False,
    ):
        assert latent_time_steps.ndim == 2
        tt_temb_1BLD, tt_timestep_proj = self.prepare_timestep_conditioning(latent_time_steps, action_mode=action_mode)
        tt_prompt_1BLP = self.prepare_text_conditioning(encoder_hidden_states, action_mode=action_mode)
        return tt_temb_1BLD, tt_timestep_proj, tt_prompt_1BLP

    def preprocess_spatial_input_host(self, spatial: torch.Tensor):
        """Patchify video (B, C, F, H, W) -> (1, B, N, C*p0*p1*p2) and pad for sequence parallel.
        Patch vector order must match reference: (C, p0, p1, p2) as in rearrange
        'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)'."""
        B, C, F, H, W = spatial.shape
        logger.debug("Preprocessing spatial input with shape {}", spatial.shape)
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W
        spatial = spatial.reshape(B, C, patch_F, pF, patch_H, pH, patch_W, pW)
        # (B, patch_F, patch_H, patch_W, C, pF, pH, pW) -> patch order (C, p0, p1, p2) to match ref
        spatial = spatial.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(1, B, N, C * pF * pH * pW)
        spatial = pad_vision_seq_parallel(spatial, num_devices=self.parallel_config.sequence_parallel.factor)
        return spatial, N

    def preprocess_spatial_input(self, spatial: torch.Tensor):
        spatial, N = self.preprocess_spatial_input_host(spatial)
        spatial = bf16_tensor(
            spatial,
            device=self.mesh_device,
            mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
            shard_dim=-2,
        )
        return spatial, N

    def preprocess_action_input(self, spatial: torch.Tensor):
        """Rearrange (B, action_dim, F, H, W) -> (1, B, N, action_dim) on device."""
        B, C, F, H, W = spatial.shape
        assert C == self.action_dim
        N = F * H * W
        # (B, C, F, H, W) -> (B, F*H*W, C)
        spatial = spatial.permute(0, 2, 3, 4, 1).reshape(1, B, N, self.action_dim)
        spatial = pad_vision_seq_parallel(spatial, num_devices=self.parallel_config.sequence_parallel.factor)
        spatial = bf16_tensor(
            spatial,
            device=self.mesh_device,
            mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
            shard_dim=-2,
        )
        return spatial, N

    def postprocess_spatial_output_host(self, spatial_1BND: torch.Tensor, F: int, H: int, W: int, N: int):
        """Reverse of preprocess_spatial_input_host. (1, B, N, out) -> (B, out_channels, F, H, W)."""
        assert len(spatial_1BND.shape) == 4 and spatial_1BND.shape[0] == 1
        B = spatial_1BND.shape[1]
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        spatial_BND = spatial_1BND.squeeze(0)
        spatial_BND = spatial_BND[:, :N]
        spatial_patches = spatial_BND.reshape(B, patch_F, patch_H, patch_W, pF, pH, pW, self.out_channels)
        spatial_BCFHW = spatial_patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, self.out_channels, F, H, W)
        return spatial_BCFHW

    def postprocess_spatial_output(self, spatial_1BND: ttnn.Tensor, F: int, H: int, W: int, N: int) -> torch.Tensor:
        spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
            spatial_1BND, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
        )
        spatial_1BND = ttnn.to_torch(ttnn.get_device_tensors(spatial_1BND)[0])
        return self.postprocess_spatial_output_host(spatial_1BND, F, H, W, N).float()

    def postprocess_action_output(self, spatial_1BND: ttnn.Tensor, N: int) -> torch.Tensor:
        """Gather SP and return (B, N, action_dim)."""
        spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
            spatial_1BND, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
        )
        out = ttnn.to_torch(ttnn.get_device_tensors(spatial_1BND)[0])
        out = out[0, :, :N, :]  # (1, B, N, action_dim) -> (B, N, action_dim)
        return out.float()

    def forward(
        self,
        spatial: torch.Tensor,
        prompt: torch.Tensor,
        timestep: torch.Tensor,
        grid_id: torch.Tensor,
        action_mode: bool = False,
        update_cache: int = 0,
        cache_name: str = "pos",
        timestep_per_frame: torch.Tensor | None = None,
        dump_iter: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            spatial: Video path (B, in_channels, F, H, W) or action path (B, action_dim, F, H, W).
            prompt: Encoder hidden states (text).
            timestep: 1D timestep tensor (used when timestep_per_frame is None).
            action_mode: If True, use action embedder and action_proj_out.
            timestep_per_frame: Optional [B, F] per-frame timesteps for per-token conditioning.

        Returns:
            Video path: (B, out_channels, F, H, W). Action path: (B, N, action_dim).
        """
        B, C, F, H, W = spatial.shape
        pF, pH, pW = self.patch_size
        if action_mode:
            patch_F, patch_H, patch_W = F, H, W
        else:
            patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W

        rope_cos_1HND, rope_sin_1HND, trans_mat = self.get_rope_features(grid_id)

        use_per_token = timestep_per_frame is not None
        if use_per_token:
            patches_per_frame = patch_H * patch_W
            N_padded = get_padded_vision_seq_len(N, self.parallel_config.sequence_parallel.factor)
            temb_1BND, timestep_proj_1BN6D, prompt_1BLP = self.prepare_per_token_conditioning(
                timestep_per_frame,
                prompt,
                action_mode=action_mode,
                patches_per_frame=patches_per_frame,
                N_padded=N_padded,
            )
        else:
            latent_ts = self._reference_latent_time_steps_cpu(timestep, B, F, H, W, action_mode)
            latent_ts = self._pad_latent_time_steps_cpu(latent_ts)
            temb_11BD, timestep_proj_1BTD, prompt_1BLP = self.prepare_conditioning(
                latent_ts, prompt, action_mode=action_mode
            )

        if action_mode:
            spatial_1BNI, N = self.preprocess_action_input(spatial)
            spatial_1BND = self.action_embedder(
                spatial_1BNI,
                compute_kernel_config=self.hifi4_compute_kernel_config,
                dtype=ttnn.bfloat16,
            )
        else:
            spatial_1BNI, N = self.preprocess_spatial_input(spatial)
            spatial_1BND = self.patch_embedding(spatial_1BNI)

        caches_list = self._attn_caches.get(cache_name)
        use_cache = caches_list is not None and len(caches_list) == len(self.blocks)
        caches = caches_list if use_cache else None

        block_temb = timestep_proj_1BN6D if use_per_token else timestep_proj_1BTD
        dump_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tests", "demo", "out_inference"))
        dump_path = "action" if action_mode else "video"
        suffix = f"_iter{dump_iter}_{dump_path}" if dump_iter is not None else ""
        os.makedirs(dump_dir, exist_ok=True)

        def _to_torch_cpu(tt_tensor: ttnn.Tensor) -> torch.Tensor:
            return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0]).detach().cpu().float()

        def _spatial_to_bnc(tt_tensor: ttnn.Tensor) -> torch.Tensor:
            t = _to_torch_cpu(tt_tensor)
            return t[0, :, :N, :].contiguous()

        if dump_iter is not None:
            torch.save(_spatial_to_bnc(spatial_1BND), os.path.join(dump_dir, f"input_tt{suffix}.pt"))
            torch.save(
                _to_torch_cpu(prompt_1BLP)[0],
                os.path.join(dump_dir, f"text_hidden_states_tt{suffix}.pt"),
            )
            if use_per_token:
                torch.save(
                    _to_torch_cpu(temb_1BND)[0, :, :N, :].contiguous(),
                    os.path.join(dump_dir, f"temb_tt{suffix}.pt"),
                )
                tproj = _to_torch_cpu(block_temb)[0, :, :N, :]
                tproj = tproj.reshape(B, N, 6, -1).contiguous()
                torch.save(tproj, os.path.join(dump_dir, f"timestep_proj_tt{suffix}.pt"))
            else:
                torch.save(
                    _to_torch_cpu(temb_11BD)[0, :, :N, :].contiguous(),
                    os.path.join(dump_dir, f"temb_tt{suffix}.pt"),
                )
                tproj = _to_torch_cpu(block_temb)[0, :, :N, :, :].contiguous()
                tproj = tproj.reshape(B, N, 6, -1).contiguous()
                torch.save(tproj, os.path.join(dump_dir, f"timestep_proj_tt{suffix}.pt"))

        for block_idx, block in enumerate(self.blocks):
            cached_k_tt = None
            cached_v_tt = None
            block_has_cache = (
                use_cache and caches is not None and block_idx < len(caches) and caches[block_idx] is not None
            )
            return_kv = block_has_cache

            if return_kv and caches[block_idx]["mask"].any():
                cache = caches[block_idx]
                valid = cache["mask"].nonzero(as_tuple=False).squeeze(-1)
                if valid.dim() == 0:
                    valid = valid.unsqueeze(0)
                ext_k = cache["k"][:, valid].contiguous()
                ext_v = cache["v"][:, valid].contiguous()
                ext_k = ext_k.permute(0, 2, 1, 3).contiguous()
                ext_v = ext_v.permute(0, 2, 1, 3).contiguous()
                cached_k_tt = float32_tensor(ext_k.float(), device=self.mesh_device)
                cached_v_tt = float32_tensor(ext_v.float(), device=self.mesh_device)

            if return_kv:
                attn1_out = block(
                    spatial_1BND=spatial_1BND,
                    prompt_1BLP=prompt_1BLP,
                    temb_1BTD=block_temb,
                    N=N,
                    rope_cos=rope_cos_1HND,
                    rope_sin=rope_sin_1HND,
                    trans_mat=trans_mat,
                    update_cache=update_cache,
                    cache_name=cache_name,
                    cached_k=cached_k_tt,
                    cached_v=cached_v_tt,
                    return_kv=True,
                )
                spatial_1BND, k_cur, v_cur = attn1_out
                k_t = ttnn.to_torch(ttnn.get_device_tensors(k_cur)[0])
                v_t = ttnn.to_torch(ttnn.get_device_tensors(v_cur)[0])
                while k_t.dim() > 4:
                    k_t = k_t.squeeze(0)
                    v_t = v_t.squeeze(0)
                k_t = k_t.permute(0, 2, 1, 3).contiguous()
                v_t = v_t.permute(0, 2, 1, 3).contiguous()
                cache = caches[block_idx]
                cache_dtype = cache["k"].dtype
                k_t = k_t.cpu().to(dtype=cache_dtype)
                v_t = v_t.cpu().to(dtype=cache_dtype)
                slots = self._cache_update(cache, k_t, v_t, is_pred=(update_cache == 1))
                if update_cache == 0:
                    self._cache_restore(cache, slots)
                if dump_iter is not None:
                    torch.save(
                        _spatial_to_bnc(attn1_out[0]),
                        os.path.join(dump_dir, f"transformers_tt_{block_idx}{suffix}.pt"),
                    )
            else:
                spatial_1BND = block(
                    spatial_1BND=spatial_1BND,
                    prompt_1BLP=prompt_1BLP,
                    temb_1BTD=block_temb,
                    N=N,
                    rope_cos=rope_cos_1HND,
                    rope_sin=rope_sin_1HND,
                    trans_mat=trans_mat,
                    update_cache=update_cache,
                    cache_name=cache_name,
                )
                if dump_iter is not None:
                    torch.save(
                        _spatial_to_bnc(spatial_1BND),
                        os.path.join(dump_dir, f"transformers_tt_{block_idx}{suffix}.pt"),
                    )

        if use_per_token:
            sst_shift, sst_scale = ttnn.chunk(self.scale_shift_table.data, 2, dim=-2)
            shift_norm = sst_shift + temb_1BND
            scale_norm = sst_scale + temb_1BND
            spatial_norm_1BND = self.norm_out(spatial_1BND, dtype=ttnn.float32)
            spatial_norm_1BND = spatial_norm_1BND * (1 + scale_norm) + shift_norm
        else:
            # Reference: scale_shift_table[None] + temb[:, :, None, :] -> [B, L, 2, D]
            sst = self.scale_shift_table.data
            sst_5d = ttnn.reshape(sst, (1, 1, 1, 2, sst.shape[-1]))
            temb_5d = ttnn.unsqueeze(temb_11BD, 3)
            scale_shift_1BSD = sst_5d + temb_5d
            shift_norm, scale_norm = ttnn.chunk(scale_shift_1BSD, 2, dim=3)
            shift_norm = ttnn.squeeze(shift_norm, dim=3)
            scale_norm = ttnn.squeeze(scale_norm, dim=3)
            # Per-token [1,1,L,D] cannot be passed as dit_layernorm dynamic_weight (gamma must be TILE with height 32).
            spatial_norm_1BND = self.norm_out(spatial_1BND, dtype=ttnn.float32)
            shift_norm = _align_modulation_to_spatial_layout(spatial_norm_1BND, shift_norm)
            scale_norm = _align_modulation_to_spatial_layout(spatial_norm_1BND, scale_norm)
            spatial_norm_1BND = spatial_norm_1BND * ttnn.typecast(1.0 + scale_norm, ttnn.float32) + ttnn.typecast(
                shift_norm, ttnn.float32
            )
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_norm_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_norm_1BND,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )
        if dump_iter is not None:
            torch.save(
                _spatial_to_bnc(spatial_norm_1BND),
                os.path.join(dump_dir, f"norm_out_tt{suffix}.pt"),
            )

        if action_mode:
            proj_out_1BNA = self.action_proj_out(
                spatial_norm_1BND,
                compute_kernel_config=self.hifi4_compute_kernel_config,
                dtype=ttnn.float32,
            )
            out_action = self.postprocess_action_output(proj_out_1BNA, N)
            if dump_iter is not None:
                torch.save(out_action.detach().cpu(), os.path.join(dump_dir, f"final_out_action_tt{suffix}.pt"))
            return out_action
        else:
            proj_out_1BNI = self.proj_out(
                spatial_norm_1BND,
                compute_kernel_config=self.hifi4_compute_kernel_config,
                dtype=ttnn.float32,
            )
            if dump_iter is not None:
                torch.save(
                    _spatial_to_bnc(proj_out_1BNI),
                    os.path.join(dump_dir, f"final_pre_rearrange_tt{suffix}.pt"),
                )
            out_video = self.postprocess_spatial_output(proj_out_1BNI, F, H, W, N)
            if dump_iter is not None:
                torch.save(out_video.detach().cpu(), os.path.join(dump_dir, f"final_out_video_tt{suffix}.pt"))
            return out_video
