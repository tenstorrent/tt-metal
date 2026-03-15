# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN WanTransformer3DModel for Lingbot-VA.

Reuses WanPatchEmbed, WanTimeTextImageEmbedding, and utilities from models.tt_dit.
Defines WanTransformerBlock and WanAttention locally (no wan2_2 dependency).
Adds Lingbot-VA-specific: in_channels=48, action_embedder, condition_embedder_action,
action_proj_out, and dual forward paths (video + action).
"""

from __future__ import annotations

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
            activation_fn="gelu",
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

        self.ff_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
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
        cached_k: ttnn.Tensor | None = None,
        cached_v: ttnn.Tensor | None = None,
        return_kv: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        spatial_1BND: fractured N on SP, fractured D on TP
        prompt_1BLP: replicated on SP, replicated D on TP
        temb_1BTD: replicated on SP, fractured D on TP
        N: logical sequence length of the spatial input
        rope_cos_BANH: fractured N on SP, A (num_heads) on TP
        rope_sin_BANH: fractured N on SP, A (num_heads) on TP
        trans_mat: replicated on SP, replicated D on TP

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        """

        assert temb_1BTD.shape[2] == 6, "wan expects 6 chunks in timestep embedding"

        shifted_temb_1BTD = self.scale_shift_table.data + temb_1BTD
        shift_msa_1B1D, scale_msa_1B1D, gate_msa_1B1D, c_shift_msa_1B1D, c_scale_msa_1B1D, c_gate_msa_1B1D = ttnn.chunk(
            shifted_temb_1BTD, 6, dim=2
        )
        # addcmul gate in bfloat16 for numerical accuracy
        gate_msa_1B1D = ttnn.typecast(gate_msa_1B1D, dtype=ttnn.bfloat16)
        c_gate_msa_1B1D = ttnn.typecast(c_gate_msa_1B1D, dtype=ttnn.bfloat16)

        spatial_normed_1BND = self.norm1(
            spatial_1BND, dynamic_weight=(1.0 + scale_msa_1B1D), dynamic_bias=shift_msa_1B1D
        )

        # Self attention on spatial with fused residual addcmul
        attn1_out = self.attn1(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
            addcmul_residual=spatial_1BND,
            addcmul_gate=gate_msa_1B1D,
            cached_k=cached_k,
            cached_v=cached_v,
            return_kv=return_kv,
        )
        if return_kv:
            spatial_1BND, k_cur, v_cur = attn1_out
        else:
            spatial_1BND = attn1_out

        # Cross attention on prompt
        spatial_normed_1BND = self.norm2(spatial_1BND)

        attn_output_1BND = self.attn2(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            prompt_1BLP=prompt_1BLP,
        )
        spatial_1BND = spatial_1BND + attn_output_1BND

        # Feed Forward
        spatial_normed_1BND = self.norm3(
            spatial_1BND, dynamic_weight=(1.0 + c_scale_msa_1B1D), dynamic_bias=c_shift_msa_1B1D
        )

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_normed_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_normed_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        spatial_ff_1BND = self.ffn(spatial_normed_1BND, compute_kernel_config=self.ff_compute_kernel_config)

        spatial_1BND = ttnn.addcmul(spatial_1BND, spatial_ff_1BND, c_gate_msa_1B1D)

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
        if cache_name in self._attn_caches:
            self._attn_caches[cache_name] = [None] * len(self.blocks)

    def clear_pred_cache(self, cache_name: str) -> None:
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
        if patch_bias is not None and patch_bias.ndim == 1:
            state["patch_embedding.bias"] = patch_bias.reshape(1, -1)

    def get_rope_features(self, grid_id: torch.Tensor):
        """Build RoPE cos/sin and transformation matrix from grid_id (B, 3, L)."""
        if tuple(grid_id.shape) not in self.cached_rope_features:
            rope_features = self.prepare_rope_features(grid_id)
            self.cached_rope_features[tuple(grid_id.shape)] = rope_features
        return self.cached_rope_features[tuple(grid_id.shape)]

    def prepare_rope_features(self, grid_id: torch.Tensor):
        """Build RoPE cos/sin and transformation matrix from grid_id (B, 3, L). Pads for sequence parallel."""
        logger.debug("Preparing rope features for shape {}", grid_id.shape)
        grid_id_tt = bf16_tensor(grid_id, device=self.mesh_device)
        rope_cos_tt, rope_sin_tt = self.rope(grid_id_tt)  # each [1, L, 128]
        B, L, D = rope_cos_tt.shape
        rope_cos_11LD = ttnn.reshape(rope_cos_tt, (1, 1, L, D))
        rope_sin_11LD = ttnn.reshape(rope_sin_tt, (1, 1, L, D))
        num_devices = self.parallel_config.sequence_parallel.factor
        padded_len = get_padded_vision_seq_len(L, num_devices)
        if padded_len > L:
            pad_len = padded_len - L
            zeros = ttnn.zeros(
                (1, 1, pad_len, D), device=self.mesh_device, dtype=rope_cos_11LD.dtype, layout=ttnn.TILE_LAYOUT
            )
            rope_cos_11LD = ttnn.concat([rope_cos_11LD, zeros], dim=2)
            rope_sin_11LD = ttnn.concat([rope_sin_11LD, zeros], dim=2)
        if num_devices == 1:
            tt_rope_cos_1HND = ttnn.typecast(rope_cos_11LD, ttnn.float32)
            tt_rope_sin_1HND = ttnn.typecast(rope_sin_11LD, ttnn.float32)
        else:
            # Multi-device SP: shard via from_torch (no in-place shard of device tensor)
            sp_axis = self.parallel_config.sequence_parallel.mesh_axis
            rope_cos_torch = ttnn.to_torch(ttnn.get_device_tensors(rope_cos_11LD)[0])
            rope_sin_torch = ttnn.to_torch(ttnn.get_device_tensors(rope_sin_11LD)[0])
            tt_rope_cos_1HND = from_torch(
                rope_cos_torch, device=self.mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None]
            )
            tt_rope_sin_1HND = from_torch(
                rope_sin_torch, device=self.mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None]
            )
        tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=self.mesh_device)
        return tt_rope_cos_1HND, tt_rope_sin_1HND, tt_trans_mat

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

    def prepare_timestep_conditioning(self, timestep: torch.Tensor, action_mode: bool = False):
        assert timestep.ndim == 1
        timestep = float32_tensor(timestep.unsqueeze(1).unsqueeze(1).unsqueeze(1), device=self.mesh_device)
        embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        tt_temb_11BD, tt_timestep_proj_1BTD = embedder.forward_timestep(timestep, timestep_seq_len=None)
        tt_timestep_proj_1BTD = unflatten(ttnn.squeeze(tt_timestep_proj_1BTD, -2), -1, (6, -1))
        return tt_temb_11BD, tt_timestep_proj_1BTD

    def prepare_conditioning(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        action_mode: bool = False,
    ):
        assert timestep.ndim == 1
        tt_temb_11BD, tt_timestep_proj_1BTD = self.prepare_timestep_conditioning(timestep, action_mode=action_mode)
        tt_prompt_1BLP = self.prepare_text_conditioning(encoder_hidden_states, action_mode=action_mode)
        return tt_temb_11BD, tt_timestep_proj_1BTD, tt_prompt_1BLP

    def preprocess_spatial_input_host(self, spatial: torch.Tensor):
        """Patchify video (B, C, F, H, W) -> (1, B, N, C*p0*p1*p2) and pad for sequence parallel.
        Patch vector order must match reference: (C, p0, p1, p2) as in rearrange
        'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)'."""
        B, C, F, H, W = spatial.shape
        if B != 1:
            raise NotImplementedError("Batch size > 1 is not currently supported")
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
        return self.postprocess_spatial_output_host(spatial_1BND, F, H, W, N)

    def postprocess_action_output(self, spatial_1BND: ttnn.Tensor, N: int) -> torch.Tensor:
        """Gather SP and return (B, N, action_dim)."""
        spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
            spatial_1BND, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
        )
        out = ttnn.to_torch(ttnn.get_device_tensors(spatial_1BND)[0])
        out = out[0, :, :N, :]  # (1, B, N, action_dim) -> (B, N, action_dim)
        return out

    def forward(
        self,
        spatial: torch.Tensor,
        prompt: torch.Tensor,
        timestep: torch.Tensor,
        grid_id: torch.Tensor,
        action_mode: bool = False,
        update_cache: int = 0,
        cache_name: str = "pos",
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            spatial: Video path (B, in_channels, F, H, W) or action path (B, action_dim, F, H, W).
            prompt: Encoder hidden states (text).
            timestep: 1D timestep tensor.
            action_mode: If True, use action embedder and action_proj_out.

        Returns:
            Video path: (B, out_channels, F, H, W). Action path: (B, N, action_dim).
        """
        B, C, F, H, W = spatial.shape
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W

        rope_cos_1HND, rope_sin_1HND, trans_mat = self.get_rope_features(grid_id)
        temb_11BD, timestep_proj_1BTD, prompt_1BLP = self.prepare_conditioning(
            timestep, prompt, action_mode=action_mode
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

        use_cache = (
            cache_name in self._attn_caches
            and self._attn_caches[cache_name] is not None
            and self.parallel_config.sequence_parallel.factor == 1
            and self.parallel_config.tensor_parallel.factor == 1
        )
        caches = self._attn_caches.get(cache_name) if use_cache else None

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
                cache_dtype = cache["k"].dtype
                ext_k = ext_k.permute(0, 2, 1, 3).contiguous().to(dtype=cache_dtype)
                ext_v = ext_v.permute(0, 2, 1, 3).contiguous().to(dtype=cache_dtype)
                cached_k_tt = bf16_tensor(ext_k.to(torch.bfloat16), device=self.mesh_device)
                cached_v_tt = bf16_tensor(ext_v.to(torch.bfloat16), device=self.mesh_device)

            if return_kv:
                attn1_out = block(
                    spatial_1BND=spatial_1BND,
                    prompt_1BLP=prompt_1BLP,
                    temb_1BTD=timestep_proj_1BTD,
                    N=N,
                    rope_cos=rope_cos_1HND,
                    rope_sin=rope_sin_1HND,
                    trans_mat=trans_mat,
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
            else:
                spatial_1BND = block(
                    spatial_1BND=spatial_1BND,
                    prompt_1BLP=prompt_1BLP,
                    temb_1BTD=timestep_proj_1BTD,
                    N=N,
                    rope_cos=rope_cos_1HND,
                    rope_sin=rope_sin_1HND,
                    trans_mat=trans_mat,
                )

        scale_shift_1BSD = self.scale_shift_table.data + temb_11BD
        shift_11BD, scale_11BD = ttnn.chunk(scale_shift_1BSD, 2, -2)
        spatial_norm_1BND = self.norm_out(
            spatial_1BND,
            dynamic_weight=(1 + scale_11BD),
            dynamic_bias=shift_11BD,
            dtype=ttnn.float32,
        )
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_norm_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_norm_1BND,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            )

        if action_mode:
            proj_out_1BNA = self.action_proj_out(
                spatial_norm_1BND,
                compute_kernel_config=self.hifi4_compute_kernel_config,
                dtype=ttnn.float32,
            )
            return self.postprocess_action_output(proj_out_1BNA, N)
        else:
            proj_out_1BNI = self.proj_out(
                spatial_norm_1BND,
                compute_kernel_config=self.hifi4_compute_kernel_config,
                dtype=ttnn.float32,
            )
            return self.postprocess_spatial_output(proj_out_1BNI, F, H, W, N)
