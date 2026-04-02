# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN WanTransformer3DModel for Lingbot-VA.

Reuses WanPatchEmbed, WanTimeTextImageEmbedding, and utilities from models.tt_dit.
``WanTransformerBlock`` uses on-device ``attention_wan.WanAttention`` (same algorithm as
``reference.transformer_wan.WanAttention``, without PyTorch attention round-trips).
Adds Lingbot-VA-specific: in_channels=48, action_embedder, condition_embedder_action,
action_proj_out, and dual forward paths (video + action).

Forward pass: hidden-state math uses ``ttnn.add`` / ``ttnn.multiply`` / ``ttnn.addcmul`` (no PyTorch
``+`` / ``*`` on activations). ``forward()`` and the public preprocess/conditioning helpers take and return
``ttnn.Tensor`` on the mesh. Per-token timestep conditioning uses ``numpy.unique`` on a small host view
(read back from ``ttnn``) plus ``ttnn.full`` / ``repeat`` / ``concat``. RoPE uses ``grid_id`` as ``ttnn``.
Weight load still uses ``_prepare_torch_state`` (``torch.Tensor`` state dict from checkpoints). Chunked self-attention KV is
stored as **device-resident** ``ttnn`` tensors per layer: a list of
``(k, v)`` segments (speecht5-style concat prefix). When ``update_cache == 1`` the current step’s
K/V is appended; when ``update_cache == 0`` it is ephemeral (no host gather/scatter).
"""

from __future__ import annotations

import gc
import numpy as np
import torch
from loguru import logger

import ttnn

from models.tt_dit.layers.embeddings import WanPatchEmbed, WanTimeTextImageEmbedding
from models.tt_dit.layers.feedforward import ParallelFeedForward
from models.tt_dit.layers.linear import ColParallelLinear, Linear
from models.tt_dit.layers.module import Module, ModuleList, Parameter
from models.tt_dit.layers.normalization import DistributedLayerNorm
from models.tt_dit.parallel.config import DiTParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.padding import get_padded_vision_seq_len
from models.tt_dit.utils.substate import pop_substate, rename_substate
from models.tt_dit.utils.tensor import unflatten

from .attention_wan import WanAttention
from .wan_rotary_pos_embed import WanRotaryPosEmbed


def _pad_vision_seq_parallel_ttnn(x: ttnn.Tensor, num_devices: int, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Right-pad sequence dim (index 2) to match ``pad_vision_seq_parallel`` / SP divisibility."""
    seq_len = int(x.shape[2])
    padded_seq_len = get_padded_vision_seq_len(seq_len, num_devices)
    pad_len = padded_seq_len - seq_len
    if pad_len <= 0:
        return x
    _, b, _, c = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    zeros = ttnn.zeros(
        (1, b, pad_len, c),
        device=mesh_device,
        dtype=x.dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    return ttnn.concat([x, zeros], dim=2)


def _safe_deallocate_tensor(tensor: ttnn.Tensor | None, label: str = "") -> None:
    if tensor is None:
        return
    try:
        ttnn.deallocate(tensor)
    except Exception as e:
        logger.warning("Failed to deallocate{}: {}", f" {label}" if label else "", e)


# Lingbot-VA config (from reference model.py)
DIM = 3072  # num_attention_heads * attention_head_dim
FFN_DIM = 14336
NUM_HEADS = 24
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


def _rope_transformation_matrix_np() -> np.ndarray:
    """Host float32 pattern for RoPE matmul (same layout as ``get_rot_transformation_mat`` in tt_dit mochi)."""
    dhead = 32
    m = np.zeros((1, 1, dhead, dhead), dtype=np.float32)
    m[0, 0, np.arange(0, dhead, 2), np.arange(1, dhead, 2)] = 1.0
    m[0, 0, np.arange(1, dhead, 2), np.arange(0, dhead, 2)] = -1.0
    return m


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
            qk_norm=True,
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
            qk_norm=True,
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
        # attn1 / attn2 are TT ``WanAttention`` modules: weights load via child ``_prepare_torch_state`` + Module loader.

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
        temb_1BTD: [1, B, 6, D] global conditioning OR [1, B, N_padded, 6*D] per-token conditioning
        N: logical sequence length of the spatial input
        rope_cos_BANH: fractured N on SP, A (num_heads) on TP
        rope_sin_BANH: fractured N on SP, A (num_heads) on TP
        trans_mat: replicated on SP, replicated D on TP

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        """

        T_dim = temb_1BTD.shape[2]
        if T_dim == 6:
            shifted_temb_1BTD = self.scale_shift_table.data + temb_1BTD
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = ttnn.chunk(
                shifted_temb_1BTD, 6, dim=2
            )
        else:
            sst = self.scale_shift_table.data
            sst_flat = ttnn.reshape(sst, (1, 1, 1, sst.shape[-2] * sst.shape[-1]))
            shifted = temb_1BTD + sst_flat
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = ttnn.chunk(shifted, 6, dim=-1)

        gate_msa = ttnn.typecast(gate_msa, dtype=ttnn.bfloat16)
        c_gate_msa = ttnn.typecast(c_gate_msa, dtype=ttnn.bfloat16)

        per_token = T_dim != 6
        if per_token:
            spatial_normed_1BND = self.norm1(spatial_1BND)
            spatial_normed_1BND = spatial_normed_1BND * ttnn.typecast(
                1.0 + scale_msa, spatial_normed_1BND.dtype
            ) + ttnn.typecast(shift_msa, spatial_normed_1BND.dtype)
        else:
            spatial_normed_1BND = self.norm1(spatial_1BND, dynamic_weight=(1.0 + scale_msa), dynamic_bias=shift_msa)

        if return_kv:
            attn_out = self.attn1(
                spatial_normed_1BND,
                N,
                prompt_1BLP=None,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                trans_mat=trans_mat,
                cached_k=cached_k,
                cached_v=cached_v,
                return_kv=True,
            )
            attn_output, k_cur, v_cur = attn_out
        else:
            attn_output = self.attn1(
                spatial_normed_1BND,
                N,
                prompt_1BLP=None,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                trans_mat=trans_mat,
                cached_k=cached_k,
                cached_v=cached_v,
                return_kv=False,
            )
        spatial_1BND = ttnn.addcmul(spatial_1BND, attn_output, gate_msa)

        spatial_normed_1BND = self.norm2(spatial_1BND) if self.norm2 is not None else spatial_1BND
        attn_output_1BND = self.attn2(
            spatial_normed_1BND,
            N,
            prompt_1BLP=prompt_1BLP,
            rope_cos=None,
            rope_sin=None,
            trans_mat=None,
        )
        spatial_1BND = ttnn.add(spatial_1BND, attn_output_1BND)

        # 3) Feed Forward
        if per_token:
            spatial_normed_1BND = self.norm3(spatial_1BND)
            spatial_normed_1BND = spatial_normed_1BND * ttnn.typecast(
                1.0 + c_scale_msa, spatial_normed_1BND.dtype
            ) + ttnn.typecast(c_shift_msa, spatial_normed_1BND.dtype)
        else:
            spatial_normed_1BND = self.norm3(spatial_1BND, dynamic_weight=(1.0 + c_scale_msa), dynamic_bias=c_shift_msa)

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_normed_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_normed_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        spatial_ff_1BND = self.ffn(spatial_normed_1BND, compute_kernel_config=self.ff_compute_kernel_config)

        spatial_1BND = ttnn.addcmul(spatial_1BND, spatial_ff_1BND, c_gate_msa)

        if return_kv:
            return (spatial_1BND, k_cur, v_cur)
        return spatial_1BND


class WanTransformer3DModel(Module):
    """
    TTNN WanTransformer3DModel for Lingbot-VA.

    ``forward`` and the ``preprocess_*`` / ``postprocess_*`` / ``prepare_*`` helpers use ``ttnn.Tensor``
    on the mesh. Checkpoint load still goes through ``_prepare_torch_state`` / ``load_torch_state_dict``.

    - Video: ``spatial`` ``(B, in_channels, F, H, W)`` → … → ``(B, out_channels, F, H, W)``.
    - Action: ``spatial`` ``(B, action_dim, F, H, W)`` → … → ``(B, N, action_dim)``.
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
        self._rope_static_trans_mat_bf16 = ttnn.from_torch(
            _rope_transformation_matrix_np(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh_device,
        )

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

        # Column-parallel on TP (same as WanPatchEmbed proj_weight) so hidden dim matches blocks / norm1.
        self.action_embedder = ColParallelLinear(
            action_dim,
            dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
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
        # Per cache_name: per-layer list that is either None (after clear_cache) or a list of
        # {"k","v","is_pred"} segments (ttnn tensors on mesh_device).
        self._attn_caches: dict[str, list[list[dict] | None]] = {}

    def cleanup_all(self) -> None:
        """Release cached TTNN tensors and attention cache state (nanobind / device memory hygiene)."""
        try:
            ttnn.synchronize_device(self.mesh_device)
        except Exception as e:
            logger.warning("cleanup_all: synchronize_device failed: {}", e)

        for cache_name in list(self._attn_caches.keys()):
            try:
                self.clear_cache(cache_name)
            except Exception as e:
                logger.warning("cleanup_all: clear_cache({}) failed: {}", cache_name, e)
        self._attn_caches.clear()

        for key, rope_pair in list(self.cached_rope_features.items()):
            try:
                cos_t, sin_t = rope_pair
                _safe_deallocate_tensor(cos_t, "cached rope cos")
                _safe_deallocate_tensor(sin_t, "cached rope sin")
            except Exception as e:
                logger.warning("cleanup_all: rope cache entry {}: {}", key, e)
        self.cached_rope_features.clear()

        _safe_deallocate_tensor(self._rope_static_trans_mat_bf16, "rope static trans_mat")
        self._rope_static_trans_mat_bf16 = None

        gc.collect()

    def __enter__(self) -> WanTransformer3DModel:
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.cleanup_all()

    def clear_cache(self, cache_name: str) -> None:
        if cache_name in self._attn_caches:
            for layer_cache in self._attn_caches[cache_name]:
                if layer_cache is None:
                    continue
                for seg in layer_cache:
                    _safe_deallocate_tensor(seg.get("k"), "clear_cache seg k")
                    _safe_deallocate_tensor(seg.get("v"), "clear_cache seg v")
            self._attn_caches[cache_name] = [None] * len(self.blocks)

    def clear_pred_cache(self, cache_name: str) -> None:
        if cache_name not in self._attn_caches:
            return
        for layer_cache in self._attn_caches[cache_name]:
            if layer_cache is None:
                continue
            kept: list[dict] = []
            for seg in layer_cache:
                if seg.get("is_pred"):
                    _safe_deallocate_tensor(seg.get("k"), "clear_pred_cache seg k")
                    _safe_deallocate_tensor(seg.get("v"), "clear_pred_cache seg v")
                else:
                    kept.append(seg)
            layer_cache[:] = kept

    def create_empty_cache(
        self,
        cache_name: str,
        attn_window: int,
        latent_token_per_chunk: int,
        action_token_per_chunk: int,
        batch_size: int = 1,
        **_: object,
    ) -> None:
        """``device`` / ``dtype`` (and other kwargs) are ignored; kept for demo/reference call compatibility."""
        _ = batch_size, attn_window, latent_token_per_chunk, action_token_per_chunk
        self._attn_caches[cache_name] = [[] for _ in range(len(self.blocks))]

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

    def _rope_grid_cache_key(self, grid_id: ttnn.Tensor) -> tuple:
        """Stable cache key from grid contents (host bytes; one readback)."""
        shape = tuple(int(x) for x in grid_id.shape)
        th = ttnn.to_torch(ttnn.get_device_tensors(grid_id)[0])
        payload = th.contiguous().float().numpy().tobytes()
        return (shape, payload)

    def get_rope_features(self, grid_id: ttnn.Tensor):
        """Build RoPE cos/sin and transformation matrix from ``grid_id`` ``(B, 3, L)`` on the mesh.

        Cache key must include grid *values*, not only shape: multi-chunk generate passes the same
        token layout with different ``frame_st_id`` (temporal offset in ``get_mesh_id``). Shape-only
        keys incorrectly reuse chunk-0 RoPE for all chunks → static-looking decoded video while
        single-chunk PCC vs torch stays high.
        """
        cache_key = self._rope_grid_cache_key(grid_id)
        if cache_key not in self.cached_rope_features:
            rope_cos, rope_sin = self._prepare_rope_cos_sin(grid_id)
            self.cached_rope_features[cache_key] = (rope_cos, rope_sin)
        cos_t, sin_t = self.cached_rope_features[cache_key]
        return cos_t, sin_t, self._rope_static_trans_mat_bf16

    def _prepare_rope_cos_sin(self, grid_id: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Build RoPE cos/sin from ``grid_id`` ``(B, 3, L)``. Pads for sequence parallel."""
        logger.debug("Preparing rope cos/sin for shape {}", grid_id.shape)
        grid_was_intermediate = False
        grid_id_tt = grid_id
        if grid_id_tt.layout != ttnn.TILE_LAYOUT:
            grid_id_tt = ttnn.to_layout(grid_id_tt, ttnn.TILE_LAYOUT)
            grid_was_intermediate = True
        if grid_id_tt.dtype != ttnn.bfloat16:
            grid_id_tt = ttnn.typecast(grid_id_tt, ttnn.bfloat16)
            grid_was_intermediate = True
        rope_cos_tt, rope_sin_tt = self.rope(grid_id_tt)  # each [1, L, 128]
        if grid_was_intermediate:
            _safe_deallocate_tensor(grid_id_tt, "prepare_rope_cos_sin grid_id")
        _l, d = int(rope_cos_tt.shape[1]), int(rope_cos_tt.shape[2])
        rope_cos_11LD = ttnn.reshape(rope_cos_tt, (1, 1, _l, d))
        rope_sin_11LD = ttnn.reshape(rope_sin_tt, (1, 1, _l, d))
        num_devices = self.parallel_config.sequence_parallel.factor
        padded_len = get_padded_vision_seq_len(_l, num_devices)
        if padded_len > _l:
            pad_len = padded_len - _l
            zeros = ttnn.zeros(
                (1, 1, pad_len, d), device=self.mesh_device, dtype=rope_cos_11LD.dtype, layout=ttnn.TILE_LAYOUT
            )
            rope_cos_11LD = ttnn.concat([rope_cos_11LD, zeros], dim=2)
            rope_sin_11LD = ttnn.concat([rope_sin_11LD, zeros], dim=2)
        if num_devices == 1:
            tt_rope_cos_1HND = ttnn.typecast(rope_cos_11LD, ttnn.float32)
            tt_rope_sin_1HND = ttnn.typecast(rope_sin_11LD, ttnn.float32)
        else:
            sp_axis = self.parallel_config.sequence_parallel.mesh_axis
            rc = ttnn.typecast(rope_cos_11LD, ttnn.float32)
            rs = ttnn.typecast(rope_sin_11LD, ttnn.float32)
            tt_rope_cos_1HND = ttnn.mesh_partition(rc, dim=2, cluster_axis=sp_axis)
            tt_rope_sin_1HND = ttnn.mesh_partition(rs, dim=2, cluster_axis=sp_axis)
        return tt_rope_cos_1HND, tt_rope_sin_1HND

    def prepare_text_conditioning(self, encoder_hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        return self.condition_embedder.forward_text(encoder_hidden_states)

    def _forward_timestep_embedder(self, tt_timestep_4d: ttnn.Tensor, action_mode: bool = False):
        embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        tt_temb_11BD, tt_timestep_proj_1BTD = embedder.forward_timestep(tt_timestep_4d, timestep_seq_len=None)
        tt_timestep_proj_1BTD = unflatten(ttnn.squeeze(tt_timestep_proj_1BTD, -2), -1, (6, -1))
        return tt_temb_11BD, tt_timestep_proj_1BTD

    def prepare_timestep_conditioning(self, timestep: ttnn.Tensor, action_mode: bool = False):
        """``timestep``: 1D ``[B]`` on the mesh (float32/bfloat16)."""
        assert len(timestep.shape) == 1
        tbf = ttnn.typecast(timestep, ttnn.float32)
        b = int(tbf.shape[0])
        tt_ts = ttnn.reshape(tbf, (b, 1, 1, 1))
        if tt_ts.layout != ttnn.TILE_LAYOUT:
            tt_ts = ttnn.to_layout(tt_ts, ttnn.TILE_LAYOUT)
        return self._forward_timestep_embedder(tt_ts, action_mode)

    def prepare_per_token_conditioning(
        self,
        timestep_per_frame: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        action_mode: bool = False,
        patches_per_frame: int = 1,
        N_padded: int | None = None,
    ):
        """Build per-token timestep conditioning from per-frame timesteps.

        Args:
            timestep_per_frame: ``[B, F]`` per-frame timestep values on the mesh.
            encoder_hidden_states: text prompt ``ttnn`` tensor.
            action_mode: use action condition embedder.
            patches_per_frame: H_patches * W_patches (spatial patches per frame).
            N_padded: padded sequence length (for SP); defaults to F * patches_per_frame.

        Returns:
            temb_1BND: per-token temb [1, B, N_padded, D] for final norm.
            timestep_proj_1BN6D: per-token conditioning [1, B, N_padded, 6*D_frac] for blocks.
            prompt_1BLP: text conditioning.
        """
        B, F_frames = int(timestep_per_frame.shape[0]), int(timestep_per_frame.shape[1])
        N = F_frames * patches_per_frame
        if N_padded is None:
            N_padded = N

        flat_np = ttnn.to_torch(ttnn.get_device_tensors(timestep_per_frame)[0]).reshape(-1).float().cpu().numpy()
        unique_ts, inverse_flat = np.unique(flat_np, return_inverse=True)
        inverse_bf = inverse_flat.reshape(B, F_frames)
        temb_by_idx: list = []
        proj_by_idx: list = []
        for t_val in unique_ts:
            tt_ts = ttnn.full(
                (1, 1, 1, 1),
                fill_value=float(t_val),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32,
                device=self.mesh_device,
            )
            temb_i, proj_i = self._forward_timestep_embedder(tt_ts, action_mode)
            _safe_deallocate_tensor(tt_ts, "per-token scalar timestep")
            proj_flat = ttnn.reshape(proj_i, (1, 1, 1, -1))
            temb_by_idx.append(temb_i)
            proj_by_idx.append(proj_flat)

        batch_temb_rows: list = []
        batch_proj_rows: list = []
        for b_idx in range(B):
            frame_temb: list = []
            frame_proj: list = []
            for f_idx in range(F_frames):
                u = int(inverse_bf[b_idx, f_idx])
                temb_u = temb_by_idx[u]
                proj_u = proj_by_idx[u]
                frame_temb.append(ttnn.repeat(temb_u, (1, 1, patches_per_frame, 1)))
                frame_proj.append(ttnn.repeat(proj_u, (1, 1, patches_per_frame, 1)))
            row_temb = ttnn.concat(frame_temb, dim=2)
            row_proj = ttnn.concat(frame_proj, dim=2)
            for t in frame_temb:
                _safe_deallocate_tensor(t, "per-token frame temb")
            for t in frame_proj:
                _safe_deallocate_tensor(t, "per-token frame proj")
            batch_temb_rows.append(row_temb)
            batch_proj_rows.append(row_proj)

        if B > 1:
            tt_per_token_temb = ttnn.concat(batch_temb_rows, dim=1)
            tt_per_token_proj = ttnn.concat(batch_proj_rows, dim=1)
            for t in batch_temb_rows:
                _safe_deallocate_tensor(t, "per-token batch temb row")
            for t in batch_proj_rows:
                _safe_deallocate_tensor(t, "per-token batch proj row")
        else:
            tt_per_token_temb = batch_temb_rows[0]
            tt_per_token_proj = batch_proj_rows[0]

        if N_padded > N:
            _, _, _, D_temb = tt_per_token_temb.shape
            _, _, _, D_proj = tt_per_token_proj.shape
            pad_len = N_padded - N
            z_temb = ttnn.zeros(
                (1, B, pad_len, D_temb),
                device=self.mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
            )
            z_proj = ttnn.zeros(
                (1, B, pad_len, D_proj),
                device=self.mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
            )
            tt_per_token_temb = ttnn.concat([tt_per_token_temb, z_temb], dim=2)
            tt_per_token_proj = ttnn.concat([tt_per_token_proj, z_proj], dim=2)
            _safe_deallocate_tensor(z_temb, "per-token pad temb")
            _safe_deallocate_tensor(z_proj, "per-token pad proj")

        for t in temb_by_idx:
            _safe_deallocate_tensor(t, "per-token unique temb")
        for t in proj_by_idx:
            _safe_deallocate_tensor(t, "per-token unique proj")

        tt_prompt_1BLP = self.prepare_text_conditioning(encoder_hidden_states)
        return tt_per_token_temb, tt_per_token_proj, tt_prompt_1BLP

    def prepare_conditioning(
        self,
        timestep: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        action_mode: bool = False,
    ):
        assert len(timestep.shape) == 1
        tt_temb_11BD, tt_timestep_proj_1BTD = self.prepare_timestep_conditioning(timestep, action_mode=action_mode)
        tt_prompt_1BLP = self.prepare_text_conditioning(encoder_hidden_states)
        return tt_temb_11BD, tt_timestep_proj_1BTD, tt_prompt_1BLP

    def _preprocess_spatial_tokens_ttnn(self, spatial: ttnn.Tensor) -> tuple[ttnn.Tensor, int]:
        """Patchify + SP pad on device: (B,C,F,H,W) -> (1,B,N,in_dim), TILE."""
        b, c, f, h, w = (int(spatial.shape[i]) for i in range(5))
        if b != 1:
            raise NotImplementedError("Batch size > 1 is not currently supported")
        logger.debug("Preprocessing spatial ttnn input with shape {}", spatial.shape)
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = f // pF, h // pH, w // pW
        n = patch_F * patch_H * patch_W
        x = ttnn.reshape(spatial, (b, c, patch_F, pF, patch_H, pH, patch_W, pW))
        x = ttnn.permute(x, (0, 2, 4, 6, 1, 3, 5, 7))
        x = ttnn.reshape(x, (1, b, n, c * pF * pH * pW))
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        return _pad_vision_seq_parallel_ttnn(x, self.parallel_config.sequence_parallel.factor, self.mesh_device), n

    def _preprocess_action_tokens_ttnn(self, spatial: ttnn.Tensor) -> tuple[ttnn.Tensor, int]:
        """(B, action_dim, F, H, W) -> (1, B, N, action_dim) + SP pad on device."""
        b, c, f, h, w = (int(spatial.shape[i]) for i in range(5))
        assert c == self.action_dim
        n = f * h * w
        x = ttnn.permute(spatial, (0, 2, 3, 4, 1))
        x = ttnn.reshape(x, (1, b, n, self.action_dim))
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        return _pad_vision_seq_parallel_ttnn(x, self.parallel_config.sequence_parallel.factor, self.mesh_device), n

    def _tokens_to_sharded_bfloat16(self, x_1bnd: ttnn.Tensor) -> ttnn.Tensor:
        """bf16 + TILE + sequence-parallel shard on dim ``N`` (index 2), matching embedder layout."""
        x = ttnn.typecast(x_1bnd, ttnn.bfloat16)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        sp_factor = int(self.parallel_config.sequence_parallel.factor)
        if sp_factor <= 1:
            return x
        mesh_axis = self.parallel_config.sequence_parallel.mesh_axis
        mapper_dims: list = [None, None]
        mapper_dims[mesh_axis] = -2
        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=mapper_dims
        )
        host = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).to(dtype=torch.bfloat16).contiguous()
        return ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def preprocess_spatial_input(self, spatial: ttnn.Tensor) -> tuple[ttnn.Tensor, int]:
        """``spatial``: ``(B, C, F, H, W)`` video on the mesh (tile)."""
        x_tt, n = self._preprocess_spatial_tokens_ttnn(spatial)
        return self._tokens_to_sharded_bfloat16(x_tt), n

    def preprocess_action_input(self, spatial: ttnn.Tensor) -> tuple[ttnn.Tensor, int]:
        """``spatial``: ``(B, action_dim, F, H, W)`` on the mesh."""
        x_tt, n = self._preprocess_action_tokens_ttnn(spatial)
        return self._tokens_to_sharded_bfloat16(x_tt), n

    def _postprocess_spatial_output_device(
        self, spatial_1BND: ttnn.Tensor, F: int, H: int, W: int, N: int
    ) -> ttnn.Tensor:
        """Gather sequence-parallel shards and unpatchify on device: (1,B,N_pad,C) -> (B,C,F,H,W)."""
        x = self.ccl_manager.all_gather_persistent_buffer(
            spatial_1BND, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
        )
        _, B, _seq, C = x.shape
        x = ttnn.slice(x, [0, 0, 0, 0], [1, B, N, C])
        x = ttnn.squeeze(x, 0)
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        x = ttnn.reshape(x, (B, patch_F, patch_H, patch_W, pF, pH, pW, self.out_channels))
        x = ttnn.permute(x, (0, 7, 1, 4, 2, 5, 3, 6))
        return ttnn.reshape(x, (B, self.out_channels, F, H, W))

    def postprocess_spatial_output(self, spatial_1BND: ttnn.Tensor, F: int, H: int, W: int, N: int) -> ttnn.Tensor:
        """Unpatchify on device → ``(B, out_channels, F, H, W)`` on the mesh."""
        return self._postprocess_spatial_output_device(spatial_1BND, F, H, W, N)

    def postprocess_action_output(self, spatial_1BND: ttnn.Tensor, N: int) -> ttnn.Tensor:
        """Gather SP and return ``(B, N, action_dim)`` on the mesh."""
        x = self.ccl_manager.all_gather_persistent_buffer(
            spatial_1BND, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
        )
        _, B, _, A = x.shape
        x = ttnn.slice(x, [0, 0, 0, 0], [1, B, N, A])
        return ttnn.squeeze(x, 0)

    def forward(
        self,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        grid_id: ttnn.Tensor | dict,
        action_mode: bool = False,
        update_cache: int = 0,
        cache_name: str = "pos",
        timestep_per_frame: ttnn.Tensor | None = None,
        dump_iter: int | None = None,
        single_run: bool = False,
    ) -> ttnn.Tensor:
        """
        Forward pass (all activations on the mesh).

        Args:
            spatial: Video ``(B, in_channels, F, H, W)`` or action ``(B, action_dim, F, H, W)``.
            prompt: Text encoder hidden states (layout expected by ``prepare_text_conditioning``).
            timestep: 1D ``[B]`` (used when ``timestep_per_frame`` is None).
            grid_id: RoPE grid ``(B, 3, L)``, or when ``single_run=True`` a metadata ``dict`` (see below).
            timestep_per_frame: Optional ``[B, F]`` for per-token conditioning.
            single_run: If True, ``spatial``/``prompt``/``timestep`` are precomputed embedded tensors;
                ``grid_id`` must be a dict with ``rope_cos``, ``rope_sin``, ``trans_mat``, ``N``, ``F``,
                ``H``, ``W``, ``use_per_token``.

        Returns:
            Video: ``(B, out_channels, F, H, W)``. Action: ``(B, N, action_dim)``.
        """
        _ = dump_iter  # Kept in signature for call compatibility; debug tensor dumps were removed.
        if single_run:
            if not isinstance(grid_id, dict):
                raise ValueError("single_run=True requires grid_id to be a metadata dict")
            required_keys = {"rope_cos", "rope_sin", "trans_mat", "N", "F", "H", "W", "use_per_token"}
            if not required_keys.issubset(grid_id.keys()):
                missing = required_keys - set(grid_id.keys())
                raise ValueError(f"single_run metadata missing keys: {sorted(missing)}")

            rope_cos_1HND = grid_id["rope_cos"]
            rope_sin_1HND = grid_id["rope_sin"]
            trans_mat = grid_id["trans_mat"]
            N = grid_id["N"]
            use_per_token = grid_id["use_per_token"]
            action_mode = grid_id.get("action_mode", action_mode)
            F = grid_id["F"]
            H = grid_id["H"]
            W = grid_id["W"]

            spatial_1BND = spatial
            prompt_1BLP = prompt
            block_temb = timestep_per_frame
            if block_temb is None:
                raise ValueError("single_run=True requires precomputed block timestep tensor in timestep_per_frame")
            if use_per_token:
                temb_1BND = timestep
            else:
                temb_11BD = timestep
        else:
            B, C, F, H, W = (int(spatial.shape[i]) for i in range(5))
            pF, pH, pW = self.patch_size
            if action_mode:
                patch_F, patch_H, patch_W = F, H, W
            else:
                patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
            N = patch_F * patch_H * patch_W

            if not isinstance(grid_id, ttnn.Tensor):
                raise TypeError("grid_id must be a ttnn.Tensor unless single_run=True")
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
                temb_11BD, timestep_proj_1BTD, prompt_1BLP = self.prepare_conditioning(
                    timestep, prompt, action_mode=action_mode
                )

            if action_mode:
                spatial_1BNI, N = self.preprocess_action_input(spatial)
                spatial_1BND = self.action_embedder(
                    spatial_1BNI,
                    compute_kernel_config=self.hifi4_compute_kernel_config,
                )
            else:
                spatial_1BNI, N = self.preprocess_spatial_input(spatial)
                spatial_1BND = self.patch_embedding(spatial_1BNI)

            block_temb = timestep_proj_1BN6D if use_per_token else timestep_proj_1BTD

        use_cache = cache_name in self._attn_caches
        caches = self._attn_caches[cache_name] if use_cache else None

        for block_idx, block in enumerate(self.blocks):
            cached_k_tt = None
            cached_v_tt = None
            prefix_owned = False
            block_has_cache = (
                use_cache and caches is not None and block_idx < len(caches) and caches[block_idx] is not None
            )
            segs = caches[block_idx] if block_has_cache else None
            need_kv_from_attn = block_has_cache and segs is not None and (update_cache != 0 or len(segs) > 0)

            if block_has_cache and segs is not None and len(segs) > 0:
                ks = [s["k"] for s in segs]
                vs = [s["v"] for s in segs]
                if len(ks) == 1:
                    cached_k_tt, cached_v_tt = ks[0], vs[0]
                else:
                    cached_k_tt = ttnn.concat(ks, dim=2)
                    cached_v_tt = ttnn.concat(vs, dim=2)
                    prefix_owned = True

            if need_kv_from_attn:
                attn1_out = block(
                    spatial_1BND=spatial_1BND,
                    prompt_1BLP=prompt_1BLP,
                    temb_1BTD=block_temb,
                    N=N,
                    rope_cos=rope_cos_1HND,
                    rope_sin=rope_sin_1HND,
                    trans_mat=trans_mat,
                    cached_k=cached_k_tt,
                    cached_v=cached_v_tt,
                    return_kv=True,
                )
                spatial_1BND, k_cur, v_cur = attn1_out
                assert segs is not None
                if update_cache == 1:
                    segs.append({"k": k_cur, "v": v_cur, "is_pred": True})
                else:
                    _safe_deallocate_tensor(k_cur, "block k_cur")
                    _safe_deallocate_tensor(v_cur, "block v_cur")
                if prefix_owned:
                    _safe_deallocate_tensor(cached_k_tt, "block cached_k concat")
                    _safe_deallocate_tensor(cached_v_tt, "block cached_v concat")
            else:
                spatial_1BND = block(
                    spatial_1BND=spatial_1BND,
                    prompt_1BLP=prompt_1BLP,
                    temb_1BTD=block_temb,
                    N=N,
                    rope_cos=rope_cos_1HND,
                    rope_sin=rope_sin_1HND,
                    trans_mat=trans_mat,
                )

        if use_per_token:
            sst_shift, sst_scale = ttnn.chunk(self.scale_shift_table.data, 2, dim=-2)
            shift_norm = ttnn.add(sst_shift, temb_1BND)
            scale_norm = ttnn.add(sst_scale, temb_1BND)
            spatial_norm_1BND = self.norm_out(spatial_1BND, dtype=ttnn.float32)
            spatial_norm_1BND = ttnn.add(
                ttnn.multiply(spatial_norm_1BND, ttnn.add(scale_norm, 1.0)),
                shift_norm,
            )
        else:
            scale_shift_1BSD = ttnn.add(self.scale_shift_table.data, temb_11BD)
            shift_norm, scale_norm = ttnn.chunk(scale_shift_1BSD, 2, -2)
            spatial_norm_1BND = self.norm_out(
                spatial_1BND,
                dynamic_weight=ttnn.add(scale_norm, 1.0),
                dynamic_bias=shift_norm,
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
            if single_run:
                return proj_out_1BNA
            return self.postprocess_action_output(proj_out_1BNA, N)
        proj_out_1BNI = self.proj_out(
            spatial_norm_1BND,
            compute_kernel_config=self.hifi4_compute_kernel_config,
            dtype=ttnn.float32,
        )
        if single_run:
            return proj_out_1BNI
        return self.postprocess_spatial_output(proj_out_1BNI, F, H, W, N)
