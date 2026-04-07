# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Image Pooling for Molmo2 Vision Adapter.

This module implements 2D image pooling using cross-attention, where:
- Query: mean of gathered patch features from pooled_patches_idx neighborhoods
- Keys/Values: gathered patch features

The pooled_patches_idx tensor maps each output visual token to a neighborhood
of K source ViT patches. This is computed by the image processor on CPU.

Dimensions:
    - input_dim: 2304 (1152 * 2, concat of ViT layers 18 and 24)
    - hidden_dim: 1152 (adapter hidden size)
    - num_heads: 16
    - head_dim: 72 (1152 / 16)

All wq, wk, wv, wo projections have bias.

**Tensor parallelism (mesh):** On a ``MeshDevice`` with ``D > 1`` devices, if
``num_heads % D == 0``, Q/K/V use **column-parallel** sharding on the head
output (dim 3), ``wo`` uses **row-parallel** sharding on the head input (dim 2),
then ``all_reduce(..., cluster_axis=1)`` and replicated output bias — same
pattern as ``TextAttention``. Otherwise weights stay **replicated** (legacy).
Set ``MOLMO2_IMAGE_POOLING_DISABLE_TP=1`` to force replication on multi-device.
"""

import logging
import math
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

_logger = logging.getLogger(__name__)


def _should_use_image_pooling_tp(is_mesh: bool, mesh_device, num_heads: int) -> bool:
    if not is_mesh or os.environ.get("MOLMO2_IMAGE_POOLING_DISABLE_TP", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False
    n = mesh_device.get_num_devices()
    return n > 1 and num_heads % n == 0


class ImagePooling(LightweightModule):
    """
    Cross-attention based image pooling for Molmo2.

    Pools multi-scale ViT features using attention with gathered patch neighborhoods.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        input_dim: int = 2304,
        hidden_dim: int = 1152,
        num_heads: int = 16,
        head_dim: int = 72,
        weight_cache_path=None,
        state_dict_prefix: str = "model.vision_backbone.image_pooling_2d",
        dtype=ttnn.bfloat8_b,
        force_replicate_attention: bool = False,
    ):
        """
        Initialize ImagePooling.

        Args:
            mesh_device: TTNN mesh device or single device
            state_dict: Model state dict containing weights
            input_dim: Input dimension (2304 = 1152 * 2 for multi-scale)
            hidden_dim: Hidden dimension for attention (1152)
            num_heads: Number of attention heads (16)
            head_dim: Dimension per head (72)
            weight_cache_path: Path to cache weights
            state_dict_prefix: Prefix for state dict keys
            dtype: Data type for weights
            force_replicate_attention: If True, never shard cross-attention on the mesh (replicated
                weights). Required when this module runs inside ``begin_trace_capture`` / ``end_trace_capture``
                because ``all_reduce`` performs reads that are not allowed during mesh trace capture.
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.tile_size = 32

        # Pad head_dim to tile boundary if needed
        self.padded_head_dim = math.ceil(head_dim / self.tile_size) * self.tile_size

        # Scale factor for attention
        self.scale = head_dim**-0.5

        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.is_mesh_device = is_mesh_device
        self.num_devices = mesh_device.get_num_devices() if is_mesh_device else 1

        self.use_attention_tp = (
            _should_use_image_pooling_tp(is_mesh_device, mesh_device, num_heads) and not force_replicate_attention
        )
        self.num_heads_per_device = num_heads // self.num_devices if self.use_attention_tp else num_heads

        if force_replicate_attention and is_mesh_device and self.num_devices > 1:
            _logger.info("ImagePooling: cross-attention TP disabled (replicated) for trace-capture-safe path")

        if is_mesh_device and self.num_devices > 1 and not self.use_attention_tp and not force_replicate_attention:
            if num_heads % self.num_devices != 0:
                _logger.info(
                    "ImagePooling: attention TP unavailable (num_heads=%s not divisible by num_devices=%s); "
                    "using replicated weights.",
                    num_heads,
                    self.num_devices,
                )
            else:
                _logger.info(
                    "ImagePooling: attention TP disabled via MOLMO2_IMAGE_POOLING_DISABLE_TP; "
                    "using replicated weights."
                )
        elif self.use_attention_tp:
            _logger.info(
                "ImagePooling: cross-attention tensor parallelism enabled (%s devices, %s heads/device)",
                self.num_devices,
                self.num_heads_per_device,
            )

        # Cache file naming
        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            suffix = ".attn_tp" if self.use_attention_tp else ""
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{suffix}"

        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None
        col_shard = ttnn.ShardTensorToMesh(mesh_device, dim=3) if self.use_attention_tp else replicate
        row_shard = ttnn.ShardTensorToMesh(mesh_device, dim=2) if self.use_attention_tp else replicate

        # Load wq: input_dim (2304) -> hidden_dim (1152)
        wq = state_dict[f"{state_dict_prefix}.wq.weight"]
        bq = state_dict[f"{state_dict_prefix}.wq.bias"]

        if self.head_dim != self.padded_head_dim:
            wq = self._pad_weight(wq)
            bq = self._pad_bias(bq)

        self.wq = self._make_qkv_weight_ttnn(wq, mesh_device, dtype, col_shard, cache_name("wq.weight"))
        self.bq = self._make_qkv_bias_ttnn(bq, mesh_device, col_shard, cache_name("wq.bias"))

        # Load wk
        wk = state_dict[f"{state_dict_prefix}.wk.weight"]
        bk = state_dict[f"{state_dict_prefix}.wk.bias"]
        if self.head_dim != self.padded_head_dim:
            wk = self._pad_weight(wk)
            bk = self._pad_bias(bk)
        self.wk = self._make_qkv_weight_ttnn(wk, mesh_device, dtype, col_shard, cache_name("wk.weight"))
        self.bk = self._make_qkv_bias_ttnn(bk, mesh_device, col_shard, cache_name("wk.bias"))

        # Load wv
        wv = state_dict[f"{state_dict_prefix}.wv.weight"]
        bv = state_dict[f"{state_dict_prefix}.wv.bias"]
        if self.head_dim != self.padded_head_dim:
            wv = self._pad_weight(wv)
            bv = self._pad_bias(bv)
        self.wv = self._make_qkv_weight_ttnn(wv, mesh_device, dtype, col_shard, cache_name("wv.weight"))
        self.bv = self._make_qkv_bias_ttnn(bv, mesh_device, col_shard, cache_name("wv.bias"))

        # Load wo: hidden_dim (1152) -> hidden_dim (1152)
        wo = state_dict[f"{state_dict_prefix}.wo.weight"]
        bo = state_dict[f"{state_dict_prefix}.wo.bias"]
        if self.head_dim != self.padded_head_dim:
            wo_reshaped = wo.reshape(-1, self.num_heads, self.head_dim)
            wo_padded = torch.nn.functional.pad(wo_reshaped, (0, self.padded_head_dim - self.head_dim))
            wo = wo_padded.reshape(-1, self.num_heads * self.padded_head_dim)

        self.wo = self._make_wo_weight_ttnn(wo, mesh_device, dtype, row_shard, cache_name("wo.weight"))

        # TP path adds bias after all_reduce; use explicit broadcast shape. Legacy: flat bias for linear.
        bo_host = bo.reshape(1, 1, 1, -1) if self.use_attention_tp else bo
        self.bo = ttnn.as_tensor(
            bo_host,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=replicate,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo.bias"),
        )

        # Compute kernel configs
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _make_qkv_weight_ttnn(self, w_padded, mesh_device, dtype, mesh_mapper, cache_file_name):
        """w_padded: [num_heads * padded_head_dim, input_dim] PyTorch layout."""
        if not self.use_attention_tp:
            wq_t = torch.transpose(w_padded, -2, -1)
            return ttnn.as_tensor(
                wq_t.unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_file_name,
            )
        in_dim = w_padded.shape[1]
        h_per = self.num_heads_per_device
        ph = self.padded_head_dim
        w_h = w_padded.view(self.num_heads, ph, in_dim)
        pieces = []
        for d in range(self.num_devices):
            wd = w_h[d * h_per : (d + 1) * h_per].reshape(-1, in_dim)
            wt = wd.transpose(0, 1).unsqueeze(0).unsqueeze(0)
            pieces.append(wt)
        wq_cat = torch.cat(pieces, dim=-1)
        return ttnn.as_tensor(
            wq_cat,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
        )

    def _make_qkv_bias_ttnn(self, b_padded, mesh_device, mesh_mapper, cache_file_name):
        if not self.use_attention_tp:
            # Match legacy layout: flat bias vector for ttnn.linear with replicated QKV.
            return ttnn.as_tensor(
                b_padded,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_file_name,
            )
        ph = self.padded_head_dim
        h_per = self.num_heads_per_device
        b_h = b_padded.view(self.num_heads, ph)
        pieces = []
        for d in range(self.num_devices):
            bd = b_h[d * h_per : (d + 1) * h_per].reshape(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            pieces.append(bd)
        b_cat = torch.cat(pieces, dim=-1)
        return ttnn.as_tensor(
            b_cat,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
        )

    def _make_wo_weight_ttnn(self, wo_padded, mesh_device, dtype, mesh_mapper, cache_file_name):
        """wo_padded: [hidden_dim, num_heads * padded_head_dim] (Linear out x in)."""
        wo_t = torch.transpose(wo_padded, -2, -1)
        if not self.use_attention_tp:
            return ttnn.as_tensor(
                wo_t.unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_file_name,
            )
        # wo_t: [in_features, out_features] = [num_heads * ph, hidden_dim]
        h_per = self.num_heads_per_device
        ph = self.padded_head_dim
        flat = wo_t
        pieces = []
        for d in range(self.num_devices):
            sl = flat[d * h_per * ph : (d + 1) * h_per * ph, :]
            pieces.append(sl.unsqueeze(0).unsqueeze(0))
        wo_cat = torch.cat(pieces, dim=2)
        return ttnn.as_tensor(
            wo_cat,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
        )

    def _pad_weight(self, w):
        """Pad weight tensor for non-tile-aligned head_dim."""
        w = w.reshape(self.num_heads, self.head_dim, -1)
        w = torch.nn.functional.pad(w, (0, 0, 0, self.padded_head_dim - self.head_dim))
        return w.reshape(self.num_heads * self.padded_head_dim, -1)

    def _pad_bias(self, b):
        """Pad bias tensor for non-tile-aligned head_dim."""
        b = b.reshape(self.num_heads, self.head_dim)
        b = torch.nn.functional.pad(b, (0, self.padded_head_dim - self.head_dim))
        return b.reshape(-1)

    def forward(
        self,
        query: ttnn.Tensor,
        key_value: ttnn.Tensor,
        attn_mask: ttnn.Tensor = None,
        matmul_output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """
        Forward pass through cross-attention pooling.

        Args:
            query: Query tensor of shape [1, 1, num_queries, input_dim]
                   (typically mean of gathered features)
            key_value: Key/Value tensor of shape [1, 1, pool_size, input_dim]
                       (gathered patch features)
            attn_mask: Optional additive mask (e.g. [1, 1, 1, pool_size]; batch may vary on dim 0).
                When present and pool_size is not a multiple of 32, K/V and the mask are padded so
                SDPA sees -inf on padded keys (matches default SDPA k-chunk alignment).

        Returns:
            Pooled features of shape [1, 1, num_queries, hidden_dim]
        """
        num_queries = query.shape[-2]
        pool_size = key_value.shape[-2]
        padded_pool_size = math.ceil(pool_size / self.tile_size) * self.tile_size
        pool_pad = padded_pool_size - pool_size

        kv_for_linear = key_value
        deallocate_kv_padded = False
        if attn_mask is not None and pool_pad > 0:
            kv_zeros = ttnn.zeros(
                shape=(key_value.shape[0], key_value.shape[1], pool_pad, key_value.shape[-1]),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=key_value.device(),
                memory_config=matmul_output_memory_config,
            )
            kv_for_linear = ttnn.concat([key_value, kv_zeros], dim=2)
            ttnn.deallocate(kv_zeros)
            deallocate_kv_padded = True

        attn_kv_len = padded_pool_size if (attn_mask is not None and pool_pad > 0) else pool_size

        # Q projection
        q = ttnn.linear(
            query,
            self.wq,
            bias=self.bq,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=matmul_output_memory_config,
        )

        # K projection
        k = ttnn.linear(
            kv_for_linear,
            self.wk,
            bias=self.bk,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=matmul_output_memory_config,
        )

        # V projection
        v = ttnn.linear(
            kv_for_linear,
            self.wv,
            bias=self.bv,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=matmul_output_memory_config,
        )

        if deallocate_kv_padded:
            ttnn.deallocate(kv_for_linear)

        batch_seq = query.shape[1]
        nh = self.num_heads_per_device
        padded_hidden = nh * self.padded_head_dim

        q = ttnn.reshape(q, [batch_seq, num_queries, nh, self.padded_head_dim])
        q = ttnn.permute(q, (0, 2, 1, 3))
        q = ttnn.typecast(q, dtype=ttnn.bfloat8_b)

        k = ttnn.reshape(k, [batch_seq, attn_kv_len, nh, self.padded_head_dim])
        k = ttnn.permute(k, (0, 2, 1, 3))
        k = ttnn.typecast(k, dtype=ttnn.bfloat8_b)

        v = ttnn.reshape(v, [batch_seq, attn_kv_len, nh, self.padded_head_dim])
        v = ttnn.permute(v, (0, 2, 1, 3))
        v = ttnn.typecast(v, dtype=ttnn.bfloat8_b)

        sdpa_mask = attn_mask
        if attn_mask is not None:
            if pool_pad > 0:
                inf_tail_shape = list(attn_mask.shape)
                inf_tail_shape[-1] = pool_pad
                mask_neg_inf = ttnn.full(
                    shape=tuple(inf_tail_shape),
                    fill_value=float("-inf"),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=attn_mask.device(),
                    memory_config=matmul_output_memory_config,
                )
                sdpa_mask = ttnn.concat([attn_mask, mask_neg_inf], dim=3)
                ttnn.deallocate(mask_neg_inf)

            mask_shape = list(sdpa_mask.shape)
            if len(mask_shape) == 4 and mask_shape[2] == 1 and num_queries > 1:
                prev_mask = sdpa_mask
                sdpa_mask = ttnn.repeat(
                    prev_mask,
                    (1, 1, num_queries, 1),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if prev_mask is not attn_mask:
                    ttnn.deallocate(prev_mask)

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            attn_mask=sdpa_mask,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        if sdpa_mask is not attn_mask and sdpa_mask is not None:
            ttnn.deallocate(sdpa_mask)

        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, [1, batch_seq, num_queries, padded_hidden])

        if self.use_attention_tp:
            output = ttnn.linear(
                attn_output,
                self.wo,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                memory_config=matmul_output_memory_config,
            )
            ttnn.deallocate(attn_output)
            output = ttnn.all_reduce(
                output,
                cluster_axis=1,
                num_links=1,
                memory_config=matmul_output_memory_config,
            )
            output = ttnn.add(output, self.bo)
            return output

        output = ttnn.linear(
            attn_output,
            self.wo,
            bias=self.bo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=matmul_output_memory_config,
        )
        ttnn.deallocate(attn_output)
        return output
