# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 image pooling 2D cross-attention adapter.

Cross-attention where:
  - Q = masked mean of patches in a pooling window [N_windows, 1, 2304]
  - K, V = all patches in the window [N_windows, pool_window, 2304]
  - wq/wk/wv: [1152, 2304] (biases present); wo: [1152, 1152] (bias present)
  - head_dim=72 → padded to 96

Based on the reference image_pooling_2d() in functional.py but implemented in TTNN.
Runs replicated (same computation on all 8 devices since input is replicated after
the ViT AllGather).

Window batching: the HF reference processes each window individually; here we
batch all windows together: effective batch = B*N_pooled.
"""


import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMolmo2ImagePooling2D(LightweightModule):
    """Cross-attention adapter for Molmo2 image pooling."""

    def __init__(self, mesh_device, state_dict, cfg, weight_cache_path):
        super().__init__()

        self.mesh_device = mesh_device
        self.n_heads = cfg.pool_n_heads  # 16
        self.head_dim = cfg.pool_head_dim  # 72
        self.padded_head_dim = cfg.pool_padded_head_dim  # 96
        self.hidden = cfg.pool_hidden  # 1152
        self.pool_dim = cfg.pool_dim  # 2304
        self.scale = self.head_dim**-0.5
        self.compute_kernel_config_hifi4 = cfg.compute_kernel_config_hifi4
        self.compute_kernel_config_hifi2_fp16 = cfg.compute_kernel_config_hifi2_fp16

        prefix = "model.vision_backbone.image_pooling_2d"

        if cfg.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"molmo2_pool.{name}"

        def _load_proj_w(key):
            w = state_dict[f"{prefix}.{key}.weight"]  # [1152, 2304] or [1152, 1152]
            n_heads = self.n_heads
            head_in = w.shape[0] // n_heads
            if head_in == self.head_dim and self.head_dim != self.padded_head_dim:
                w = w.reshape(n_heads, self.head_dim, -1)
                w = torch.nn.functional.pad(w, (0, 0, 0, self.padded_head_dim - self.head_dim))
                w = w.reshape(n_heads * self.padded_head_dim, -1)
            return w.T

        def _load_bias(key):
            b = state_dict.get(f"{prefix}.{key}.bias")
            if b is None:
                return None
            # If output dim is 1152 (i.e., n_heads * head_dim) and head_dim needs padding
            if b.shape[0] == self.n_heads * self.head_dim and self.head_dim != self.padded_head_dim:
                b = b.reshape(self.n_heads, self.head_dim)
                b = torch.nn.functional.pad(b, (0, self.padded_head_dim - self.head_dim))
                b = b.reshape(-1)
            return b

        def _tt(tensor, name, layout=ttnn.TILE_LAYOUT):
            return ttnn.as_tensor(
                tensor,
                dtype=ttnn.bfloat16,
                layout=layout,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=cache_name(name),
            )

        # Build fused QKV weight: [pool_dim, n_heads*padded_head_dim*3]
        wq = _load_proj_w("wq")  # [2304, n_heads*padded_head_dim]
        wk = _load_proj_w("wk")
        wv = _load_proj_w("wv")
        wqkv = torch.cat([wq, wk, wv], dim=-1).unsqueeze(0).unsqueeze(0)
        self.wqkv = _tt(wqkv, "wqkv")

        bq = _load_bias("wq")
        bk = _load_bias("wk")
        bv = _load_bias("wv")
        self.wqkv_bias = None
        if bq is not None:
            qkv_bias = torch.cat([bq, bk, bv])
            self.wqkv_bias = _tt(qkv_bias, "wqkv_bias")

        # Output wo: [1152, 1152], with possible wo side head-dim padding
        wo_raw = state_dict[f"{prefix}.wo.weight"]  # [1152, 1152]
        if self.head_dim != self.padded_head_dim:
            wo_reshaped = wo_raw.reshape(self.hidden, self.n_heads, self.head_dim)
            wo_padded = torch.nn.functional.pad(wo_reshaped, (0, self.padded_head_dim - self.head_dim))
            wo_raw = wo_padded.reshape(self.hidden, self.n_heads * self.padded_head_dim)

        self.wo = _tt(wo_raw.T.unsqueeze(0).unsqueeze(0), "wo")

        self.wo_bias = None
        wo_b = state_dict.get(f"{prefix}.wo.bias")
        if wo_b is not None:
            self.wo_bias = _tt(wo_b, "wo_bias")

    def forward(
        self,
        image_features: torch.Tensor,  # [B, n_crops, 729, 2304] CPU tensor
        pooled_patches_idx: torch.Tensor,  # [B, N_pooled, pool_window] CPU tensor
    ) -> torch.Tensor:
        """Run image pooling cross-attention.

        Returns CPU tensor [B, N_pooled, 1152].

        This operation is run on CPU via PyTorch (matching the reference) and then
        moved to the device. The pooling gather is non-contiguous (irregular indices)
        which is easier to handle in PyTorch before moving to device.
        """

        B, n_crops, n_patches, dim = image_features.shape
        N_pooled = pooled_patches_idx.shape[1]
        pool_window = pooled_patches_idx.shape[2]

        valid = pooled_patches_idx >= 0  # [B, N_pooled, pool_window]
        valid_token = valid.any(dim=-1)  # [B, N_pooled]

        # Gather patches for each pooling window
        batch_idx = torch.arange(B).view(B, 1, 1).expand(B, N_pooled, pool_window)
        flat_features = image_features.reshape(B, -1, dim)  # [B, n_crops*729, dim]
        clipped = pooled_patches_idx.clamp(min=0)
        to_pool = flat_features[batch_idx, clipped]  # [B, N_pooled, pool_window, dim]
        to_pool = to_pool * valid.to(to_pool.dtype).unsqueeze(-1)

        to_pool_flat = to_pool.reshape(B * N_pooled, pool_window, dim)  # [B*N, pw, 2304]
        valid_flat = valid.reshape(B * N_pooled, pool_window)

        # Query = masked mean of patches in window [B*N, 1, 2304]
        denom = valid_flat.float().sum(dim=-1).clamp(min=1)
        query = to_pool_flat.sum(dim=1, keepdim=True) / denom.unsqueeze(-1).unsqueeze(-1).to(to_pool_flat.dtype)

        # Build additive attention mask for invalid patches [B*N, 1, 1, pool_window]
        attn_mask_pool = torch.where(
            valid_flat.unsqueeze(1).unsqueeze(1),
            torch.zeros(1),
            torch.full((1,), float("-inf")),
        ).to(to_pool_flat.dtype)

        # Concatenate query and kv for cross-attention: [B*N, 1+pw, 2304]
        # Use torch for this non-standard gather operation, then do QKV via TTNN
        cross_in = torch.cat([query, to_pool_flat], dim=1)  # [B*N, 1+pw, 2304]

        # Move to device and run cross-attention
        cross_ttnn = ttnn.from_torch(
            cross_in.unsqueeze(0).to(torch.bfloat16),  # [1, 1, B*N*(1+pw), 2304]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Reshape: [1, B*N, 1+pw, 2304]
        n_windows = B * N_pooled
        cross_ttnn = ttnn.reshape(cross_ttnn, [1, n_windows, 1 + pool_window, dim])

        # QKV projection
        xqkv = ttnn.linear(
            cross_ttnn,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
        )
        if self.wqkv_bias is not None:
            xqkv = ttnn.add(xqkv, self.wqkv_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(cross_ttnn)

        # Split Q (from query position 0) and KV (from patch positions 1:)
        # xqkv: [1, B*N, 1+pw, 3*n_heads*padded_head_dim]
        xqkv_rm = ttnn.to_layout(xqkv, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(xqkv)

        qkv_per_head = self.n_heads * self.padded_head_dim
        q_part = xqkv_rm[:, :, :1, :qkv_per_head]  # Q from query
        kv_part = xqkv_rm[:, :, 1:, qkv_per_head:]  # K and V from patches

        k_part = kv_part[:, :, :, :qkv_per_head]
        v_part = kv_part[:, :, :, qkv_per_head:]
        ttnn.deallocate(xqkv_rm)
        ttnn.deallocate(kv_part)

        # Reshape to [B*N, n_heads, 1, padded_head_dim] for Q
        #              [B*N, n_heads, pw, padded_head_dim] for K, V
        def _heads(t, seq):
            t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
            t = ttnn.reshape(t, [n_windows, seq, self.n_heads, self.padded_head_dim])
            return ttnn.permute(t, [0, 2, 1, 3])  # [B*N, n_heads, seq, head_dim]

        q_heads = _heads(q_part, 1)
        k_heads = _heads(k_part, pool_window)
        v_heads = _heads(v_part, pool_window)
        ttnn.deallocate(q_part)
        ttnn.deallocate(k_part)
        ttnn.deallocate(v_part)

        # Attention mask: [B*N, 1, 1, pool_window]
        attn_mask_ttnn = ttnn.from_torch(
            attn_mask_pool.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # SDPA: [B*N, n_heads, 1, padded_head_dim]
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            attn_mask=attn_mask_ttnn,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)
        ttnn.deallocate(attn_mask_ttnn)

        # attn_out: [B*N, n_heads, 1, padded_head_dim]
        # → concat heads → [1, B*N, 1, n_heads*padded_head_dim]
        attn_out = ttnn.permute(attn_out, [0, 2, 1, 3])  # [B*N, 1, n_heads, phd]
        attn_out = ttnn.reshape(attn_out, [1, n_windows, 1, self.n_heads * self.padded_head_dim])

        # Output projection
        out = ttnn.linear(
            attn_out,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.wo_bias is not None:
            out = ttnn.add(out, self.wo_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # out: [1, B*N, 1, 1152] → CPU tensor [B, N_pooled, 1152]
        out_cpu = ttnn.to_torch(
            ttnn.get_device_tensors(out)[0],  # from first device (replicated)
        ).float()
        ttnn.deallocate(out)

        # Reshape [1, B*N, 1, 1152] → [B, N_pooled, 1152]
        out_cpu = out_cpu.reshape(B, N_pooled, self.hidden)
        return out_cpu
