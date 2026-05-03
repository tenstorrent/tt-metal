# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 image pooling 2D cross-attention adapter.

Cross-attention where:
  - Q = masked mean of patches in a pooling window   [N_windows, 1, 2304]
  - K, V = all patches in the window                 [N_windows, k_pool, 2304]
  - wq/wk/wv: [1152, 2304] with bias; wo: [1152, 1152] with bias
  - head_dim=72 padded to 96 for tile alignment

TP=8 layout (T3K):
  - wq/wk/wv: column-parallel (ShardTensorToMesh dim=3) — each device 2 of 16 heads
  - wo:        row-parallel (ShardTensorToMesh dim=2) + ttnn.all_reduce after linear
  - biases:    column-parallel for proj biases, replicated for wo bias

API: forward(query, key_value, attn_mask)
  - caller pre-computes query and key_value on device (see model._run_chunked_ttnn_pooling)
  - uses manual matmul attention (not SDPA) to handle cross-attn mask correctly
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtMolmo2ImagePooling2D(LightweightModule):
    """Cross-attention image pooling adapter for Molmo2."""

    def __init__(self, mesh_device, state_dict, cfg, weight_cache_path):
        super().__init__()

        self.mesh_device = mesh_device
        self.n_heads = cfg.pool_n_heads  # 16
        self.head_dim = cfg.pool_head_dim  # 72
        self.padded_head_dim = cfg.pool_padded_head_dim  # 96
        self.scale = self.head_dim**-0.5
        self.compute_kernel_config_hifi2 = cfg.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi4 = cfg.compute_kernel_config_hifi4

        is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
        self.num_devices = mesh_device.get_num_devices() if is_mesh else 1
        self.n_local_heads = self.n_heads // self.num_devices  # 2 on T3K

        col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3) if is_mesh else None
        row_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2) if is_mesh else None
        bias_col = ttnn.ShardTensorToMesh(mesh_device, dim=-1) if is_mesh else None
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        pfx = "model.vision_backbone.image_pooling_2d"

        if cfg.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"molmo2_pool2.{name}"

        def _load_qkv(key):
            w = state_dict[f"{pfx}.{key}.weight"]  # [n_heads*head_dim, input_dim]
            b = state_dict[f"{pfx}.{key}.bias"]  # [n_heads*head_dim]
            if self.head_dim != self.padded_head_dim:
                w = w.reshape(self.n_heads, self.head_dim, -1)
                w = torch.nn.functional.pad(w, (0, 0, 0, self.padded_head_dim - self.head_dim))
                w = w.reshape(self.n_heads * self.padded_head_dim, -1)
                b = b.reshape(self.n_heads, self.head_dim)
                b = torch.nn.functional.pad(b, (0, self.padded_head_dim - self.head_dim))
                b = b.reshape(-1)
            return (
                ttnn.as_tensor(
                    w.T.unsqueeze(0).unsqueeze(0).to(torch.float32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=col_mapper,
                    cache_file_name=cache_name(f"{key}.col"),
                ),
                ttnn.as_tensor(
                    b.to(torch.float32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=bias_col,
                    cache_file_name=cache_name(f"{key}_b.col"),
                ),
            )

        self.wq, self.bq = _load_qkv("wq")
        self.wk, self.bk = _load_qkv("wk")
        self.wv, self.bv = _load_qkv("wv")

        # wo: row-parallel — input dim is n_heads*padded_head_dim (sharded by heads)
        wo = state_dict[f"{pfx}.wo.weight"]  # [hidden_dim, n_heads*head_dim]
        if self.head_dim != self.padded_head_dim:
            wo = wo.reshape(-1, self.n_heads, self.head_dim)
            wo = torch.nn.functional.pad(wo, (0, self.padded_head_dim - self.head_dim))
            wo = wo.reshape(-1, self.n_heads * self.padded_head_dim)
        self.wo = ttnn.as_tensor(
            wo.T.unsqueeze(0).unsqueeze(0).to(torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_mapper,
            cache_file_name=cache_name("wo.row"),
        )
        self.bo = ttnn.as_tensor(
            state_dict[f"{pfx}.wo.bias"].to(torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_file_name=cache_name("wo_b"),
        )

    def forward(
        self,
        query: ttnn.Tensor,  # [1, N_windows, 1, POOL_DIM]
        key_value: ttnn.Tensor,  # [1, N_windows, k_pool, POOL_DIM]
        attn_mask: ttnn.Tensor = None,  # [N_windows, 1, 1, k_pool] or None
    ) -> ttnn.Tensor:
        """Cross-attention pooling. Returns [1, N_windows, 1, HIDDEN_DIM]."""
        n_windows = query.shape[1]
        k_pool = key_value.shape[2]

        # Column-parallel QKV — each device computes n_local_heads
        q = ttnn.add(
            ttnn.linear(
                query,
                self.wq,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            self.bq,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k = ttnn.add(
            ttnn.linear(
                key_value,
                self.wk,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            self.bk,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = ttnn.add(
            ttnn.linear(
                key_value,
                self.wv,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            self.bv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape to [N_windows, n_local_heads, seq, padded_head_dim]
        q = ttnn.permute(ttnn.reshape(q, [n_windows, 1, self.n_local_heads, self.padded_head_dim]), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, [n_windows, k_pool, self.n_local_heads, self.padded_head_dim]), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v, [n_windows, k_pool, self.n_local_heads, self.padded_head_dim]), (0, 2, 1, 3))

        # Manual matmul attention
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        attn_w = ttnn.mul(
            ttnn.matmul(
                q, k_t, compute_kernel_config=self.compute_kernel_config_hifi4, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(k_t)
        if attn_mask is not None:
            # [N_windows, 1, 1, k_pool] broadcasts to [N_windows, n_local_heads, 1, k_pool]
            attn_w = ttnn.add(attn_w, attn_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_p = ttnn.softmax(attn_w, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_w)
        attn_out = ttnn.matmul(
            attn_p, v, compute_kernel_config=self.compute_kernel_config_hifi4, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for t in (attn_p, q, k, v):
            ttnn.deallocate(t)

        # [N_windows, n_local_heads, 1, padded] → [1, N_windows, 1, n_local_heads*padded]
        attn_out = ttnn.reshape(
            ttnn.permute(attn_out, (0, 2, 1, 3)), [1, n_windows, 1, self.n_local_heads * self.padded_head_dim]
        )

        # Row-parallel wo + all_reduce + bias
        out = ttnn.linear(
            attn_out,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_out)
        if self.num_devices > 1:
            out = ttnn.all_reduce(out, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.add(out, self.bo, memory_config=ttnn.DRAM_MEMORY_CONFIG)
