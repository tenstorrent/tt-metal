# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 ViT residual block — TP=8 tensor-parallel weights.

Weight sharding (T3K, 8 devices):
  attention wqkv : ShardTensorToMesh(dim=3) — column-parallel, n_local_heads=2/device
  attention wo   : ShardTensorToMesh(dim=2) — row-parallel
  MLP w1         : ShardTensorToMesh(dim=3) — column-parallel
  MLP w2         : ShardTensorToMesh(dim=2) — row-parallel
  biases, norms  : replicated

After each row-parallel projection, ttnn.all_reduce(cluster_axis=1) combines partial
sums from 8 devices.

Input/output format: [n_crops, 1, 729, hidden] — per-crop batch dimension keeps
attention strictly within each crop's 729 patches (fuse_batch=False).
"""


import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_vl.tt.vision_layernorm import LayerNorm

# Note: use ttnn.all_reduce (native TTNN) not tt_all_reduce (which is a no-op for T3K cluster_axis=1)


class _ViTAttention(LightweightModule):
    """ViT MHA — DATA PARALLEL weights (replicated), no CCL.

    All 16 heads run on every device. Input crops are sharded across devices
    (1 crop/device), so each device independently processes its crop with all
    heads and full MLP — zero AllGather/ReduceScatter per block.
    Tracy profile: 4.5× faster than TP=8 (44% CCL eliminated).
    """

    def __init__(self, mesh_device, state_dict, layer_prefix, vit_cfg, weight_cache_path):
        super().__init__()

        self.mesh_device = mesh_device
        self.n_heads = vit_cfg.vit_n_heads  # 16
        self.head_dim = vit_cfg.vit_head_dim  # 72
        self.padded_head_dim = vit_cfg.vit_padded_head_dim  # 96
        self.hidden = vit_cfg.vit_hidden  # 1152
        self.scale = self.head_dim**-0.5
        self.tile_size = vit_cfg.tile_size
        self.num_devices = vit_cfg.num_devices  # 8 for T3K
        self.n_local_heads = self.n_heads  # DP: all 16 heads on each device

        self.compute_kernel_config_hifi2 = vit_cfg.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi4 = vit_cfg.compute_kernel_config_hifi4
        self.compute_kernel_config_hifi2_fp16 = vit_cfg.compute_kernel_config_hifi2_fp16

        is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
        # DP: ALL weights replicated — each device has the full weight matrices
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        col_mapper = row_mapper = bias_col = replicate

        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda n: weight_cache_path / f"{layer_prefix}.{n}"

        # ---- wq, wk, wv: load, pad head_dim if needed, fuse into wqkv ----
        def load_w(key):
            w = state_dict[f"{layer_prefix}.attention.{key}.weight"]  # [1152, 1152]
            if self.head_dim != self.padded_head_dim:
                w = w.reshape(self.n_heads, self.head_dim, -1)
                w = torch.nn.functional.pad(w, (0, 0, 0, self.padded_head_dim - self.head_dim))
                w = w.reshape(self.n_heads * self.padded_head_dim, -1)
            return w.T  # [1152, n_heads*padded_head_dim]

        def load_b(key):
            b = state_dict.get(f"{layer_prefix}.attention.{key}.bias")
            if b is None:
                return None
            if self.head_dim != self.padded_head_dim:
                b = b.reshape(self.n_heads, self.head_dim)
                b = torch.nn.functional.pad(b, (0, self.padded_head_dim - self.head_dim))
                b = b.reshape(-1)
            return b

        # DP: fuse wqkv as simple cat — all heads on all devices, replicated
        wq_full = load_w("wq")  # [1152, n_heads*padded_head_dim]
        wk_full = load_w("wk")
        wv_full = load_w("wv")
        wqkv = torch.cat([wq_full, wk_full, wv_full], dim=-1)  # [1152, 3*n_heads*padded=4608]
        self.wqkv = ttnn.as_tensor(
            wqkv.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_file_name=cache_name("wqkv.dp"),
        )

        bq, bk, bv = load_b("wq"), load_b("wk"), load_b("wv")
        if bq is not None:
            bqkv = torch.cat([bq, bk, bv])
            self.bqkv = ttnn.as_tensor(
                bqkv,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
                cache_file_name=cache_name("bqkv.dp"),
            )
        else:
            self.bqkv = None

        # ---- wo: replicated (DP — no row-parallel needed) ----
        wo = state_dict[f"{layer_prefix}.attention.wo.weight"]  # [1152, 1152]
        if self.head_dim != self.padded_head_dim:
            wo = wo.reshape(-1, self.n_heads, self.head_dim)
            wo = torch.nn.functional.pad(wo, (0, self.padded_head_dim - self.head_dim))
            wo = wo.reshape(-1, self.n_heads * self.padded_head_dim)
        wo_t = wo.T  # [n_heads*padded_head_dim, 1152]
        self.wo = ttnn.as_tensor(
            wo_t.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_file_name=cache_name("wo.dp"),
        )

        bo = state_dict.get(f"{layer_prefix}.attention.wo.bias")
        self.bo = None
        if bo is not None:
            self.bo = ttnn.as_tensor(
                bo,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
                cache_file_name=cache_name("bo"),
            )

        # SDPA program config: 128-token chunks (DP: 16 heads per device, [1,16,729,96])
        self._sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [n_crops_per_dev, 1, 729, 1152] — DP: each device has its own crops."""
        # DP QKV: full wqkv replicated, all 16 heads on each device
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        if self.bqkv is not None:
            xqkv = ttnn.add(xqkv, self.bqkv, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)

        # Split heads: [n_crops, n_local_heads, 729, padded_head_dim]
        # shape[1] == 1 is satisfied by the per-crop [n_crops, 1, 729, ...] format
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # SDPA (non-causal, no RoPE; chunked for L1 efficiency)
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            program_config=self._sdpa_cfg,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn_out = ttnn.reshape(attn_out, [attn_out.shape[0], self.n_local_heads, -1, self.padded_head_dim])
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Row-parallel wo
        out = ttnn.linear(
            attn_out,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_out)

        # DP: no all_reduce — each device already has the full output
        if self.bo is not None:
            out = ttnn.add(out, self.bo, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return out


class _ViTMLP(LightweightModule):
    """ViT GELU MLP — DATA PARALLEL (replicated weights, no CCL)."""

    def __init__(self, mesh_device, state_dict, layer_prefix, vit_cfg, weight_cache_path):
        super().__init__()
        self.compute_kernel_config = vit_cfg.compute_kernel_config_hifi2_fp16
        self.num_devices = vit_cfg.num_devices

        is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda n: weight_cache_path / f"{layer_prefix}.ff.{n}"

        def _tt_weight(key, name):
            return ttnn.as_tensor(
                state_dict[key].T.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
                cache_file_name=cache_name(name),
            )

        def _tt_bias(key, name):
            b = state_dict.get(key)
            if b is None:
                return None
            return ttnn.as_tensor(
                b,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
                cache_file_name=cache_name(name),
            )

        # DP: all weights replicated — full 4304-dim MLP on each device
        self.w1 = _tt_weight(f"{layer_prefix}.feed_forward.w1.weight", "w1.dp")
        self.b1 = _tt_bias(f"{layer_prefix}.feed_forward.w1.bias", "b1.dp")
        self.w2 = _tt_weight(f"{layer_prefix}.feed_forward.w2.weight", "w2.dp")
        self.b2 = _tt_bias(f"{layer_prefix}.feed_forward.w2.bias", "b2.dp")
        self.mesh_device = mesh_device

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        hidden = ttnn.linear(
            x,
            self.w1,
            bias=self.b1,
            activation="gelu",
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.linear(
            hidden,
            self.w2,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)
        # DP: no all_reduce — each device has the full output already
        if self.b2 is not None:
            out = ttnn.add(out, self.b2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out


class TtMolmo2ViTBlock(LightweightModule):
    """Single Molmo2 ViT residual block: pre-norm attn + pre-norm MLP."""

    def __init__(self, mesh_device, state_dict, layer_num, vit_cfg, weight_cache_path):
        super().__init__()

        layer_prefix = f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}"

        self.attention_norm = LayerNorm(
            device=mesh_device,
            dim=vit_cfg.vit_hidden,
            eps=vit_cfg.vit_norm_eps,
            state_dict=state_dict,
            state_dict_prefix=f"{layer_prefix}.attention_norm",
            weight_cache_path=None if vit_cfg.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
        )
        self.ffn_norm = LayerNorm(
            device=mesh_device,
            dim=vit_cfg.vit_hidden,
            eps=vit_cfg.vit_norm_eps,
            state_dict=state_dict,
            state_dict_prefix=f"{layer_prefix}.ffn_norm",
            weight_cache_path=None if vit_cfg.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
        )
        self.attention = _ViTAttention(mesh_device, state_dict, layer_prefix, vit_cfg, weight_cache_path)
        self.mlp = _ViTMLP(mesh_device, state_dict, layer_prefix, vit_cfg, weight_cache_path)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [n_crops, 1, 729, hidden] — input NOT deallocated (needed for capture)."""
        skip_cfg = ttnn.DRAM_MEMORY_CONFIG

        attn_in = self.attention_norm(x)
        attn_out = self.attention.forward(attn_in)
        # attn_in was already deallocated inside attention.forward
        h = ttnn.add(x, attn_out, memory_config=skip_cfg)
        ttnn.deallocate(attn_out)
        # x is NOT deallocated — caller (encoder) manages capture lifetime

        ff_in = self.ffn_norm(h)
        ff_out = self.mlp.forward(ff_in)
        ttnn.deallocate(ff_in)
        out = ttnn.add(h, ff_out, memory_config=skip_cfg, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        ttnn.deallocate(ff_out)
        return out
