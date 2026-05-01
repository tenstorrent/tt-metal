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

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_vl.tt.vision_layernorm import LayerNorm

# Note: use ttnn.all_reduce (native TTNN) not tt_all_reduce (which is a no-op for T3K cluster_axis=1)


class _ViTAttention(LightweightModule):
    """ViT MHA — column-parallel wqkv, row-parallel wo, tt_all_reduce after wo."""

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
        self.n_local_heads = self.n_heads // self.num_devices  # 2

        self.compute_kernel_config_hifi2 = vit_cfg.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi4 = vit_cfg.compute_kernel_config_hifi4
        self.compute_kernel_config_hifi2_fp16 = vit_cfg.compute_kernel_config_hifi2_fp16

        is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
        if is_mesh:
            col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
            row_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
            bias_col = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
            replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        else:
            col_mapper = row_mapper = bias_col = replicate = None

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

        # Build TP-interleaved wqkv: device i gets [wq_i, wk_i, wv_i] consecutively.
        # Simple cat([wq,wk,wv]) + ShardTensorToMesh(dim=3) gives each device a slice
        # from the *Q block only*, not the correct Q/K/V mix.
        # Correct approach: for each device, select its n_local_heads from Q, K, V
        # and concatenate — then stack devices. Mirrors qwen3_vl attention.py pattern.
        wq_full = load_w("wq")  # [1152, n_heads*padded_head_dim]
        wk_full = load_w("wk")
        wv_full = load_w("wv")
        cols = self.n_local_heads * self.padded_head_dim  # 2*96=192 per device
        qkv_chunks = []
        for i in range(self.num_devices):
            qkv_chunks.append(
                torch.cat(
                    [
                        wq_full[:, i * cols : (i + 1) * cols],  # Q for this device's heads
                        wk_full[:, i * cols : (i + 1) * cols],  # K
                        wv_full[:, i * cols : (i + 1) * cols],  # V
                    ],
                    dim=-1,
                )
            )  # [1152, 576]
        # [1, 1, 1152, num_devices*576=4608]; ShardTensorToMesh(dim=3) gives each device [1152, 576]
        wqkv = torch.cat(qkv_chunks, dim=-1)
        self.wqkv = ttnn.as_tensor(
            wqkv.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col_mapper,
            cache_file_name=cache_name("wqkv.tp8"),
        )

        bqkv = None
        bq, bk, bv = load_b("wq"), load_b("wk"), load_b("wv")
        if bq is not None:
            # Same TP-interleaved order as wqkv: each device's [bq_i, bk_i, bv_i]
            b_chunks = []
            for i in range(self.num_devices):
                b_chunks.append(
                    torch.cat(
                        [bq[i * cols : (i + 1) * cols], bk[i * cols : (i + 1) * cols], bv[i * cols : (i + 1) * cols]]
                    )
                )
            bqkv = torch.cat(b_chunks)
            self.bqkv = ttnn.as_tensor(
                bqkv,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=bias_col,
                cache_file_name=cache_name("bqkv.tp8"),
            )
        else:
            self.bqkv = None

        # ---- wo: row-parallel ----
        wo = state_dict[f"{layer_prefix}.attention.wo.weight"]  # [1152, 1152]
        if self.head_dim != self.padded_head_dim:
            wo = wo.reshape(-1, self.n_heads, self.head_dim)
            wo = torch.nn.functional.pad(wo, (0, self.padded_head_dim - self.head_dim))
            wo = wo.reshape(-1, self.n_heads * self.padded_head_dim)
        wo_t = wo.T  # [n_heads*padded_head_dim, 1152]
        # ShardTensorToMesh(dim=2) shards the input dim (head outputs)
        self.wo = ttnn.as_tensor(
            wo_t.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_mapper,
            cache_file_name=cache_name("wo.tp8"),
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

        # SDPA program config: 128-token chunks — keeps L1 well within budget
        # for the 2-local-head-per-device ViT SDPA ([n_crops, 2, 729, 96]).
        self._sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        # Program config for the column-parallel wqkv linear
        # in0_block_w=4: load 4 tiles (128 elements) of hidden dim per step for better
        # register reuse (was 1 tile = 32 elements, causing excessive DRAM re-reads).
        qkv_out_per_dev = 3 * self.n_local_heads * self.padded_head_dim  # 3*2*96=576
        self.xqkv_progcfg = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=max(1, math.ceil(seq_len / self.tile_size / 8)),
            per_core_N=max(1, math.ceil(qkv_out_per_dev / self.tile_size / 8)),
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [n_crops, 1, 729, 1152] — per-crop batch keeps per-crop attention."""
        seq_len = x.shape[-2]  # 729

        # Column-parallel QKV
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            program_config=self.xqkv_progcfg(seq_len),
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

        # All-reduce across T3K devices (row-parallel partial sums → full result)
        if self.num_devices > 1:
            out = ttnn.all_reduce(
                out,
                cluster_axis=1,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if self.bo is not None:
            out = ttnn.add(out, self.bo, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return out


class _ViTMLP(LightweightModule):
    """ViT GELU MLP — column-parallel w1, row-parallel w2, tt_all_reduce after w2."""

    def __init__(self, mesh_device, state_dict, layer_prefix, vit_cfg, weight_cache_path):
        super().__init__()
        self.compute_kernel_config = vit_cfg.compute_kernel_config_hifi2_fp16
        self.num_devices = vit_cfg.num_devices

        is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
        col_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3) if is_mesh else None
        row_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2) if is_mesh else None
        bias_col = ttnn.ShardTensorToMesh(mesh_device, dim=-1) if is_mesh else None
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        if weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda n: weight_cache_path / f"{layer_prefix}.ff.{n}"

        def _tt_weight(key, mapper, name):
            return ttnn.as_tensor(
                state_dict[key].T.unsqueeze(0).unsqueeze(0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
                cache_file_name=cache_name(name),
            )

        def _tt_bias(key, mapper, name):
            b = state_dict.get(key)
            if b is None:
                return None
            return ttnn.as_tensor(
                b,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
                cache_file_name=cache_name(name),
            )

        self.w1 = _tt_weight(f"{layer_prefix}.feed_forward.w1.weight", col_mapper, "w1.tp8")
        self.b1 = _tt_bias(f"{layer_prefix}.feed_forward.w1.bias", bias_col, "b1.tp8")
        self.w2 = _tt_weight(f"{layer_prefix}.feed_forward.w2.weight", row_mapper, "w2.tp8")
        self.b2 = _tt_bias(f"{layer_prefix}.feed_forward.w2.bias", replicate, "b2")
        self.mesh_device = mesh_device

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Column-parallel w1 + GELU
        hidden = ttnn.linear(
            x,
            self.w1,
            bias=self.b1,
            activation="gelu",
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Row-parallel w2
        out = ttnn.linear(
            hidden,
            self.w2,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hidden)

        # All-reduce combines row-parallel partial sums
        if self.num_devices > 1:
            out = ttnn.all_reduce(
                out,
                cluster_axis=1,
                num_links=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

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
