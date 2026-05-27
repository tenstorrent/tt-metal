# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ZImageTransformer TTNN — optimized inference model.

All performance optimizations are active:
  - ttnn.rms_norm           (replaces manual 9-op F32 sequence, ×153/pass)
  - ttnn.layer_norm         (final layer)
  - minimal_matmul          (attention + MLP projections, pre-transposed weights)
  - minimal_matmul_split    (fused Q/K/V in one kernel)
  - nlp_create_qkv_heads    (fused V head-reshape)
  - CCLManager async CCL    (reduce_scatter_minimal_async + all_gather_async)

4-way tensor parallelism, (1,4) MeshDevice, 8 heads/device.
Fixed resolution: 512×512 px (64×64 latent, 1024 image patches, 32 caption tokens).
"""

import math

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.z_image_turbo.tt.dit import consteval, model_pt, params
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.matmul import get_matmul_config as _get_matmul_config

# ── Architecture constants ─────────────────────────────────────────────────────

HIDDEN_DIM = 3840
PADDED_HEADS = 32  # 30 original → padded to 32 for TP
ORIGINAL_HEADS = 30
HEAD_DIM = 128
MLP_HIDDEN = 10240
TP = 4
HEADS_PER_DEV = PADDED_HEADS // TP  # 8
MLP_PER_DEV = MLP_HIDDEN // TP  # 2560
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)

EXTRA_DIM = (PADDED_HEADS - ORIGINAL_HEADS) * HEAD_DIM  # 256
IMG_PATCHES = 1024  # 32×32 patches
CAP_TOKENS = 128
PATCH_SIZE = 2
PATCH_DIM = 16 * PATCH_SIZE * PATCH_SIZE  # 64
ADALN_EMBED_DIM = 256

RMS_EPS = 1e-5  # all RMSNorm layers
LN_EPS = 1e-6  # final LayerNorm

# ── TTNN config ────────────────────────────────────────────────────────────────


REDUCE_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ── Main model class ───────────────────────────────────────────────────────────


class ZImageTransformerTTNN(LightweightModule):
    """Optimized TTNN ZImageTransformer with all perf improvements baked in."""

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device

        # ── Load + pad the reference PyTorch model ─────────────────────────────
        tr_pt = model_pt.load_model()
        model_pt.pad_heads(tr_pt)

        # ── Load weights ───────────────────────────────────────────────────────
        print("  Loading params...")
        self.weights = params.load_weights(mesh_device, tr_pt)
        del tr_pt
        print("  Running consteval ...")
        consteval.run_const_evals(self.weights, mesh_device)
        print("  Consteval complete.")

        # RoPE tables
        def _to_bf16_rm(t):
            host = ttnn.to_torch(
                ttnn.from_device(t),
                mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
            )
            host = host[: host.shape[0] // 4]
            if host.dim() > 2:
                host = host.reshape(host.shape[0], -1)
            return ttnn.from_torch(
                host.bfloat16(),
                dtype=ttnn.DataType.BFLOAT16,
                layout=ttnn.Layout.ROW_MAJOR,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        self.weights["_freqs_F"] = _to_bf16_rm(self.weights["__rope_freqs_F__"])
        self.weights["_freqs_H"] = _to_bf16_rm(self.weights["__rope_freqs_H__"])
        self.weights["_freqs_W"] = _to_bf16_rm(self.weights["__rope_freqs_W__"])
        self.weights["_x_pad_token"] = self.weights["x_pad_token"]
        self.weights["_cap_pad_token"] = self.weights["cap_pad_token"]

        # ── Detect compute grid ────────────────────────────────────────────────
        d = mesh_device.get_device(0) if hasattr(mesh_device, "get_device") else mesh_device
        self._core_grid = d.compute_with_storage_grid_size()
        print(f"  Compute grid: {self._core_grid.x}×{self._core_grid.y}")

        # ── Pre-process weights for optimized ops ──────────────────────────────
        print("  Pre-processing weights ...")
        self._prep_qk_norm_weights()
        self._prep_fused_qkv_weights()
        self._prep_parallel_weights()

        # ── Async CCL (reduce_scatter_minimal_async + all_gather_async) ────────
        self._ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Ring)
        print("  Async CCL initialized.")

        # ── cap_feats persistent buffer (updated per-prompt) ───────────────────
        self._cap_feats_buf = None

    # ── Weight preprocessing ───────────────────────────────────────────────────

    def _prep_qk_norm_weights(self):
        """Reshape QK norm weights [1,1,1,64,2] → [1,1,1,128] for ttnn.rms_norm."""
        for key in list(self.weights):
            if (".norm_q." in key or ".norm_k." in key) and "weight" in key:
                w = self.weights[key]
                w_host = ttnn.to_torch(
                    ttnn.from_device(w),
                    mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
                )
                shard = w_host[: w_host.shape[0] // 4].reshape(1, 1, 1, HEAD_DIM).bfloat16()
                self.weights[key + "_flat"] = ttnn.from_torch(
                    shard,
                    dtype=ttnn.DataType.BFLOAT16,
                    layout=ttnn.Layout.TILE,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )

    def _prep_fused_qkv_weights(self):
        """Fuse Q/K/V weights → [HIDDEN, 3*N] for minimal_matmul_split."""
        prefixes = (
            [f"noise_refiner.{i}" for i in range(2)]
            + [f"context_refiner.{i}" for i in range(2)]
            + [f"layers.{i}" for i in range(30)]
        )
        for prefix in prefixes:
            q_key = f"{prefix}.attention.to_q.weight"
            if q_key not in self.weights:
                continue
            q_T = ttnn.permute(self.weights[q_key], [1, 0], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            k_T = ttnn.permute(
                self.weights[f"{prefix}.attention.to_k.weight"], [1, 0], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            v_T = ttnn.permute(
                self.weights[f"{prefix}.attention.to_v.weight"], [1, 0], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self.weights[f"{prefix}.attention.qkv_fused_mmT"] = ttnn.concat(
                [q_T, k_T, v_T], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

    def _prep_parallel_weights(self):
        """Pre-transpose to_out, w1/w2/w3 weights for minimal_matmul."""
        suffixes = (
            "attention.to_out.0.weight",
            "feed_forward.w1.weight",
            "feed_forward.w2.weight",
            "feed_forward.w3.weight",
        )
        for key in list(self.weights):
            if any(key.endswith(s) for s in suffixes) and len(self.weights[key].shape) == 2:
                self.weights[key + "_mmT"] = ttnn.permute(
                    self.weights[key], [1, 0], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _ensure_tile(self, x):
        if x.get_layout() != ttnn.Layout.TILE:
            x = ttnn.to_layout(x, ttnn.Layout.TILE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    def _mm(self, x, weight, M, K, N, dtype=ttnn.DataType.BFLOAT16):
        """minimal_matmul with auto-configured blocking."""
        config = _get_matmul_config(M, K, N, self._core_grid)
        return ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=weight,
            config=config,
            compute_kernel_config=REDUCE_KERNEL,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ── Optimized norm ─────────────────────────────────────────────────────────

    def _rms_norm(self, x, norm_weight):
        """Fused RMS norm via ttnn.rms_norm."""
        x = self._ensure_tile(x)
        return ttnn.rms_norm(
            x,
            epsilon=RMS_EPS,
            weight=norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=REDUCE_KERNEL,
        )

    def _qk_norm(self, qk, norm_weight, seq_len, num_heads):
        """Fused RMS norm on head dimension using pre-reshaped [1,1,1,128] weight."""
        # Find the flat weight (reshaped from [1,1,1,64,2] to [1,1,1,128] at init)
        flat_w = None
        for key, val in self.weights.items():
            if val is norm_weight:
                flat_w = self.weights.get(key + "_flat")
                break
        assert (
            flat_w is not None
        ), "QK norm weight missing preprocessed '_flat' variant — _prep_qk_norm_weights must run first"

        qk = self._ensure_tile(qk)
        return ttnn.rms_norm(
            qk,
            epsilon=RMS_EPS,
            weight=flat_w,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=REDUCE_KERNEL,
        )

    # ── Optimized attention ────────────────────────────────────────────────────

    def _attention(self, x, seq_len, block_prefix, is_caption=False):
        """Attention with fused QKV (minimal_matmul_split) + nlp_create_qkv_heads for V."""
        x_2d = ttnn.reshape(x, [seq_len, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        N = HEADS_PER_DEV * HEAD_DIM  # 1024

        # ── Fused QKV projection ──────────────────────────────────────────────
        fused_qkv = self.weights[f"{block_prefix}.attention.qkv_fused_mmT"]
        config = _get_matmul_config(seq_len, HIDDEN_DIM, 3 * N, self._core_grid)
        q_2d, k_2d, v_2d = ttnn.experimental.minimal_matmul_split(
            x_2d,
            fused_qkv,
            chunks=3,
            dim=-1,
            config=config,
            compute_kernel_config=REDUCE_KERNEL,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # each [seq, N] BF16
        ttnn.deallocate(x_2d, False)

        # Q: reshape → QK norm → RoPE
        q_reshaped = ttnn.reshape(q_2d, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_2d, False)
        q_normed = self._qk_norm(
            q_reshaped, self.weights[f"{block_prefix}.attention.norm_q.weight"], seq_len, HEADS_PER_DEV
        )
        ttnn.deallocate(q_reshaped, False)
        q = self._apply_rope(q_normed, seq_len, HEADS_PER_DEV, is_caption=is_caption)
        ttnn.deallocate(q_normed, False)

        # K: same
        k_reshaped = ttnn.reshape(k_2d, [1, seq_len, HEADS_PER_DEV, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(k_2d, False)
        k_normed = self._qk_norm(
            k_reshaped, self.weights[f"{block_prefix}.attention.norm_k.weight"], seq_len, HEADS_PER_DEV
        )
        ttnn.deallocate(k_reshaped, False)
        k = self._apply_rope(k_normed, seq_len, HEADS_PER_DEV, is_caption=is_caption)
        ttnn.deallocate(k_normed, False)

        # V: nlp_create_qkv_heads handles head-reshape in one fused op
        v_reshaped = ttnn.reshape(v_2d, [1, 1, seq_len, N], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(v_2d, False)
        v_tiled = self._ensure_tile(v_reshaped)
        if v_tiled is not v_reshaped:
            ttnn.deallocate(v_reshaped, False)
        v, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            v_tiled,
            num_heads=HEADS_PER_DEV,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, HEADS_PER_DEV, seq, HEAD_DIM] BF16
        ttnn.deallocate(v_tiled, False)

        # ── SDPA ─────────────────────────────────────────────────────────────
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            scale=ATTN_SCALE,
            sliding_window_size=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q, False)
        ttnn.deallocate(k, False)
        ttnn.deallocate(v, False)
        attn_concat = ttnn.transformer.concatenate_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out, False)
        attn_2d = ttnn.reshape(attn_concat, [seq_len, N], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_concat, False)

        # ── to_out projection (row_par) ───────────────────────────────────────
        out_wT = self.weights[f"{block_prefix}.attention.to_out.0.weight_mmT"]
        attn_out = self._mm(attn_2d, out_wT, seq_len, N, HIDDEN_DIM)
        ttnn.deallocate(attn_2d, False)
        return self._all_reduce(attn_out, seq_len)

    # ── Optimized MLP ──────────────────────────────────────────────────────────

    def _mlp(self, x, seq_len, block_prefix):
        """SwiGLU MLP with minimal_matmul for all three projections."""
        w1T = self.weights[f"{block_prefix}.feed_forward.w1.weight_mmT"]
        w3T = self.weights[f"{block_prefix}.feed_forward.w3.weight_mmT"]
        w2T = self.weights[f"{block_prefix}.feed_forward.w2.weight_mmT"]

        gate_matmul = self._mm(x, w1T, seq_len, HIDDEN_DIM, MLP_PER_DEV)
        gate = ttnn.silu(gate_matmul, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_matmul, False)
        up = self._mm(x, w3T, seq_len, HIDDEN_DIM, MLP_PER_DEV)
        h = ttnn.multiply(gate, up, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate, False)
        ttnn.deallocate(up, False)
        out = self._mm(h, w2T, seq_len, MLP_PER_DEV, HIDDEN_DIM)
        ttnn.deallocate(h, False)
        return self._all_reduce(out, seq_len)

    # ── Async all-reduce ───────────────────────────────────────────────────────

    def _all_reduce(self, x, seq_len):
        """Async ring all-reduce via CCLManager (persistent ping-pong buffers)."""
        x_4d = ttnn.reshape(x, [1, 1, seq_len, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x, False)
        x_reduced = self._ccl.reduce_scatter(x_4d, dim=3, mesh_axis=1, use_persistent_buffer=True)
        ttnn.deallocate(x_4d, False)
        x_gathered = self._ccl.all_gather(
            x_reduced, dim=3, mesh_axis=1, use_hyperparams=True, use_persistent_buffer=True
        )
        ttnn.deallocate(x_reduced, False)
        x = ttnn.reshape(x_gathered, [1, seq_len, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_gathered, False)
        return x

    # ── Optimized final layer ──────────────────────────────────────────────────

    def _final_layer(self, x, adaln_input, seq_len):
        """Final norm (ttnn.layer_norm) + adaLN scale + linear projection."""
        final_prefix = "all_final_layer.2-1"

        cond = ttnn.silu(adaln_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scale = ttnn.matmul(
            cond,
            self.weights[f"{final_prefix}.adaLN_modulation.1.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
        )
        ttnn.deallocate(cond, False)
        scale_biased = ttnn.add(
            scale,
            self.weights[f"{final_prefix}.adaLN_modulation.1.bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(scale, False)
        scale_shifted = ttnn.add(
            self.weights["_one"], scale_biased, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(scale_biased, False)
        scale = ttnn.reshape(scale_shifted, [1, 1, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scale_shifted, False)

        x_reshaped = ttnn.reshape(x, [1, seq_len, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_3d = self._ensure_tile(x_reshaped)
        if x_3d is not x_reshaped:
            ttnn.deallocate(x_reshaped, False)

        x_norm = ttnn.layer_norm(
            x_3d,
            epsilon=LN_EPS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=REDUCE_KERNEL,
        )
        ttnn.deallocate(x_3d, False)

        x_scaled = ttnn.multiply(x_norm, scale, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_norm, False)
        ttnn.deallocate(scale, False)
        x_2d = ttnn.reshape(x_scaled, [seq_len, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_scaled, False)

        out = ttnn.matmul(
            x_2d,
            self.weights[f"{final_prefix}.linear.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            compute_kernel_config=REDUCE_KERNEL,
        )
        ttnn.deallocate(x_2d, False)
        out_biased = ttnn.add(
            out,
            self.weights[f"{final_prefix}.linear.bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(out, False)
        out = ttnn.reshape(out_biased, [1, seq_len, PATCH_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_biased, False)
        return out

    # ── Unchanged infrastructure (patchify, embed, RoPE, blocks, unpatchify) ──

    def _patchify_and_embed(self, latent):
        x_7d = ttnn.reshape(latent, [16, 1, 1, 32, 2, 32, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_permuted = ttnn.permute(x_7d, [1, 3, 5, 2, 4, 6, 0], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(x_7d, False)
        x_2d = ttnn.reshape(x_permuted, [IMG_PATCHES, PATCH_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_permuted, False)
        x_tiled = ttnn.to_layout(x_2d, ttnn.Layout.TILE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_2d, False)
        x_matmul = ttnn.matmul(
            x_tiled,
            self.weights["all_x_embedder.2-1.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
        )
        ttnn.deallocate(x_tiled, False)
        x_biased = ttnn.add(
            x_matmul,
            self.weights["all_x_embedder.2-1.bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_matmul, False)
        x = ttnn.reshape(x_biased, [1, IMG_PATCHES, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_biased, False)
        return x

    def _cap_embed(self, cap_feats):
        if len(cap_feats.shape) == 3:
            cap_feats_2d = ttnn.reshape(cap_feats, [CAP_TOKENS, 2560], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            cap_feats_2d = cap_feats
        x_normed = self._rms_norm(
            cap_feats_2d,
            norm_weight=self.weights["cap_embedder.0.weight"],
        )
        ttnn.deallocate(cap_feats_2d, False)
        x_matmul = ttnn.matmul(
            x_normed,
            self.weights["cap_embedder.1.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
        )
        ttnn.deallocate(x_normed, False)
        x_biased = ttnn.add(
            x_matmul,
            self.weights["cap_embedder.1.bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_matmul, False)
        x = ttnn.reshape(x_biased, [1, CAP_TOKENS, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_biased, False)
        return x

    def _timestep_embed(self, timestep):
        t_scaled = ttnn.multiply(
            timestep, self.weights["_t_scale"], dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        t_f32 = ttnn.typecast(t_scaled, ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(t_scaled, False)
        t = ttnn.reshape(t_f32, [1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(t_f32, False)
        freqs = ttnn.multiply(
            t, self.weights["_t_freqs"], dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(t, False)
        cos_freqs = ttnn.cos(freqs, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin_freqs = ttnn.sin(freqs, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(freqs, False)
        t_emb_concat = ttnn.concat(
            [cos_freqs, sin_freqs],
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(cos_freqs, False)
        ttnn.deallocate(sin_freqs, False)
        t_emb_typecast = ttnn.typecast(t_emb_concat, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(t_emb_concat, False)
        t_emb_matmul0 = ttnn.matmul(
            t_emb_typecast,
            self.weights["t_embedder.mlp.0.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
        )
        ttnn.deallocate(t_emb_typecast, False)
        t_emb_add0 = ttnn.add(
            t_emb_matmul0,
            self.weights["t_embedder.mlp.0.bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(t_emb_matmul0, False)
        t_emb_silu = ttnn.silu(t_emb_add0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(t_emb_add0, False)
        t_emb_matmul2 = ttnn.matmul(
            t_emb_silu,
            self.weights["t_embedder.mlp.2.weight"],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
        )
        ttnn.deallocate(t_emb_silu, False)
        t_emb_add2 = ttnn.add(
            t_emb_matmul2,
            self.weights["t_embedder.mlp.2.bias"],
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(t_emb_matmul2, False)
        t_emb = ttnn.reshape(t_emb_add2, [1, ADALN_EMBED_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(t_emb_add2, False)
        return t_emb

    def _adaLN_modulation(self, adaln_input, block_prefix):
        if block_prefix.startswith("all_final_layer"):
            w_key = f"{block_prefix}.adaLN_modulation.1.weight"
            b_key = f"{block_prefix}.adaLN_modulation.1.bias"
        else:
            w_key = f"{block_prefix}.adaLN_modulation.0.weight"
            b_key = f"{block_prefix}.adaLN_modulation.0.bias"
        mod_matmul = ttnn.matmul(
            adaln_input,
            self.weights[w_key],
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
        )
        mod_biased = ttnn.add(
            mod_matmul, self.weights[b_key], dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(mod_matmul, False)
        mod = ttnn.reshape(mod_biased, [1, 1, 4 * HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mod_biased, False)
        H = HIDDEN_DIM
        s = lambda a, b: ttnn.slice(mod, [0, 0, a], [1, 1, b], [1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scale_msa = ttnn.add(
            self.weights["_one"], s(0, H), dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate_msa = ttnn.tanh(s(H, 2 * H), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scale_mlp = ttnn.add(
            self.weights["_one"], s(2 * H, 3 * H), dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate_mlp = ttnn.tanh(s(3 * H, 4 * H), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mod, False)
        return scale_msa, gate_msa, scale_mlp, gate_mlp

    def _apply_rope(self, q_f32, seq_len, num_heads, is_caption=False):
        freqs_rm = self._build_freqs_cis(seq_len, is_caption)
        freqs_cis = ttnn.to_layout(freqs_rm, ttnn.Layout.TILE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(freqs_rm, False)
        q = ttnn.reshape(q_f32, [1, seq_len, num_heads, HEAD_DIM // 2, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_real = ttnn.slice(
            q,
            [0, 0, 0, 0, 0],
            [1, seq_len, num_heads, HEAD_DIM // 2, 1],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_imag = ttnn.slice(
            q,
            [0, 0, 0, 0, 1],
            [1, seq_len, num_heads, HEAD_DIM // 2, 2],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q, False)
        f_real = ttnn.slice(
            freqs_cis,
            [0, 0, 0, 0, 0],
            [1, seq_len, 1, HEAD_DIM // 2, 1],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        f_imag = ttnn.slice(
            freqs_cis,
            [0, 0, 0, 0, 1],
            [1, seq_len, 1, HEAD_DIM // 2, 2],
            [1, 1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(freqs_cis, False)
        # out_real = q_real * f_real - q_imag * f_imag
        qr_fr = ttnn.multiply(q_real, f_real, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qi_fi = ttnn.multiply(q_imag, f_imag, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out_real = ttnn.subtract(
            qr_fr,
            qi_fi,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qr_fr, False)
        ttnn.deallocate(qi_fi, False)
        # out_imag = q_real * f_imag + q_imag * f_real
        qr_fi = ttnn.multiply(q_real, f_imag, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_real, False)
        ttnn.deallocate(f_imag, False)
        qi_fr = ttnn.multiply(q_imag, f_real, dtype=ttnn.DataType.FLOAT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_imag, False)
        ttnn.deallocate(f_real, False)
        out_imag = ttnn.add(
            qr_fi,
            qi_fr,
            dtype=ttnn.DataType.FLOAT32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qr_fi, False)
        ttnn.deallocate(qi_fr, False)
        q_rot_concat = ttnn.concat([out_real, out_imag], dim=4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_real, False)
        ttnn.deallocate(out_imag, False)
        q_rot_reshaped = ttnn.reshape(
            q_rot_concat, [1, seq_len, num_heads, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(q_rot_concat, False)
        q_rot_permuted = ttnn.permute(
            q_rot_reshaped, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0
        )
        ttnn.deallocate(q_rot_reshaped, False)
        q_rot = ttnn.typecast(q_rot_permuted, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_rot_permuted, False)
        return q_rot

    def _build_freqs_cis(self, seq_len, is_caption=False):
        if seq_len == IMG_PATCHES:
            return self._build_freqs_img()
        elif seq_len == CAP_TOKENS:
            return self._build_freqs_cap()
        else:
            return ttnn.concat(
                [self._build_freqs_img(), self._build_freqs_cap()], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

    def _build_freqs_img(self):
        ff = self._embed_freq(self.weights["_img_f_ids"], self.weights["_freqs_F"], IMG_PATCHES, 16)
        fh = self._embed_freq(self.weights["_img_h_ids"], self.weights["_freqs_H"], IMG_PATCHES, 24)
        fw = self._embed_freq(self.weights["_img_w_ids"], self.weights["_freqs_W"], IMG_PATCHES, 24)
        return ttnn.concat([ff, fh, fw], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _build_freqs_cap(self):
        ff = self._embed_freq(self.weights["_cap_f_ids"], self.weights["_freqs_F"], CAP_TOKENS, 16)
        fhw = self._embed_freq(self.weights["_cap_hw_ids"], self.weights["_freqs_H"], CAP_TOKENS, 24)
        return ttnn.concat([ff, fhw, fhw], dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _embed_freq(self, ids, freq_table, seq_len, out_half_dim):
        ids_flat = ttnn.reshape(ids, [seq_len], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        emb = ttnn.embedding(
            ids_flat,
            freq_table,
            padding_idx=None,
            layout=ttnn.Layout.ROW_MAJOR,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(emb, [1, seq_len, 1, out_half_dim, 2], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _block_with_adaLN(self, x, adaln_input, seq_len, block_prefix):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self._adaLN_modulation(adaln_input, block_prefix)
        x_3d = ttnn.reshape(x, [1, seq_len, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x, False)

        norm1_normed = self._rms_norm(
            x_3d,
            self.weights[f"{block_prefix}.attention_norm1.weight"],
        )
        norm1_x = ttnn.multiply(
            norm1_normed, scale_msa, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(norm1_normed, False)
        ttnn.deallocate(scale_msa, False)
        attn_out = self._attention(norm1_x, seq_len, block_prefix)
        ttnn.deallocate(norm1_x, False)
        norm2_out = self._rms_norm(
            attn_out,
            self.weights[f"{block_prefix}.attention_norm2.weight"],
        )
        ttnn.deallocate(attn_out, False)
        gated_attn = ttnn.multiply(
            gate_msa, norm2_out, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(gate_msa, False)
        ttnn.deallocate(norm2_out, False)
        x = ttnn.add(
            x_3d,
            gated_attn,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_3d, False)
        ttnn.deallocate(gated_attn, False)

        norm3_normed = self._rms_norm(
            x,
            self.weights[f"{block_prefix}.ffn_norm1.weight"],
        )
        norm3_x = ttnn.multiply(
            norm3_normed, scale_mlp, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(norm3_normed, False)
        ttnn.deallocate(scale_mlp, False)
        norm3_2d = ttnn.reshape(norm3_x, [seq_len, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(norm3_x, False)
        mlp_out = self._mlp(norm3_2d, seq_len, block_prefix)
        ttnn.deallocate(norm3_2d, False)
        norm4_out = self._rms_norm(
            mlp_out,
            self.weights[f"{block_prefix}.ffn_norm2.weight"],
        )
        ttnn.deallocate(mlp_out, False)
        gated_mlp = ttnn.multiply(
            gate_mlp, norm4_out, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(gate_mlp, False)
        ttnn.deallocate(norm4_out, False)
        x_residual = ttnn.add(
            x,
            gated_mlp,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x, False)
        ttnn.deallocate(gated_mlp, False)
        return x_residual

    def _block_no_adaLN(self, x, seq_len, block_prefix, is_caption=False):
        x_3d = ttnn.reshape(x, [1, seq_len, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x, False)

        norm1_x = self._rms_norm(
            x_3d,
            self.weights[f"{block_prefix}.attention_norm1.weight"],
        )
        attn_out = self._attention(norm1_x, seq_len, block_prefix, is_caption=is_caption)
        ttnn.deallocate(norm1_x, False)
        norm2_out = self._rms_norm(
            attn_out,
            self.weights[f"{block_prefix}.attention_norm2.weight"],
        )
        ttnn.deallocate(attn_out, False)
        x = ttnn.add(
            x_3d,
            norm2_out,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_3d, False)
        ttnn.deallocate(norm2_out, False)

        norm3_x = self._rms_norm(
            x,
            self.weights[f"{block_prefix}.ffn_norm1.weight"],
        )
        norm3_2d = ttnn.reshape(norm3_x, [seq_len, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(norm3_x, False)
        mlp_out = self._mlp(norm3_2d, seq_len, block_prefix)
        ttnn.deallocate(norm3_2d, False)
        norm4_out = self._rms_norm(
            mlp_out,
            self.weights[f"{block_prefix}.ffn_norm2.weight"],
        )
        ttnn.deallocate(mlp_out, False)
        x_residual = ttnn.add(x, norm4_out, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x, False)
        ttnn.deallocate(norm4_out, False)
        return x_residual

    def _unpatchify(self, x):
        x_2d = ttnn.reshape(x, [IMG_PATCHES, PATCH_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x, False)
        x_7d = ttnn.reshape(x_2d, [1, 32, 32, 1, 2, 2, 16], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_2d, False)
        x_permuted = ttnn.permute(x_7d, [6, 0, 3, 1, 4, 2, 5], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(x_7d, False)
        x = ttnn.reshape(x_permuted, [16, 1, 64, 64], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_permuted, False)
        return x

    # ── Forward ────────────────────────────────────────────────────────────────

    def set_cap_feats(self, cap_cpu_bf16):
        """Upload caption features for a new prompt.

        Must be called before each prompt's denoising loop.  from_torch with
        ReplicateTensorToMesh is the only API that reliably writes to all 4
        BH devices; in-place copy helpers (copy_host_to_device_tensor without
        mesh_mapper) only reach device 0, leaving the other 3 with stale
        caption data that then propagates through the all_reduce ring.

        Args:
            cap_cpu_bf16: CPU BF16 tensor of shape [1, CAP_TOKENS, 2560].
        """
        self._cap_feats_buf = ttnn.from_torch(
            cap_cpu_bf16,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.ROW_MAJOR,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _forward_impl(self, latents, timestep):
        """Full forward pass."""
        latent = latents[0]
        cap_feats = self._cap_feats_buf
        x = self._patchify_and_embed(latent)
        adaln_input = self._timestep_embed(timestep)
        cap = self._cap_embed(cap_feats)

        x_3d = ttnn.reshape(x, [1, IMG_PATCHES, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x, False)
        cap_3d = ttnn.reshape(cap, [1, CAP_TOKENS, HIDDEN_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(cap, False)

        for i in range(2):
            x_3d = self._block_with_adaLN(x_3d, adaln_input, IMG_PATCHES, f"noise_refiner.{i}")
        for i in range(2):
            cap_3d = self._block_no_adaLN(cap_3d, CAP_TOKENS, f"context_refiner.{i}", is_caption=True)

        concat_xc = ttnn.concat([x_3d, cap_3d], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_3d, False)
        ttnn.deallocate(cap_3d, False)
        joint = ttnn.reshape(
            concat_xc,
            [1, IMG_PATCHES + CAP_TOKENS, HIDDEN_DIM],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(concat_xc, False)
        for i in range(30):
            joint = self._block_with_adaLN(joint, adaln_input, IMG_PATCHES + CAP_TOKENS, f"layers.{i}")

        x_sliced = ttnn.slice(
            joint, [0, 0, 0], [1, IMG_PATCHES, HIDDEN_DIM], [1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(joint, False)
        x = self._final_layer(x_sliced, adaln_input, IMG_PATCHES)
        ttnn.deallocate(x_sliced, False)
        ttnn.deallocate(adaln_input, False)
        return [self._unpatchify(x)]

    def forward(self, latents, timestep):
        """Forward pass. Call set_cap_feats(cap_cpu_bf16) before each new prompt."""
        return self._forward_impl(latents, timestep)
