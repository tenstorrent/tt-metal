# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TextEncoder (Qwen3) TTNN model — clean LightweightModule implementation.

Architecture (Qwen3Model from Tongyi-MAI/Z-Image-Turbo / text_encoder):
  - 36 transformer layers, hidden_size=2560, num_heads=32, num_kv_heads=8, head_dim=128
  - intermediate_size=9728, rms_norm_eps=1e-6, rope_theta=1000000
  - Causal self-attention (decoder-style, no KV cache)
  - 4-way tensor parallelism on (1,4) mesh:
      Q/K/V/gate/up projections: column-parallel (ShardTensorToMesh dim=0)
      O/down projections: row-parallel (ShardTensorToMesh dim=1) + all-reduce
      Norms and embedding: replicated

Weights are loaded directly from the HuggingFace PyTorch model — no tensorbin files.
RoPE tables and causal mask are precomputed at init time for a fixed sequence length.
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.z_image_turbo.tt.text_encoder import params

# ── Constants ──────────────────────────────────────────────────────────────────


NORM_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

REDUCE_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


# ── Main model class ───────────────────────────────────────────────────────────


class TextEncoderTTNN(LightweightModule):
    """
    Qwen3 text encoder in TTNN with 4-way tensor parallelism.

    Loads weights directly from a PyTorch Qwen3Model and converts them
    to TTNN with TP sharding. Uses a fixed sequence length (default 128).

    Usage::

        model = TextEncoderTTNN(mesh_device, pt_model, seq_len=128)
        tt_out = model(input_ids_tt)  # [1, 128] INT32 on device
    """

    # Architecture constants
    HIDDEN_SIZE = 2560
    NUM_HEADS = 32
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    INTERMEDIATE_SIZE = 9728
    NUM_LAYERS = 36
    RMS_NORM_EPS = 1e-6
    ROPE_THETA = 1_000_000.0
    TP = 4  # tensor parallelism degree
    ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)  # 1/sqrt(head_dim) ≈ 0.08839

    def __init__(self, mesh_device, pt_model=None, seq_len=128):
        """
        Initialize and load all weights.

        Args:
            mesh_device: TTNN MeshDevice ((1,4) mesh, already opened).
            pt_model: PyTorch Qwen3Model in eval mode. If None, loads from
                      Tongyi-MAI/Z-Image-Turbo automatically.
            seq_len: Fixed sequence length for RoPE tables and causal mask.
        """
        self.mesh_device = mesh_device
        self.seq_len = seq_len
        self.q_per_dev = self.NUM_HEADS // self.TP  # 8 Q heads per device
        self.kv_per_dev = self.NUM_KV_HEADS // self.TP  # 2 KV heads per device

        if pt_model is None:
            from transformers import AutoModel

            print("  Loading Qwen3 text encoder from HuggingFace ...")
            pt_model = AutoModel.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
                use_cache=False,
            ).eval()

        print("  Converting weights to TTNN (TP sharding) ...")
        self.weights = params.load_weights(mesh_device, pt_model)
        print("  Weight loading complete.")

        print(f"  Precomputing RoPE tables and causal mask for seq_len={seq_len} ...")
        self.cos, self.sin = self._compute_rope_tables(seq_len)
        self.attn_mask = self._compute_causal_mask(seq_len)
        print("  Precomputation complete.")

    # ── Forward pass ───────────────────────────────────────────────────────────

    def forward(self, input_ids):
        """
        Run the Qwen3 text encoder forward pass.

        Args:
            input_ids: [1, seq_len] INT32 TTNN tensor on device (replicated).

        Returns:
            [seq_len, hidden_size] bfloat16 TTNN tensor — last hidden state.
        """
        seq_len = input_ids.shape[-1]
        assert seq_len == self.seq_len, f"Input seq_len={seq_len} != precomputed seq_len={self.seq_len}"

        cos, sin, attn_mask = self.cos, self.sin, self.attn_mask

        # Token embedding: [1, seq_len] INT32 → [seq_len, 2560] BF16
        hidden_states = self._embedding(input_ids, seq_len)  # [seq_len, 2560]

        # 36 Qwen3 decoder layers
        for layer_idx in range(self.NUM_LAYERS):
            layer_out = self._decoder_layer(hidden_states, cos, sin, attn_mask, layer_idx)
            ttnn.deallocate(hidden_states, False)
            hidden_states = layer_out

        # Return pre-norm hidden states (equivalent to hidden_states[-2] in HuggingFace
        # Qwen3Model with output_hidden_states=True). The ZImagePipeline uses
        # hidden_states[-2] as cap_feats input, not last_hidden_state.
        # The cap_embedder in the transformer applies its own RMSNorm, so the
        # text encoder's final norm should NOT be applied here.
        if len(hidden_states.shape) == 3:
            hidden_states_2d = ttnn.reshape(
                hidden_states, [seq_len, self.HIDDEN_SIZE], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(hidden_states, False)
            return hidden_states_2d

        return hidden_states

    # ── Embedding ──────────────────────────────────────────────────────────────

    def _embedding(self, input_ids, seq_len):
        """
        Token embedding lookup.

        Input:  [1, seq_len] INT32 TTNN tensor on device
        Output: [seq_len, 2560] BF16 TTNN tensor

        Follows the traced op sequence:
          INT32 → TILE → UINT32 (for embedding) → ROW_MAJOR → embedding → reshape
        """
        x_tiled = ttnn.to_layout(input_ids, ttnn.Layout.TILE, None, memory_config=None)
        x_typecast = ttnn.typecast(x_tiled, ttnn.DataType.UINT32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_tiled, False)
        x_rowmajor = ttnn.to_layout(x_typecast, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
        ttnn.deallocate(x_typecast, False)

        x_embedded = ttnn.embedding(
            x_rowmajor,
            self.weights["embed_tokens.weight"],
            padding_idx=None,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, seq_len, 2560]
        ttnn.deallocate(x_rowmajor, False)

        x = ttnn.reshape(x_embedded, [seq_len, self.HIDDEN_SIZE], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_embedded, False)
        return x

    # ── RoPE ───────────────────────────────────────────────────────────────────

    def _compute_rope_tables(self, seq_len):
        """
        Compute RoPE cosine and sine tables for the given sequence length.

          inv_freq [64] × positions [0..seq_len-1] → angles [seq_len, 64]
          → doubled [seq_len, 128] → cos/sin → BF16 → [1, 1, seq_len, 128]

        Args:
            seq_len: number of tokens in the sequence

        Returns:
            cos, sin: each [1, 1, seq_len, 128] BF16 TTNN tensors (replicated)
        """
        inv_freq = self.weights["inv_freq"]  # [64] float32 on CPU
        positions = torch.arange(seq_len, dtype=torch.float32)  # [seq_len]

        # [seq_len, 64] = positions[:, None] * inv_freq[None, :]
        angles = torch.outer(positions, inv_freq)

        # Double: [seq_len, 128]
        angles_doubled = torch.cat([angles, angles], dim=-1)

        cos_table = torch.cos(angles_doubled).bfloat16().unsqueeze(0).unsqueeze(0)
        sin_table = torch.sin(angles_doubled).bfloat16().unsqueeze(0).unsqueeze(0)
        # Each: [1, 1, seq_len, 128]

        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)

        cos_tt = ttnn.from_torch(
            cos_table,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        sin_tt = ttnn.from_torch(
            sin_table,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        return cos_tt, sin_tt

    # ── Attention mask ──────────────────────────────────────────────────────────

    def _compute_causal_mask(self, seq_len):
        """
        Build a causal (lower-triangular) attention mask.

        Valid positions (i >= j): 0.0
        Masked positions (i < j): -inf

        Output: [1, 1, seq_len, seq_len] BF16 TTNN tensor (replicated)
        """
        mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.bfloat16)
        mask = mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )

        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return ttnn.from_torch(
            mask,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )

    # ── Decoder layer ──────────────────────────────────────────────────────────

    def _decoder_layer(self, hidden_states, cos, sin, attn_mask, layer_idx):
        """
        Single Qwen3 decoder layer (pre-norm, Post-RoPE).

        Structure:
            residual = hidden_states
            x = input_layernorm(hidden_states)
            x = self_attention(x, cos, sin, attn_mask)      → [1, S, H]
            hidden_states = residual + x                     broadcasts

            x2d = reshape(hidden_states, [S, H])
            residual = x2d
            x = post_attention_layernorm(x2d)
            x = mlp(x)                                       → [1, S, H]
            hidden_states = residual + x                     broadcasts

        Args:
            hidden_states: [S, H] or [1, S, H] BF16 TTNN tensor
            cos, sin:      [1, 1, S, 128] BF16 RoPE tables
            attn_mask:     [1, 1, S, S] BF16 causal mask
            layer_idx:     0-35

        Returns:
            [1, seq_len, 2560] BF16 TTNN tensor
        """
        seq_len = hidden_states.shape[-2] if len(hidden_states.shape) == 3 else hidden_states.shape[0]

        # Ensure [seq_len, hidden_size] for layernorm
        x = ttnn.reshape(hidden_states, [seq_len, self.HIDDEN_SIZE], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # ── Attention sub-layer ───────────────────────────────────────────────
        residual = x  # residual and x point to the same tensor

        x_normed = ttnn.rms_norm(
            x,
            epsilon=self.RMS_NORM_EPS,
            weight=self.weights[f"layers.{layer_idx}.input_layernorm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
            compute_kernel_config=NORM_KERNEL,
        )  # [seq_len, 2560]
        # Don't deallocate x here — residual still points to it

        x_attn_out = self._attention(x_normed, cos, sin, attn_mask, seq_len, layer_idx)
        ttnn.deallocate(x_normed, False)
        # x_attn_out: [1, seq_len, 2560]

        hidden_states = ttnn.add(
            residual,
            x_attn_out,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, seq_len, 2560] (broadcast from [S, H] + [1, S, H])
        ttnn.deallocate(residual, False)
        ttnn.deallocate(x_attn_out, False)

        # ── MLP sub-layer ─────────────────────────────────────────────────────
        x = ttnn.reshape(hidden_states, [seq_len, self.HIDDEN_SIZE], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(hidden_states, False)
        residual = x  # residual and x point to the same tensor

        x_normed = ttnn.rms_norm(
            x,
            epsilon=self.RMS_NORM_EPS,
            weight=self.weights[f"layers.{layer_idx}.post_attention_layernorm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
            compute_kernel_config=NORM_KERNEL,
        )  # [seq_len, 2560]
        # Don't deallocate x here — residual still points to it

        x_mlp_out = self._mlp(x_normed, seq_len, layer_idx)
        ttnn.deallocate(x_normed, False)
        # x_mlp_out: [1, seq_len, 2560]

        hidden_states = ttnn.add(
            residual,
            x_mlp_out,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, seq_len, 2560]
        ttnn.deallocate(residual, False)
        ttnn.deallocate(x_mlp_out, False)

        return hidden_states

    # ── Self-Attention ──────────────────────────────────────────────────────────

    def _attention(self, x, cos, sin, attn_mask, seq_len, layer_idx):
        """
        Qwen3 grouped-query self-attention with RoPE and 4-way TP.

        Per device:
          Q: 8 heads × 128 dim  (col_par on q_proj)
          K: 2 heads × 128 dim  (col_par on k_proj)
          V: 2 heads × 128 dim  (col_par on v_proj)
          O: row_par on o_proj  + all-reduce

        GQA ratio: 4:1  (8 Q heads per 2 KV heads per device)

        Args:
            x:          [seq_len, 2560] BF16
            cos, sin:   [1, 1, seq_len, 128] BF16
            attn_mask:  [1, 1, seq_len, seq_len] BF16
            seq_len:    sequence length (int)
            layer_idx:  layer index

        Returns:
            [1, seq_len, 2560] BF16 (after all-reduce)
        """
        li = layer_idx  # shorthand

        # ── Q projection (col_par: 8 Q heads per device) ─────────────────────
        q = ttnn.matmul(
            x,
            self.weights[f"layers.{li}.self_attn.q_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 1024] per device

        q_reshaped = ttnn.reshape(
            q, [1, seq_len, self.q_per_dev, self.HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, seq_len, 8, 128]
        ttnn.deallocate(q, False)

        q_normed = ttnn.rms_norm(
            q_reshaped,
            epsilon=self.RMS_NORM_EPS,
            weight=self.weights[f"layers.{li}.self_attn.q_norm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
            compute_kernel_config=NORM_KERNEL,
        )  # [1, seq_len, 8, 128]
        ttnn.deallocate(q_reshaped, False)

        q_permuted = ttnn.permute(q_normed, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(q_normed, False)
        # [1, 8, seq_len, 128]

        q_embedded = ttnn.experimental.rotary_embedding(
            q_permuted, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(q_permuted, False)
        q = ttnn.slice(
            q_embedded,
            [0, 0, 0, 0],
            [1, self.q_per_dev, seq_len, self.HEAD_DIM],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 8, seq_len, 128] — trims TILE padding from rotary_embedding
        ttnn.deallocate(q_embedded, False)

        # ── K projection (col_par: 2 KV heads per device) ────────────────────
        k = ttnn.matmul(
            x,
            self.weights[f"layers.{li}.self_attn.k_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 256] per device

        k_reshaped = ttnn.reshape(
            k, [1, seq_len, self.kv_per_dev, self.HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, seq_len, 2, 128]
        ttnn.deallocate(k, False)

        k_normed = ttnn.rms_norm(
            k_reshaped,
            epsilon=self.RMS_NORM_EPS,
            weight=self.weights[f"layers.{li}.self_attn.k_norm.weight"],
            bias=None,
            residual_input_tensor=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
            compute_kernel_config=NORM_KERNEL,
        )  # [1, seq_len, 2, 128]
        ttnn.deallocate(k_reshaped, False)

        k_permuted = ttnn.permute(k_normed, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(k_normed, False)
        # [1, 2, seq_len, 128]

        k_embedded = ttnn.experimental.rotary_embedding(
            k_permuted, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(k_permuted, False)
        k = ttnn.slice(
            k_embedded,
            [0, 0, 0, 0],
            [1, self.kv_per_dev, seq_len, self.HEAD_DIM],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 2, seq_len, 128]
        ttnn.deallocate(k_embedded, False)

        # ── V projection (col_par: 2 KV heads per device) ────────────────────
        v = ttnn.matmul(
            x,
            self.weights[f"layers.{li}.self_attn.v_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 256] per device

        v_reshaped = ttnn.reshape(
            v, [1, seq_len, self.kv_per_dev, self.HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1, seq_len, 2, 128]
        ttnn.deallocate(v, False)

        v = ttnn.permute(v_reshaped, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG, pad_value=0.0)
        ttnn.deallocate(v_reshaped, False)
        # [1, 2, seq_len, 128] — no RoPE on V

        # ── Scaled Dot-Product Attention (GQA: 4 Q heads per KV head) ─────────
        # Expand K/V from [1, 2, S, 128] to [1, 8, S, 128] using grouped
        # (repeat_interleave) style: each KV head is repeated 4 times.
        # NOTE: ttnn.transformer.scaled_dot_product_attention with GQA (unequal
        # Q/KV head counts) produces incorrect results in this TTNN version.
        # Manual SDPA (matmul + scale + mask + softmax + matmul) is used instead.
        grp = self.q_per_dev // self.kv_per_dev  # 4
        k_exp = self._expand_kv(k, self.kv_per_dev, grp, seq_len)
        ttnn.deallocate(k, False)
        v_exp = self._expand_kv(v, self.kv_per_dev, grp, seq_len)
        ttnn.deallocate(v, False)
        # k_exp, v_exp: [1, 8, seq_len, 128] each

        scores = ttnn.matmul(
            q,
            k_exp,
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
        )  # [1, 8, seq_len, seq_len]
        ttnn.deallocate(q, False)
        ttnn.deallocate(k_exp, False)

        scores_scaled = ttnn.multiply(
            scores,
            self.ATTN_SCALE,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(scores, False)

        scores = ttnn.add(
            scores_scaled,
            attn_mask,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # causal mask broadcast: [1,1,S,S] → [1,8,S,S]
        ttnn.deallocate(scores_scaled, False)

        attn_w = ttnn.softmax(scores, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(scores, False)

        out = ttnn.matmul(
            attn_w,
            v_exp,
            transpose_a=False,
            transpose_b=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
        )  # [1, 8, seq_len, 128]
        ttnn.deallocate(attn_w, False)
        ttnn.deallocate(v_exp, False)

        out_concat = ttnn.transformer.concatenate_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out, False)
        # [1, seq_len, 1024]

        out_reshaped = ttnn.reshape(
            out_concat, [seq_len, self.q_per_dev * self.HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [seq_len, 1024]
        ttnn.deallocate(out_concat, False)

        # ── O projection (row_par) + TP all-reduce ────────────────────────────
        out = ttnn.matmul(
            out_reshaped,
            self.weights[f"layers.{li}.self_attn.o_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 2560] — partial sum per device
        ttnn.deallocate(out_reshaped, False)

        return self._all_reduce(out, seq_len)  # [1, seq_len, 2560]

    # ── KV head expansion for GQA ──────────────────────────────────────────────

    def _expand_kv(self, kv, kv_heads, grp, seq_len):
        """
        Expand KV from [1, kv_heads, seq_len, head_dim] to
        [1, kv_heads * grp, seq_len, head_dim] using grouped repeat_interleave
        style: each KV head is repeated 'grp' times contiguously.

        Example (kv_heads=2, grp=4):
          KV[0] KV[0] KV[0] KV[0] KV[1] KV[1] KV[1] KV[1]

        Args:
            kv:       [1, kv_heads, seq_len, head_dim] BF16 TTNN tensor
            kv_heads: number of KV heads per device (2)
            grp:      repeat count per KV head (4 = q_per_dev // kv_per_dev)
            seq_len:  sequence length

        Returns:
            [1, kv_heads * grp, seq_len, head_dim] BF16 TTNN tensor
        """
        slices = []
        unique_slices = []
        for kv_i in range(kv_heads):
            s = ttnn.slice(
                kv,
                [0, kv_i, 0, 0],
                [1, kv_i + 1, seq_len, self.HEAD_DIM],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            unique_slices.append(s)
            for _ in range(grp):
                slices.append(s)
        result = ttnn.concat(slices, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for s in unique_slices:
            ttnn.deallocate(s, False)
        return result

    # ── MLP ────────────────────────────────────────────────────────────────────

    def _mlp(self, x, seq_len, layer_idx):
        """
        Qwen3 SwiGLU MLP with 4-way TP.

        Structure:
            gate = SiLU(gate_proj(x))   col_par: 2432 per device
            up   = up_proj(x)           col_par: 2432 per device
            h    = gate * up
            out  = down_proj(h)         row_par + all-reduce

        Args:
            x:         [seq_len, 2560] BF16
            seq_len:   sequence length (int)
            layer_idx: layer index

        Returns:
            [1, seq_len, 2560] BF16 (after all-reduce)
        """
        li = layer_idx

        # Gate projection with SiLU fused (col_par)
        gate = ttnn.matmul(
            x,
            self.weights[f"layers.{li}.mlp.gate_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation="silu",
            compute_kernel_config=None,
        )  # [seq_len, 2432]

        # Up projection (col_par)
        up = ttnn.matmul(
            x,
            self.weights[f"layers.{li}.mlp.up_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 2432]

        # Element-wise product (SwiGLU)
        h = ttnn.multiply(gate, up, dtype=ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate, False)
        ttnn.deallocate(up, False)
        # [seq_len, 2432]

        # Down projection (row_par) + all-reduce
        out = ttnn.matmul(
            h,
            self.weights[f"layers.{li}.mlp.down_proj.weight"],
            transpose_a=False,
            transpose_b=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.DataType.BFLOAT16,
            program_config=None,
            activation=None,
            compute_kernel_config=None,
        )  # [seq_len, 2560] — partial sum per device
        ttnn.deallocate(h, False)

        return self._all_reduce(out, seq_len)  # [1, seq_len, 2560]

    # ── TP All-Reduce (reduce_scatter + all_gather) ────────────────────────────

    def _all_reduce(self, x, seq_len):
        """
        Tensor-parallel all-reduce via reduce_scatter + all_gather (ring topology).

        Matches the traced pattern exactly:
          [S, H] → reshape [1,1,S,H] → reduce_scatter(dim=3) → [1,1,S,H//4]
          → reshape [S, H//4] → all_gather(dim=1) → [S, H]
          → reshape [1, S, H]

        Args:
            x:       [seq_len, hidden_size] BF16 — partial sum on each device
            seq_len: sequence length (int)

        Returns:
            [1, seq_len, hidden_size] BF16 — fully summed, same on all devices
        """
        H = self.HIDDEN_SIZE  # 2560

        x_reshaped = ttnn.reshape(x, [1, 1, seq_len, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x, False)

        x_reduced = ttnn.reduce_scatter(
            input_tensor=x_reshaped,
            dim=3,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=None,
            topology=ttnn.Topology.Ring,
            compute_kernel_config=REDUCE_KERNEL,
        )  # [1, 1, seq_len, H//4] per device
        ttnn.deallocate(x_reshaped, False)

        x_reduced_2d = ttnn.reshape(x_reduced, [seq_len, H // self.TP], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_reduced, False)
        # [seq_len, 640]

        x_gathered = ttnn.all_gather(
            input_tensor=x_reduced_2d,
            dim=1,
            cluster_axis=1,
            subdevice_id=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            num_links=None,
            topology=ttnn.Topology.Ring,
        )  # [seq_len, H]
        ttnn.deallocate(x_reduced_2d, False)

        x = ttnn.reshape(x_gathered, [1, seq_len, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_gathered, False)
        # [1, seq_len, 2560]

        return x
