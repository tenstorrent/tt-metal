# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 embeddings connector: transformer blocks + final norm that refine the
per-modality GemmaFeatureExtractor features into the dims the DiT cross-attention
consumes.

Reference: ltx_core.text_encoders.gemma.embeddings_connector
"""

from __future__ import annotations

import torch

import ttnn

from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import RMSNorm
from ...models.transformers.ltx.rope_ltx import LTXRopeType, precompute_freqs_cis, reshape_interleaved_to_bhnd
from ...utils.substate import rename_substate
from ...utils.tensor import bf16_tensor


class ConnectorBlock(Module):
    """Single transformer block for the embeddings connector (pre-norm, self-attn + FF)."""

    def __init__(self, dim: int, ff_dim: int, num_heads: int, eps: float, mesh_device, ccl_manager, parallel_config):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.eps = eps
        tp_axis = parallel_config.tensor_parallel.mesh_axis

        # fp32 weights + HiFi4: bf16 rounding here costs audio PCC at no latency saving on BH.
        lin_dtype = ttnn.float32
        fidelity = ttnn.MathFidelity.HiFi4

        # FSDP: shard weights on the sequence-parallel axis (gathered per-op).
        sp = parallel_config.sequence_parallel
        fsdp_mesh_axis = sp.mesh_axis if (sp is not None and sp.factor > 1) else None

        col_kwargs = {"bias": True, "mesh_device": mesh_device, "mesh_axis": tp_axis, "dtype": lin_dtype}
        if fsdp_mesh_axis is not None:
            col_kwargs["fsdp_mesh_axis"] = fsdp_mesh_axis
            col_kwargs["ccl_manager"] = ccl_manager
        self.to_q = ColParallelLinear(dim, dim, **col_kwargs)
        self.to_k = ColParallelLinear(dim, dim, **col_kwargs)
        self.to_v = ColParallelLinear(dim, dim, **col_kwargs)
        self.to_out = Linear(dim, dim, bias=True, mesh_device=mesh_device, dtype=lin_dtype)

        # Gemma-style QK normalization over the full inner dim, applied before RoPE
        # (reference Attention uses torch.nn.RMSNorm(inner_dim): raw weight, no +1).
        self.q_norm = RMSNorm(dim, norm_eps=eps, bias=False, mesh_device=mesh_device)
        self.k_norm = RMSNorm(dim, norm_eps=eps, bias=False, mesh_device=mesh_device)
        # Per-head gated attention: gates = 2*sigmoid(to_gate_logits(x)), applied to attn output.
        # dtype=float32 routes the matmul through HiFi4 + fp32 dest acc to match the host fp32
        # baseline (mirrors attention_ltx; the gate is precision-sensitive over 8 blocks).
        # Column-parallel on heads so the gate is sharded to match the head-split SDPA output and
        # multiplies in BHNE before concatenate_heads (no full-activation reshape).
        self.to_gate_logits = ColParallelLinear(
            dim, num_heads, bias=True, dtype=ttnn.float32, mesh_device=mesh_device, mesh_axis=tp_axis
        )

        # Feed-forward (GELU gated)
        self.ff1 = ColParallelLinear(
            dim,
            ff_dim,
            bias=True,
            activation_fn="gelu",
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            dtype=lin_dtype,
            **({"fsdp_mesh_axis": fsdp_mesh_axis, "ccl_manager": ccl_manager} if fsdp_mesh_axis is not None else {}),
        )
        self.ff2 = RowParallelLinear(
            ff_dim,
            dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
            dtype=lin_dtype,
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # HiFi4 + fp32 dest-acc for the parameter-free RMS norms: the native kernel's default
        # fidelity costs ~1e-3 PCC.
        self.rmsnorm_cc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state):
        rename_substate(state, "attn1.to_q", "to_q")
        rename_substate(state, "attn1.to_k", "to_k")
        rename_substate(state, "attn1.to_v", "to_v")
        rename_substate(state, "attn1.to_out.0", "to_out")
        rename_substate(state, "ff.net.0.proj", "ff1")
        rename_substate(state, "ff.net.2", "ff2")
        # QK norms and the gated-attention gate (reference Attention uses these).
        rename_substate(state, "attn1.q_norm", "q_norm")
        rename_substate(state, "attn1.k_norm", "k_norm")
        rename_substate(state, "attn1.to_gate_logits", "to_gate_logits")

        # SPLIT-rotation checkpoint → INTERLEAVED rotary_embedding_llama: permute Q/K
        # output channels per head so the SPLIT pair (i, i+D/2) lands at adjacent
        # interleaved slots (2i, 2i+1). q_norm/k_norm share the channel layout. Mirrors
        # attention_ltx; without it interleaved RoPE on the unpermuted layout gives PCC ~0.09.
        D = self.head_dim
        perm = torch.empty(D, dtype=torch.long)
        perm[0::2] = torch.arange(D // 2)
        perm[1::2] = torch.arange(D // 2, D)

        def _permute_qk(t: torch.Tensor) -> torch.Tensor:
            rest = t.shape[1:]
            return t.reshape(self.num_heads, D, *rest).index_select(1, perm).reshape(self.num_heads * D, *rest)

        for key in ("to_q.weight", "to_q.bias", "to_k.weight", "to_k.bias", "q_norm.weight", "k_norm.weight"):
            if key in state:
                state[key] = _permute_qk(state[key])

    def forward(self, x, rope_cos=None, rope_sin=None, trans_mat=None):
        # Self-attention with residual
        residual = x
        x = ttnn.experimental.dit_rms_norm_unary_fused(
            x, weight=None, epsilon=self.eps, compute_kernel_config=self.rmsnorm_cc
        )
        attn_in = x  # normed input, reused for the per-head gate
        q = self.to_q(x, compute_kernel_config=self.compute_config)
        k = self.to_k(x, compute_kernel_config=self.compute_config)
        v = self.to_v(x, compute_kernel_config=self.compute_config)

        tp = self.parallel_config.tensor_parallel.factor
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        apply_rope = rope_cos is not None and rope_sin is not None

        # Gemma QK-norm over the full inner dim, BEFORE RoPE (raw-weight RMS, matching the
        # reference torch.nn.RMSNorm). Done on device — the q/k channels and the q_norm/k_norm
        # weights were permuted at load (SPLIT→INTERLEAVED), so RoPE below is the fast
        # interleaved rotary_embedding_llama kernel, no host round-trip. QK-norm runs over the
        # full inner dim, so TP gathers to full then re-shards for the head split.
        if apply_rope:
            if tp > 1:
                q = self.ccl_manager.all_gather(q, dim=2, mesh_axis=tp_axis, use_hyperparams=True)
                k = self.ccl_manager.all_gather(k, dim=2, mesh_axis=tp_axis, use_hyperparams=True)
            q = self.q_norm(q, compute_kernel_config=self.rmsnorm_cc)
            k = self.k_norm(k, compute_kernel_config=self.rmsnorm_cc)
            if tp > 1:
                q = ttnn.mesh_partition(q, dim=2, cluster_axis=tp_axis)
                k = ttnn.mesh_partition(k, dim=2, cluster_axis=tp_axis)

        n_local_heads = self.num_heads // tp

        # Fused head split: QK-norm ran on the full inner dim above, so split here — after norm —
        # by concatenating q|k|v and using the fused nlp_create_qkv_heads. The SPLIT→INTERLEAVED
        # channel permute (done at load) is preserved, so RoPE below is unaffected.
        B, S = q.shape[0], q.shape[1]
        qkv = ttnn.concat([q, k, v], dim=-1)  # (B, S, 3*n_local_heads*D)
        qkv = ttnn.reshape(qkv, (B, 1, S, 3 * n_local_heads * self.head_dim))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=n_local_heads,
            num_kv_heads=n_local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Interleaved RoPE on the head-split Q/K (cos/sin/trans_mat prepared by the caller).
        if apply_rope:
            q = ttnn.experimental.rotary_embedding_llama(
                q, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.compute_config
            )
            k = ttnn.experimental.rotary_embedding_llama(
                k, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.compute_config
            )

        grid = self.mesh_device.compute_with_storage_grid_size()
        sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,
        )
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=sdpa_config,
            compute_kernel_config=self.compute_config,
        )  # (B, n_local_heads, T, D)

        # Per-head gate applied in BHNE, before concatenate_heads: gates = 2*sigmoid(...) are
        # sharded on heads (ColParallel) to match, so each head's output scales by its gate with a
        # broadcast over head_dim — no reshape of the full (B,T,H*D) activation. Mirrors attention_ltx.
        gate_logits = self.to_gate_logits(attn_in)  # (B, T, n_local_heads)
        gate = ttnn.multiply(ttnn.sigmoid(gate_logits), 2.0)
        gate = ttnn.unsqueeze(ttnn.permute(gate, (0, 2, 1)), -1)  # (B, n_local_heads, T, 1)
        attn_out = ttnn.multiply(attn_out, gate)

        attn_out = ttnn.transformer.concatenate_heads(attn_out)
        attn_out = ttnn.unsqueeze(attn_out, 0)
        if tp > 1:
            attn_out = self.ccl_manager.all_gather(
                attn_out,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=True,
            )
        attn_out = ttnn.squeeze(attn_out, 0)  # (B, T, H*D), full

        attn_out = self.to_out(attn_out, compute_kernel_config=self.compute_config)
        x = attn_out + residual

        # FF with residual
        residual = x
        x = ttnn.experimental.dit_rms_norm_unary_fused(
            x, weight=None, epsilon=self.eps, compute_kernel_config=self.rmsnorm_cc
        )
        x = self.ff1(x, compute_kernel_config=self.compute_config)
        # ff2 is RowParallel: reduce_scatter internally → each device has dim/TP
        x = self.ff2(x, compute_kernel_config=self.compute_config)
        x = ttnn.unsqueeze(x, 0)
        if tp > 1:
            x = self.ccl_manager.all_gather(
                x,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=True,
            )
        x = ttnn.squeeze(x, 0)
        return x + residual


class EmbeddingsConnector(Module):
    """Transformer blocks + learnable registers + final norm. ``forward`` consumes the projected
    features from GemmaFeatureExtractor (B, seq, output_dim) and the caller-built rotation matrix,
    runs register replacement + RoPE + the blocks, and returns the host conditioning."""

    def __init__(
        self,
        *,
        output_dim: int,  # 4096 (video) or 2048 (audio)
        num_blocks: int,  # 8 for both video and audio
        num_heads: int = 32,
        ff_mult: int = 4,
        num_learnable_registers: int = 128,
        eps: float = 1e-6,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_learnable_registers = num_learnable_registers
        self.mesh_device = mesh_device
        # HiFi4 + fp32 dest-acc RMS-norm config; used by the encoder pair's final norm.
        self.rmsnorm_cc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # aggregate_embed lives in GemmaFeatureExtractor (reference FeatureExtractorV2 boundary);
        # this connector consumes the projected features.

        # Learnable registers (replace padding tokens)
        if num_learnable_registers > 0:
            self.learnable_registers = Parameter(
                total_shape=[num_learnable_registers, output_dim],
                device=mesh_device,
                dtype=ttnn.float32,
            )

        # Transformer blocks
        ff_dim = output_dim * ff_mult
        self.transformer_1d_blocks = ModuleList(
            ConnectorBlock(output_dim, ff_dim, num_heads, eps, mesh_device, ccl_manager, parallel_config)
            for _ in range(num_blocks)
        )

        # RoPE cos/sin are fixed per seq_len; cache the on-device tensors (built once, not per encode).
        self._rope_cache: dict[int, tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        # Learnable registers tiled to [1, seq, dim], cached (constant after load).
        self._reg_cache: dict[int, ttnn.Tensor] = {}

    def _rope_cos_sin(self, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Connector RoPE cos/sin for ``seq_len``, cached on device.

        Q/K weights are permuted SPLIT→INTERLEAVED at load, so the interleaved kernel matches the
        SPLIT checkpoint. Head dim is sharded on the connector TP axis (TP=1 → no-op)."""
        cached = self._rope_cache.get(seq_len)
        if cached is not None:
            return cached
        dim = self.output_dim
        num_heads = self.transformer_1d_blocks[0].num_heads
        indices_grid = torch.arange(seq_len, dtype=torch.float32).reshape(1, seq_len, 1)
        cos_freq, sin_freq = precompute_freqs_cis(
            indices_grid,
            dim=dim,
            out_dtype=torch.float32,
            theta=10000.0,
            max_pos=[4096],
            num_attention_heads=num_heads,
            rope_type=LTXRopeType.INTERLEAVED,
        )
        cos_freq = reshape_interleaved_to_bhnd(cos_freq, num_heads)
        sin_freq = reshape_interleaved_to_bhnd(sin_freq, num_heads)
        conn_tp = self.transformer_1d_blocks[0].parallel_config.tensor_parallel
        shard_kw = {"mesh_axis": conn_tp.mesh_axis, "shard_dim": 1} if conn_tp.factor > 1 else {}
        rope_cos = bf16_tensor(cos_freq, device=self.mesh_device, **shard_kw)
        rope_sin = bf16_tensor(sin_freq, device=self.mesh_device, **shard_kw)
        self._rope_cache[seq_len] = (rope_cos, rope_sin)
        return rope_cos, rope_sin

    def _tiled_registers(self, seq_len: int, dim: int) -> ttnn.Tensor:
        """Learnable registers tiled to [1, seq, dim] on device, cached (constant). Forced to
        bf16 to match the bf16 activation stream — fp32 registers (fp32-connector mode) blended
        against bf16 gathered features via ttnn.where produce NaN."""
        cached = self._reg_cache.get(seq_len)
        if cached is None:
            reps = seq_len // self.num_learnable_registers
            tiled = ttnn.repeat(self.learnable_registers.data, [reps, 1])  # [seq, dim]
            tiled = ttnn.reshape(tiled, (1, seq_len, dim))
            cached = ttnn.typecast(tiled, ttnn.bfloat16)
            self._reg_cache[seq_len] = cached
        return cached

    def build_indices(self, attention_mask: torch.Tensor, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """From the left-padded attention_mask, build the device tensors for on-device register
        replacement: ``src_idx`` [B, seq] = seq positions of the real tokens packed to the front
        (ttnn.embedding row-gather), ``keep_mask`` [B, seq, 1] selects real vs register positions.
        Tiny per-prompt inputs, static shapes → trace-safe."""
        mask = attention_mask.bool()
        B = mask.shape[0]
        idx = torch.zeros(B, seq_len, dtype=torch.int32)
        keep = torch.zeros(B, seq_len, 1)
        for b in range(B):
            real = torch.nonzero(mask[b], as_tuple=False).flatten()  # ascending real positions
            k = real.numel()
            idx[b, :k] = real.to(torch.int32)
            keep[b, :k] = 1.0
        src_idx = ttnn.from_torch(idx, device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        keep_mask = ttnn.from_torch(keep, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return src_idx, keep_mask

    def forward(
        self, features: ttnn.Tensor, src_idx: ttnn.Tensor, keep_mask: ttnn.Tensor, *, trans_mat: ttnn.Tensor
    ) -> ttnn.Tensor:
        """On-device register replacement (gather real tokens left, fill padding with registers) →
        RoPE transformer blocks → final norm, on the aggregate_embed ``features`` from
        GemmaFeatureExtractor. ``src_idx``/``keep_mask`` are the (shared) device index/mask built by
        the caller (build_indices); ``trans_mat`` is a shared constant. Returns a DEVICE tensor so
        the whole encode traces — the caller does the final to_torch."""
        seq_len, dim = features.shape[1], features.shape[-1]

        if self.num_learnable_registers > 0:
            # Row-gather real tokens to the front via ttnn.embedding (tiny [B,seq] index vs a full
            # [B,seq,dim] gather index). embedding wants a row-major table → convert features.
            features_tbl = ttnn.reshape(ttnn.to_layout(features, ttnn.ROW_MAJOR_LAYOUT), (seq_len, dim))
            packed = ttnn.embedding(src_idx, features_tbl, layout=ttnn.TILE_LAYOUT)  # [B, seq, dim]
            ttnn.deallocate(features)
            tt_x = ttnn.where(keep_mask, packed, self._tiled_registers(seq_len, dim))
        else:
            tt_x = features

        # Connector RoPE on device, cached per seq_len (see _rope_cos_sin).
        rope_cos, rope_sin = self._rope_cos_sin(seq_len)

        for block in self.transformer_1d_blocks:
            tt_x = block(tt_x, rope_cos=rope_cos, rope_sin=rope_sin, trans_mat=trans_mat)
        # Do NOT zero register positions: the reference replaces padding with learnable registers
        # then masks with all-zeros, so every token carries information after the blocks.
        return ttnn.experimental.dit_rms_norm_unary_fused(
            tt_x, weight=None, epsilon=1e-6, compute_kernel_config=self.rmsnorm_cc
        )
