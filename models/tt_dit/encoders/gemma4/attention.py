# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 text attention (sliding GQA + full GQA, with K=V quirk for full).

Reference:
    transformers.models.diffusion_gemma.modeling_diffusion_gemma.
    DiffusionGemmaEncoderTextAttention / DiffusionGemmaDecoderTextAttention.

Single class with ``is_sliding`` flag controlling hyperparameters; the forward takes
an optional ``encoder_kv=(K, V)`` for decoder use (cross-attention to a read-only
cached encoder KV). The encoder/decoder distinction is otherwise made by the caller
via the supplied attention mask.

Layer-type differences:
  sliding_attention  →  head_dim=256, num_kv_heads=8,  separate v_proj.
  full_attention     →  head_dim=512, num_kv_heads=2,  v_proj=None  (V is fed from
                        the raw k_proj output, prior to k_norm/RoPE), with its own
                        v_norm (no scale).

TP sharding: q/k/v projections are column-parallel on the *head* axis (heads split
across the TP mesh axis); o_proj is row-parallel. Per-head RMSNorm runs locally over
the head-dim axis (no CCL needed).

NOTE on KV-head TP: ``num_kv_heads`` is 8 (sliding) or 2 (full). TP factor must
divide both — i.e. TP ∈ {1, 2}. Higher TP factors require KV-head replication
(deferred until we land the padding/replication wrapper).
"""

from __future__ import annotations

import ttnn

from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module
from ...layers.normalization import RMSNorm
from ...parallel.config import DiTParallelConfig

TILE = ttnn.TILE_SIZE


class Gemma4Attention(Module):
    """Sliding/full GQA attention. Used by both encoder and decoder text layers."""

    def __init__(
        self,
        *,
        is_sliding: bool,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        sliding_window: int | None,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()

        self.is_sliding = is_sliding
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window if is_sliding else None
        self.parallel_config = parallel_config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager

        tp_factor = parallel_config.tensor_parallel.factor
        assert (
            num_attention_heads % tp_factor == 0
        ), f"num_attention_heads={num_attention_heads} must divide tp_factor={tp_factor}"
        assert head_dim % TILE == 0, f"head_dim={head_dim} must be tile-aligned"
        assert hidden_size % TILE == 0

        # KV-head sharding: either num_kv_heads divides tp_factor (standard shard, replication=1)
        # OR tp_factor divides num_kv_heads (also standard shard, more KV heads per device) OR
        # tp_factor > num_kv_heads AND num_kv_heads divides tp_factor (replicate each KV head
        # ``tp_factor // num_kv_heads`` times so each rank gets exactly 1 effective KV head).
        # This is necessary for full-attention layers where num_global_key_value_heads = 2 — without
        # replication, TP > 2 would be impossible on the full layers.
        if tp_factor <= num_kv_heads:
            assert (
                num_kv_heads % tp_factor == 0
            ), f"num_kv_heads={num_kv_heads} must divide tp_factor={tp_factor} when tp_factor ≤ num_kv_heads"
            self._kv_replication = 1
            effective_num_kv_heads = num_kv_heads
        else:
            assert (
                tp_factor % num_kv_heads == 0
            ), f"tp_factor={tp_factor} must be a multiple of num_kv_heads={num_kv_heads} for KV replication"
            self._kv_replication = tp_factor // num_kv_heads
            effective_num_kv_heads = num_kv_heads * self._kv_replication  # == tp_factor

        self.num_local_heads = num_attention_heads // tp_factor
        self.num_local_kv_heads = effective_num_kv_heads // tp_factor
        self._effective_num_kv_heads = effective_num_kv_heads

        col_kwargs = dict(
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        # Projections. K/V output dim uses ``effective_num_kv_heads`` so the per-rank slice
        # is ``self.num_local_kv_heads`` heads regardless of whether we replicated.
        self.q_proj = ColParallelLinear(hidden_size, num_attention_heads * head_dim, **col_kwargs)
        self.k_proj = ColParallelLinear(hidden_size, effective_num_kv_heads * head_dim, **col_kwargs)
        # Full attention has no v_proj — V is sourced from the raw k_proj output (pre-k_norm, pre-RoPE).
        self.v_proj = (
            ColParallelLinear(hidden_size, effective_num_kv_heads * head_dim, **col_kwargs) if is_sliding else None
        )

        # o_proj: heads are already TP-sharded → row-parallel into hidden_size.
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        # Per-head RMSNorm over head_dim (head_dim is fully local — no CCL needed).
        # q/k norms have learned weight; v_norm has no scale (with_scale=False in HF).
        norm_kwargs = dict(
            embedding_dim=head_dim,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.q_norm = RMSNorm(**norm_kwargs)
        self.k_norm = RMSNorm(**norm_kwargs)
        self.v_norm = RMSNorm(
            embedding_dim=head_dim,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=mesh_device,
        )

        # SDPA + compute kernel configs (matches gemma3 / Wan conventions).
        # ``q_chunk_size`` / ``k_chunk_size`` are starting values; tune per mesh shape after first
        # hardware run (see Wan's ``sdpa_chunk_size_map`` for an example of per-mesh tuning).
        self.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        # Accuracy-first compute config. Reasoning per knob:
        #   HiFi4: max bf16 accumulator fidelity — 4 fidelity passes per matmul/norm step.
        #   fp32_dest_acc_en=True: intermediate accumulator in fp32 (else bf16 partial sums).
        #   packer_l1_acc=False: skip the packer's low-precision L1 accumulation; each tile
        #     writes go through the fp32 dest, then external accumulation — mirrors tt_dit's
        #     LayerNorm default (normalization.py:98). ``True`` is faster but drops precision
        #     over deep reductions (head_dim=256 or 512 here).
        # Post-validation, revisit ``packer_l1_acc=True`` / HiFi2 for perf if PCC still meets
        # the layer/model-level thresholds.
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict) -> None:
        """When TP > num_kv_heads, replicate each K/V head ``kv_replication`` times along the
        head axis so the per-rank slice has exactly one (replicated) KV head."""
        import torch  # local import to keep top-level deps minimal

        if self._kv_replication == 1:
            return

        H_kv = self.num_kv_heads
        D = self.head_dim
        rep = self._kv_replication
        for name in ("k_proj", "v_proj"):
            w = state.get(f"{name}.weight")
            if w is None:
                continue
            # HF weight shape: [num_kv_heads * head_dim, hidden_size]. Repeat each KV head ``rep``
            # times contiguously so the column-parallel split puts the right replica on each rank.
            hidden = w.shape[-1]
            w = w.reshape(H_kv, D, hidden)
            w = w.repeat_interleave(rep, dim=0)
            state[f"{name}.weight"] = w.reshape(H_kv * rep * D, hidden)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        encoder_kv: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            hidden_states:   replicated [1, B, seq, hidden_size] (tt_dit ``1BND`` convention).
            cos, sin:        [B, 1, seq, head_dim/2]  (from Gemma4RotaryEmbedding.get_cos_sin)
            attention_mask:  None or bf16 additive mask broadcastable to attention scores.
                              Mutually exclusive with implicit causal — when None, SDPA runs
                              with `is_causal=True`; when set, SDPA uses the provided mask.
            encoder_kv:      decoder mode — (K_enc, V_enc) sharded ``[B, num_local_kv_heads, src_seq, head_dim]``
                              (K_enc already has RoPE applied).

        Returns:
            (out, K_local, V_local) — `out` replicated [1, B, seq, hidden_size]; the K_local/V_local
            tensors are the post-RoPE K and post-v_norm V *from this call* (TP-sharded on heads,
            shape [B, num_local_kv_heads, seq, head_dim]). The text-encoder model collects these
            per layer to feed the decoder later. Pass ``None`` through in non-encoder use.
        """
        # Read batch/seq from the trailing 3 dims so both 3D and 4D ``1BND`` inputs work.
        B, S = hidden_states.shape[-3], hidden_states.shape[-2]

        # 1. Projections (column-parallel over head axis). ``parallel_config`` is only passed
        # when the input is TP-fractured on its last dim (triggers the fused
        # ``all_gather_minimal_matmul_async`` path). Here ``hidden_states`` is replicated —
        # each device runs ``minimal_matmul`` against its local weight shard.
        # ``compute_kernel_config=self.compute_config`` overrides Linear's default HiFi2 with
        # our HiFi4 config; the four chained proj matmuls (Q/K/V/O) accumulate drift at HiFi2
        # that lands ~1e-3 short of the 0.999 attention PCC target vs the fp32 HF reference.
        q = self.q_proj(hidden_states, compute_kernel_config=self.compute_config)
        k = self.k_proj(hidden_states, compute_kernel_config=self.compute_config)
        # K=V quirk for full attention: V uses the raw k_proj output BEFORE k_norm / RoPE.
        # `v_raw = k` is a tensor alias — safe because ttnn ops are non-in-place by default
        # (RMSNorm/RoPE return new tensors). The underlying k_proj output lives until both
        # the k-path (k_norm → RoPE) and v-path (v_norm) have consumed it.
        v_raw = k if not self.is_sliding else self.v_proj(hidden_states, compute_kernel_config=self.compute_config)

        # 2. Reshape (B, S, H_local * D) → (B, S, H_local, D) → permute (B, H_local, S, D).
        q = ttnn.permute(ttnn.reshape(q, (B, S, self.num_local_heads, self.head_dim)), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, (B, S, self.num_local_kv_heads, self.head_dim)), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v_raw, (B, S, self.num_local_kv_heads, self.head_dim)), (0, 2, 1, 3))

        # 3. Per-head RMSNorm over head_dim (last axis); fully local.
        # Pass our HiFi4 / fp32-dest-acc config: RMSNorm's default is unset (device default is
        # HiFi2 / bf16 acc), which squares bf16 head_dim values into a bf16 accumulator and
        # noticeably degrades PCC of every downstream matmul.
        q = self.q_norm(q, compute_kernel_config=self.compute_config)
        k = self.k_norm(k, compute_kernel_config=self.compute_config)
        v = self.v_norm(v, compute_kernel_config=self.compute_config)

        # 4. RoPE applied to Q and K. V is NOT rotated.
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        # Capture per-layer K/V (post-RoPE for K, post-v_norm for V) for the decoder's later cross-attention.
        # We yield references — caller decides whether to retain or deallocate.
        k_out = k
        v_out = v

        # 5. Decoder cross-attention: prepend the encoder cache along the seq axis.
        # For sliding_attention layers with long encoder prompts, HF *slices* the encoder
        # cache to its last ``sliding_window`` tokens before concatenating with the canvas
        # (see ``create_diffusion_decoder_attention_mask`` in HF's modeling_diffusion_gemma:
        # ``sliding_mask = full_mask[..., sliding_start_idx:sliding_end_idx]`` +
        # ``pad(sliding_mask, (0, canvas_length), value=True)``). Every canvas query then
        # attends to (last-W of encoder cache) + (all canvas tokens) with a bidirectional
        # zero mask. This is a *different* mechanism from encoder self-attention's
        # per-query sliding, so we implement it by slicing the cache here — NOT by passing
        # ``sliding_window_size`` to SDPA in the decoder path.
        # For short encoder prompts (src_seq ≤ sliding_window) the slice is a no-op.
        if encoder_kv is not None:
            k_enc, v_enc = encoder_kv
            if self.is_sliding and self.sliding_window is not None:
                src_seq = k_enc.shape[2]
                if src_seq > self.sliding_window:
                    start = src_seq - self.sliding_window
                    # ttnn.slice is [start_indices, end_indices] over all dims.
                    k_enc = ttnn.slice(k_enc, [0, 0, start, 0], [*k_enc.shape[:2], src_seq, k_enc.shape[3]])
                    v_enc = ttnn.slice(v_enc, [0, 0, start, 0], [*v_enc.shape[:2], src_seq, v_enc.shape[3]])
            k = ttnn.concat([k_enc, k], dim=2)
            v = ttnn.concat([v_enc, v], dim=2)

        # 6. GQA expand K and V to match Q heads. repeat_interleave ([kv0,kv0,kv1,kv1,...]) matches HF repeat_kv.
        if self.num_local_kv_heads < self.num_local_heads:
            repeats = self.num_local_heads // self.num_local_kv_heads
            k = ttnn.repeat_interleave(k, repeats, dim=1)
            v = ttnn.repeat_interleave(v, repeats, dim=1)

        # Ensure DRAM interleaved layout for SDPA.
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)

        # 7. SDPA with scale=1.0 (HF Gemma 4 explicitly sets scaling=1.0 — q_norm replaces the 1/sqrt(d)
        # normalization). ttnn SDPA: is_causal and attn_mask are mutually exclusive.
        #
        # sliding_window_size (encoder-mode only): HF Gemma 4 enforces per-query sliding-window
        # for ``sliding_attention`` layers via the SDPA op (see
        # ``demos/gemma4/tt/attention/operations.py::chunked_prefill_sdpa_sliding``). This
        # applies to encoder self-attention. For decoder cross-attention HF uses a different
        # mechanism (slice the encoder cache to last-W, then bidirectional attention), which
        # we handle above by slicing ``encoder_kv``. So: only pass ``sliding_window_size``
        # when we're in encoder mode (``encoder_kv is None``).
        _sliding_window_size = (
            self.sliding_window
            if (self.is_sliding and self.sliding_window is not None and encoder_kv is None)
            else None
        )
        _sdpa_kwargs = {"sliding_window_size": _sliding_window_size} if _sliding_window_size is not None else {}
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=(attention_mask is None),
            attn_mask=attention_mask,
            scale=1.0,
            program_config=self.sdpa_config,
            compute_kernel_config=self.compute_config,
            **_sdpa_kwargs,
        )

        # 8. (B, H_local, S, D) → (B, S, H_local * D) → 4D → o_proj → 4D [1, B, S, hidden_size].
        attn = ttnn.transformer.concatenate_heads(attn)  # (B, S, H_local * D), 3D
        attn = ttnn.unsqueeze(attn, 0)  # back to 4D 1BND for o_proj + downstream ops
        out = self.o_proj(attn, compute_kernel_config=self.compute_config)
        # RowParallelLinear's output is N-fractured on the TP axis (tt_dit convention — the impl
        # does a per-device matmul, no allreduce). Gather along the last dim so downstream
        # residual adds and the returned tensor are replicated.
        if self.parallel_config.tensor_parallel.factor > 1:
            out = self.ccl_manager.all_gather_persistent_buffer(
                out, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        return out, k_out, v_out

    @staticmethod
    def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """Half-dim interleaved rotation.

        x:   [B, H, S, head_dim]; cos/sin: [B, 1, S, head_dim/2].
        Returns [B, H, S, head_dim] with the first half rotated by (cos, sin):
            out1 = x1 * cos - x2 * sin
            out2 = x2 * cos + x1 * sin
            out  = concat(out1, out2, -1)
        """
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        out1 = ttnn.subtract(ttnn.multiply(x1, cos), ttnn.multiply(x2, sin))
        out2 = ttnn.add(ttnn.multiply(x2, cos), ttnn.multiply(x1, sin))
        return ttnn.concat([out1, out2], dim=-1)
