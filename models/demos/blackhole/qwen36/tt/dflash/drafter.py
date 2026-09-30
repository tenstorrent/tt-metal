# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The DFlash drafter on device (ttnn), replicated across the mesh.

The target's taps never leave the mesh, and the draft logits come off the target's own resident LM
head.

Per speculative step:

1. :meth:`project_taps` — all-gather the target's fractured residual taps, then a local ``fc`` and
   ``hidden_norm``. This is the K/V source, shared by all 5 layers, and the drafter's only
   collective.
2. :meth:`forward` — Q comes from the noise block only; K/V span ``[context, noise]``. Four causal
   sliding-window layers (2048) then one bidirectional full-attention layer, which is what lets the
   15 masked slots see each other, so it must stay unmasked.
3. The final norm feeds the target's LM head.

The drafter is replicated rather than tensor-parallel because it is bound by collectives and
dispatch, not arithmetic; see :mod:`.weights`.

Two further departures from the target:

* KV history is a contiguous per-layer tensor, not a paged cache, since speculation advances 1-16
  tokens a step and the drafter attends over its own context. By
  default the history grows by concatenation each step; with ``ctx_capacity`` it is a persistent
  fixed-size buffer written in place, so every step has a constant shape that can be compiled before
  the verify trace is captured, and nothing is reallocated under it. 8 KV heads x 128 dim is
  2 KB per token per layer, so a 4096-token context is 8 MB per layer per tensor.
* RoPE tables are resident on device and sliced per step, rather than recomputed on host.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_prefill
from models.demos.blackhole.qwen36.tt.dflash.config import MAX_SEQ_LEN, DFlashDrafterConfig
from models.demos.blackhole.qwen36.tt.dflash.weights import DrafterWeights, load_drafter_weights

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG
_TILE = ttnn.TILE_SIZE

#: Additive-mask "invisible" value for the fixed-capacity path; finite, not -inf.
#:
#: The growing-history path uses -inf safely because its masks are causal over a short span, so no
#: 32-column tile is ever entirely masked. The fixed-capacity path masks the whole unwritten tail of
#: a C-row buffer, which hands the kernel fully masked blocks; a flash-style softmax takes a
#: per-block max, and for such a block that max is -inf, so it evaluates exp(-inf - -inf). A large
#: finite sentinel makes the block max finite and the weights underflow to zero instead.
_MASK_NEG = -1e9

#: Default depth of the resident RoPE tables. The drafter's positions track the target's, so this
#: caps the sequence a drafter instance can serve; it is host trig once at construction.
DEFAULT_MAX_SEQ_LEN = MAX_SEQ_LEN


class TtDFlashDrafter:
    """Device DFlash drafter, replicated on every device. Borrows the target's embedding + LM head."""

    def __init__(
        self,
        mesh_device,
        cfg: DFlashDrafterConfig,
        state_dict,
        *,
        tt_ccl=None,
        cache_path=None,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        ctx_capacity=None,
    ):
        self.device = mesh_device
        self.cfg = cfg
        self.tp = mesh_device.get_num_devices()
        self.multi = self.tp > 1
        # Needed only for the tap all-gather; a replicated drafter has no other collective.
        if tt_ccl is None and self.multi:
            from models.tt_transformers.tt.ccl import TT_CCL

            tt_ccl = TT_CCL(mesh_device)
        self.tt_ccl = tt_ccl
        # Replicated: every device runs all heads, not a fracture of them.
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.scale = self.hd**-0.5
        # nlp_create_qkv_heads' tied mode rejects a fused width that also parses as untied, i.e. it
        # needs 2*nkv*hd not to divide into (nkv + 2*nkv) sections -- head_dim not a multiple of 3.
        # Asserted at construction so a config change fails here, not mid-step inside _kv_heads.
        assert (2 * self.nkv * self.hd) % (3 * self.nkv) != 0, (
            f"head_dim {self.hd} is a multiple of 3, so the fused [k|v] block is ambiguous to "
            "nlp_create_qkv_heads(kv_tied=True); _kv_heads would need two split projections again"
        )
        self.max_seq_len = max_seq_len
        # Fixed-capacity KV mode (opt-in). None keeps the growing-concat history; an int C makes
        # every per-step shape constant and the history addresses stable. See _alloc_ctx_buffers.
        assert ctx_capacity is None or (
            ctx_capacity % 32 == 0 and ctx_capacity > 0
        ), f"ctx_capacity must be a positive multiple of the 32-row tile height, got {ctx_capacity}"
        self._cap = ctx_capacity
        self.weights: DrafterWeights = load_drafter_weights(mesh_device, state_dict, cfg, cache_path=cache_path)
        self.compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self._exact_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self._cos, self._sin = self._build_rope_tables()
        # Contiguous per-layer K/V history for the committed context; see the module docstring.
        self._ctx_k: list[ttnn.Tensor | None] = [None] * cfg.num_hidden_layers
        self._ctx_v: list[ttnn.Tensor | None] = [None] * cfg.num_hidden_layers
        self._ctx_len = 0
        if self._cap is not None:
            # A sliding layer only ever attends to the last `sliding_window` positions, so its history
            # is a ring of that size (plus slack) instead of the full capacity; position p lives in
            # row p % L. The full-attention layer keeps every position (L == capacity, so p % L == p).
            ring = -(-(cfg.sliding_window + 32) // 32) * 32
            self._ring = min(self._cap, ring)
            self._lcap = [
                min(self._cap, ring) if cfg.is_sliding(i) else self._cap for i in range(cfg.num_hidden_layers)
            ]
            self._alloc_ctx_buffers()
        # Projection matmul program configs, keyed (m_tiles, K, N); see _proj_pc.
        self._mm_pc = {}

    # ---- properties the speculative loop reads ---------------------------------------------

    @property
    def block_size(self) -> int:
        return self.cfg.block_size

    @property
    def mask_token_id(self) -> int:
        return self.cfg.mask_token_id

    @property
    def target_layer_ids(self):
        return list(self.cfg.target_layer_ids)

    @property
    def context_len(self) -> int:
        return self._ctx_len

    # ---- state -----------------------------------------------------------------------------

    def warm_block_widths(self, widths=None, seed=0):
        """Compile one program set per block width, before any trace is captured.

        A program first compiled while a verify trace is captured does not raise; it hangs the host
        on a dispatch queue that never drains. The loop reaches a narrow block whenever a generation
        ends mid-block (``verify_size = min(block_size, max_length - start,
        target.max_block(start))``), so ``q_len`` can take any value in ``1..block_size`` and every
        width has to be compiled up front.

        Requires ``ctx_capacity``. On the growing-history path the key length is
        ``hist_len + new_ctx + q_len`` and changes every step, so the shape space is unbounded. Fixed
        capacity makes the history a persistent ``[1, nkv, C, hd]`` buffer and pads the accepted
        context to a constant 16 rows, leaving ``q_len`` as the only shape variable, i.e.
        ``block_size`` program sets.

        Leaves the drafter reset, so call it before the first real generation.
        """
        import torch

        assert self._cap is not None, (
            "warm_block_widths needs ctx_capacity: on the growing-history path the key length moves "
            "with hist_len every step, so the shape space is unbounded and warming it is impossible"
        )
        widths = list(range(1, self.block_size + 1)) if widths is None else list(widths)
        gen = torch.Generator().manual_seed(seed)

        def _mk(rows):
            t = torch.randn(1, 1, rows, self.cfg.hidden_size, generator=gen, dtype=torch.float32) * 0.05
            return ttnn.from_torch(
                t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=_DRAM,
                **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)) if self.multi else {}),
            )

        for q_len in widths:
            self.reset()
            # Two steps per width: the first with no context (the prompt-shaped entry), the second
            # with one accepted row, which is the steady-state shape the loop actually runs.
            for new_ctx in (0, 1):
                kv_source = _mk(new_ctx) if new_ctx else None
                noise = _mk(q_len)
                ttnn.deallocate(self.forward(kv_source, noise, new_ctx))
                ttnn.deallocate(noise)
                if kv_source is not None:
                    ttnn.deallocate(kv_source)
        self.reset()
        ttnn.synchronize_device(self.device)

    def reset(self) -> None:
        """Drop the drafter's KV history and start a new sequence.

        On the fixed-capacity path this rewinds ``_ctx_len`` and keeps the buffers instead of freeing
        and reallocating them. From the second generation onward, reallocating would allocate while
        the verify trace is captured, which can leave corrupted buffers.

        The kept buffers are not zeroed: rows past ``_ctx_len`` are masked out rather than trusted
        (see :meth:`_alloc_ctx_buffers` and :meth:`_fixed_mask_tensors`), so stale content beyond the
        live length cannot reach attention.
        """
        if self._cap is not None and self._ctx_k[0] is not None:
            self._ctx_len = 0
            return
        for store in (self._ctx_k, self._ctx_v):
            for i, t in enumerate(store):
                if t is not None:
                    ttnn.deallocate(t)
                store[i] = None
        self._ctx_len = 0
        if self._cap is not None:
            self._alloc_ctx_buffers()

    def _replicate(self, t, layout=ttnn.TILE_LAYOUT):
        """Upload a host tensor, replicated across the mesh (the drafter runs every head everywhere)."""
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=layout,
            device=self.device,
            memory_config=_DRAM,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)) if self.multi else {}),
        )

    def _alloc_ctx_buffers(self) -> None:
        """Allocate the persistent per-layer K/V history, once, at a fixed ``[1, nkv, C, hd]``.

        The growing path rebinds ``self._ctx_k[i]`` to a freshly concatenated tensor every step, so
        both its shape and its address move, and the concat allocates while the verify trace is
        parked. Here the buffer is allocated once and only ever written through
        (``ttnn.experimental.slice_write`` into the live tensor), so its shape and address are fixed.

        Rows past ``_ctx_len`` are masked out rather than trusted (see :meth:`_fixed_mask_tensors`):
        a zero K row is not a no-op under softmax, it is a key that scores 0 against every query.
        """
        for i in range(self.cfg.num_hidden_layers):
            zeros = torch.zeros(1, self.nkv, self._lcap[i], self.hd, dtype=torch.bfloat16)
            self._ctx_k[i] = self._replicate(zeros)
            self._ctx_v[i] = self._replicate(zeros)

    def _hist_columns(self, L: int, prev: int):
        """Absolute position stored in each of a layer's ``L`` history rows, and which are live.

        Position ``p`` lives in row ``p % L``; the last ``L`` positions before ``prev`` are live.
        """
        pos = torch.arange(max(0, prev - L), prev)
        k_pos = torch.zeros(L, dtype=torch.long)
        k_real = torch.zeros(L, dtype=torch.bool)
        k_pos[pos % L] = pos
        k_real[pos % L] = True
        return k_pos, k_real

    def _fixed_mask_tensors(self, q_len: int, new_ctx: int, ctx_pad: int):
        """The two additive masks the fixed-capacity path needs: ``[1, 1, q_len, L + ctx_pad + q_len]``
        for the sliding layers (``L`` = ring size) and for the full-attention layer (``L`` = capacity).

        Key columns are laid out ``[ history(L) | padded context(16) | block(q_len) ]``:

        * ``history`` is the persistent buffer; a column is real only while it holds a position
          before this step's context (:meth:`_hist_columns`).
        * ``padded context`` is this step's newly accepted rows, left-padded to a constant 16 so the
          fused kv_proj's M is constant. Left-padding makes the position map uniform: row ``i`` of
          the 32-row kv_src is absolute position ``start - 16 + i``, contiguous across both regions.
          The pad rows duplicate positions that are already in the history buffer, so masking them is
          exact -- those positions are still attended, once, via ``history``.
        * ``block`` is the drafted slots at ``[start, start + q_len)``.

        Two masks because the layers differ: the four sliding layers need validity, causality and the
        window, while the bidirectional layer needs validity only, which matches the unmasked
        growing-history path because every row it sees there is real.
        """
        prev, start = self._ctx_len, self._ctx_len + new_ctx
        q_pos = torch.arange(start, start + q_len).unsqueeze(1)

        def _build(L: int, sliding: bool):
            kv_len = L + ctx_pad + q_len
            k_pos = torch.empty(kv_len, dtype=torch.long)
            k_real = torch.zeros(kv_len, dtype=torch.bool)
            k_pos[:L], k_real[:L] = self._hist_columns(L, prev)
            k_pos[L : L + ctx_pad] = torch.arange(start - ctx_pad, start)
            k_real[L : L + ctx_pad] = torch.arange(ctx_pad) >= (ctx_pad - new_ctx)
            k_pos[L + ctx_pad :] = torch.arange(start, start + q_len)
            k_real[L + ctx_pad :] = True
            k_pos, k_real = k_pos.unsqueeze(0), k_real.unsqueeze(0)
            visible = k_real.expand(q_len, kv_len)
            if sliding:
                visible = visible & (k_pos <= q_pos) & ((q_pos - k_pos) < self.cfg.sliding_window)
            return torch.where(visible, 0.0, _MASK_NEG).reshape(1, 1, q_len, kv_len).to(torch.bfloat16)

        return _build(self._ring, True), _build(self._cap, False)

    # ---- setup -----------------------------------------------------------------------------

    def _build_rope_tables(self):
        """Host trig once at construction; every step then slices these on device.

        The drafter cannot borrow the target's tables: it rotates the full 128-dim head at
        ``rope_theta = 1e7``, where the target rotates 64 of 256 dims at its own theta.
        """
        rope_dim = self.hd
        inv_freq = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        freqs = torch.outer(torch.arange(self.max_seq_len).float(), inv_freq)
        # HF half-split layout: [freqs, freqs], which is what rotary_embedding_hf consumes.
        emb = torch.cat([freqs, freqs], dim=-1)

        def _upload(t):
            return ttnn.from_torch(
                t.reshape(1, 1, self.max_seq_len, rope_dim).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,  # ROW_MAJOR so a non-tile-aligned row slice is legal
                device=self.device,
                memory_config=_DRAM,
                **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)) if self.multi else {}),
            )

        # Keep the host trig too: the fixed-capacity path gathers rows from it (see _fixed_rope)
        # rather than slicing the device table. bfloat16 and the same cast as the upload, so the
        # gathered rows are bit-identical to the sliced ones.
        self._cos_host = emb.cos().reshape(1, 1, self.max_seq_len, rope_dim).to(torch.bfloat16)
        self._sin_host = emb.sin().reshape(1, 1, self.max_seq_len, rope_dim).to(torch.bfloat16)
        return _upload(emb.cos()), _upload(emb.sin())

    def _fixed_rope(self, start: int, q_len: int, ctx_pad: int):
        """cos/sin for the fixed 32-row kv_src span and for the block's own queries.

        The kv_src span is ``[start - 16, start + q_len)`` -- the left-padded context then the
        block. ``start`` can be smaller than 16 on the first speculative step (the prompt may be
        shorter), which would ask for negative positions; those rows are pad and are masked out, so
        the index is clamped to 0 and they take position 0's rotation. Clamping cannot touch a real
        row: the real context occupies the last ``new_ctx`` of the 16, at positions
        ``[start - new_ctx, start)``, all non-negative.
        """
        span = torch.arange(start - ctx_pad, start + q_len).clamp_min(0)
        blk = torch.arange(start, start + q_len)
        assert start + q_len <= self.max_seq_len, (
            f"positions [{start}, {start + q_len}) exceed the drafter's {self.max_seq_len}-row RoPE "
            "tables; raise max_seq_len"
        )
        g = lambda tbl, idx: self._replicate(tbl[:, :, idx, :])  # noqa: E731
        return (
            g(self._cos_host, span),
            g(self._sin_host, span),
            g(self._cos_host, blk),
            g(self._sin_host, blk),
        )

    def _rope_slice(self, pos0: int, span: int):
        """cos/sin for absolute positions ``[pos0, pos0 + span)``, sliced on device."""
        assert pos0 + span <= self.max_seq_len, (
            f"positions [{pos0}, {pos0 + span}) exceed the drafter's {self.max_seq_len}-row RoPE "
            "tables; raise max_seq_len"
        )

        def _slice(tbl):
            r = ttnn.slice(tbl, [0, 0, pos0, 0], [1, 1, pos0 + span, self.hd])
            return ttnn.to_layout(r, ttnn.TILE_LAYOUT)

        return _slice(self._cos), _slice(self._sin)

    # ---- pieces ----------------------------------------------------------------------------

    def _proj_pc(self, m, weight, fused_activation=None):
        """Cached 1D (mcast_in0) program config for a projection matmul at this step's M.

        The drafter's projections run at M = one tile row, where ttnn's auto config sits well off the
        DRAM roofline for ``q_proj`` and especially the fused ``kv_proj`` (N = 2048, 64 tiles).
        ``o_proj``, ``up_proj`` and ``down_proj`` stay on the auto config.

        M is not constant -- ``q_len`` is the block and ``kv_seq`` is that plus the newly accepted
        context -- so configs are cached per ``(m_tiles, K, N, fused_activation)``.

        ``fused_activation`` is used only by ``_layer_mlp``'s ``gate_proj``; see there.
        """
        k, n = weight.shape[-2], weight.shape[-1]
        key = (-(-m // 32), k, n, fused_activation)
        pc = self._mm_pc.get(key)
        if pc is None:
            # Full 64-core grid, shaped wide-first (8 cols) to shorten the in0 multicast column.
            pc = self._mm_pc[key] = tpc.create_matmul_1d_decode_progcfg(
                m, k, n, num_cores=64, fused_activation=fused_activation, grid_w=8
            )
        return pc

    def _rms(self, x, weight, memory_config=None):
        """Standard RMSNorm. The gain is used as-is — the drafter is Qwen3, not zero-centered Qwen3.5."""
        return ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.rms_norm_eps, memory_config=memory_config)

    def project_taps(self, taps: list[ttnn.Tensor]) -> ttnn.Tensor:
        """``hidden_norm(fc(gather(concat(taps))))`` — the K/V source, computed once per step.

        ``taps`` are the target's residual streams at ``cfg.target_layer_ids``, in that order, each
        ``[1, 1, S, hidden/tp]`` fractured on the hidden dim. Concatenating them locally and
        all-gathering gives every device the full 25,600-wide feature in device-major column
        order, which is why ``fc``'s rows were permuted to match at load time
        (:func:`~.weights.reorder_fc_rows`). This gather is the drafter's only collective.
        """
        assert len(taps) == len(self.cfg.target_layer_ids), (
            f"got {len(taps)} taps, expected {len(self.cfg.target_layer_ids)} "
            f"({list(self.cfg.target_layer_ids)}) — order matters, fc was trained on it"
        )
        joined = ttnn.concat(list(taps), dim=-1, memory_config=_DRAM) if len(taps) > 1 else taps[0]
        if self.multi:
            from models.tt_transformers.tt.ccl import tt_all_gather

            # Full-mesh gather (cluster_axis=None): on a (1, N) mesh an explicit axis 0 holds one
            # device, which all_gather_async rejects ("num_devices > 1, but has 1").
            gathered = tt_all_gather(
                joined,
                self.device,
                self.tt_ccl,
                cluster_axis=None,
                dim=3,
                topology=ttnn.Topology.Linear,
                num_workers_per_link=2,
                chunks_per_sync=10,
            )
        else:
            gathered = joined
        projected = ttnn.linear(gathered, self.weights.fc, compute_kernel_config=self.compute_cfg, memory_config=_DRAM)
        if gathered is not joined:
            ttnn.deallocate(gathered)
        out = self._rms(projected, self.weights.hidden_norm, memory_config=_DRAM)
        ttnn.deallocate(projected)
        return out

    def _heads(self, proj, n_heads, seq):
        """``[1, 1, S, n*hd]`` -> ``[1, n, S, hd]``. Q only — K/V go through :meth:`_kv_heads`."""
        x = ttnn.reshape(proj, (1, seq, n_heads, self.hd))
        x = ttnn.transpose(x, 1, 2)
        return ttnn.to_memory_config(x, _DRAM)

    def _kv_heads(self, kv):
        """Split the fused ``kv_proj`` output ``[1, 1, S, 2*nkv*hd]`` into head-major K and V.

        One op for both, using ``nlp_create_qkv_heads``' tied-KV mode. That mode is meant for a
        ``[q | kv]`` block whose single K/V section serves both K and V, so handing it our ``[k | v]``
        block with ``num_heads == num_kv_heads == nkv`` makes the "q" section K and the tied section
        V — it returns ``(K, V, V)`` and the third output is V again, dropped here. The result is
        identical to a ``reshape`` + ``transpose`` per tensor, in one launch instead of four.
        """
        k, v, v_dup = ttnn.experimental.nlp_create_qkv_heads(
            kv,
            num_heads=self.nkv,
            num_kv_heads=self.nkv,
            transpose_k_heads=False,
            kv_tied=True,
            memory_config=_DRAM,
        )
        ttnn.deallocate(v_dup)
        return k, v

    def _sliding_mask(self, q_len, kv_len):
        """Additive ``[1, 1, q_len, kv_len]`` mask for the causal sliding-window layers.

        Built once per step, not once per layer: every layer's KV history is the same length (they
        commit in lockstep), so all four sliding layers see the same ``(q_len, kv_len)``. The
        bidirectional layer takes no mask at all.
        """
        q_pos = torch.arange(kv_len - q_len, kv_len).unsqueeze(1)
        k_pos = torch.arange(kv_len).unsqueeze(0)
        visible = (k_pos <= q_pos) & ((q_pos - k_pos) < self.cfg.sliding_window)
        mask = torch.where(visible, 0.0, float("-inf")).reshape(1, 1, q_len, kv_len)
        return ttnn.from_torch(
            mask.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_DRAM,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)) if self.multi else {}),
        )

    def _layer_attention(
        self,
        layer_idx,
        lw,
        hidden,
        ctx_rm,
        cos,
        sin,
        q_cos,
        q_sin,
        mask,
        *,
        hist_len,
        mask_full=None,
        real_ctx=None,
        ctx_pad=16,
        select=None,
    ):
        """One layer's attention. Q from ``hidden`` (the block); K/V from ``[ctx_rm, hidden]``.

        ``ctx_rm`` arrives ROW_MAJOR and ``q_cos``/``q_sin`` arrive already sliced, both because they
        are identical across the five layers — see :meth:`forward`.

        ``ctx_rm`` is the shared :meth:`project_taps` output and goes to ``kv_proj`` raw, without
        this layer's ``input_layernorm``; only the noise branch is normed. That asymmetry is the
        drafter's (see the reference's ``Qwen3DFlashAttention``), and norming it here silently costs
        acceptance.

        ``hist_len`` is how many positions this layer's KV history already holds. Every row of
        ``ctx_rm`` is accepted context by construction, so all of them are committed; the block's own
        K/V is scratch and is dropped when this call returns.
        """
        q_len = hidden.shape[-2]
        new_ctx = ctx_rm.shape[-2] if ctx_rm is not None else 0

        # Row-axis concat of two part-tile row counts (16 ctx + 16 block) cannot happen in TILE, so
        # it goes through ROW_MAJOR. ``ctx_rm`` is converted once per step in forward(), so only
        # `hidden` is untilized per layer.
        if new_ctx:
            hidden_rm = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
            kv_src = ttnn.concat([ctx_rm, hidden_rm], dim=-2, memory_config=_DRAM)
            ttnn.deallocate(hidden_rm)
            kv_src = ttnn.to_layout(kv_src, ttnn.TILE_LAYOUT)
        else:
            kv_src = hidden
        kv_seq = kv_src.shape[-2]

        # Explicit 1D progcfgs (see _proj_pc) and L1 outputs. The L1 output is legal for both
        # consumers -- _heads reshapes q and then moves it to DRAM itself, and nlp_create_qkv_heads
        # only requires an interleaved input -- and the tensors are 256 KB (q) and 128 KB (kv).
        q = ttnn.linear(
            hidden,
            lw.q_proj,
            compute_kernel_config=self.compute_cfg,
            program_config=self._proj_pc(q_len, lw.q_proj),
            memory_config=_L1,
        )
        # K and V read the same rows, so they are one fused matmul and one tied head-split. The 1D
        # progcfg and L1 output are sized for the steady state (at most one tile row); the prompt
        # step hands over the whole prompt's context at once, which must take the auto config and a
        # DRAM output or its circular buffers and output overflow L1.
        steady = kv_seq <= _TILE
        kv = ttnn.linear(
            kv_src,
            lw.kv_proj,
            compute_kernel_config=self.compute_cfg,
            program_config=self._proj_pc(kv_seq, lw.kv_proj) if steady else None,
            memory_config=_L1 if steady else _DRAM,
        )
        if new_ctx:
            ttnn.deallocate(kv_src)

        q = self._heads(q, self.nh, q_len)
        k, v = self._kv_heads(kv)
        ttnn.deallocate(kv)

        q = self._rms(q, lw.q_norm)
        k = self._rms(k, lw.k_norm)

        # RoPE: K takes every position it spans; Q takes only the trailing block positions. That
        # split is what pins the drafted block to its absolute positions. Q's cos/sin are sliced
        # once per step in forward(), not per layer.
        # K spans the whole prompt on the prompt step, which outgrows the helper's default L1 output.
        k = apply_partial_rope_prefill(k, cos, sin, self.nkv, self.hd, memory_config=None if steady else _DRAM)
        q = apply_partial_rope_prefill(q, q_cos, q_sin, self.nh, self.hd)

        hist_k, hist_v = self._ctx_k[layer_idx], self._ctx_v[layer_idx]
        if self._cap is not None:
            # Fixed-capacity: the buffer is persistent, so commit by writing through it rather than
            # by rebinding to a fresh concat. The real rows are the last `real_ctx` of the 16-row
            # padded context region (left-padded; see _fixed_mask_tensors), and they land at
            # positions `hist_len...` -- the length BEFORE this step. The commit happens after the
            # concat below, so attention reads the pre-commit history (which is what the mask
            # describes) and the rows are counted once, via the context region.
            # With `select` (the traced step) the destination rows are tensor contents, not Python
            # ints: a one-hot matrix applied in place.
            full_k = ttnn.concat([hist_k, k], dim=-2, memory_config=_DRAM)
            full_v = ttnn.concat([hist_v, v], dim=-2, memory_config=_DRAM)
            if select is not None:
                self._select_commit(k, hist_k, *select)
                self._select_commit(v, hist_v, *select)
            elif real_ctx:
                for src, dst in ((k, hist_k), (v, hist_v)):
                    rows = ttnn.slice(
                        src, (0, 0, ctx_pad - real_ctx, 0), (1, self.nkv, ctx_pad, self.hd), memory_config=_DRAM
                    )
                    self._ring_write(dst, rows, hist_len)
                    ttnn.deallocate(rows)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        else:
            # Prepend this layer's committed history, then commit the newly accepted rows.
            full_k = ttnn.concat([hist_k, k], dim=-2, memory_config=_DRAM) if hist_k is not None else k
            full_v = ttnn.concat([hist_v, v], dim=-2, memory_config=_DRAM) if hist_v is not None else v

            if new_ctx:
                keep = hist_len + new_ctx
                new_hist_k = ttnn.slice(full_k, (0, 0, 0, 0), (1, self.nkv, keep, self.hd), memory_config=_DRAM)
                new_hist_v = ttnn.slice(full_v, (0, 0, 0, 0), (1, self.nkv, keep, self.hd), memory_config=_DRAM)
                if hist_k is not None:
                    ttnn.deallocate(hist_k)
                    ttnn.deallocate(hist_v)
                self._ctx_k[layer_idx], self._ctx_v[layer_idx] = new_hist_k, new_hist_v

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            full_k,
            full_v,
            attn_mask=mask if self.cfg.is_sliding(layer_idx) else mask_full,
            is_causal=False,  # Q is the block only, so SDPA's own causal alignment would be wrong.
            scale=self.scale,
            memory_config=_DRAM,
        )
        for t in (q, full_k, full_v):
            ttnn.deallocate(t)

        # transpose + reshape rather than nlp_concat_heads: its work split is
        # num_blocks = B * S/TILE_HEIGHT, so a 16-slot block runs on a single core. It only pays at
        # prefill seq lengths; the drafter never has more than a tile row of queries.
        attn = ttnn.transpose(attn, 1, 2)
        attn = ttnn.reshape(attn, (1, 1, q_len, self.nh * self.hd))
        # Replicated, so o_proj is a plain local matmul — no all-reduce. Left on the auto program
        # config, which is already efficient at M=16 K=4096 N=5120.
        out = ttnn.linear(attn, lw.o_proj, compute_kernel_config=self.compute_cfg, memory_config=_DRAM)
        ttnn.deallocate(attn)
        return out

    def _layer_mlp(self, lw, x):
        """SwiGLU. ``up_proj`` and ``down_proj`` use the auto program config.

        ``gate_proj`` carries an explicit 1D config solely to fuse its SiLU into the packer:
        ``activation="silu"`` on the auto config emits a separate unary op, so the explicit config
        saves one dispatch per layer. ``ttnn.swiglu`` is not used because it returns the tile-padded
        height (M=32 for a 16-row input), which a 16-slot block's residual add cannot consume.

        ``gate_proj`` / ``up_proj`` are bf4 and ``down_proj`` bf8; see :data:`~.weights.MLP_DTYPE`
        and :data:`~.weights.MLP_DOWN_DTYPE`.
        """
        gate = ttnn.linear(
            x,
            lw.gate_proj,
            compute_kernel_config=self.compute_cfg,
            program_config=self._proj_pc(x.shape[-2], lw.gate_proj, fused_activation=ttnn.UnaryOpType.SILU),
            memory_config=_DRAM,
        )
        up = ttnn.linear(x, lw.up_proj, compute_kernel_config=self.compute_cfg, memory_config=_DRAM)
        hidden = ttnn.mul(gate, up, memory_config=_DRAM)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        # Replicated, so down_proj is a plain local matmul — no all-reduce.
        out = ttnn.linear(hidden, lw.down_proj, compute_kernel_config=self.compute_cfg, memory_config=_DRAM)
        ttnn.deallocate(hidden)
        return out

    def ingest_context(self, taps: list[ttnn.Tensor]) -> None:
        """Commit context rows straight into every layer's K/V history, without running a block.

        ``taps`` are the target's tap tensors for the next ``n`` positions, ``[1, 1, n, hidden/tp]``
        each. The rows go through the same ``project_taps`` -> ``kv_proj`` -> ``k_norm`` -> RoPE path
        a block step uses, so the history is the same either way; this only lets a long prompt be fed
        in bounded pieces instead of as one prompt-sized tensor. Fixed-capacity history only.
        """
        assert self._cap is not None, "ingest_context needs the fixed-capacity history"
        n = taps[0].shape[-2]
        pos0 = self._ctx_len
        assert pos0 + n <= self._cap, f"context would reach {pos0 + n} rows, past the {self._cap}-row capacity"
        src = ttnn.to_layout(self.project_taps(taps), ttnn.TILE_LAYOUT)
        cos, sin = self._rope_slice(pos0, n)
        for i, lw in enumerate(self.weights.layers):
            kv = ttnn.linear(src, lw.kv_proj, compute_kernel_config=self.compute_cfg, memory_config=_DRAM)
            k, v = self._kv_heads(kv)
            ttnn.deallocate(kv)
            k = self._rms(k, lw.k_norm)
            k = apply_partial_rope_prefill(k, cos, sin, self.nkv, self.hd, memory_config=_DRAM)
            for rows, dst in ((k, self._ctx_k[i]), (v, self._ctx_v[i])):
                self._ring_write(dst, rows, pos0)
                ttnn.deallocate(rows)
        for t in (src, cos, sin):
            ttnn.deallocate(t)
        self._ctx_len = pos0 + n

    def _ring_write(self, dst, rows, first_pos: int) -> None:
        """Write ``rows`` (``[1, nkv, n, hd]``, positions ``first_pos...``) into a history buffer whose
        row for position ``p`` is ``p % L``. Only the last ``L`` positions survive, and a write that
        crosses the end of the buffer wraps into two."""
        L = dst.shape[-2]
        n = rows.shape[-2]
        owned = []
        if n > L:
            rows = ttnn.slice(rows, (0, 0, n - L, 0), (1, self.nkv, n, self.hd), memory_config=_DRAM)
            owned.append(rows)
            first_pos, n = first_pos + n - L, L
        r0 = first_pos % L
        n1 = min(n, L - r0)
        for a, b, row in [(0, n1, r0)] + ([(n1, n, 0)] if n1 < n else []):
            part = rows
            if (a, b) != (0, n):
                part = ttnn.slice(rows, (0, 0, a, 0), (1, self.nkv, b, self.hd), memory_config=_DRAM)
                owned.append(part)
            ttnn.experimental.slice_write(
                part, dst, (0, 0, row, 0), (1, self.nkv, row + (b - a), self.hd), (1, 1, 1, 1)
            )
        for t in owned:
            ttnn.deallocate(t)

    def _select_commit(self, src, dst, sel, keep):
        """``dst[pos] = src[i]`` for the one-hot ``sel[pos, i]``; every other row of ``dst`` is kept.

        ``dst`` is written in place, so its address is unchanged. The select is exact (one 1.0 per
        selected row, HiFi4 with fp32 accumulation), so this commits the same bits a ``slice_write``
        would, with the destination rows carried in tensor contents instead of Python ints.
        """
        contrib = ttnn.matmul(sel, src, compute_kernel_config=self._exact_cfg, memory_config=_DRAM)
        kept = ttnn.multiply(dst, keep, memory_config=_DRAM)
        new = ttnn.add(kept, contrib, memory_config=_DRAM)
        ttnn.copy(new, dst)
        for t in (contrib, kept, new):
            ttnn.deallocate(t)

    # ---- traced steady-state step ----------------------------------------------------------
    #
    # In the loop's steady state the drafter's inputs are the verify trace's own tap buffers (16
    # rows, the first `produced` of them accepted) and a 16-slot block, so every shape is constant
    # and a step can be captured. What varies per step -- positions, masks, which rows are committed
    # -- lives in persistent tensors refreshed by a host copy before each replay.

    def alloc_trace_io(self) -> None:
        """Persistent per-step inputs of the traced step (RoPE, masks, commit selectors)."""
        assert self._cap is not None, "the traced step needs the fixed-capacity history"
        Q, C, R, hd = self.block_size, self._cap, self._ring, self.hd

        def _t(shape, fill=0.0):
            return self._replicate(torch.full(shape, fill, dtype=torch.bfloat16))

        self._tio = {
            "cos": _t((1, 1, 16 + Q, hd)),
            "sin": _t((1, 1, 16 + Q, hd)),
            "q_cos": _t((1, 1, Q, hd)),
            "q_sin": _t((1, 1, Q, hd)),
            "mask": _t((1, 1, Q, R + 16 + Q)),
            "mask_full": _t((1, 1, Q, C + 16 + Q)),
            "sel_ring": _t((1, 1, R, 16 + Q)),
            "keep_ring": _t((1, 1, R, 1), 1.0),
            "sel": _t((1, 1, C, 16 + Q)),
            "keep": _t((1, 1, C, 1), 1.0),
        }

    def stage_trace_inputs(self, start: int, produced: int) -> None:
        """Refresh the traced step's inputs for a block at ``start`` whose context is the last
        verify's first ``produced`` rows (positions ``[start - produced, start)``).

        The 16 context rows are the verify's rows in order, so row ``i`` sits at position
        ``start - produced + i``; only the first ``produced`` are real. Advances the history length.
        """
        Q, C, R = self.block_size, self._cap, self._ring
        prev = start - produced
        assert prev == self._ctx_len, f"drafter context is {self._ctx_len} rows, expected {prev}"
        assert start + Q <= self.max_seq_len, f"positions up to {start + Q} exceed the {self.max_seq_len}-row RoPE"
        q_pos = torch.arange(start, start + Q).unsqueeze(1)

        def _mask(L, sliding):
            kv_len = L + 16 + Q
            k_pos = torch.empty(kv_len, dtype=torch.long)
            k_real = torch.zeros(kv_len, dtype=torch.bool)
            k_pos[:L], k_real[:L] = self._hist_columns(L, prev)
            k_pos[L : L + 16] = torch.arange(prev, prev + 16)
            k_real[L : L + 16] = torch.arange(16) < produced
            k_pos[L + 16 :] = torch.arange(start, start + Q)
            k_real[L + 16 :] = True
            k_pos, k_real = k_pos.unsqueeze(0), k_real.unsqueeze(0)
            visible = k_real.expand(Q, kv_len)
            if sliding:
                visible = visible & (k_pos <= q_pos) & ((q_pos - k_pos) < self.cfg.sliding_window)
            return torch.where(visible, 0.0, _MASK_NEG).reshape(1, 1, Q, kv_len).to(torch.bfloat16)

        def _select(L):
            sel = torch.zeros(1, 1, L, 16 + Q, dtype=torch.bfloat16)
            keep = torch.ones(1, 1, L, 1, dtype=torch.bfloat16)
            rows = torch.arange(produced)
            sel[0, 0, (prev + rows) % L, rows] = 1.0
            keep[0, 0, (prev + rows) % L, 0] = 0.0
            return sel, keep

        span = torch.cat([torch.arange(prev, prev + 16), torch.arange(start, start + Q)])
        blk = torch.arange(start, start + Q)
        sel_r, keep_r = _select(R)
        sel_f, keep_f = _select(C)
        host = {
            "cos": self._cos_host[:, :, span, :],
            "sin": self._sin_host[:, :, span, :],
            "q_cos": self._cos_host[:, :, blk, :],
            "q_sin": self._sin_host[:, :, blk, :],
            "mask": _mask(R, True),
            "mask_full": _mask(C, False),
            "sel_ring": sel_r,
            "keep_ring": keep_r,
            "sel": sel_f,
            "keep": keep_f,
        }
        for name, t in host.items():
            h = ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.device)) if self.multi else {}),
            )
            ttnn.copy_host_to_device_tensor(h, self._tio[name])
        self._ctx_len = start

    def forward_traced_body(self, noise: ttnn.Tensor, taps: list) -> ttnn.Tensor:
        """The steady-state step over persistent inputs: ``taps`` are 16-row tap buffers, ``noise`` the
        ``[1, 1, block_size, hidden]`` block embedding. Returns the final-normed block hidden."""
        io = self._tio
        kv_source = self.project_taps(taps)
        ctx_rm = ttnn.to_layout(kv_source, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(kv_source)
        # One copy of the one-hot per KV head: a batch-broadcast matmul is not available to every
        # program config, and the select is per head anyway.
        sel_ring = ttnn.repeat(io["sel_ring"], ttnn.Shape([1, self.nkv, 1, 1]))
        sel_full = ttnn.repeat(io["sel"], ttnn.Shape([1, self.nkv, 1, 1]))
        x = noise
        for i, lw in enumerate(self.weights.layers):
            residual = x
            normed = self._rms(x, lw.input_layernorm, memory_config=_DRAM)
            attn = self._layer_attention(
                i,
                lw,
                normed,
                ctx_rm,
                io["cos"],
                io["sin"],
                io["q_cos"],
                io["q_sin"],
                io["mask"],
                hist_len=0,
                mask_full=io["mask_full"],
                real_ctx=16,
                ctx_pad=16,
                select=(sel_ring, io["keep_ring"]) if self.cfg.is_sliding(i) else (sel_full, io["keep"]),
            )
            ttnn.deallocate(normed)
            x = ttnn.add(residual, attn, memory_config=_DRAM)
            ttnn.deallocate(attn)
            if residual is not noise:
                ttnn.deallocate(residual)

            residual = x
            normed = self._rms(x, lw.post_attention_layernorm, memory_config=_DRAM)
            mlp = self._layer_mlp(lw, normed)
            ttnn.deallocate(normed)
            x = ttnn.add(residual, mlp, memory_config=_DRAM)
            ttnn.deallocate(mlp)
            ttnn.deallocate(residual)
        ttnn.deallocate(ctx_rm)
        ttnn.deallocate(sel_ring)
        ttnn.deallocate(sel_full)
        out = self._rms(x, self.weights.norm, memory_config=_DRAM)
        ttnn.deallocate(x)
        return out

    # ---- forward ---------------------------------------------------------------------------

    def forward(self, kv_source: ttnn.Tensor, noise: ttnn.Tensor, start: int):
        """Run the drafter over one block.

        Args:
            kv_source: :meth:`project_taps` output, ``[1, 1, n_new, hidden]`` replicated — the newly
                accepted context rows, at absolute positions ``[start - n_new, start)``.
            noise: the block's input embeddings, ``[1, 1, q_len, hidden]`` replicated.
            start: absolute position of the block's first slot.

        Returns:
            ``[1, 1, q_len, hidden]`` — the final-normed hidden states for the block.
        """
        q_len = noise.shape[-2]
        new_ctx = kv_source.shape[-2] if kv_source is not None else 0
        hist_len = self._ctx_len
        assert hist_len + new_ctx == start, (
            f"drafter context is at {hist_len} + {new_ctx} new rows but the block starts at {start}; "
            "the loop must hand over every accepted token's taps exactly once"
        )

        fixed = self._cap is not None
        if fixed:
            assert hist_len + new_ctx <= self._cap, (
                f"drafter context would reach {hist_len + new_ctx} rows, past its {self._cap}-row "
                "fixed capacity; raise ctx_capacity or fall back to the growing-history path"
            )
            # Context pads up to a multiple of 16. In the steady state new_ctx is the accept count
            # (1..block_size) so this is a constant 16 and every shape is constant. The only step
            # that exceeds it is the first, where dflash_generate hands over the whole prompt's taps
            # at once; it is the one step whose shape may differ.
            ctx_pad = max(16, -(-new_ctx // 16) * 16)
            # Everything per-step is a constant shape here: 16 padded context rows + q_len block
            # rows for kv_src, a [1,1,q_len,C+16+q_len] mask pair, and a persistent [1,nkv,C,hd]
            # history. Only tensor contents vary.
            cos, sin, q_cos, q_sin = self._fixed_rope(start, q_len, ctx_pad)
            mask, mask_full = (self._replicate(t) for t in self._fixed_mask_tensors(q_len, new_ctx, ctx_pad))
            ctx_rm = ttnn.to_layout(kv_source, ttnn.ROW_MAJOR_LAYOUT) if new_ctx else None
            if new_ctx < ctx_pad:
                pad = self._replicate(
                    # noise, not kv_source: the very first step has no accepted context at all
                    # (kv_source is None) and both carry the same hidden width.
                    torch.zeros(1, 1, ctx_pad - new_ctx, noise.shape[-1], dtype=torch.bfloat16),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                parts = [pad, ctx_rm] if ctx_rm is not None else [pad]
                ctx_rm = ttnn.concat(parts, dim=-2, memory_config=_DRAM) if len(parts) > 1 else pad
            x = noise
            for i, lw in enumerate(self.weights.layers):
                residual = x
                normed = self._rms(x, lw.input_layernorm, memory_config=_DRAM)
                attn = self._layer_attention(
                    i,
                    lw,
                    normed,
                    ctx_rm,
                    cos,
                    sin,
                    q_cos,
                    q_sin,
                    mask,
                    hist_len=hist_len,
                    mask_full=mask_full,
                    real_ctx=new_ctx,
                    ctx_pad=ctx_pad,
                )
                ttnn.deallocate(normed)
                x = ttnn.add(residual, attn, memory_config=_DRAM)
                ttnn.deallocate(attn)
                if residual is not noise:
                    ttnn.deallocate(residual)

                residual = x
                normed = self._rms(x, lw.post_attention_layernorm, memory_config=_DRAM)
                mlp = self._layer_mlp(lw, normed)
                ttnn.deallocate(normed)
                x = ttnn.add(residual, mlp, memory_config=_DRAM)
                ttnn.deallocate(mlp)
                ttnn.deallocate(residual)

            for t in (cos, sin, q_cos, q_sin, mask, mask_full, ctx_rm):
                if t is not None:
                    ttnn.deallocate(t)
            self._ctx_len = hist_len + new_ctx
            out = self._rms(x, self.weights.norm, memory_config=_DRAM)
            if x is not noise:
                ttnn.deallocate(x)
            return out

        # cos/sin cover [start - new_ctx, start + q_len): the new context rows then the block, which
        # is exactly the span K sees.
        cos, sin = self._rope_slice(start - new_ctx, new_ctx + q_len)
        # Q's own cos/sin are the trailing q_len positions, [start, start + q_len), sliced once per
        # step straight off the resident ROW_MAJOR tables (see _build_rope_tables): a
        # non-tile-aligned row slice is legal there and costs a Slice plus one Tilize.
        q_cos, q_sin = self._rope_slice(start, q_len)
        # One ROW_MAJOR copy of the context for all five layers' K/V concat (see _layer_attention).
        ctx_rm = ttnn.to_layout(kv_source, ttnn.ROW_MAJOR_LAYOUT) if new_ctx else None
        # Every layer's KV history is the same length, so one mask serves all the sliding layers.
        mask = self._sliding_mask(q_len, hist_len + new_ctx + q_len)

        x = noise
        for i, lw in enumerate(self.weights.layers):
            residual = x
            normed = self._rms(x, lw.input_layernorm, memory_config=_DRAM)
            attn = self._layer_attention(i, lw, normed, ctx_rm, cos, sin, q_cos, q_sin, mask, hist_len=hist_len)
            ttnn.deallocate(normed)
            x = ttnn.add(residual, attn, memory_config=_DRAM)
            ttnn.deallocate(attn)
            if residual is not noise:
                ttnn.deallocate(residual)

            residual = x
            normed = self._rms(x, lw.post_attention_layernorm, memory_config=_DRAM)
            mlp = self._layer_mlp(lw, normed)
            ttnn.deallocate(normed)
            x = ttnn.add(residual, mlp, memory_config=_DRAM)
            ttnn.deallocate(mlp)
            ttnn.deallocate(residual)

        for t in (cos, sin, q_cos, q_sin, mask, ctx_rm):
            if t is not None:
                ttnn.deallocate(t)
        self._ctx_len = hist_len + new_ctx

        out = self._rms(x, self.weights.norm, memory_config=_DRAM)
        if x is not noise:
            ttnn.deallocate(x)
        return out
