# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The DFlash drafter on device (ttnn), **replicated** across the mesh.

Replaces the host PyTorch drafter: the target's taps never leave the mesh, and the draft logits come
off the target's own resident LM head.

Architecture, per speculative step:

1. :meth:`project_taps` — all-gather the target's fractured residual taps, then a **local** ``fc``
   and ``hidden_norm``. This is the K/V source, shared by all 5 layers. That gather is the drafter's
   only collective.
2. :meth:`forward` — Q comes from the noise block only; K/V span ``[context, noise]``. Four causal
   sliding-window layers (2048) then one **bidirectional** full-attention layer — that last one is
   what lets the 15 masked slots see each other, so it must stay unmasked.
3. The final norm feeds the target's LM head.

Why replicated and not tensor-parallel
--------------------------------------
See :mod:`.weights` for the measurement. Short version: the drafter's arithmetic is trivial (~55
GFLOP for a 16-token block) and a TP version was bound by collectives — 22 CCL ops per step, 45% of
wall clock, 0.70x the host PyTorch drafter. Replicating makes every projection local and leaves one
tap all-gather.

Two further departures from the target, both deliberate:

* **KV history is plain growing tensors, concatenated per step**, not a paged cache. The target's
  paged path can only write a multi-token run at a bucket-aligned offset (see
  :class:`~...reference.dflash.targets.TtTarget`) and speculation advances 1-16 tokens a step, so a
  contiguous history sidesteps the problem entirely. It is cheap at drafter shapes: 8 KV heads x 128
  dim is 2 KB per token per layer, so a 4096-token context is 8 MB per layer per tensor.
* **RoPE tables are resident on device** and sliced per step, rather than recomputed on host.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_prefill
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig
from models.demos.blackhole.qwen36.tt.dflash.weights import DrafterWeights, load_drafter_weights

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG

#: Default depth of the resident RoPE tables. The drafter's positions track the target's, so this
#: caps the sequence a drafter instance can serve; it is host trig once at construction.
DEFAULT_MAX_SEQ_LEN = 4096


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
        **weight_dtypes,
    ):
        """``weight_dtypes`` is passed through to :func:`~.weights.load_drafter_weights`
        (``proj_dtype`` / ``mlp_dtype`` / ``mlp_down_dtype``), so a caller can build two drafters
        that differ only in weight precision. That is what
        ``tests/reference/test_dflash_acceptance.py`` uses to price the bf4 MLP against a bf8 one
        under a single target load; the defaults are the shipped choice.
        """
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
        # needs 2*nkv*hd NOT to divide into (nkv + 2*nkv) sections -- head_dim not a multiple of 3.
        # Asserted at construction so a config change fails here, not mid-step inside _kv_heads.
        assert (2 * self.nkv * self.hd) % (3 * self.nkv) != 0, (
            f"head_dim {self.hd} is a multiple of 3, so the fused [k|v] block is ambiguous to "
            "nlp_create_qkv_heads(kv_tied=True); _kv_heads would need two split projections again"
        )
        self.max_seq_len = max_seq_len
        self.weights: DrafterWeights = load_drafter_weights(
            mesh_device, state_dict, cfg, cache_path=cache_path, **weight_dtypes
        )
        self.compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self._cos, self._sin = self._build_rope_tables()
        # Contiguous per-layer K/V history for the committed context; see the module docstring.
        self._ctx_k: list[ttnn.Tensor | None] = [None] * cfg.num_hidden_layers
        self._ctx_v: list[ttnn.Tensor | None] = [None] * cfg.num_hidden_layers
        self._ctx_len = 0
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

    def reset(self) -> None:
        """Drop the drafter's KV history and start a new sequence."""
        for store in (self._ctx_k, self._ctx_v):
            for i, t in enumerate(store):
                if t is not None:
                    ttnn.deallocate(t)
                store[i] = None
        self._ctx_len = 0

    # ---- setup -----------------------------------------------------------------------------

    def _build_rope_tables(self):
        """Host trig ONCE at construction; every step then slices these on device.

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

        return _upload(emb.cos()), _upload(emb.sin())

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

        The drafter's projections all run at M = one tile row, where ttnn's **auto** config sits far
        off the DRAM roofline -- catastrophically so for the fused ``kv_proj``, whose N = 2048 is 64
        tiles. MEASURED by ``tests/perf/test_dflash_attn_matmul_sweep.py`` (full 8x8 grid, the
        drafter's own dtypes and compute config), against auto:

            q_proj   M=16 K=5120 N=4096   120.0 -> 108.7 us   (-9.4 %)
            kv_proj  M=32 K=5120 N=2048   116.5 ->  53.5 us   (-54.0 %, 39 us roofline)
            o_proj   M=16 K=4096 N=5120   111.1 -> 111.0 us   (auto is already optimal -- o_proj is
                                                               deliberately left on auto)

        PCC is unchanged (0.999967 either way). Three families LOST and should not be retried here:
        the DRAM-sharded weight layout (+11 % q, +24 % o, and its in0 reshard is a second op),
        ``create_prefill_mlp_matmul_program_config`` (+23 % to +37 %, it is a 2D config for M in the
        thousands), and ``COMPUTE_HIFI2_NO_FP32_ACC``, which was slower AND cost PCC.
        ``create_prefill_kpass1_matmul_program_config`` is illegal at M = 32.

        M is not constant -- ``q_len`` is the block and ``kv_seq`` is that plus the newly accepted
        context -- so configs are cached per ``(m_tiles, K, N, fused_activation)`` instead of built
        once.

        ``fused_activation`` is used by exactly one caller, ``_layer_mlp``'s ``gate_proj``, and NOT
        for speed -- see there.
        """
        k, n = weight.shape[-2], weight.shape[-1]
        key = (-(-m // 32), k, n, fused_activation)
        pc = self._mm_pc.get(key)
        if pc is None:
            # Full grid: the sweep put nc=64 first at both shapes, and the factory shapes it
            # wide-first (8 cols) which is what shortens the in0 multicast column.
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
        all-gathering gives every device the full 25,600-wide feature in **device-major** column
        order, which is why ``fc``'s rows were permuted to match at load time
        (:func:`~.weights.reorder_fc_rows`). This gather is the drafter's only collective.
        """
        assert len(taps) == len(self.cfg.target_layer_ids), (
            f"got {len(taps)} taps, expected {len(self.cfg.target_layer_ids)} "
            f"({list(self.cfg.target_layer_ids)}) — order matters, fc was trained on it"
        )
        joined = ttnn.concat(list(taps), dim=-1, memory_config=_DRAM) if len(taps) > 1 else taps[0]
        if self.multi:
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            # The repo's full-mesh (cluster_axis=None) gather. NOT ccl.tt_all_gather: that takes
            # cluster_axis literally and on a (1, N) mesh axis 0 holds one device, which
            # all_gather_async rejects ("num_devices > 1, but has 1").
            gathered = tpc.tuned_vocab_all_gather(
                joined,
                self.device,
                self.tt_ccl,
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

        One op for both, using ``nlp_create_qkv_heads``' **tied-KV** mode. That mode is meant for a
        ``[q | kv]`` block whose single K/V section serves both K and V, so handing it our ``[k | v]``
        block with ``num_heads == num_kv_heads == nkv`` makes the "q" section K and the tied section
        V — it returns ``(K, V, V)`` and the third output is V again, dropped here.

        VERIFIED BIT-EXACT against the ``reshape`` + ``transpose`` form it replaces: K, V and the
        duplicate all ``torch.equal``, at NKV=8 / HD=128 / S=32 (the drafter's own geometry).

        MEASURED (T3K, ctx64, 5 layers): 251 us against 287 us for the 2 ``ReshapeView`` + 2
        ``Transpose`` it replaces, on top of the ``kv_proj`` fusion's own -273 us of matmul. The
        margin is thin because this op's work split gives a 32-row input a single core (the same
        reason ``nlp_concat_heads`` LOSES on the way out -- see :meth:`_layer_attention`); it wins
        only because it does K and V in one launch instead of four.
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

    def _layer_attention(self, layer_idx, lw, hidden, ctx_rm, cos, sin, q_cos, q_sin, mask, *, hist_len):
        """One layer's attention. Q from ``hidden`` (the block); K/V from ``[ctx_rm, hidden]``.

        ``ctx_rm`` arrives **ROW_MAJOR** and ``q_cos``/``q_sin`` arrive already sliced, both because
        they are identical across the five layers — see :meth:`forward`.

        ``ctx_kv`` is the shared :meth:`project_taps` output — note it goes to ``kv_proj``
        **raw**, without this layer's ``input_layernorm``. Only the noise branch is normed. That
        asymmetry is the drafter's, not an oversight (see the reference's ``Qwen3DFlashAttention``),
        and norming it here silently costs acceptance.

        ``hist_len`` is how many positions this layer's KV history already holds. Every row of
        ``ctx_kv`` is accepted context by construction, so all of them are committed; the block's own
        K/V is scratch and is dropped when this call returns.
        """
        q_len = hidden.shape[-2]
        new_ctx = ctx_rm.shape[-2] if ctx_rm is not None else 0

        # Row-axis concat of two part-tile row counts (16 ctx + 16 block) cannot happen in TILE, so
        # it goes through ROW_MAJOR. Doing that explicitly, with ``ctx_rm`` converted ONCE per step
        # in forward(), leaves only `hidden`'s untilize per layer -- ttnn's implicit conversion
        # untilized BOTH operands every layer (MEASURED: 2 x 11 us per layer, 5 of the 10
        # UntilizeWithUnpadding ops in the step were re-doing identical work).
        if new_ctx:
            hidden_rm = ttnn.to_layout(hidden, ttnn.ROW_MAJOR_LAYOUT)
            kv_src = ttnn.concat([ctx_rm, hidden_rm], dim=-2, memory_config=_DRAM)
            ttnn.deallocate(hidden_rm)
            kv_src = ttnn.to_layout(kv_src, ttnn.TILE_LAYOUT)
        else:
            kv_src = hidden
        kv_seq = kv_src.shape[-2]

        # Explicit 1D progcfgs, and L1 outputs: both are measured wins at these shapes (see
        # _proj_pc). The L1 output is legal for both consumers -- _heads reshapes q and then hands
        # it back to DRAM itself, and nlp_create_qkv_heads only requires an INTERLEAVED input --
        # and the tensors are 256 KB (q) and 128 KB (kv).
        q = ttnn.linear(
            hidden,
            lw.q_proj,
            compute_kernel_config=self.compute_cfg,
            program_config=self._proj_pc(q_len, lw.q_proj),
            memory_config=_L1,
        )
        # K and V read the same rows, so they are one fused matmul and one tied head-split.
        kv = ttnn.linear(
            kv_src,
            lw.kv_proj,
            compute_kernel_config=self.compute_cfg,
            program_config=self._proj_pc(kv_seq, lw.kv_proj),
            memory_config=_L1,
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
        # once per step in forward(), not here -- see there for why that mattered.
        k = apply_partial_rope_prefill(k, cos, sin, self.nkv, self.hd)
        q = apply_partial_rope_prefill(q, q_cos, q_sin, self.nh, self.hd)

        # Prepend this layer's committed history, then commit the newly accepted rows.
        hist_k, hist_v = self._ctx_k[layer_idx], self._ctx_v[layer_idx]
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
            attn_mask=mask if self.cfg.is_sliding(layer_idx) else None,
            is_causal=False,  # Q is the block only, so SDPA's own causal alignment would be wrong.
            scale=self.scale,
            memory_config=_DRAM,
        )
        for t in (q, full_k, full_v):
            ttnn.deallocate(t)

        # NOT nlp_concat_heads, though it is bit-identical here (verified torch.equal at q_len=16)
        # and one op instead of two: its work split is num_blocks = B * S/TILE_HEIGHT, so a 16-slot
        # block is ONE block and it runs on ONE core -- 63 us/layer against 49 us (42 reshape + 7
        # transpose) for this pair on 64 cores (MEASURED, T3K ctx64: 314 us vs 245 us over 5 layers,
        # +69 us). It only pays at prefill seq lengths; the drafter never has more than a tile row
        # of queries.
        attn = ttnn.transpose(attn, 1, 2)
        attn = ttnn.reshape(attn, (1, 1, q_len, self.nh * self.hd))
        # Replicated, so o_proj is a plain local matmul — no all-reduce. Left on the AUTO program
        # config deliberately: the sweep in _proj_pc found nothing that beats it at M=16 K=4096
        # N=5120 (every explicit config tied or lost), unlike q_proj and kv_proj.
        out = ttnn.linear(attn, lw.o_proj, compute_kernel_config=self.compute_cfg, memory_config=_DRAM)
        ttnn.deallocate(attn)
        return out

    def _layer_mlp(self, lw, x):
        """SwiGLU. All three matmuls stay on the **auto** program config, deliberately.

        ``up_proj`` and ``down_proj`` stay on the **auto** program config; ``gate_proj`` carries an
                explicit one solely to fuse its SiLU into the packer (below).

                SWEPT and nothing beat auto on TIME — the opposite of ``_layer_attention``, where the same
                families won big (see :meth:`_proj_pc`). ``tests/perf/test_dflash_mlp_matmul_sweep.py``,
                M=16 and the drafter's own dtypes, best explicit candidate of ~14 per shape:

                    gate  M=16 K=5120  N=17408   auto 490 us   best 1D 511 us   +4.3 %
                    up    M=16 K=5120  N=17408   auto 469 us   best 1D 495 us   +5.5 %
                    down  M=16 K=17408 N=5120    auto 447 us   best 1D 454 us   +1.7 %

                Three results worth keeping:

                * **DRAM-sharded is a trap here.** On ``down`` it is +207 % (1,371 us against 447 us), and on
                  ``gate`` it is not even legal -- the DRAM-sharded matmul rejects a fused activation
                  (``!parameters.user_fused_activation.has_value``). tt-perf-report prints "try a DRAM-sharded
                  program config" for these ops; at M = one tile row that advice is wrong.
                * **Fusing gate|up into one matmul does NOT pay**, unlike the same trick on ``kv_proj``:
                  975 us fused against 490 + 469 = 959 us split, and that is *before* the slice + silu the
                  fused form needs (SILU cannot be packer-fused across the ``up`` half).
                * ``activation="silu"`` here does **not** fuse into the packer on the auto path -- it emits a
                  separate ~18 us ``Unary`` (visible as 5 ops / 94 us in the profile). An explicit progcfg
                  does genuinely fuse it, and still loses overall (511 us as one op vs 472 + 18).

                What DID pay here is not a program config at all but the **weight dtype**: ``gate_proj`` and
                ``up_proj`` are bf4 (-43 %/-45 %, these are pure weight streaming so half the bytes is half
                the time), while ``down_proj`` stays bf8 because bf4 on it fails the drafter's PCC gate. The
                measurements and the accuracy trade are in :data:`~.weights.MLP_DOWN_DTYPE`.

                THE SILU IS FUSED INTO THE PACKER, and that is a DISPATCH-COUNT change, not a speed one.
                ``activation="silu"`` on the auto config does **not** fuse -- it emits a separate
                ``UnaryDeviceOperation`` (18 us x 5 layers, 94 us/step in the profile). An explicit progcfg
                with ``fused_activation=SILU`` genuinely fuses it. MEASURED over the whole SwiGLU block
                (``tests/perf/test_dflash_swiglu_fusion_sweep.py``, BFP4 weights):

                    current       561.5 us   4 ops   matmul 521 + mul 23 + unary 18
                    packer_fused  563.0 us   3 ops   matmul 540 + mul 23           <- shipped
                    manual_silu   578.6 us   4 ops   the control: explicit progcfg, silu NOT fused

                So the fusion itself is free and the +19 us is the explicit progcfg -- ``manual_silu``
                isolates that, being +17 us with no fusion at all. Device time is a wash (+1.5 us, the
                difference of two real numbers); what it buys is **5 fewer dispatches per step**, which is
                worth having in an untraced module at ~13 % device utilization, on the same reasoning that
                kept the tied K/V head-split in :meth:`_kv_heads`.

                ``ttnn.swiglu`` was also measured and is NOT a fused kernel: it decomposes into
                Slice + Unary + BinaryNg (5 ops, 623.5 us, +62 us) and returns the tile-PADDED height
                (M=32 for a 16-row input), which a 16-slot block's residual add cannot consume.
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

        # cos/sin cover [start - new_ctx, start + q_len): the new context rows then the block, which
        # is exactly the span K sees.
        cos, sin = self._rope_slice(start - new_ctx, new_ctx + q_len)
        # Q's own cos/sin are the trailing q_len positions, [start, start + q_len). Sliced HERE,
        # once, straight off the resident ROW_MAJOR tables -- which is what those tables are
        # ROW_MAJOR for (see _build_rope_tables): a non-tile-aligned row slice is legal there and
        # costs a Slice plus one Tilize. _layer_attention used to slice the already-TILIZED cos/sin
        # instead, and once per layer, so each layer paid untilize + slice + retilize for an
        # identical result. MEASURED: 130 us and 30 ops per step became ~9 us and 4 ops.
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
