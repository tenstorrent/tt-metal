# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel prefill on the SINGLE-CHIP (SC) model classes.

Same wavefront as sp_prefill.py's ``SPPrefill`` (one long prompt split into n_spans contiguous
spans, one per die of a (1, n_spans) mesh; GDN state and cumulative K/V prefixes flow strictly
forward die d -> d+1 over MeshSockets; the host builds the program LAYER-MAJOR across dies), but
each die is the plain single-device ``Qwen36Model`` (num_devices == 1, sequence_parallel=False ->
Qwen36GatedAttention / Qwen36GatedDeltaNet / the SC MLP and all their single-device prefill
optimizations) instead of the TP-path classes at tp=1, which are ~1.6x slower per layer.

Span lengths: uniform span_len per die by default; ``spans=[L_0, ..., L_{n-1}]`` (or env QWEN36_SP_SPANS,
e.g. "1152,1024,1024,896") gives die d its own length L_d (each a multiple of 128, sum = the prompt length),
so later dies (longer SDPA prefixes; the last die also runs the LM-head tail) can take fewer tokens.
Below, s_d = L_0 + ... + L_{d-1} is die d's span start (uniform: s_d = d * span_len).

Die d processes its span as one chunk of the SC chunked prefill (chunk_size = L_d, chunk start s_d), i.e. the
same per-layer calls ``Qwen36Model._forward_prefill_chunk`` makes, on the persistent chunk buffers
``Qwen36Model._prepare_prefill_trace_chunked_setup`` creates:
  * cos/sin = the RoPE table slice (s_d, L_d), chunk start = s_d, own-chunk page table = blocks
    [s_d / 64, (s_d + L_d) / 64), M4 R4B last-row position = s_d + L_d - 1. All fixed per die, written once
    at construction.
  * Full-attention layer: die d > 0 first receives the K/V prefix [0, s_d) from die d - 1 and
    paged_fill_cache's it into its own paged cache (blocks [0, s_d / 64)); the layer then fills its own
    blocks and runs the flexible chunked SDPA over [0, s_d + L_d). Right after the mixer
    (Qwen36DecoderLayer.forward's post_mixer_hook) die d < last sends the cumulative prefix
    [0, s_d + L_d) = [0, s_{d+1}) to die d + 1.
  * GDN layer: the SC GDN runs with ``_chunk_inplace_state`` (the persistent external state is both
    the initial state and the in-place output). Die d > 0 receives die d - 1's recurrent + conv state
    DIRECTLY into that persistent state; die 0 zeroes it at the start of every request. Right after
    the mixer die d < last sends its (updated in place) state to die d + 1.
  * Tail (last die): final norm + LM head on the last row (greedy uint32 token on device via
    ``_exact_multiple_tail_device``, or the full logits row with return_logits=True).

Traced path: ``capture`` warms (one eager layer-major pass that also compiles both tails on the real
SP hidden), then opens a trace on EVERY die, builds the same layer-major pass once more (no host writes:
the per-request input is only the persistent ``_chunk_token_buf``, written by ``prefill_traced`` before
each replay; die 0's GDN state zeroing is a device copy inside its trace), and closes all traces. The
trace outputs are the last die's greedy uint32 token and its (DRAM) final hidden; ``prefill_traced``
reads the token (TTFT) and, outside the timed window, runs the eager logits tail on the hidden.
"""
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.common import create_tt_model
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.sp_prefill import (
    BLOCK_SIZE,
    _build_hop_sockets,
    _identity_page_table,
    _kv_paged_to_seq,
    _resolve_socket_config,
    _socket_recv_op,
    _socket_send_op,
    _sp,
)
from models.tt_transformers.tt.common import Mode


class SPPrefillSC:
    """Sequence-parallel prefill engine on the single-chip model classes: n_spans dies, each a 1x1
    submesh running a full-weight single-device ``Qwen36Model``. Public API mirrors ``SPPrefill``
    (see sp_prefill.py for the socket modes / FIFO arguments)."""

    def __init__(
        self,
        mesh_device,
        *,
        n_spans=4,
        span_len=1024,
        max_seq_len=4096,
        hf_model=None,
        layer_indices=None,
        state_socket_mode="direct",
        kv_socket_mode="direct",
        state_fifo_bytes=None,
        kv_fifo_bytes=None,
        spans=None,
    ):
        """spans: optional per-die span lengths (list of n_spans ints, each a multiple of 128, summing to
        n_spans * span_len, the prompt length); None -> env QWEN36_SP_SPANS (comma list) if set, else uniform
        span_len on every die (the original behavior)."""
        (
            state_socket_mode,
            kv_socket_mode,
            state_fifo_bytes,
            kv_fifo_bytes,
            socket_storage,
        ) = _resolve_socket_config(state_socket_mode, kv_socket_mode, state_fifo_bytes, kv_fifo_bytes)
        self.state_socket_mode = state_socket_mode
        self.kv_socket_mode = kv_socket_mode

        assert tuple(mesh_device.shape) == (1, n_spans), (
            f"SPPrefillSC assumes a (1, {n_spans}) mesh for the forward die d -> d+1 adjacency, "
            f"got {tuple(mesh_device.shape)}"
        )
        self.n_spans = n_spans
        spans_src = "spans="
        if spans is None:
            env = os.environ.get("QWEN36_SP_SPANS", "").strip()
            if env:
                spans = [int(v) for v in env.replace(" ", "").split(",") if v != ""]
                spans_src = "QWEN36_SP_SPANS="
        if spans is None:
            spans = [int(span_len)] * n_spans
        else:
            spans = [int(v) for v in spans]
            assert len(spans) == n_spans, f"SPPrefillSC: {spans_src}{spans} has {len(spans)} entries, need {n_spans}"
            assert sum(spans) == n_spans * span_len, (
                f"SPPrefillSC: {spans_src}{spans} sums to {sum(spans)}, but the prompt length is "
                f"n_spans * span_len = {n_spans * span_len}"
            )
            logger.info(
                f"[SPPrefillSC] per-die span lengths {spans} (from {spans_src.rstrip('=')}; uniform would be "
                f"{span_len}); span starts {[sum(spans[:d]) for d in range(n_spans)]}"
            )
        for d, L in enumerate(spans):
            # _prepare_prefill_trace_chunked_setup: chunk_size (== L_d) must be a multiple of 128; hence every
            # span start s_d is a multiple of 128 too: of the paged-KV block (64) and of the flexible SDPA
            # q_chunk (64 / 128), as the device chunk start requires.
            assert L > 0 and L % 128 == 0, f"die {d}: span length {L} must be a positive multiple of 128"
        self.spans = spans
        self.span_starts = [sum(spans[:d]) for d in range(n_spans)]
        self.total_len = sum(spans)
        # Uniform spans: the common span length (the original attribute); uneven spans: None (use self.spans).
        self.span_len = spans[0] if len(set(spans)) == 1 else None
        # The SC model's RoPE table / paged cache must cover the whole prompt (die d reads [0, s_d + L_d)).
        self.max_seq_len = max(int(max_seq_len), self.total_len)
        assert self.max_seq_len % BLOCK_SIZE == 0, "max_seq_len must be a multiple of the paged-KV block size (64)"
        self.num_blocks = self.max_seq_len // BLOCK_SIZE

        self.mesh_device = mesh_device
        self.subs = mesh_device.create_submeshes(ttnn.MeshShape(1, 1))
        assert len(self.subs) == n_spans, f"expected {n_spans} submeshes, got {len(self.subs)}"

        # Safe defaults so close() can always run cleanly, even if construction fails partway.
        self._trace_ids = [None] * n_spans
        self.state_sockets = []
        self.kv_sockets = []
        self.rbuf = {}
        self.models = []
        self.args_list = []
        self.last_first_token = None
        # ---- trace state (populated by capture(), released by _release_traces()) ----
        self._traced_tok = None  # last die: persistent greedy uint32 token the trace writes
        self._traced_hidden = None  # last die: persistent DRAM final hidden the trace writes (logits tail input)
        self._trace_pc_entries = None  # per-die program-cache entry counts at capture (no compile after park)
        self._trace_refs_snap = None  # tensors whose addresses the traces bake in (identity-checked on replay)

        try:
            # ---- build die 0 (warms the tensor cache), then dies 1..n_spans-1 (reuse the state_dict).
            # Plain single-device args (sequence_parallel=False) on a 1x1 submesh -> the SC classes. ----
            t0 = time.perf_counter()
            args0, model0, state_dict = create_tt_model(
                self.subs[0],
                max_batch_size=1,
                max_seq_len=self.max_seq_len,
                hf_model=hf_model,
                sequence_parallel=False,
                layer_indices=layer_indices,
            )
            self.models.append(model0)
            self.args_list.append(args0)
            for d in range(1, n_spans):
                args_d = Qwen36ModelArgs(
                    mesh_device=self.subs[d], max_batch_size=1, max_seq_len=self.max_seq_len, sequence_parallel=False
                )
                if layer_indices is not None:
                    # Mirror create_tt_model's layer_indices handling so every die builds the SAME layers.
                    args_d.layer_indices = list(layer_indices)
                    args_d.n_layers = len(args_d.layer_indices)
                model_d = Qwen36Model(self.subs[d], args_d, state_dict, tensor_cache_path=args_d.weight_cache_path())
                self.models.append(model_d)
                self.args_list.append(args_d)
            del state_dict
            for d, m in enumerate(self.models):
                assert m._single_device, f"die {d}: expected the single-device (SC) model classes"
            logger.info(f"[SPPrefillSC] built {n_spans} SC die models in {time.perf_counter() - t0:.1f}s")

            self.nkv = self.args_list[0].n_kv_heads
            self.hd = self.args_list[0].head_dim
            self.vocab_size = self.args_list[0].vocab_size

            # ---- per die: paged KV + GDN external state, then the SC chunked-prefill setup (persistent
            # chunk buffers, GDN in-place binding, zero buffers, compile warm-up of one chunk forward). ----
            self._page_table_host = torch.arange(self.num_blocks, dtype=torch.int32).unsqueeze(0)
            kv_shape = [self.num_blocks, self.nkv, BLOCK_SIZE, self.hd]
            self.kv_dtype = ttnn.bfloat8_b if os.environ.get("QWEN36_ATTN_KV_BF8", "0") == "1" else ttnn.bfloat16
            if (
                os.environ.get("QWEN36_ATTN_KV_BF8", "0") == "1"
                and os.environ.get("QWEN36_SP_BF8_GATE_UNFUSE", "1") == "1"
                and os.environ.get("QWEN36_ATTN_GATE_CAST", "1") == "0"
            ):
                # QWEN36_ATTN_KV_BF8=1 also typecasts Q to bfloat8_b, so the SDPA output (and the head concat)
                # is bfloat8_b while the attention gate stays bfloat16. The fused gate multiply
                # (ttnn.multiply(a, b, input_tensor_b_activations=[SIGMOID]), QWEN36_ATTN_GATE_FUSED=1) is wrong
                # for mixed a/b dtypes (standalone: PCC 0.66 bf8 x bf16, 0.74 bf16 x bf8; 1.0 same-dtype), which
                # took the SP HF PCC to 0.41. By default (QWEN36_ATTN_GATE_CAST=1) ttnn_gated_attention.py
                # typecasts the gate to the SDPA output dtype before the fused multiply, so the fused path stays
                # on; only with QWEN36_ATTN_GATE_CAST=0 fall back to the two-op sigmoid + multiply path (read per
                # call by ttnn_gated_attention.py, so it covers the warm-up and the traces).
                # QWEN36_SP_BF8_GATE_UNFUSE=0 keeps the fused (broken) path.
                if os.environ.get("QWEN36_ATTN_GATE_FUSED", "1") != "0":
                    logger.warning(
                        "[SPPrefillSC] QWEN36_ATTN_KV_BF8=1: forcing QWEN36_ATTN_GATE_FUSED=0 (fused sigmoid-gate "
                        "multiply is inaccurate for bf8 x bf16 operands)"
                    )
                os.environ["QWEN36_ATTN_GATE_FUSED"] = "0"
            t1 = time.perf_counter()
            for d, (sub, model) in enumerate(zip(self.subs, self.models)):
                # QWEN36_ATTN_KV_BF8=1: bfloat8_b paged KV (the SC attention then typecasts K/V to the cache
                # dtype before paged_fill_cache, which requires input dtype == cache dtype). Default bf16.
                model.allocate_kv_caches(kv_shape, self.kv_dtype, batch_size=1)
                if d == n_spans - 1:
                    # Greedy on-device token for the tail: set before the setup so its warm-up compiles it.
                    model.set_greedy_token_output(True)
                model._prepare_prefill_trace_chunked_setup(
                    sub,
                    self._page_table_host,
                    self.spans[d],
                    warmup_masked_buckets=False,
                    request_warm_len=0,
                )
                self._write_die_fixed_inputs(d)
            for sub in self.subs:
                ttnn.synchronize_device(sub)
            logger.info(f"[SPPrefillSC] per-die SC chunk setup + warm-up in {time.perf_counter() - t1:.1f}s")
            self._check_die_models_consistent()

            # ---- recv buffers (K/V prefixes only; GDN state is received straight into the persistent
            # per-layer state) and the prefix page tables (d > 0: blocks [0, s_d / 64)) ----
            self._alloc_rbuf()
            self.prefix_page_tables = [None] + [
                _identity_page_table(self.subs[d], self.span_starts[d] // BLOCK_SIZE) for d in range(1, n_spans)
            ]

            _build_hop_sockets(
                self.subs, socket_storage, state_fifo_bytes, kv_fifo_bytes, self.state_sockets, self.kv_sockets
            )
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------------------------------------

    def _write_die_fixed_inputs(self, d):
        """Die d's per-chunk inputs of the SC chunked prefill, which are fixed per die (chunk d of the
        sequence): written once, eagerly, the same way prefill_traced_chunked writes them per chunk."""
        m = self.models[d]
        L = self.spans[d]
        cs = self.span_starts[d]
        csi_host = ttnn.from_torch(
            torch.tensor([cs], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(csi_host, m._chunk_start_idx_tensor)
        if m._m4_r4b:
            # M4 R4B: absolute position of this chunk's last row (the last layer's decode-SDPA cur_pos).
            lp_host = ttnn.from_torch(
                torch.tensor([cs + L - 1], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(lp_host, m._chunk_last_pos_tensor)
            m._m4_last_pos_host = cs + L - 1
        blk0 = cs // BLOCK_SIZE
        cpt_host = ttnn.from_torch(
            self._page_table_host[:, blk0 : blk0 + L // BLOCK_SIZE].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(cpt_host, m._chunk_page_table_buf)
        # Full page table: the setup already wrote arange(num_blocks) into _chunk_full_page_table_buf.
        if m.rope.rope_device_table_enabled() and m.rope._req_cos is None:
            cos_slice, sin_slice = m.rope.get_prefill_rot_mats_table_slice(cs, L)
            ttnn.copy(cos_slice, m._chunk_cos_buf)
            ttnn.copy(sin_slice, m._chunk_sin_buf)
            ttnn.deallocate(cos_slice)
            ttnn.deallocate(sin_slice)
        else:
            cos_seq, sin_seq = m.rope.prefill_cos_sin_torch(cs, L)
            cos_host = ttnn.from_torch(cos_seq.unsqueeze(0).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            sin_host = ttnn.from_torch(sin_seq.unsqueeze(0).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            ttnn.copy_host_to_device_tensor(cos_host, m._chunk_cos_buf)
            ttnn.copy_host_to_device_tensor(sin_host, m._chunk_sin_buf)

    def _check_die_models_consistent(self):
        """Every die must run the same layer list and the same prefill variants, and a GDN layer's state
        spec (the socket payload, received in place) must be identical on the sending and receiving die."""
        ref = self.models[0]
        for d, m in enumerate(self.models[1:], start=1):
            assert len(m.layers) == len(ref.layers), f"die {d}: {len(m.layers)} layers vs {len(ref.layers)}"
            for key in ("_m2_nowhere", "_m2_lastrow", "_m4_r4a", "_m4_r4b", "_m5_addnorm"):
                assert getattr(m, key) == getattr(ref, key), f"die {d}: {key} differs from die 0"
            for li, (la, lb) in enumerate(zip(ref.layers, m.layers)):
                assert la.is_full_attention == lb.is_full_attention, f"die {d} layer {li}: type differs"
                if la.is_full_attention:
                    continue
                for name in ("recurrent_state", "fused_conv_state"):
                    ta, tb = getattr(la.attention, name), getattr(lb.attention, name)
                    assert tuple(ta.spec.shape) == tuple(tb.spec.shape) and ta.spec.dtype == tb.spec.dtype, (
                        f"die {d} layer {li} GDN {name} spec {tb.spec} != die 0 {ta.spec} "
                        "(it is received in place from the previous die)"
                    )
                    assert ta.layout == ttnn.TILE_LAYOUT and tb.layout == ttnn.TILE_LAYOUT

    def _alloc_rbuf(self):
        """Persistent K/V prefix recv buffers, keyed (die, layer_idx, "k"/"v"), d > 0 full-attention layers
        only: [1, nkv, s_d, hd] TILE DRAM in the paged cache's dtype (bf16, or bfloat8_b under
        QWEN36_ATTN_KV_BF8=1 -- s_d = die d's span start; the _kv_paged_to_seq layout the sender
        produces). The dtype must match the cache: _send_kv asserts it and paged_fill_cache requires it."""
        self.rbuf = {}
        for d in range(1, self.n_spans):
            sub = self.subs[d]
            kv_shape = (1, self.nkv, self.span_starts[d], self.hd)
            for li, layer in enumerate(self.models[d].layers):
                if not layer.is_full_attention:
                    continue
                for name in ("k", "v"):
                    self.rbuf[(d, li, name)] = ttnn.zeros(
                        kv_shape,
                        device=sub,
                        dtype=layer.attention.paged_kv_cache_key.dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

    # ------------------------------------------------------------------------------------------------
    # cross-die transfers (each call touches ONLY one die's queue; see SPPrefill._recv_kv)
    # ------------------------------------------------------------------------------------------------

    def _send_kv(self, d, li):
        """Die d, right after layer li's mixer: forward the K/V prefix [0, s_d + L_d) of its paged cache
        (prefix received from die d-1 + its own span, just filled) to die d+1's rbuf."""
        attn = self.models[d].layers[li].attention
        S = self.span_starts[d] + self.spans[d]
        blocks = S // BLOCK_SIZE
        k_t = _kv_paged_to_seq(attn.paged_kv_cache_key, blocks, S, self.nkv, self.hd)
        v_t = _kv_paged_to_seq(attn.paged_kv_cache_value, blocks, S, self.nkv, self.hd)
        rb_k = self.rbuf[(d + 1, li, "k")]
        assert (
            tuple(k_t.spec.shape) == tuple(rb_k.spec.shape) and k_t.spec.dtype == rb_k.spec.dtype
        ), f"KV send/recv spec mismatch at layer {li}: sent {k_t.spec} vs rbuf {rb_k.spec}"
        send_sock, _ = self.kv_sockets[d]
        send_op = _socket_send_op(self.kv_socket_mode)
        send_op(k_t, send_sock)
        send_op(v_t, send_sock)
        ttnn.deallocate(k_t)
        ttnn.deallocate(v_t)

    def _recv_kv(self, d, li):
        """Die d > 0, before layer li: receive the K/V prefix [0, s_d) from die d-1 into the rbuf, then
        write it into this die's paged cache (blocks [0, s_d / 64))."""
        _, recv_sock = self.kv_sockets[d - 1]
        recv_op = _socket_recv_op(self.kv_socket_mode)
        rb_k, rb_v = self.rbuf[(d, li, "k")], self.rbuf[(d, li, "v")]
        recv_op(rb_k, recv_sock)
        recv_op(rb_v, recv_sock)
        attn = self.models[d].layers[li].attention
        ttnn.experimental.paged_fill_cache(attn.paged_kv_cache_key, rb_k, self.prefix_page_tables[d], batch_idx=0)
        ttnn.experimental.paged_fill_cache(attn.paged_kv_cache_value, rb_v, self.prefix_page_tables[d], batch_idx=0)

    def _send_gdn(self, d, li):
        """Die d, right after layer li's mixer: forward its GDN recurrent + conv state (the persistent
        tensors the SC GDN just updated in place; NOT deallocated) to die d+1."""
        dn = self.models[d].layers[li].attention
        send_sock, _ = self.state_sockets[d]
        send_op = _socket_send_op(self.state_socket_mode)
        send_op(dn.recurrent_state, send_sock)
        send_op(dn.fused_conv_state, send_sock)

    def _recv_gdn(self, d, li):
        """Die d > 0, before layer li: receive die d-1's GDN state straight into this die's persistent
        state (the SC GDN reads it as the initial state and overwrites it in place)."""
        dn = self.models[d].layers[li].attention
        _, recv_sock = self.state_sockets[d - 1]
        recv_op = _socket_recv_op(self.state_socket_mode)
        recv_op(dn.recurrent_state, recv_sock)
        recv_op(dn.fused_conv_state, recv_sock)

    # ------------------------------------------------------------------------------------------------
    # per-request program
    # ------------------------------------------------------------------------------------------------

    def _zero_gdn_state(self, d):
        """Sequence start: zero die d's GDN recurrent + conv state in place (die 0 only: every other die
        receives its state before each GDN layer runs)."""
        m = self.models[d]
        for layer in m.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            ttnn.copy(m._dn_zero_recurrent, dn.recurrent_state)
            ttnn.copy(m._dn_zero_conv, dn.fused_conv_state)

    def _embed(self, d, tokens):
        """Tokens of span d -> the persistent chunk token buffer -> embedding (mirrors the head of
        Qwen36Model._forward_prefill_chunk). tokens=None skips the host write (traced pass: the buffer is
        written before each replay). Returns the per-die loop state."""
        m = self.models[d]
        if tokens is not None:
            # tokens=None (the traced pass): no host write; prefill_traced writes _chunk_token_buf before replay.
            tok_host = ttnn.from_torch(self._span_tokens(tokens, d), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(tok_host, m._chunk_token_buf)
        x = m.embd(m._chunk_token_buf)
        _resid_l1 = os.environ.get("QWEN36_LAYER_RESID_L1", "0") == "1" and x.shape[1] <= 2048
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG if _resid_l1 else ttnn.DRAM_MEMORY_CONFIG)
        if not m._m2_nowhere:
            x = m._apply_vision_merge(x, length=x.shape[1])
        # M5 ADDNORM applies to T in tpc.M5_ADDNORM_T_SET chunks only (env QWEN36_M5_ADDNORM_T, default 2048; a
        # comma list of the per-die span lengths, e.g. 1024 or "896,1024,1152", enables it per die; the same
        # membership test gates it in the setup's warm-up forward, model.py). The post-mixer hook then fires right
        # before the fused attention residual add + ffn_norm (layer.py): it only sends the mixer state.
        m5 = m._m5_addnorm and m.num_devices == 1 and x.shape[1] in tpc.M5_ADDNORM_T_SET
        return {"x": x, "pending": None, "m5": m5}

    def _span_tokens(self, tokens, d):
        """Die d's tokens [1, L_d] int32 (host): prompt positions [s_d, s_d + L_d)."""
        s0 = self.span_starts[d]
        return tokens[:, s0 : s0 + self.spans[d]].to(torch.int32)

    def _layer_step(self, d, li, st, hook):
        """One layer of die d: the body of Qwen36Model._forward_prefill_chunk's layer loop, plus the
        post-mixer hook (the cross-die send). Updates st in place."""
        m = self.models[d]
        layer = m.layers[li]
        last = len(m.layers) - 1
        x, pending, m5 = st["x"], st["pending"], st["m5"]
        m5_kw = dict(m5_addnorm=True, pending=pending, defer_out=li < last) if m5 else {}
        x_in = x if pending is None else None
        if layer.is_full_attention:
            lastrow = m._m2_lastrow and li == last
            m4 = {}
            if lastrow and m._m4_r4a:
                m4["last_row_tile_slices"] = True
            if lastrow and m._m4_r4b:
                m4["last_row_pos_tensor"] = m._chunk_last_pos_tensor
            x_new = layer.forward(
                x_in,
                cos=m._chunk_cos_buf,
                sin=m._chunk_sin_buf,
                mode="prefill",
                page_table=m._chunk_full_page_table_buf,
                chunk_page_table=m._chunk_page_table_buf,
                chunk_start_idx_tensor=m._chunk_start_idx_tensor,
                last_row_only=lastrow,
                post_mixer_hook=hook,
                **m4,
                **m5_kw,
            )
        else:
            x_new = layer.forward(
                x_in,
                mode="prefill",
                chunk_size=layer.attention.long_prefill_chunk_size,
                post_mixer_hook=hook,
                **m5_kw,
            )
        if pending is None:
            ttnn.deallocate(x)  # (M5 ADDNORM with pending: the layer made and freed its own x)
        if m5 and li < last:
            st["pending"], st["x"] = x_new, None
        else:
            st["pending"], st["x"] = None, x_new

    def _tail_logits(self, model, hidden):
        """Last die: full last-token logits row (final norm + LM head, no argmax) -> host [vocab]."""
        L = self.spans[-1]
        x_last = hidden if hidden.shape[1] == 1 else hidden[:, L - 1 : L, :]
        x_last = ttnn.to_layout(x_last, ttnn.TILE_LAYOUT)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = model.norm(x_last, mode=Mode.PREFILL)
        logits = model._lm_head(x_last)
        lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.device, dim=0))
        ttnn.deallocate(logits)
        return lt[0].reshape(-1)[: self.vocab_size]

    def _tail_token(self, model, hidden):
        """Last die: greedy token on device (same ops as the SC exact-multiple tail) -> host int."""
        tok = model._exact_multiple_tail_device(hidden, self.spans[-1])
        tt = ttnn.to_torch(tok, mesh_composer=ttnn.ConcatMeshToTensor(model.device, dim=0))
        ttnn.deallocate(tok)
        return int(tt.reshape(-1)[0])

    def _tail_traced(self, model, hidden):
        """Last die, device ops only (no readback; valid inside a trace): the greedy uint32 token
        (_exact_multiple_tail_device; set_greedy_token_output(True) is on for the last die) plus the final
        hidden kept for the logits tail, moved to DRAM if it is in L1 (so a trace output never pins L1).
        Consumes ``hidden``. Returns (tok, keep)."""
        tok = model._exact_multiple_tail_device(hidden, self.spans[-1])
        keep = hidden
        if hidden.memory_config().buffer_type == ttnn.BufferType.L1:
            keep = ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(hidden)
        return tok, keep

    def _run_layer_major(self, tokens, *, return_logits, traced=False, warm=False):
        """Build the whole prefill program LAYER-MAJOR across dies (required by the direct-mode
        rendezvous, see SPPrefill._run_layer_major): embed every die, then for each layer run ALL dies
        (die d: recv from d-1 -> layer, whose post-mixer hook sends to d+1) before the next layer; tail
        on the last die.

        traced=False, warm=False: eager; synchronizes every die; returns logits [vocab] or the token.
        warm=True (eager, capture's warm-up): runs BOTH tails (_tail_traced, then the logits tail on its
        kept hidden) so every program the capture and prefill_traced's logits read need is compiled on
        the real SP hidden; synchronizes; returns the logits.
        traced=True (inside capture's open traces): tokens is None (no host writes), no signposts, no
        readback, no sync; returns the last die's (tok, keep) device tensors (the trace outputs)."""
        n = self.n_spans
        last = n - 1
        if not traced:
            _sp("prefill start")
        st = {}
        for d in range(n):
            if d == 0:
                self._zero_gdn_state(0)  # device copies (inside die 0's trace when traced)
            st[d] = self._embed(d, None if traced else tokens)

        ref = self.models[0]
        for li in range(len(ref.layers)):
            is_fa = ref.layers[li].is_full_attention
            if not traced:
                _sp(f"prefill L{ref.layer_indices[li]} {'attn' if is_fa else 'gdn'}")
            for d in range(n):
                fired = []
                hook = None
                if d < last:
                    send = self._send_kv if is_fa else self._send_gdn

                    def hook(d=d, li=li, send=send, fired=fired):
                        fired.append(True)
                        send(d, li)

                if d > 0:
                    (self._recv_kv if is_fa else self._recv_gdn)(d, li)
                self._layer_step(d, li, st[d], hook)
                assert hook is None or len(fired) == 1, f"die {d} layer {li}: post_mixer_hook fired {len(fired)}x"

        for d in range(last):
            ttnn.deallocate(st[d]["x"])
        model = self.models[last]
        hidden = st[last]["x"]
        if traced:
            return self._tail_traced(model, hidden)
        _sp("prefill tail")
        if warm:
            tok, keep = self._tail_traced(model, hidden)
            out = self._tail_logits(model, keep)
            self.last_first_token = int(torch.argmax(out.float()))
            ttnn.deallocate(tok)
            ttnn.deallocate(keep)
        elif return_logits:
            out = self._tail_logits(model, hidden)
            self.last_first_token = int(torch.argmax(out.float()))
            ttnn.deallocate(hidden)
        else:
            out = self._tail_token(model, hidden)
            self.last_first_token = out
            ttnn.deallocate(hidden)
        for sub in self.subs:
            ttnn.synchronize_device(sub)
        return out

    def prefill(self, tokens: torch.Tensor, return_logits: bool = True):
        """tokens [1, sum(spans)] -> last-token logits [vocab] (host torch; return_logits=True, the
        SPPrefill API) or the greedy first token (int, on-device argmax; return_logits=False)."""
        assert tokens.shape[0] == 1, "SPPrefillSC is B=1 only"
        assert tokens.shape[-1] == self.total_len, f"tokens {tuple(tokens.shape)}: expected {self.total_len} tokens"
        t0 = time.perf_counter()
        out = self._run_layer_major(tokens, return_logits=return_logits)
        logger.info(
            f"[SPPrefillSC] prefill wall time: {(time.perf_counter() - t0) * 1000:.2f} ms "
            f"(first token {self.last_first_token})"
        )
        return out

    def _trace_refs(self):
        """Every per-die tensor whose device address the captured traces bake in (persistent inputs,
        GDN / KV state, recv buffers, prefix page tables). prefill_traced checks each is still the SAME
        object as at capture (a rebind would make the replay read / write a freed address)."""
        refs = []
        for m in self.models:
            for name in (
                "_chunk_token_buf",
                "_chunk_cos_buf",
                "_chunk_sin_buf",
                "_chunk_start_idx_tensor",
                "_chunk_full_page_table_buf",
                "_chunk_page_table_buf",
                "_chunk_last_pos_tensor",
                "_dn_zero_recurrent",
                "_dn_zero_conv",
            ):
                refs.append(getattr(m, name, None))
            for layer in m.layers:
                attn = layer.attention
                for name in ("recurrent_state", "fused_conv_state", "paged_kv_cache_key", "paged_kv_cache_value"):
                    refs.append(getattr(attn, name, None))
        refs.extend(self.rbuf[k] for k in sorted(self.rbuf))
        refs.extend(self.prefix_page_tables)
        return tuple(refs)

    def capture(self, tokens: torch.Tensor, comm: bool = True, dies=None):
        """Warm up (eager layer-major pass that also compiles both tails on the real SP hidden), then
        capture the whole program as ONE layer-major pass with a trace open on every die (MeshSocket recv
        is device-blocking, so all captures open before the pass and close after it; see
        SPPrefill.capture). The capture must compile nothing (a compile after a trace is parked puts a
        kernel binary in parked-trace scratch, #48536): checked via the program-cache entry counts. Only
        the full-comm, all-dies capture is supported. Never leaves a trace open (_safe_end_traces)."""
        assert comm and dies is None, "SPPrefillSC.capture: only comm=True, dies=None (all dies) is supported"
        want = (1, self.total_len)
        assert tuple(tokens.shape) == want, f"SPPrefillSC.capture: tokens {tuple(tokens.shape)} != {want}"
        self._release_traces()
        self._run_layer_major(tokens, return_logits=True, warm=True)
        for sub in self.subs:
            ttnn.synchronize_device(sub)
        n0 = [sub.num_program_cache_entries() for sub in self.subs]
        self._trace_refs_snap = self._trace_refs()

        t0 = time.perf_counter()
        opened = {}
        try:
            for d, sub in enumerate(self.subs):
                opened[d] = ttnn.begin_trace_capture(sub, cq_id=0)
            tok, keep = self._run_layer_major(None, return_logits=False, traced=True)
            for d, sub in enumerate(self.subs):
                ttnn.end_trace_capture(sub, opened[d], cq_id=0)
                self._trace_ids[d] = opened.pop(d)
        except Exception:
            self._safe_end_traces(opened)
            for d, tid in opened.items():
                self._trace_ids[d] = tid  # force-closed traces: released by _release_traces / close()
            raise
        # Trace outputs: stored first so _release_traces frees them on the failure path below.
        self._traced_tok, self._traced_hidden = tok, keep
        n2 = [sub.num_program_cache_entries() for sub in self.subs]
        if n2 != n0:
            self._release_traces()
            raise RuntimeError(f"SPPrefillSC.capture compiled: program cache entries {n0} -> {n2}")
        self._trace_pc_entries = n0
        logger.info(f"[SPPrefillSC] captured {self.n_spans} die traces in {(time.perf_counter() - t0) * 1000:.1f} ms")

    def _safe_end_traces(self, opened):
        """A trace left open (no matching end_trace_capture) wedges every later op on that
        device, including close_mesh_device -- so always try to close what capture() opened."""
        for d, tid in opened.items():
            try:
                ttnn.end_trace_capture(self.subs[d], tid, cq_id=0)
            except Exception as e:
                logger.warning(f"[SPPrefillSC] safe_end_traces: could not close trace on die {d}: {e!r}")

    def prefill_traced(self, tokens: torch.Tensor, return_logits: bool = True):
        """Replay the captured per-die traces on a new prompt (capture() must have run). Only the token
        buffers are written (host -> device) before the replay. Returns (logits, wavefront_s, total_s):
        wavefront_s = all traces launched + synchronized; total_s additionally includes the on-device
        greedy token readback (TTFT). The logits (return_logits=True; else None) come from the eager
        logits tail on the trace's kept hidden, AFTER total_s is stamped, and cross-check the token."""
        assert all(t is not None for t in self._trace_ids), "capture() must run before prefill_traced()"
        want = (1, self.total_len)
        assert tuple(tokens.shape) == want, f"SPPrefillSC.prefill_traced: tokens {tuple(tokens.shape)} != {want}"
        for d, (sub, m) in enumerate(zip(self.subs, self.models)):
            tok_host = ttnn.from_torch(self._span_tokens(tokens, d), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(tok_host, m._chunk_token_buf)
            n = sub.num_program_cache_entries()
            assert n == self._trace_pc_entries[d], (
                f"die {d}: program cache entries {self._trace_pc_entries[d]} at capture -> {n} now: an op "
                "compiled after the trace was parked (compile after park, #48536)"
            )
        refs = self._trace_refs()
        assert len(refs) == len(self._trace_refs_snap) and all(
            a is b for a, b in zip(refs, self._trace_refs_snap)
        ), "SPPrefillSC.prefill_traced: a tensor the traces bake in was rebound since capture()"

        t0 = time.perf_counter()
        for d, sub in enumerate(self.subs):
            ttnn.execute_trace(sub, self._trace_ids[d], cq_id=0, blocking=False)
        for sub in self.subs:
            ttnn.synchronize_device(sub)
        t1 = time.perf_counter()
        tok_t = ttnn.to_torch(self._traced_tok, mesh_composer=ttnn.ConcatMeshToTensor(self.subs[-1], dim=0))
        first = int(tok_t.reshape(-1)[0])
        t2 = time.perf_counter()
        self.last_first_token = first

        logits = None
        if return_logits:
            # Outside the timed window: eager logits tail (compiled in capture's warm-up) on the kept hidden.
            logits = self._tail_logits(self.models[-1], self._traced_hidden).float()
            host_argmax = int(torch.argmax(logits))
            if host_argmax != first:
                top2 = torch.topk(logits, 2)
                msg = (
                    f"[SPPrefillSC] traced: device token {first} (logit {float(logits[first]):.6f}) != host argmax "
                    f"{host_argmax}; top-2 ids {top2.indices.tolist()} values {top2.values.tolist()}"
                )
                # An exact bf16 tie may resolve to a different index on device vs torch: warn, don't fail.
                assert float(logits[first]) == float(top2.values[0]), msg
                logger.warning(msg + " (exact tie)")
        logger.info(
            f"[SPPrefillSC traced] wavefront={(t1 - t0) * 1000:.2f} ms ttft={(t2 - t0) * 1000:.2f} ms "
            f"first_token={first}"
        )
        return logits, (t1 - t0), (t2 - t0)

    def _release_traces(self):
        """Release every captured trace and free the trace outputs. Idempotent."""
        for d, tid in enumerate(self._trace_ids):
            if tid is not None:
                ttnn.release_trace(self.subs[d], tid)
        self._trace_ids = [None] * self.n_spans
        for name in ("_traced_tok", "_traced_hidden"):
            t = getattr(self, name, None)
            if t is not None:
                ttnn.deallocate(t)
            setattr(self, name, None)
        self._trace_pc_entries = None
        self._trace_refs_snap = None

    def export_state_host(self):
        """Read the LAST die's post-prefill state to host torch in the layout
        ``sp_handoff.inject_into_tp_model`` expects (same as SPPrefill.export_state_host): full-attention KV
        [1, n_kv_heads, S, head_dim] bf16 (S = sum(spans), natural position order from the paged
        cache), GDN (rec [1, Nv, Dk, Dv] fp32, conv [1, K-1, q_dim+k_dim+v_dim] bf16). Keys are positions in
        model.layers. Returns (kv_per_attn_layer, gdn_per_layer, export_s)."""
        t0 = time.perf_counter()
        model = self.models[self.n_spans - 1]
        S = self.total_len
        blocks = S // BLOCK_SIZE
        comp = ttnn.ConcatMeshToTensor(model.device, dim=0)

        kv_per_attn_layer, gdn_per_layer = {}, {}
        for li, layer in enumerate(model.layers):
            if layer.is_full_attention:
                attn = layer.attention
                k_seq = _kv_paged_to_seq(attn.paged_kv_cache_key, blocks, S, self.nkv, self.hd)
                v_seq = _kv_paged_to_seq(attn.paged_kv_cache_value, blocks, S, self.nkv, self.hd)
                K = ttnn.to_torch(k_seq, mesh_composer=comp).to(torch.bfloat16)
                V = ttnn.to_torch(v_seq, mesh_composer=comp).to(torch.bfloat16)
                ttnn.deallocate(k_seq)
                ttnn.deallocate(v_seq)
                kv_per_attn_layer[li] = (K, V)
            else:
                dn = layer.attention
                rec = ttnn.to_torch(dn.recurrent_state, mesh_composer=comp).to(torch.float32)
                conv = ttnn.to_torch(dn.fused_conv_state, mesh_composer=comp).to(torch.bfloat16)
                gdn_per_layer[li] = (rec, conv)

        export_s = time.perf_counter() - t0
        logger.info(f"[SPPrefillSC] export_state_host: {export_s * 1000:.2f} ms")
        return kv_per_attn_layer, gdn_per_layer, export_s

    def close(self):
        """Release captured traces and drop socket / recv-buffer references. Idempotent (safe from a
        failed __init__'s except clause and again from a test's finally block)."""
        self._release_traces()
        self.state_sockets = []
        self.kv_sockets = []
        self.rbuf = {}
