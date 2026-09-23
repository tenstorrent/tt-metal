# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel prefill: split one long prompt into N contiguous spans, one per die.

Each die (a 1x1 submesh of a (1,N) mesh) runs a full-weight model instance built with
``sequence_parallel=True`` (the TP-path classes at tp=1 — see model_config.py /
gdn/tp.py / attention/tp.py). GDN recurrent state and attention K/V flow strictly
forward die d -> d+1 over MeshSockets:

  * Full-attention layers: die d fills its own paged KV cache with its local span,
    then forwards the CUMULATIVE K/V prefix [0, span_len*(d+1)) to die d+1, which
    fills its own (independent) paged cache with that prefix before running the
    same layer.
  * GDN layers: die d forwards its final recurrent + conv state to die d+1, which
    feeds it in as ``initial_state`` / ``initial_conv_state`` for the same layer.

Correctness-first / untraced: every die's forward pass is plain eager ttnn ops.
``_run_layer_major`` builds the whole `dies` program LAYER-MAJOR (all dies process layer L
before any die moves to layer L+1) -- required so a direct-mode send never blocks waiting for a
recv the host hasn't issued yet (a die-major loop deadlocks: see the task report). The same
method is reused, unmodified, as the body ``capture()`` wraps in begin/end_trace_capture.
"""
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.common import create_tt_model
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.tt_transformers.tt.common import Mode

BLOCK_SIZE = 64  # paged-KV block size (fixed; matches forward_prefill_paged's k_paged.shape[2] usage)


def _sp(header):
    """Tracy signpost, gated on QWEN36_SIGNPOSTS=1 (unset/0 -> no-op, no tracy import at all).
    Only meaningful from an UNTRACED op sequence -- a captured trace's replay is one command
    and never re-enters Python, so callers gate on `traced=False` before calling this."""
    if os.environ.get("QWEN36_SIGNPOSTS") != "1":
        return
    try:
        from tracy import signpost
    except ImportError:
        return
    signpost(header)


# The box has exactly 2 fabric links between adjacent dies (num_cores <= num_available_links),
# so every socket uses 2 SocketConnections. Two socket *pairs* per hop, on disjoint cores so
# neither role nor purpose collides on a die that is both a receiver (incoming hop) and a
# sender (outgoing hop): STATE (GDN rec state + conv carry) and KV (attention K/V prefixes),
# both direct-mode (send_direct_async/recv_direct_async) with a 128B L1 FIFO. tt-metal's L1
# allocator uses bank-uniform addresses, so a socket's FIFO reserves address space on EVERY
# core, not just the socket's own cores -- a 64KB async FIFO (2 sockets x 2 connections) ate
# ~260KB of L1 on every core and collided with gdn/tp.py's native ttnn.conv1d (GDN prefill
# depthwise conv) at span_len=1024 (T=4096); relocating the socket cores did not help (same
# clash, same address) because the reservation isn't core-local. 128B direct-mode FIFOs are a
# wash on bandwidth for the ~1MB GDN state hop (63 vs 59us measured) and avoid the L1 pressure.
_STATE_SEND_CORES = [ttnn.CoreCoord(0, 9), ttnn.CoreCoord(1, 9)]
_STATE_RECV_CORES = [ttnn.CoreCoord(0, 8), ttnn.CoreCoord(1, 8)]
_KV_SEND_CORES = [ttnn.CoreCoord(2, 9), ttnn.CoreCoord(3, 9)]
_KV_RECV_CORES = [ttnn.CoreCoord(2, 8), ttnn.CoreCoord(3, 8)]


def _build_socket_pair(sub_a, sub_b, send_cores, recv_cores, storage, fifo_bytes):
    conns = [
        ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, sc), ttnn.MeshCoreCoord(coord, rc))
        for coord in ttnn.MeshCoordinateRange(sub_a.shape)
        for sc, rc in zip(send_cores, recv_cores)
    ]
    cfg = ttnn.SocketConfig(conns, ttnn.SocketMemoryConfig(storage, fifo_bytes))
    return ttnn.create_socket_pair(sub_a, sub_b, cfg)


def _sp_layer_mixer(layer, x, *, cos, sin, d, span_len, page_table, chunk_page_table, gdn_in):
    """Mirror Qwen36DecoderLayer.forward's TP prefill branch through attn_norm -> token
    mixer -> residual add ONLY (split out of the old `_sp_layer_prefill` so the cross-die
    send can be issued right after the mixer, before the MLP -- the data die d+1 needs
    (GDN final state + conv carry, or the paged K/V the mixer just filled) is produced
    here, not by the MLP; see SP_PREFILL_HANDOFF.md sec 3.3/5).

    Returns (h, gdn_out): gdn_out is (final_state, conv_new_state) for a GDN layer (to
    relay to die d+1), or None for a full-attention layer (whose K/V the caller reads
    straight off `layer.attention.paged_k/v`, already filled by forward_prefill_paged).
    """
    args = layer.args
    # No distributed_output_mem_config override here: at tp=1 (SP) the norm is never a
    # DistributedNorm (num_devices == 1), so its output placement just follows the input
    # (ttnn.rms_norm, memory_config=None -> matches the input's memory config, rmsnorm.cpp:29).
    attn_norm_cfg = args.get_norm_config("attn", Mode.PREFILL)

    attn_input = layer.attention_norm(x, mode=Mode.PREFILL, norm_config=attn_norm_cfg)

    gdn_out = None
    if layer.is_full_attention:
        attn_output = layer.attention.forward_prefill_paged(
            attn_input,
            cos,
            sin,
            page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=d * span_len,
            user_id=0,
        )
    else:
        init_state, init_conv = gdn_in
        attn_output, final_state, conv_new_state = layer.attention.forward_prefill(
            attn_input, return_state=True, initial_state=init_state, initial_conv_state=init_conv
        )
        gdn_out = (final_state, conv_new_state)
    ttnn.deallocate(attn_input)

    h = ttnn.add(x, attn_output)
    ttnn.deallocate(attn_output)
    return h, gdn_out


def _sp_layer_mlp(layer, h):
    """ff_norm -> MLP -> residual add: the back half of the old `_sp_layer_prefill`, run
    AFTER the caller has issued the cross-die send for this layer (see `_sp_layer_mixer`)."""
    ff_norm_cfg = layer.args.get_norm_config("ff", Mode.PREFILL)

    ff_input = layer.ffn_norm(h, mode=Mode.PREFILL, norm_config=ff_norm_cfg)
    ff_output = layer.feed_forward.forward(ff_input, mode="prefill")
    ttnn.deallocate(ff_input)

    output = ttnn.add(h, ff_output)
    ttnn.deallocate(h)
    ttnn.deallocate(ff_output)
    return output


def _identity_page_table(sub, blocks, offset=0):
    rows = [list(range(offset, offset + blocks))]
    return ttnn.from_torch(
        torch.tensor(rows, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=sub,
        mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
    )


class SPPrefill:
    """Sequence-parallel prefill engine: n_spans dies, each a 1x1 submesh.

    state_socket_mode / kv_socket_mode select the MeshSocket transport used for the GDN-state
    and KV-prefix hops (see the module docstring). "direct" (send_direct_async/recv_direct_async,
    a 128 B FIFO) is the correctness-validated default. "async" (send_async/recv_async, a real
    streamed FIFO, default 64 KiB) is experimental: measured within 4 ms of "direct" at T=4096,
    and its FIFO reserves L1 address space on EVERY core (tt-metal's L1 allocator uses
    bank-uniform addresses, not per-core ones), so a large async FIFO can collide with other L1
    users on the same cores (e.g. gdn/tp.py's native ttnn.conv1d).
    """

    def __init__(
        self,
        mesh_device,
        *,
        n_spans=4,
        tp=1,
        span_len=1024,
        max_seq_len=4096,
        hf_model=None,
        layer_indices=None,
        state_socket_mode="direct",
        kv_socket_mode="direct",
        state_fifo_bytes=None,
        kv_fifo_bytes=None,
    ):
        # fifo_bytes overrides the per-mode default (128 direct / 64KiB async) -- e.g. a smaller
        # async FIFO if 64KiB clashes with other L1 users (see the class docstring).
        assert state_socket_mode in ("direct", "async") and kv_socket_mode in ("direct", "async")
        self.state_socket_mode = state_socket_mode
        self.kv_socket_mode = kv_socket_mode
        _default_fifo = {"direct": 128, "async": 64 * 1024}
        state_fifo_bytes = state_fifo_bytes or _default_fifo[state_socket_mode]
        kv_fifo_bytes = kv_fifo_bytes or _default_fifo[kv_socket_mode]

        assert max_seq_len % BLOCK_SIZE == 0, "max_seq_len must be a multiple of the paged-KV block size (64)"
        assert span_len % BLOCK_SIZE == 0, "span_len must be a multiple of the paged-KV block size (64)"
        # 128 is the SDPA q-chunk size assumed by forward_prefill_paged's chunk_start_idx; FIFO
        # sizes and the L1 budget in this module were validated at span_len=1024 only.
        assert span_len % 128 == 0, "span_len must be a multiple of 128 (the SDPA q-chunk size)"
        # tp>1 gives each span a TENSOR-PARALLEL GROUP instead of a single die, so the two
        # parallelisms compose: SP splits the prompt across groups, TP splits each layer across
        # the dies within a group. The mesh must hold n_spans * tp dies; a (1, N) line works, and
        # so does any 2-D shape whose total size matches (a Blackhole Galaxy caps at 8 columns,
        # so SP=8 x TP=4 has to come from (4, 8) carved into 8 groups of (1, 4)).
        n_dies = mesh_device.shape[0] * mesh_device.shape[1]
        assert n_dies == n_spans * tp, (
            f"SPPrefill needs n_spans*tp = {n_spans}*{tp} = {n_spans * tp} dies, "
            f"got mesh {tuple(mesh_device.shape)} = {n_dies}"
        )
        self.tp = tp
        self.n_spans = n_spans
        self.span_len = span_len
        self.max_seq_len = max_seq_len
        self.num_blocks = max_seq_len // BLOCK_SIZE
        self.blocks_per_span = span_len // BLOCK_SIZE

        self.mesh_device = mesh_device
        # One submesh per SPAN; each is a (1, tp) tensor-parallel group (tp=1 -> one die, the
        # original behaviour). _build_socket_pair already iterates MeshCoordinateRange(sub.shape),
        # so a group-to-group socket fans out to one connection per die pair automatically.
        self.subs = mesh_device.create_submeshes(ttnn.MeshShape(1, tp))
        assert len(self.subs) == n_spans, f"expected {n_spans} submeshes, got {len(self.subs)}"
        # SNAKE the group order when the spans tile a 2-D mesh. The wavefront sockets connect
        # die j of group g to die j of group g+1, and 1D Line Fabric requires sender and
        # receiver to share a row or a column. create_submeshes hands back row-major order, so
        # on a (4,8) mesh with tp=4 the wrap from (row r, block 1) to (row r+1, block 0) is
        # diagonal and the hop dies with:
        #   mesh_socket_utils.cpp:115 Sender and receiver chips must be in the same row or
        #   column when using 1D Line Fabric
        # Reversing every other row makes consecutive groups differ in exactly one axis
        # (measured: 3 illegal hops row-major -> 0 snaked, for (4,8)/tp=4).
        # FABRIC_2D is NOT an alternative: all_gather_minimal_matmul_async's dm_in0_sender
        # kernel is written against the linear (1D) fabric API and fails to compile under 2D.
        _rows, _cols = mesh_device.shape[0], mesh_device.shape[1]
        _per_row = _cols // tp
        if _rows > 1 and _per_row > 1 and _rows * _per_row == n_spans:
            snake = []
            for r in range(_rows):
                blocks = range(_per_row) if r % 2 == 0 else reversed(range(_per_row))
                snake += [r * _per_row + b for b in blocks]
            self.subs = [self.subs[i] for i in snake]
            logger.info(f"[SPPrefill] snaked span order for 1D-fabric collinearity: {snake}")

        # Safe defaults so close() can always run cleanly, even if construction below fails
        # partway through (see the try/except immediately below).
        self._trace_ids = [None] * n_spans
        self.state_sockets = []
        self.kv_sockets = []
        self.rbuf = {}

        try:
            # ---- build die 0 (warms the tensor cache), then dies 1..n_spans-1 (reuse the state_dict) ----
            t0 = time.perf_counter()
            args0, model0, state_dict = create_tt_model(
                self.subs[0],
                max_batch_size=1,
                max_seq_len=max_seq_len,
                hf_model=hf_model,
                sequence_parallel=True,
                layer_indices=layer_indices,
            )
            self.models = [model0]
            self.args_list = [args0]
            for d in range(1, n_spans):
                args_d = Qwen36ModelArgs(
                    mesh_device=self.subs[d], max_batch_size=1, max_seq_len=max_seq_len, sequence_parallel=True
                )
                if layer_indices is not None:
                    # Mirror create_tt_model's layer_indices handling (args0, above) so every
                    # die builds the SAME checkpoint layers -- Qwen36Model.__init__ reads
                    # args.layer_indices directly (falls back to range(args.n_layers) if unset).
                    args_d.layer_indices = list(layer_indices)
                    args_d.n_layers = len(args_d.layer_indices)
                model_d = Qwen36Model(self.subs[d], args_d, state_dict, tensor_cache_path=args_d.weight_cache_path())
                self.models.append(model_d)
                self.args_list.append(args_d)
            logger.info(f"[SPPrefill] built {n_spans} die models in {time.perf_counter() - t0:.1f}s")

            self.nkv = self.args_list[0].n_local_kv_heads
            self.hd = self.args_list[0].head_dim
            self._alloc_rbuf()  # persistent recv buffers (analytical shapes; RESTRUCTURE, see docstring)

            # ---- paged KV caches (one per die per full-attention layer) ----
            for model_d in self.models:
                for layer in model_d.layers:
                    if layer.is_full_attention:
                        k_cache = self._mk_kv_cache(model_d.device)
                        v_cache = self._mk_kv_cache(model_d.device)
                        layer.attention.set_paged_kv_cache(k_cache, v_cache)

            # ---- page tables: full identity (SDPA scan), own-span slice (this die's fill), prefix
            # slice (d>0 only: fill from the received K/V) ----
            self.full_page_tables = [_identity_page_table(sub, self.num_blocks) for sub in self.subs]
            self.own_chunk_page_tables = [
                _identity_page_table(self.subs[d], self.blocks_per_span, offset=d * self.blocks_per_span)
                for d in range(n_spans)
            ]
            self.prefix_page_tables = [None] + [
                _identity_page_table(self.subs[d], self.blocks_per_span * d) for d in range(1, n_spans)
            ]

            # ---- two socket pairs per hop d -> d+1 (both forward), reused for every transfer on
            # that hop: STATE (GDN) and KV (attention); mode/FIFO size per the constructor args above ----
            for d in range(n_spans - 1):
                logger.info(
                    f"[SPPrefill] hop {d}: sub[{d}] ids={self.subs[d].get_device_ids()} "
                    f"-> sub[{d + 1}] ids={self.subs[d + 1].get_device_ids()}"
                )
                self.state_sockets.append(
                    _build_socket_pair(
                        self.subs[d],
                        self.subs[d + 1],
                        _STATE_SEND_CORES,
                        _STATE_RECV_CORES,
                        ttnn.BufferType.L1,
                        state_fifo_bytes,
                    )
                )
                self.kv_sockets.append(
                    _build_socket_pair(
                        self.subs[d],
                        self.subs[d + 1],
                        _KV_SEND_CORES,
                        _KV_RECV_CORES,
                        ttnn.BufferType.L1,
                        kv_fifo_bytes,
                    )
                )

            # Last die's post-layer (final_state, conv_new_state) per GDN layer_idx -- retained
            # (never sent onward, since there is no die n_spans) for export_state_host. In the
            # traced path these are the SAME persistent tensors the trace writes into every replay.
            self.last_die_gdn_state = {}

            # ---- trace-capture state (populated by capture()) ----
            self._tok_buf = [None] * n_spans  # persistent per-die token-id input buffer
            self._cos_buf = [None] * n_spans  # persistent per-die RoPE cos/sin (positions are fixed per die)
            self._sin_buf = [None] * n_spans
            self._sel_buf = None  # persistent one-hot last-position selector (last die's tail only)
            self._traced_logits = None  # persistent device tensor the trace writes logits into (last die)
        except Exception:
            self.close()
            raise

    def _alloc_rbuf(self):
        """Preallocate EVERY persistent recv buffer analytically from model dims, once, before
        any trace capture opens (a host-write zeros() is fine here -- the "no host writes"
        constraint is only for ops INSIDE a captured region). Keyed by (die, layer_idx,
        "state"/"conv"/"k"/"v").

        RESTRUCTURE (fixes the die-major capture serialization bug): each hop's send and recv are
        now two SEPARATE calls (_send_kv/_recv_kv, _send_gdn/_recv_gdn), each touching only ONE
        die's queue. _recv_kv/_recv_gdn are called from the RECEIVING die's OWN run_die, in that
        die's own per-layer program order -- never issued while "on" a different die's turn. That
        needs the destination buffer to exist before either side runs, hence this eager,
        analytical (not warmup-derived) allocation.

        Die 0's GDN entries are PERMANENT zeros (die 0 never receives, but forward_prefill still
        needs a real initial_state/initial_conv_state, not None -- see run_die). Dies 1..n_spans-1's
        entries are zero-inited placeholders that _recv_kv/_recv_gdn overwrite every call;
        _send_kv/_send_gdn assert the sender's tensor matches this shape/dtype exactly.
        """
        args0 = self.args_list[0]
        state_shape = (1, args0.gdn_nv_tp, args0.gdn_dk, args0.gdn_dv)
        conv_shape = (1, args0.linear_conv_kernel_dim - 1, args0.gdn_qkv_dim_tp)
        self.rbuf = {}
        for d in range(self.n_spans):
            sub = self.subs[d]
            for li, layer in enumerate(self.models[d].layers):
                if layer.is_full_attention:
                    if d > 0:
                        kv_shape = (1, self.nkv, self.span_len * d, self.hd)
                        for name in ("k", "v"):
                            self.rbuf[(d, li, name)] = ttnn.zeros(
                                kv_shape,
                                device=sub,
                                dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            )
                else:
                    self.rbuf[(d, li, "state")] = ttnn.zeros(
                        state_shape,
                        device=sub,
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    self.rbuf[(d, li, "conv")] = ttnn.zeros(
                        conv_shape,
                        device=sub,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

    def _mk_kv_cache(self, sub):
        return ttnn.from_torch(
            torch.zeros(self.num_blocks, self.nkv, BLOCK_SIZE, self.hd, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=sub,
            mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _kv_paged_to_seq(self, paged, blocks, S):
        """[num_blocks, nkv, BLOCK_SIZE, hd] paged cache -> [1, nkv, S, hd] sequence layout
        (blocks * BLOCK_SIZE == S). Shared by _send_kv (this die's cumulative prefix, forwarded
        to die d+1) and export_state_host (the last die's full sequence, read to host)."""
        sl = ttnn.slice(paged, (0, 0, 0, 0), (blocks, self.nkv, BLOCK_SIZE, self.hd))
        pm = ttnn.permute(sl, (1, 0, 2, 3))
        rs = ttnn.reshape(pm, (1, self.nkv, S, self.hd))
        ttnn.deallocate(sl)
        return rs

    def _send_kv(self, d, li):
        """Die d, AFTER its own layer li ran: forward the K/V prefix [0, span_len*(d+1)) to die
        d+1's persistent rbuf. Touches ONLY die d's queue (send_sock lives on subs[d])."""
        attn = self.models[d].layers[li].attention
        blocks = self.blocks_per_span * (d + 1)
        S = self.span_len * (d + 1)
        k_t = self._kv_paged_to_seq(attn.paged_k, blocks, S)
        v_t = self._kv_paged_to_seq(attn.paged_v, blocks, S)
        rb_k, rb_v = self.rbuf[(d + 1, li, "k")], self.rbuf[(d + 1, li, "v")]
        assert (
            tuple(k_t.spec.shape) == tuple(rb_k.spec.shape) and k_t.spec.dtype == rb_k.spec.dtype
        ), f"KV send/recv spec mismatch at layer {li}: sent {k_t.spec} vs rbuf {rb_k.spec}"
        send_sock, _ = self.kv_sockets[d]
        send_op = (
            ttnn.experimental.send_async if self.kv_socket_mode == "async" else ttnn.experimental.send_direct_async
        )
        send_op(k_t, send_sock)
        send_op(v_t, send_sock)
        ttnn.deallocate(k_t)
        ttnn.deallocate(v_t)

    def _recv_kv(self, d, li):
        """Die d, BEFORE its own layer li runs: receive layer li's K/V prefix from die d-1 into
        the persistent rbuf. Touches ONLY die d's queue (recv_sock lives on subs[d]) -- called
        from die d's OWN run_die, in die d's own per-layer program order. This is the fix for the
        capture-order bug: the recv is never issued while "on" a different die's turn."""
        _, recv_sock = self.kv_sockets[d - 1]
        recv_op = (
            ttnn.experimental.recv_async if self.kv_socket_mode == "async" else ttnn.experimental.recv_direct_async
        )
        recv_op(self.rbuf[(d, li, "k")], recv_sock)
        recv_op(self.rbuf[(d, li, "v")], recv_sock)

    def _send_gdn(self, d, li, gdn_out):
        """Die d, AFTER its own layer li ran: forward the GDN state to die d+1's persistent
        rbuf. Touches ONLY die d's queue."""
        final_state, conv_new_state = gdn_out
        rb_state, rb_conv = self.rbuf[(d + 1, li, "state")], self.rbuf[(d + 1, li, "conv")]
        assert (
            tuple(final_state.spec.shape) == tuple(rb_state.spec.shape)
            and final_state.spec.dtype == rb_state.spec.dtype
        ), f"GDN state send/recv spec mismatch at layer {li}: sent {final_state.spec} vs rbuf {rb_state.spec}"
        send_sock, _ = self.state_sockets[d]
        send_op = (
            ttnn.experimental.send_async if self.state_socket_mode == "async" else ttnn.experimental.send_direct_async
        )
        send_op(final_state, send_sock)
        send_op(conv_new_state, send_sock)
        ttnn.deallocate(final_state)
        ttnn.deallocate(conv_new_state)

    def _recv_gdn(self, d, li):
        """Die d, BEFORE its own layer li runs: receive layer li's GDN state from die d-1 into
        the persistent rbuf. Touches ONLY die d's queue -- see _recv_kv."""
        _, recv_sock = self.state_sockets[d - 1]
        recv_op = (
            ttnn.experimental.recv_async if self.state_socket_mode == "async" else ttnn.experimental.recv_direct_async
        )
        recv_op(self.rbuf[(d, li, "state")], recv_sock)
        recv_op(self.rbuf[(d, li, "conv")], recv_sock)

    def _tail(self, model, x, traced=False):
        """Last die only: select the last token, final norm, LM head.

        traced=True: use the PERSISTENT one-hot selector (span_len -- and so the selected
        position -- never changes, so it needs no per-call update) and return the raw device
        logits tensor (no to_torch: that is a host readback, not allowed inside a captured
        trace). The caller reads it back AFTER replay, in prefill_traced().
        """
        T = self.span_len
        if traced:
            sel_tt = self._sel_buf
        else:
            sel = torch.zeros(1, 1, 1, T, dtype=torch.float32)
            sel[0, 0, 0, T - 1] = 1.0
            sel_tt = ttnn.from_torch(
                sel,
                dtype=x.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=model.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(model.device),
            )
        x_last = ttnn.matmul(sel_tt, x)
        if not traced:
            ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = model.norm(x_last, mode=Mode.PREFILL)
        logits = model._lm_head(x_last)
        if traced:
            return logits
        lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.device, dim=0))
        return lt[0].reshape(-1)[: model.args.vocab_size]

    def _run_layer_major(self, tokens, *, traced=False, comm=True, dies=None):
        """Construct the whole prefill program LAYER-MAJOR across dies: embed every die in
        `dies`, then for each layer index, process ALL dies before moving to the next layer;
        tail on the last die (if included) once all layers are done.

        This ordering is REQUIRED for the untraced/eager path: direct-mode send_direct_async
        blocks (device-side) until the matching recv is issued, so a DIE-MAJOR host loop (all of
        die 0's ~1400 ops enqueued before die 1's first op) deadlocks -- die 0's queue fills
        waiting for a recv only die 1 can issue, and the host never reaches die 1's run. Layer-
        major guarantees die d's recv for layer L is issued (by die d itself; see _recv_kv/
        _recv_gdn) immediately after die (d-1)'s matching send for layer L, before die (d-1) is
        asked to send layer L+1. Mirrored in the traced path (capture()) for one shared
        implementation; tracing itself only records ops (no blocking), but this keeps the
        untraced warmup and the captured trace structurally identical.

        Each die issues ONLY its own queue's ops (recv/send split; see _recv_kv/_send_kv/
        _recv_gdn/_send_gdn) -- neither ever touches another die's queue.

        traced=True: read inputs from the PERSISTENT per-die buffers (populated by capture())
        instead of building fresh ones from `tokens`. No ttnn.from_torch/to_torch on this path.
        comm=False (EXP A/D): skip every send AND recv (no cross-die ops at all).
        dies=None: all n_spans dies (normal use); a subset for solo/no-dep timing experiments.
        """
        dies = list(range(self.n_spans)) if dies is None else list(dies)
        xs, cos_map, sin_map = {}, {}, {}
        if not traced:
            _sp("prefill start")

        for d in dies:
            sub, model = self.subs[d], self.models[d]
            if traced:
                tok = self._tok_buf[d]
                cos_map[d], sin_map[d] = self._cos_buf[d], self._sin_buf[d]
            else:
                tokens_d = tokens[:, d * self.span_len : (d + 1) * self.span_len]
                tok = ttnn.from_torch(
                    tokens_d.to(torch.int32),
                    dtype=ttnn.uint32,
                    device=sub,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
                )
                # _rope_tp_cos_sin_torch returns TORCH tensors (see model.py::prefill_tp).
                cos_t, sin_t = model._rope_tp_cos_sin_torch(d * self.span_len, self.span_len)
                rep = ttnn.ReplicateTensorToMesh(sub)
                cos_map[d] = ttnn.from_torch(
                    cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=sub, mesh_mapper=rep
                )
                sin_map[d] = ttnn.from_torch(
                    sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=sub, mesh_mapper=rep
                )
            # L1 seeds the whole residual/norm chain: rms_norm and the residual add both inherit
            # input_a's memory config, so DRAM here (the default) would force every norm/add in DRAM.
            x = model.embd(tok, memory_config=ttnn.L1_MEMORY_CONFIG)
            xs[d] = ttnn.reshape(x, (1, 1, self.span_len, x.shape[-1]))
            from models.demos.blackhole.qwen36.tt import tp_common as _tpc

            if _tpc.repl_residual_enabled(model.args):
                # One all-gather per die for the whole model; see tp_common.replicate_residual.
                xs[d] = _tpc.replicate_residual(xs[d], model)
            if not traced:
                logger.debug(f"[SPPrefill] die {d} embed enqueued")

        n_layers = len(self.models[dies[0]].layers)
        for li in range(n_layers):
            if not traced:
                ref_model = self.models[dies[0]]
                ckpt_idx = ref_model.layer_indices[li] if hasattr(ref_model, "layer_indices") else li
                kind = "attn" if ref_model.layers[li].is_full_attention else "gdn"
                _sp(f"prefill L{ckpt_idx} {kind}")
            for d in dies:
                layer = self.models[d].layers[li]
                x = xs[d]
                if layer.is_full_attention:
                    if d > 0 and comm:
                        self._recv_kv(d, li)
                    if d > 0:
                        ttnn.experimental.paged_fill_cache(
                            layer.attention.paged_k, self.rbuf[(d, li, "k")], self.prefix_page_tables[d], batch_idx=0
                        )
                        ttnn.experimental.paged_fill_cache(
                            layer.attention.paged_v, self.rbuf[(d, li, "v")], self.prefix_page_tables[d], batch_idx=0
                        )
                    x, _ = _sp_layer_mixer(
                        layer,
                        x,
                        cos=cos_map[d],
                        sin=sin_map[d],
                        d=d,
                        span_len=self.span_len,
                        page_table=self.full_page_tables[d],
                        chunk_page_table=self.own_chunk_page_tables[d],
                        gdn_in=None,
                    )
                    # The mixer's forward_prefill_paged already filled this die's paged K/V
                    # cache -- send it to die d+1 now, before the MLP, so the wavefront hop
                    # doesn't wait on ~0.4 ms of MLP work it doesn't need.
                    if d < self.n_spans - 1 and comm:
                        self._send_kv(d, li)
                    x = _sp_layer_mlp(layer, x)
                else:
                    if d > 0 and comm:
                        self._recv_gdn(d, li)
                    # Always a real (persistent) tensor -- zeros for die 0 / not-yet-received,
                    # real data once _recv_gdn has run. Never None: forward_prefill's
                    # conv_state=None path does a host-write zeros() that isn't capturable.
                    gdn_in = (self.rbuf[(d, li, "state")], self.rbuf[(d, li, "conv")])
                    x, gdn_out = _sp_layer_mixer(
                        layer,
                        x,
                        cos=cos_map[d],
                        sin=sin_map[d],
                        d=d,
                        span_len=self.span_len,
                        page_table=None,
                        chunk_page_table=None,
                        gdn_in=gdn_in,
                    )
                    if d == self.n_spans - 1:
                        self.last_die_gdn_state[li] = gdn_out
                    # gdn_out (final_state, conv_new_state) is the mixer's return value, not
                    # yet deallocated -- send it to die d+1 now, before the MLP runs on `x`.
                    if d < self.n_spans - 1 and comm:
                        self._send_gdn(d, li, gdn_out)
                    x = _sp_layer_mlp(layer, x)
                xs[d] = x
                if not traced:
                    logger.debug(f"[SPPrefill] die {d} layer {li} enqueued")

        last = self.n_spans - 1
        if not traced and last in dies:
            _sp("prefill tail")
        logits = self._tail(self.models[last], xs[last], traced=traced) if last in dies else None

        if not traced:
            for d in dies:
                ttnn.synchronize_device(self.subs[d])
            logger.info(f"[SPPrefill] layer-major pass done for dies={dies}")
        return logits

    def prefill(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [1, n_spans*span_len] -> last-die logits [vocab]. Layer-major across dies
        (see _run_layer_major) -- REQUIRED so direct-mode sends never block waiting for a recv
        the host hasn't reached yet."""
        assert tokens.shape[0] == 1, "SPPrefill is B=1 only (batch_idx=0/user_id=0 throughout)"
        assert tokens.shape[-1] == self.n_spans * self.span_len
        t0 = time.perf_counter()
        logits = self._run_layer_major(tokens, traced=False, comm=True)
        logger.info(f"[SPPrefill] total wall time: {(time.perf_counter() - t0) * 1000:.2f} ms")
        return logits

    def capture(self, tokens: torch.Tensor, comm: bool = True, dies=None):
        """Warm up (untraced, always full comm -- compiles programs; the rbuf spec asserts in
        _send_kv/_send_gdn fire here), allocate persistent per-die input buffers, then capture
        the whole `dies` program as one layer-major pass (begin_trace_capture on all `dies` ->
        _run_layer_major(traced=True, comm=comm, dies=dies) -> end_trace_capture on all `dies`).
        MeshSocket recv is device-blocking, so every die's capture must be opened before the
        shared layer-major pass runs, and only closed after all dies are done; each die issues
        its own recvs, in its own program order, and the host loop that builds the pass is
        layer-major, so no die's queue can back up behind another die's blocking send. Safe
        against a mid-capture failure: never leaves a trace open (see _safe_end_traces).

        Recv buffers (self.rbuf) are allocated once in __init__ (_alloc_rbuf), not here -- they
        are analytical (model dims), not warmup-derived, so recv/send are always valid.

        dies=None captures all n_spans dies (normal use); a subset with comm=False captures a
        single die's trace with no cross-die ops at all -- useful for isolating one die's own
        compute time from socket traffic.
        """
        assert tokens.shape[0] == 1, "SPPrefill is B=1 only (batch_idx=0/user_id=0 throughout)"
        assert tokens.shape[-1] == self.n_spans * self.span_len
        dies = list(range(self.n_spans)) if dies is None else list(dies)
        self.prefill(tokens)  # untraced warmup (always full comm); compiles kernels + validates rbuf specs

        for d in dies:
            model, sub = self.models[d], self.subs[d]
            tokens_d = tokens[:, d * self.span_len : (d + 1) * self.span_len].to(torch.int32)
            rep = ttnn.ReplicateTensorToMesh(sub)
            self._tok_buf[d] = ttnn.from_torch(
                tokens_d, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=sub, mesh_mapper=rep
            )
            cos_t, sin_t = model._rope_tp_cos_sin_torch(d * self.span_len, self.span_len)
            self._cos_buf[d] = ttnn.from_torch(
                cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=sub, mesh_mapper=rep
            )
            self._sin_buf[d] = ttnn.from_torch(
                sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=sub, mesh_mapper=rep
            )

        last = self.n_spans - 1
        if last in dies:
            sel = torch.zeros(1, 1, 1, self.span_len, dtype=torch.float32)
            sel[0, 0, 0, self.span_len - 1] = 1.0
            self._sel_buf = ttnn.from_torch(
                sel,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.subs[last],
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.subs[last]),
            )

        # Open ALL captures (for `dies`) before running the single layer-major pass, then close
        # all. Each die's own recv/send only ever touch that die's queue (see _recv_kv/_send_kv),
        # so every op lands in whichever device's trace is live when it targets that device.
        opened = {}
        try:
            for d in dies:
                opened[d] = ttnn.begin_trace_capture(self.subs[d], cq_id=0)
            out = self._run_layer_major(tokens, traced=True, comm=comm, dies=dies)
            for d in dies:
                ttnn.end_trace_capture(self.subs[d], opened[d], cq_id=0)
                self._trace_ids[d] = opened[d]
            if last in dies:
                self._traced_logits = out
        except Exception:
            self._safe_end_traces(opened)
            raise

    def _safe_end_traces(self, opened):
        """A trace left open (no matching end_trace_capture) wedges every later op on that
        device, including close_mesh_device -- so always try to close what capture() opened."""
        for d, tid in opened.items():
            try:
                ttnn.end_trace_capture(self.subs[d], tid, cq_id=0)
            except Exception as e:
                logger.warning(f"[SPPrefill] safe_end_traces: could not close trace on die {d}: {e!r}")

    def prefill_traced(self, tokens: torch.Tensor):
        """Replay the captured per-die traces on a new prompt. capture() must have run first.

        Returns (logits, wavefront_s, total_s): wavefront_s is device time (all 4 traces launched
        + synchronized); total_s additionally includes the host readback + argmax-ready logits.
        """
        assert all(t is not None for t in self._trace_ids), "capture() must run before prefill_traced()"
        assert tokens.shape[0] == 1, "SPPrefill is B=1 only (batch_idx=0/user_id=0 throughout)"
        assert tokens.shape[-1] == self.n_spans * self.span_len

        for d in range(self.n_spans):
            tokens_d = tokens[:, d * self.span_len : (d + 1) * self.span_len].to(torch.int32)
            tok_host = ttnn.from_torch(
                tokens_d,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.subs[d]),
            )
            ttnn.copy_host_to_device_tensor(tok_host, self._tok_buf[d])

        t0 = time.perf_counter()
        for d in range(self.n_spans):
            ttnn.execute_trace(self.subs[d], self._trace_ids[d], cq_id=0, blocking=False)
        for d in range(self.n_spans):
            ttnn.synchronize_device(self.subs[d])
        t1 = time.perf_counter()
        lt = ttnn.to_torch(self._traced_logits, mesh_composer=ttnn.ConcatMeshToTensor(self.subs[-1], dim=0))
        logits = lt[0].reshape(-1)[: self.args_list[-1].vocab_size].float()
        _ = int(torch.argmax(logits))
        t2 = time.perf_counter()
        return logits, (t1 - t0), (t2 - t0)

    def export_state_host(self):
        """Read the LAST die's post-prefill state back to host torch, in the exact layout
        ``sp_handoff.inject_into_tp_model`` expects: full-attention KV as [1, n_kv_heads, S,
        head_dim] bf16 (S = n_spans*span_len, the sequence this instance actually filled), GDN
        as (rec [1, Nv, Dk, Dv] fp32, conv [1, K-1, gdn_qkv_dim] bf16). Call after prefill()/
        prefill_traced() has run (untraced: reads that call's fresh tensors; traced: reads the
        persistent tensors the last replay wrote into, same as prefill_traced's own logits read).

        Returns (kv_per_attn_layer, gdn_per_layer, export_s).
        """
        t0 = time.perf_counter()
        last = self.n_spans - 1
        model = self.models[last]
        S = self.n_spans * self.span_len
        blocks = S // BLOCK_SIZE

        kv_per_attn_layer, gdn_per_layer = {}, {}
        for li, layer in enumerate(model.layers):
            if layer.is_full_attention:
                attn = layer.attention
                k_seq = self._kv_paged_to_seq(attn.paged_k, blocks, S)
                v_seq = self._kv_paged_to_seq(attn.paged_v, blocks, S)
                K = ttnn.to_torch(k_seq, mesh_composer=ttnn.ConcatMeshToTensor(model.device, dim=0))
                V = ttnn.to_torch(v_seq, mesh_composer=ttnn.ConcatMeshToTensor(model.device, dim=0))
                ttnn.deallocate(k_seq)
                ttnn.deallocate(v_seq)
                kv_per_attn_layer[li] = (K, V)
            else:
                final_state, conv_new_state = self.last_die_gdn_state[li]
                rec = ttnn.to_torch(final_state, mesh_composer=ttnn.ConcatMeshToTensor(model.device, dim=0))
                conv = ttnn.to_torch(conv_new_state, mesh_composer=ttnn.ConcatMeshToTensor(model.device, dim=0))
                gdn_per_layer[li] = (rec, conv)

        export_s = time.perf_counter() - t0
        logger.info(f"[SPPrefill] export_state_host: {export_s * 1000:.2f} ms")
        return kv_per_attn_layer, gdn_per_layer, export_s

    def close(self):
        """Release captured traces and drop socket / recv-buffer references. Idempotent: safe to
        call more than once, e.g. once from a failed __init__'s except clause and again from a
        test's own finally block."""
        for d, tid in enumerate(self._trace_ids):
            if tid is not None:
                ttnn.release_trace(self.subs[d], tid)
        self._trace_ids = [None] * self.n_spans
        self.state_sockets = []
        self.kv_sockets = []
        self.rbuf = {}
