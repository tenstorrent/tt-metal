# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vocab-sharded LM head across the SP dies (SPPrefillSC; QWEN36_SP_LMHEAD_SHARD, default "1"; "0" = the whole
greedy tail on the last die). The slice matmuls give bit-identical logits to the single-die A3 head for the same
normalized row; -0.85 ms SP TTFT.

Without it the last die runs the whole greedy tail: final norm + the A3 LM head (10 x [2048, 24832] bf8 DRAM
width-sharded chunks, ~1.2 ms at the DRAM floor) + concat + argmax, while dies 0..n-2 have finished. With it:

  * die d holds only vocab slice d: columns [d * V/n, (d + 1) * V/n), rebuilt at construction from that die's
    A3 chunks into ``_CHUNKS_PER_DIE`` (2) A3-style DRAM width-sharded chunks (same bf8 tiles, same DRAM-sharded
    matmul program config; QWEN36_SP_LMHEAD_DTYPE=bf4 re-quantizes them to bfloat4_b at load); the full-vocab
    chunks are freed on every die.
  * die n-1 normalizes its last row (the 64-core model._tail_norm_fast, written in the A3 in0 layout) and sends it
    down a reverse chain of new sockets n-1 -> n-2 -> ... -> 0 (the 1D line fabric here only connects
    neighbouring dies: a direct 3 -> 0 socket TT_FATALs); each die forwards it before its own slice matmuls. The
    recv on dies 0..n-2 is the last thing in their traces (they are idle by then).
  * each die computes its slice logits as one row-major row (per chunk: DRAM-sharded matmul -> untilize ->
    concat, as the A3 want_token path does).
  * reduce: the slice rows are relayed forward 0 -> 1 -> ... -> n-1 on a second new socket chain; die n-1
    concatenates the n rows in vocab order and runs one argmax (first max index == torch.argmax). One small
    readback (the token) as before. (A host argmax over the n slice rows was not adopted; a per-die (max value,
    index) pair was measured slower: ttnn.max / topk on a one-tile-row 62080-wide slice cost 220-320 us of
    device time vs 25 us for the argmax alone.)
  * the full logits (eager checks, prefill_traced's untimed logits) come from a host bounce: die n-1's
    normalized row -> host -> every die's slice matmuls -> host concat.
"""
import math
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.sp_prefill import _build_socket_pair
from models.tt_transformers.tt.common import Mode

# Socket cores (disjoint from the STATE / KV hop sockets on (0..3, 8..9)).
_BCAST_SEND_CORES = [ttnn.CoreCoord(4, 9), ttnn.CoreCoord(5, 9)]
_BCAST_RECV_CORES = [ttnn.CoreCoord(4, 8), ttnn.CoreCoord(5, 8)]
_GATHER_SEND_CORES = [ttnn.CoreCoord(6, 9), ttnn.CoreCoord(7, 9)]
_GATHER_RECV_CORES = [ttnn.CoreCoord(6, 8), ttnn.CoreCoord(7, 8)]


# A3-style DRAM width-sharded chunks per die slice.
_CHUNKS_PER_DIE = 2


def lmhead_dtype():
    """QWEN36_SP_LMHEAD_DTYPE: weight dtype of the SP LM-head slices. Unset / "" / "bf8" = bfloat8_b (default);
    "bf4" = bfloat4_b (half the DRAM read; the slices are re-quantized from the loaded bf8 values at load time;
    changes the logits, numerics gate required)."""
    v = os.environ.get("QWEN36_SP_LMHEAD_DTYPE", "").strip().lower()
    if v in ("", "bf8"):
        return ttnn.bfloat8_b
    if v == "bf4":
        return ttnn.bfloat4_b
    raise ValueError(f"QWEN36_SP_LMHEAD_DTYPE={v!r}: expected unset, 'bf8' or 'bf4'")


def lmhead_shard_enabled():
    """QWEN36_SP_LMHEAD_SHARD (default "1"): the vocab-sharded SP LM head (needs the A3 LM head, QWEN36_I3_LMHEAD=A3)."""
    return os.environ.get("QWEN36_SP_LMHEAD_SHARD", "1") == "1"


class SPLMHeadShard:
    def __init__(self, subs, models, vocab_size, static_send=False):
        """static_send (QWEN36_SP_STATIC_SEND=1): every LM-head transfer is fire-and-forget, as the layer hops:
        send_direct_async(static_dst_address=<the receiver's persistent buffer>) + recv_direct_async(wait_only=True),
        with each socket's FIFO sized to its transfers per request (same host-sync-per-replay requirement)."""
        self.static = bool(static_send)
        self.subs = subs
        self.models = models
        self.n = len(subs)
        self.vocab_size = vocab_size
        self.n_per = _CHUNKS_PER_DIE
        self.chunks = [None] * self.n
        self.xbuf = [None] * self.n  # d < n-1: the received normalized row
        self.gbuf = {}  # (d, j), j < d: die j's slice row relayed to die d
        self.bcast = []  # bcast[d] = pair(die d+1 -> die d)
        self.gather = []  # gather[d] = pair(die d -> die d+1)
        self._build_chunks()
        dim = int(models[0].args.dim)
        self.dim = dim
        for d in range(self.n - 1):
            self.xbuf[d] = ttnn.from_torch(
                torch.zeros(1, 1, dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=subs[d],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        for d in range(1, self.n):
            for j in range(d):
                self.gbuf[(d, j)] = ttnn.from_torch(
                    torch.zeros(1, 1, self.cols_d[j]),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=subs[d],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        st = ttnn.BufferType.L1
        for d in range(self.n - 1):
            # FIFO pages (64 B): bcast 1 transfer per request per hop, gather[d] d + 1 (static: one request's worth).
            self.bcast.append(_build_socket_pair(subs[d + 1], subs[d], _BCAST_SEND_CORES, _BCAST_RECV_CORES, st, 128))
            fifo = 64 * max(2, d + 1) if self.static else 128
            self.gather.append(
                _build_socket_pair(subs[d], subs[d + 1], _GATHER_SEND_CORES, _GATHER_RECV_CORES, st, fifo)
            )
        logger.info(
            f"[SPLMHeadShard] {self.n} dies x {self.n_per} chunks ({self.weight_dtype}); per-die slice cols "
            f"{self.cols_d}, static={int(self.static)}"
        )

    # ---------------------------------------------------------------------------------------------
    def _build_chunks(self):
        """Per die: vocab slice d of the A3 chunks -> n_per A3-style DRAM width-sharded chunks; free the rest."""
        full = self.models[0]._a3_lm_chunks
        assert full is not None, "QWEN36_SP_LMHEAD_SHARD needs the A3 LM head (QWEN36_I3_LMHEAD=A3)"
        K = int(full[0].shape[0])
        cw = int(full[0].shape[1])
        total = cw * len(full)
        assert (
            total % (self.n * self.n_per * 32) == 0
        ), f"vocab {total} does not split into {self.n} x {self.n_per} tiles"
        self.cols_d = [total // self.n] * self.n
        for v in self.cols_d:
            assert v > 0 and v % (32 * self.n_per) == 0, f"slice {v} cols is not {self.n_per} whole-tile chunks"
        self.col0 = [sum(self.cols_d[:d]) for d in range(self.n)]
        self.cols = max(self.cols_d)  # (log only)
        c = tpc.I3_A3_LM_CFG
        wdt = lmhead_dtype()  # QWEN36_SP_LMHEAD_DTYPE (default bfloat8_b)
        self.weight_dtype = wdt
        for d, (sub, m) in enumerate(zip(self.subs, self.models)):
            sub_cols = self.cols_d[d] // self.n_per
            nt = sub_cols // 32
            banks = tpc.i3_dram_banks(sub)
            shard_w = c["wpb"] * math.ceil(nt / (banks * c["wpb"]))
            used = banks - (banks * shard_w - nt) // shard_w
            assert shard_w == c["wpb"] * math.ceil(nt / (c["wpb"] * used)), (nt, shard_w, used)
            assert (
                math.ceil(nt / c["pcn"])
                <= sub.compute_with_storage_grid_size().x * sub.compute_with_storage_grid_size().y
            )
            mc = tpc.i3_dram_width_memcfg(K, shard_w, num_banks=banks)
            lo, hi = self.col0[d], self.col0[d] + self.cols_d[d]
            # Host round trip of the die's slice (on-device concat of bf8 column pieces segfaults): bf8 -> exact
            # float -> bf8 at tile-aligned columns re-quantizes to the same tiles (checked below).
            comp = ttnn.ConcatMeshToTensor(sub, dim=0)
            pieces = []
            for j, ch in enumerate(m._a3_lm_chunks):
                a, b = max(lo, j * cw), min(hi, (j + 1) * cw)
                if a >= b:
                    continue
                w = ttnn.to_torch(ch, mesh_composer=comp).reshape(K, cw)
                pieces.append(w[:, a - j * cw : b - j * cw])
            slab = torch.cat(pieces, dim=1)
            out = []
            for i in range(self.n_per):
                wi = slab[:, i * sub_cols : (i + 1) * sub_cols].contiguous()
                t = ttnn.from_torch(wi, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=sub, memory_config=mc)
                back = ttnn.to_torch(t, mesh_composer=comp).reshape(K, sub_cols)
                if wdt == ttnn.bfloat8_b:
                    assert torch.equal(back, wi), f"[SPLMHeadShard] die {d} chunk {i}: bf8 re-quantization is not exact"
                else:
                    # bf4 (QWEN36_SP_LMHEAD_DTYPE=bf4): lossy from the bf8 values; check it is a fixed point of the
                    # conversion (re-quantizing the dequantized tiles gives the same tiles).
                    t2 = ttnn.from_torch(back, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=sub, memory_config=mc)
                    back2 = ttnn.to_torch(t2, mesh_composer=comp).reshape(K, sub_cols)
                    ttnn.deallocate(t2)
                    assert torch.equal(back2, back), f"[SPLMHeadShard] die {d} chunk {i}: bf4 conversion not idempotent"
                    if d == 0 and i == 0:
                        rel = float((back.float() - wi.float()).norm() / wi.float().norm())
                        logger.info(f"[SPLMHeadShard] bf4 slice weights: rel L2 error vs the bf8 values {rel:.4f}")
                out.append(t)
            del slab, pieces
            for ch in m._a3_lm_chunks:
                ttnn.deallocate(ch)
            m._a3_lm_chunks = None  # the full-vocab A3 LM head is gone on every die (misuse asserts)
            self.chunks[d] = out
            ttnn.synchronize_device(sub)

    def refs(self):
        r = [t for d in range(self.n) for t in self.chunks[d]]
        r += [t for t in self.xbuf if t is not None]
        r += [self.gbuf[k] for k in sorted(self.gbuf)]
        return r

    # ---------------------------------------------------------------------------------------------
    def norm_row(self, model, hidden, L):
        """Last die: row L-1 of the final hidden -> TILE -> final norm. Returns (xn, x_own): xn = the normalized
        [1, 1, dim] bf16 TILE DRAM row (the broadcast payload); x_own = the tensor die n-1's own slice consumes
        (the norm's 8x8 L1 width-sharded output, which is the A3 in0 layout: model._tail_norm_fast, then one S2I for
        the payload; or xn itself when the row does not fit that layout). Does not consume hidden."""
        x_last = hidden if hidden.shape[1] == 1 else hidden[:, L - 1 : L, :]
        x_last = ttnn.to_layout(x_last, ttnn.TILE_LAYOUT)
        gx, gy = tpc.I3_A3_LM_CFG["in0_grid"]
        if (
            tpc.i3_one_tile_row(x_last)
            and not x_last.memory_config().is_sharded()
            and int(x_last.shape[-1]) == gx * gy * tpc.TILE_SIZE
        ):
            x_own = model._tail_norm_fast(x_last)
            return ttnn.sharded_to_interleaved(x_own, ttnn.DRAM_MEMORY_CONFIG), x_own
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        xn = model.norm(x_last, mode=Mode.PREFILL)
        return xn, xn

    def _slice_outs(self, d, x, want_rows):
        m = self.models[d]
        gx, gy = tpc.I3_A3_LM_CFG["in0_grid"]
        in0_mc = tpc.i3_l1_width_memcfg(int(x.shape[-1]), gx, gy)
        xs = ttnn.to_memory_config(x, in0_mc)
        pc = tpc.i3_a3_lm_progcfg()
        ck = m._i3_lm_ckc_hifi2()
        outs = []
        for c in self.chunks[d]:
            p = ttnn.linear(
                xs, c, compute_kernel_config=ck, program_config=pc, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            )
            if want_rows:
                outs.append(ttnn.to_layout(p, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG))
            else:
                outs.append(ttnn.sharded_to_interleaved(p, ttnn.DRAM_MEMORY_CONFIG))
            ttnn.deallocate(p)
        ttnn.deallocate(xs)
        return outs

    def slice_row(self, d, x):
        """Die d: slice logits as one row-major [1, 1, cols] bf16 DRAM row (a single page: the socket payload)."""
        outs = self._slice_outs(d, x, True)
        row = ttnn.concat(outs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for o in outs:
            ttnn.deallocate(o)
        return row

    def _check_spec(self, t, ref, what):
        assert (
            tuple(t.spec.shape) == tuple(ref.spec.shape)
            and t.dtype == ref.dtype
            and t.layout == ref.layout
            and t.memory_config() == ref.memory_config()
        ), f"[SPLMHeadShard] {what}: sent {t.spec} != recv buffer {ref.spec}"

    def program(self, model_last, hidden, L):
        """Enqueue the sharded greedy tail on every die (device ops only; valid inside the traces). Returns the
        per-request results: {"tok": die n-1 uint32 token}."""
        n, last = self.n, self.n - 1
        _send, _recv = ttnn.experimental.send_direct_async, ttnn.experimental.recv_direct_async
        if self.static:
            send = lambda t, sock, dst: _send(t, sock, static_dst_address=dst.buffer_address())  # noqa: E731
            recv = lambda t, sock: _recv(t, sock, wait_only=True)  # noqa: E731
        else:
            send = lambda t, sock, dst: _send(t, sock)  # noqa: E731
            recv = _recv
        xn, x_own = self.norm_row(model_last, hidden, L)
        self._check_spec(xn, self.xbuf[last - 1], "normalized row")
        send(xn, self.bcast[last - 1][0], self.xbuf[last - 1])
        rows = [None] * n
        rows[last] = self.slice_row(last, x_own)  # (consumes x_own when it is already the in0 layout)
        ttnn.deallocate(xn)
        for d in range(last - 1, -1, -1):
            recv(self.xbuf[d], self.bcast[d][1])
            if d > 0:
                send(self.xbuf[d], self.bcast[d - 1][0], self.xbuf[d - 1])
            rows[d] = self.slice_row(d, self.xbuf[d])
        # Relay rows forward. Die d sends its own row first (ready first), then forwards each row it receives
        # from die d-1 as soon as it lands; rows therefore arrive at die d in the order d-1, d-2, ..., 0.
        for d in range(n):
            if d < last:
                self._check_spec(rows[d], self.gbuf[(d + 1, d)], f"die {d} slice row")
                send(rows[d], self.gather[d][0], self.gbuf[(d + 1, d)])
                ttnn.deallocate(rows[d])
            for j in range(d - 1, -1, -1):
                recv(self.gbuf[(d, j)], self.gather[d - 1][1])
                if d < last:
                    send(self.gbuf[(d, j)], self.gather[d][0], self.gbuf[(d + 1, j)])
        # DRAM (as the A3 tail's concat): an L1 concat would put the argmax token -- a persistent trace output --
        # low in L1, under the static CB region of later programs.
        full = ttnn.concat(
            [self.gbuf[(last, j)] for j in range(last)] + [rows[last]], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(rows[last])
        tok = ttnn.argmax(full, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(full)
        return {"tok": tok}

    def token_from_results(self, res):
        """Host: the greedy token from program()'s results (reads it)."""
        t = ttnn.to_torch(res["tok"], mesh_composer=ttnn.ConcatMeshToTensor(self.subs[-1], dim=0))
        return int(t.reshape(-1)[0])

    def logits_host(self, model_last, hidden, L):
        """Eager (not traced): full last-token logits [vocab] (host) via a host bounce of the normalized row."""
        xn, x_own = self.norm_row(model_last, hidden, L)
        xh = ttnn.to_torch(xn, mesh_composer=ttnn.ConcatMeshToTensor(self.subs[-1], dim=0))
        ttnn.deallocate(xn)
        if x_own is not xn:
            ttnn.deallocate(x_own)
        parts = []
        for d, sub in enumerate(self.subs):
            x = ttnn.from_torch(
                xh.reshape(1, 1, self.dim).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=sub,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            outs = self._slice_outs(d, x, False)
            ttnn.deallocate(x)
            lg = ttnn.concat(outs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for o in outs:
                ttnn.deallocate(o)
            lt = ttnn.to_torch(lg, mesh_composer=ttnn.ConcatMeshToTensor(sub, dim=0))
            ttnn.deallocate(lg)
            parts.append(lt.reshape(-1, lt.shape[-1])[0])
        return torch.cat(parts)[: self.vocab_size]
