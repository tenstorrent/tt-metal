# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-resident decode loop of the Hy4 decoders (D39-1: nothing on the host in the steady decode loop).

Resident state per mesh row (one user per row, T=1): ``pos_f`` (fp32 position of the token being fed), ``act_f`` (1 = live row, 0 = free row), ``rng_seed`` / ``rng_ctr`` (uint32), the decoder's own token buffer
``dec.tok``. Every per-step input that ``set_inputs`` / ``PoolCtx.set_step`` uploaded (rope tables, write rows, key counts, col info, ids, update index) is a PURE FUNCTION of the position and is derived on device
(``derive``, first ops of the step trace; fp32-exact arithmetic, CPU proof in tests/samp_state_cpu.py; RoPE = two-level tables, more accurate than the host fp32 tables at long context).
The step trace ends at the logits (persistent); the SAMPLER traces (G greedy / S shared TTSampling / F full-vocabulary threshold sampler, see ``variant``) are replayed right behind it on the same CQ, write the
sampled ids into ``dec.tok`` (the next step's embedding input) and advance pos / rng: ``submit()`` = two non-blocking trace replays, ``read()`` = a non-blocking read of 4 token ids.
The host uploads only on a transition (admission, slot change: ``load_inputs``, ``set_sampling``), as the plugin's decode reload contract (decode_input_update_contract 1) commands.
"""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

ROWS = 4
NC = HP.COLS
K = 2048
CH, SL = HP.CHUNK, HP.SLICE
VOCAB = 120832
SHARD = VOCAB // NC
B2 = 1024  # rope two-level split: pos = hi * 1024 + lo


def _rope_tables_two_level(dim, theta, interleave):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    out = []
    for step in (B2, 1):
        ang = torch.outer(torch.arange(B2, dtype=torch.float64) * step, inv)
        c, s = ang.cos().float(), ang.sin().float()
        if interleave:
            c, s = c.repeat_interleave(2, dim=-1), s.repeat_interleave(2, dim=-1)
        else:
            c, s = torch.cat([c, c], -1), torch.cat([s, s], -1)
        out += [c, s]
    return out  # ch, sh, cl, sl  each [1024, dim] fp32


class DeviceLoop:
    def __init__(self, dec):
        self.dec, self.md, self.ch = dec, dec.md, dec.ch
        md = self.md
        self.pool = hasattr(dec, "ctx")
        rows = lambda t, dt, lay=ttnn.TILE_LAYOUT: upload(md, t, dt, lay, row_dim=0)
        rep = lambda t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: upload(md, t, dt, lay)
        self.pos_f = rows(torch.zeros(ROWS, 1, 1, 1), ttnn.float32)
        self.act_f = rows(torch.ones(ROWS, 1, 1, 1), ttnn.float32)
        self.seed = rows(torch.zeros(ROWS, 1, 1, 1, dtype=torch.int32), ttnn.uint32)
        self.ctr = rows(torch.zeros(ROWS, 1, 1, 1, dtype=torch.int32), ttnn.uint32)
        self.own_col = upload(
            md, torch.arange(NC, dtype=torch.float32).reshape(1, NC, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT, shard_dim=1
        )
        oh = lambda i: rep(torch.nn.functional.one_hot(torch.tensor(i), 16).float().reshape(1, 1, 1, 16))
        self.oh0, self.oh1 = oh(0), oh(1)
        self.not_oh0 = rep(1.0 - torch.nn.functional.one_hot(torch.tensor(0), 16).float().reshape(1, 1, 1, 16))
        self.ar = rep(torch.arange(K, dtype=torch.float32).reshape(1, 1, 1, K))
        rope = self.ch.layers[0].attn.rope
        t = _rope_tables_two_level(rope.dim, rope.theta, rope.interleave)
        # embedding needs bf16 weights: every fp32 table entry is stored as bf16 hi + bf16 residual (16 bits); hi-index table [1024, 4 x 64] = (cos hi, cos lo, sin hi, sin lo) and the same for the low index
        ch_, sh_, cl_, sl_ = t
        pack = lambda c, s_: torch.cat([self._split(c), self._split(s_)], dim=-1).reshape(B2, 4 * rope.dim)
        self.rt = [
            rep(pack(ch_, sh_), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            rep(pack(cl_, sl_), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ]
        self.rope_dim = rope.dim
        self._cid = None
        self.barrier = os.environ.get("HY4_LOOP_BARRIER", "1") == "1"
        self.dummy_rm = upload(
            self.md, torch.zeros(1, 1, 1, 32), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT
        )  # parity-pad gather payload (allocated before any capture)
        self._col_ids()  # persistent constants MUST exist before any trace is captured (a tensor allocated after a capture can be overwritten by the replay of its intermediates)
        if self.pool:
            first = dec.ch.layers[0].attn.indexer.idx_rope_first
            self.first = first
            self.ones64, self.zeros64 = rep(torch.ones(1, 1, 1, 64), ttnn.bfloat16), rep(
                torch.zeros(1, 1, 1, 64), ttnn.bfloat16
            )

    @staticmethod
    def _split(x):  # fp32 -> [hi bf16, residual bf16] along the last dim (as fp32 tensors holding bf16-exact values)
        hi = x.bfloat16().float()
        return torch.cat([hi, (x - hi).bfloat16().float()], dim=-1)

    # ------------------------------------------------------------------ host side (transitions only)
    def load_inputs(self, tokens, pos, active=None):
        """reload_inputs: tokens [4], pos [4] (position of the fed token), active [4] (default: all live)."""
        md, dec = self.md, self.dec
        mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
        cp = lambda t, dt, lay, tgt: ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(t, dtype=dt, layout=lay, mesh_mapper=mp), tgt
        )
        cp(
            torch.tensor([int(x) for x in tokens], dtype=torch.int32).reshape(ROWS, 1, 1),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            dec.tok,
        )
        cp(torch.tensor([float(p) for p in pos]).reshape(ROWS, 1, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT, self.pos_f)
        a = [1.0] * ROWS if active is None else [float(x) for x in active]
        cp(torch.tensor(a).reshape(ROWS, 1, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT, self.act_f)

    # ------------------------------------------------------------------ device side (traceable)
    def _i(self, x, dt):  # fp32 [..] -> dt (int32 / uint32) ROW_MAJOR
        return ttnn.to_layout(ttnn.typecast(x, dt), ttnn.ROW_MAJOR_LAYOUT)

    def derive(self):
        """pos_f -> every per-step input of the decoder (copied into its persistent tensors)."""
        dec, p = self.dec, self.pos_f
        cp = ttnn.copy
        # rope (two-level)
        hi = ttnn.floor(ttnn.multiply(p, 1.0 / B2))
        lo = ttnn.subtract(p, ttnn.multiply(hi, float(B2)))
        ih = ttnn.reshape(self._i(hi, ttnn.uint32), [1, 1])
        il = ttnn.reshape(self._i(lo, ttnn.uint32), [1, 1])
        D = self.rope_dim

        def g(tbl, idx):  # -> (cos fp32, sin fp32) [1,1,1,D]
            e = ttnn.reshape(ttnn.embedding(idx, tbl, layout=ttnn.TILE_LAYOUT), [1, 1, 1, 4 * D])
            f = ttnn.typecast(e, ttnn.float32)
            sl = lambda i: ttnn.slice(f, [0, 0, 0, i * D], [1, 1, 1, (i + 1) * D])
            return ttnn.add(sl(0), sl(1)), ttnn.add(sl(2), sl(3))

        (ch_, sh_), (cl_, sl_) = g(self.rt[0], ih), g(self.rt[1], il)
        cos = ttnn.subtract(ttnn.multiply(ch_, cl_), ttnn.multiply(sh_, sl_))
        sin = ttnn.add(ttnn.multiply(sh_, cl_), ttnn.multiply(ch_, sl_))
        cosb, sinb = ttnn.typecast(cos, ttnn.bfloat16), ttnn.typecast(sin, ttnn.bfloat16)
        cp(cosb, dec.cos)
        cp(sinb, dec.sin)
        # ids [1,1,1,2048] = arange where <= pos else -1
        valid = ttnn.le(self.ar, p)
        ids = ttnn.subtract(ttnn.multiply(valid, ttnn.add(self.ar, 1.0)), 1.0)
        ids_i = ttnn.to_layout(ttnn.typecast(ttnn.typecast(ids, ttnn.int32), ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        if not self.pool:
            cp(ttnn.reshape(self._i(p, ttnn.int32), [1]), dec.upd)
            cp(ttnn.reshape(ids_i, [1, 1, 1, K]), dec.ids)
            return
        c128 = ttnn.concat(
            [cosb, self.ones64] if self.first else [self.ones64, cosb], dim=3
        )  # product9 fix: same order as hy4_indexer / hy4_attn_opt / Hy4PoolDecoder.set_inputs (was flipped: cos on the non-rotated half)
        s128 = ttnn.concat([sinb, self.zeros64] if self.first else [self.zeros64, sinb], dim=3)
        cp(c128, dec.c128)
        cp(s128, dec.s128)
        ctx = dec.ctx
        cp(ttnn.reshape(ids_i, [1, 1, 1, K]), ctx.ids_g)
        # KV geometry
        h5 = ttnn.floor(ttnn.multiply(p, 1.0 / CH))
        l5 = ttnn.subtract(p, ttnn.multiply(h5, float(CH)))
        col = ttnn.floor(ttnn.multiply(l5, 1.0 / SL))
        loc = ttnn.add(ttnn.multiply(h5, float(SL)), ttnn.subtract(l5, ttnn.multiply(col, float(SL))))
        mine = ttnn.eq(col, self.own_col)
        e0 = ttnn.subtract(ttnn.multiply(mine, ttnn.add(loc, 1.0)), 1.0)  # write row on the owner column, -1 elsewhere
        wr = ttnn.subtract(ttnn.multiply(e0, self.oh0), self.not_oh0)  # [1,1,1,16]: e0 then -1 x15
        cp(self._i(wr, ttnn.int32), ctx.wr)
        cp(ttnn.reshape(self._i(e0, ttnn.int32), [1, 1]), ctx.upd)
        n = ttnn.add(p, 1.0)
        fu = ttnn.floor(ttnn.multiply(n, 1.0 / CH))
        rem = ttnn.subtract(n, ttnn.multiply(fu, float(CH)))
        kv = ttnn.add(
            ttnn.multiply(fu, float(SL)),
            ttnn.minimum(ttnn.maximum(ttnn.subtract(rem, ttnn.multiply(self.own_col, float(SL))), 0.0), float(SL)),
        )
        cp(self._i(ttnn.multiply(kv, self.oh0), ttnn.uint32), ctx.kvcnt)
        ci = ttnn.add(ttnn.multiply(self.own_col, self.oh0), ttnn.multiply(n, self.oh1))
        cp(self._i(ci, ttnn.uint32), ctx.colinfo)

    # ------------------------------------------------------------------ RNG (counter based: the stream of a request is a function of (seed, position): nothing to carry or remap)
    def draw(self):
        """uint32 [1,1,1,1]: lowbias32(seed ^ pos * golden); pos = the position of the token being sampled FROM (the fed token's)."""
        pu = ttnn.typecast(self.pos_f, ttnn.uint32)
        x = ttnn.bitwise_xor(self.seed, ttnn.multiply(pu, 0x9E3779B9))
        x = ttnn.bitwise_xor(x, ttnn.logical_right_shift(x, 16))
        x = ttnn.multiply(x, 0x7FEB352D)
        x = ttnn.bitwise_xor(x, ttnn.logical_right_shift(x, 15))
        x = ttnn.multiply(x, 0x846CA68B)
        return ttnn.bitwise_xor(x, ttnn.logical_right_shift(x, 16))

    def u01(self):
        """fp32 [1,1,1,1] uniform in [0, 1) (24 bits)."""
        return ttnn.multiply(ttnn.typecast(ttnn.logical_right_shift(self.draw(), 8), ttnn.float32), 1.0 / (1 << 24))

    # ------------------------------------------------------------------ samplers
    def _col_ids(self):
        if getattr(self, "_cid", None) is None:
            self._cid = upload(self.md, torch.arange(NC, dtype=torch.float32).reshape(1, 1, 1, NC), ttnn.float32)
        return self._cid

    def greedy(self, logits):
        """exact fp32 argmax over the whole vocabulary (lowest index among ties) -> [1,1] uint32 (identical on all devices of a row). logits [1,1,1,SHARD] fp32 TILE."""
        ch = self.ch
        lg = logits if logits.dtype == ttnn.float32 else ttnn.typecast(logits, ttnn.float32)
        rm = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT)
        idx = ttnn.argmax(rm, dim=-1, keepdim=True)
        mx = ttnn.to_layout(ttnn.gather(rm, -1, index=idx), ttnn.TILE_LAYOUT)
        idxf = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)
        g = ch.mc.allgather(ttnn.concat([mx, idxf], dim=-1), ch.ccl, axis=1, dim=3)  # [1,1,1,NC*2]
        g = ttnn.reshape(g, [1, 1, NC, 2])
        mx_all = ttnn.reshape(ttnn.slice(g, [0, 0, 0, 0], [1, 1, NC, 1]), [1, 1, 1, NC])
        ix_all = ttnn.reshape(ttnn.slice(g, [0, 0, 0, 1], [1, 1, NC, 2]), [1, 1, 1, NC])
        best = ttnn.argmax(ttnn.to_layout(mx_all, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
        bestf = ttnn.typecast(ttnn.to_layout(best, ttnn.TILE_LAYOUT), ttnn.float32)
        sel = ttnn.sum(ttnn.multiply(ttnn.eq(bestf, self._col_ids()), ix_all), dim=-1, keepdim=True)
        tok = ttnn.add(ttnn.multiply(bestf, float(SHARD)), sel)
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, 1])

    def greedy_v2(self, logits):
        """greedy without tiny TILE reshapes: the (max, index) pair of every column is all-gathered as a ROW_MAJOR [1,1,1,2] tensor along dim 2 -> [1,1,NC,2]."""
        ch = self.ch
        lg = logits if logits.dtype == ttnn.float32 else ttnn.typecast(logits, ttnn.float32)
        rm = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT)
        idx = ttnn.argmax(rm, dim=-1, keepdim=True)  # [1,1,1,1] uint32 RM
        mx = ttnn.gather(rm, -1, index=idx)
        idxf = ttnn.typecast(idx, ttnn.float32)
        pair = ttnn.concat([mx, idxf], dim=-1)  # RM [1,1,1,2]
        g = ch.mc.allgather(pair, ch.ccl, axis=1, dim=2)  # [1,1,NC,2] RM
        t = ttnn.transpose(ttnn.to_layout(g, ttnn.TILE_LAYOUT), 2, 3)  # [1,1,2,NC]
        mx_all = ttnn.slice(t, [0, 0, 0, 0], [1, 1, 1, NC])
        ix_all = ttnn.slice(t, [0, 0, 1, 0], [1, 1, 2, NC])
        best = ttnn.argmax(ttnn.to_layout(mx_all, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
        bestf = ttnn.typecast(ttnn.to_layout(best, ttnn.TILE_LAYOUT), ttnn.float32)
        sel = ttnn.sum(ttnn.multiply(ttnn.eq(bestf, self._col_ids()), ix_all), dim=-1, keepdim=True)
        tok = ttnn.add(ttnn.multiply(bestf, float(SHARD)), sel)
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, 1])

    def _build_shared(self):
        import types

        from models.common.sampling.tt_sampling import TTSampling

        args = types.SimpleNamespace(
            vocab_size=VOCAB,
            padded_vocab_size=VOCAB,
            cluster_shape=(ROWS, NC),
            max_batch_size=32,
            max_top_k=32,
            sampling_dp=ROWS,
            sampling_all_gather_axis=1,
            sub_core_grids=None,
        )
        self.tts = TTSampling(self.md, self.ch.ccl, args)
        self.zeros32u = upload(
            self.md, torch.zeros(ROWS, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.TILE_LAYOUT, row_dim=0
        )

    def shared(self, logits):
        """shared tt_transformers TTSampling (top-k <= 32 candidates path; bf16 candidates): seeds derived on device from (seed, pos)."""
        seeds = ttnn.bitwise_and(self.draw(), 0x7FFFFFFF)  # MAX_UINT32 is the sampler's skip sentinel
        s32 = ttnn.add(self.zeros32u, seeds)  # [1,1,1,32] all 32 tile-row users of this mesh row get the draw
        ttnn.copy(ttnn.reshape(ttnn.to_layout(s32, ttnn.ROW_MAJOR_LAYOUT), [32]), self.tts.seeds_tt_tensor)
        x32 = ttnn.pad(logits, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
        o, _ = self.tts(x32)
        return ttnn.reshape(ttnn.slice(o, [0, 0, 0, 0], [1, 1, 1, 1]), [1, 1])

    def _build_full(self):
        from models.demos.blackhole.hy4_preview.tt.hy4_sampler import Hy4DeviceSampler

        self.fs = Hy4DeviceSampler(self.md, self.ch.mc, self.ch.ccl, users_per_row=1)
        self.fs.alloc_params()

    def full(self, logits, plain):
        return self.fs.forward(logits, self.fs.params, plain=plain, u_dev=ttnn.reshape(self.u01(), [1, 1, 1, 1]))

    # ------------------------------------------------------------------ epilogue + traces
    def epilogue(self, tok_new):
        ttnn.copy(ttnn.reshape(tok_new, [1, 1, 1]), self.dec.tok)
        ttnn.copy(ttnn.add(self.pos_f, self.act_f), self.pos_f)

    def _body(self, variant, logits=None):
        logits = self.out if logits is None else logits
        if variant == "G":
            tok = self.greedy_v2(logits)
        elif variant == "S":
            tok = self.shared(logits)
        else:
            tok = self.full(logits, plain=(variant == "Fp"))
        self.epilogue(tok)
        if self.barrier:
            self._row_barrier()

    def _row_barrier(self):
        """tiny gather over the 4 mesh ROWS (axis 0, linear): the last op of every sampler trace. With the per-row gather before it, no chip leaves the sampler before every chip has reached its end, so the NEXT step's
        cross-row writes (MoE reduce-scatter / gathers into the buffers the sampler's temporaries were allocated in) can not land while a slower row is still inside the sampler.
        """
        ch = self.ch
        ch.mc.allgather(self.dummy_rm, ch.ccl, axis=0, dim=2, linear=True)

    def _prep(self, variants):
        if "S" in variants and getattr(self, "tts", None) is None:
            self._build_shared()
        if any(v.startswith("F") for v in variants) and getattr(self, "fs", None) is None:
            self._build_full()
        os.environ["HY4_HEAD_FP32"] = "1"

    def capture(self, variants=("G",), warm_pos=64):
        """PRODUCTION: one trace per sampler variant = derive + step + head + sampler + epilogue (a single trace: the allocator plans every temporary, no cross-trace buffer overlap; the replays of
        consecutive steps can be pipelined without a host sync). State (tok / pos) is garbage afterwards: call load_inputs.
        """
        dec, md, ccl = self.dec, self.md, self.ch.ccl
        self._prep(variants)
        self.ftid = {}
        for v in variants:
            self.load_inputs([1000, 2000, 3000, 4000], [warm_pos] * ROWS)
            i0 = ccl.ag_ping_pong_idx
            self.derive()
            lg = dec.step()
            self._body(v, lg)
            ttnn.synchronize_device(md)
            ttnn.deallocate(lg)
            self._dummy_gather()
            ttnn.synchronize_device(md)
            odd = (ccl.ag_ping_pong_idx - i0) % 2  # parity of the gathers of one iteration (eager count = traced count)
            ccl.ag_ping_pong_idx = ccl.barrier_idx = i0 % 2
            self.ftid[v] = ttnn.begin_trace_capture(md, cq_id=0)
            self.derive()
            lg = dec.step()
            self._body(v, lg)
            if odd:  # an odd iteration would run the same ping-pong set back to back across consecutive replays
                self._dummy_gather()
            ttnn.end_trace_capture(md, self.ftid[v], cq_id=0)
            ttnn.synchronize_device(md)
        self.trace = None

    def capture_split(self, variants=("G",), warm_pos=64):
        """TEST ONLY: step trace (logits persistent in ``out``) + one separate sampler trace per variant (lets the sampler replay alone on frozen logits). Pipelined [step, sampler] replays are NOT safe (cross-trace buffer overlap)."""
        dec, md = self.dec, self.md
        if "S" in variants and getattr(self, "tts", None) is None:
            self._build_shared()
        if any(v.startswith("F") for v in variants) and getattr(self, "fs", None) is None:
            self._build_full()
        os.environ["HY4_HEAD_FP32"] = "1"
        ccl = self.ch.ccl
        idx = lambda: (ccl.ag_ping_pong_idx, ccl.barrier_idx)
        self.load_inputs([1000, 2000, 3000, 4000], [warm_pos] * ROWS)
        self.derive()
        lg = dec.step()
        ttnn.synchronize_device(md)
        ttnn.deallocate(lg)
        # CCL ping-pong parity: consecutive replays of [step, sampler] run WITHOUT a host sync, so the semaphore sets baked into the traces must alternate over the whole iteration and across the
        # iteration boundary (a fast chip's next-step gather must not reuse the set a slow chip's last gather is still using): the sampler capture starts where the step's calls left off and is padded
        # with one dummy gather when the iteration has an odd number of them (found by pipelined-replay divergence; sync-per-step replays hide it).
        a = idx()
        self.trace = ttnn.begin_trace_capture(md, cq_id=0)
        self.derive()
        self.out = dec.step()
        ttnn.end_trace_capture(md, self.trace, cq_id=0)
        ttnn.synchronize_device(md)
        b = idx()
        step_par = ((b[0] - a[0]) % 2, (b[1] - a[1]) % 2)
        dec.trace, dec.out = self.trace, self.out
        self.stid, self.parity = {}, {}
        for v in variants:
            w0 = idx()
            self._body(v)  # compile (eager)
            ttnn.synchronize_device(md)
            w1 = idx()
            self._dummy_gather()  # compile the pad program too (a first dispatch inside a capture writes binaries: TT_FATAL)
            ttnn.synchronize_device(md)
            n_par = ((w1[0] - w0[0]) % 2, (w1[1] - w0[1]) % 2)
            ccl.ag_ping_pong_idx, ccl.barrier_idx = (a[0] + step_par[0]) % 2, (a[1] + step_par[1]) % 2
            pad = ((step_par[0] + n_par[0]) % 2, (step_par[1] + n_par[1]) % 2)
            self.stid[v] = ttnn.begin_trace_capture(md, cq_id=0)
            self._body(v)
            for _ in range(
                2 if pad[0] != pad[1] else pad[0]
            ):  # one dummy gather flips both indices; unequal parities (not expected) need two to settle
                self._dummy_gather()
            ttnn.end_trace_capture(md, self.stid[v], cq_id=0)
            ttnn.synchronize_device(md)
            e = idx()
            self.parity[v] = dict(
                step=step_par, sampler=n_par, pad=pad, end_vs_start=((e[0] - a[0]) % 2, (e[1] - a[1]) % 2)
            )
        ccl.ag_ping_pong_idx, ccl.barrier_idx = a

    def _dummy_gather(self):
        ch = self.ch
        ch.mc.allgather(self.dummy_rm, ch.ccl, axis=1, dim=2)

    def sample_first(self, logits, temperature, top_k, top_p, seed, pos):
        """EAGER (prefill, not traced): the first token of one request from its last-prompt-token logits [1,1,1,SHARD] (all four mesh rows compute the same draw from the request's own row, the caller reads its row).
        Same sampler bodies and RNG convention as decode (draw = f(seed, position of the token the logits belong to)); clobbers pos / seed / params (the plugin reloads them at the next decode).
        """
        t4, k4, p4 = [float(temperature)] * ROWS, [int(top_k)] * ROWS, [float(top_p)] * ROWS
        self.load_inputs([0] * ROWS, [pos] * ROWS)
        v = self.set_sampling(t4, k4, p4, [int(seed)] * ROWS)
        if v == "G":
            return self.greedy_v2(logits)
        if v == "S":
            return self.shared(logits)
        return self.full(logits, plain=(v == "Fp"))

    def submit(self, variant="G"):
        """one decode step, non-blocking: step trace then the sampler trace; nothing is uploaded."""
        if getattr(self, "ftid", None) and variant in self.ftid:
            ttnn.execute_trace(self.md, self.ftid[variant], cq_id=0, blocking=False)
            return
        ttnn.execute_trace(self.md, self.trace, cq_id=0, blocking=False)
        ttnn.execute_trace(self.md, self.stid[variant], cq_id=0, blocking=False)

    def read_async(self):
        """enqueue the 4-id read behind the sampler trace; returns a handle for ``tokens``."""
        return ttnn.from_device(self.dec.tok, blocking=False)

    def tokens(self, handle):
        ttnn.synchronize_device(self.md)
        ts = ttnn.get_device_tensors(handle)
        return [int(ttnn.to_torch(ts[r * NC]).reshape(-1)[0]) for r in range(ROWS)]

    def release(self):
        for t in getattr(self, "ftid", {}).values():
            ttnn.release_trace(self.md, t)
        self.ftid = {}
        for t in getattr(self, "stid", {}).values():
            ttnn.release_trace(self.md, t)
        self.stid = {}
        if getattr(self, "trace", None) is not None:
            ttnn.release_trace(self.md, self.trace)
            self.trace = None
            self.dec.trace = None

    # ------------------------------------------------------------------ sampling state (transitions: reload_sampling_params / reset_sampling_state)
    def choose_variant(self, temperature, top_k, top_p, active=None):
        """G: every live row greedy (temperature 0 or top_k 1); S: (HY4_SAMPLER_TOPK=shared only) every live row sampled with 1 < top_k <= 32 (shared TTSampling); Fp: no live row has top_k or top_p (the generation_config default);
        else F (full-vocabulary threshold search). Rows may be given as lists of 4."""
        act = [True] * ROWS if active is None else [bool(a) for a in active]
        rows = [
            (float(t or 0.0), int(k if k is not None else 0), float(p if p is not None else 1.0))
            for t, k, p, a in zip(temperature, top_k, top_p, act)
            if a
        ]
        greedy = [t <= 0 or k == 1 for t, k, p in rows]
        if all(greedy):
            return "G"
        if (
            not any(greedy)
            and all(1 < k <= 32 for t, k, p in rows)
            and os.environ.get("HY4_SAMPLER_TOPK", "exact") == "shared"
        ):
            return "S"  # opt-in only: measured NOT exact on device (TV excess 0.002-0.017, chi-square z 6-25 at 100k draws); the default sends top-k <= 32 rows to the exact full-vocabulary sampler F
        if all(g or (k <= 0 and p >= 1.0) for g, (t, k, p) in zip(greedy, rows)):
            return "Fp"
        return "F"

    def set_sampling(self, temperature, top_k, top_p, seeds, active=None):
        """Upload the per-row sampling configuration and seeds (host -> device ONLY on a transition). Returns the variant to submit."""
        md = self.md
        v = self.choose_variant(temperature, top_k, top_p, active)
        mp = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
        sd = torch.tensor([int(s) & 0x7FFFFFFF for s in seeds], dtype=torch.int32).reshape(ROWS, 1, 1, 1)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(sd, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mp), self.seed
        )
        t = torch.tensor([float(x or 0.0) for x in temperature])
        k = torch.tensor([int(x if x is not None else 0) for x in top_k])
        p = torch.tensor([float(x if x is not None else 1.0) for x in top_p])
        greedy = ((t <= 0) | (k == 1)).float()
        if v in ("F", "Fp") or getattr(self, "fs", None) is not None:
            if getattr(self, "fs", None) is not None:
                invT = torch.where(t > 0, 1.0 / t.clamp(min=1e-6), torch.ones(ROWS))
                self.fs.set_params(invT, k.clamp(min=0).float(), p, torch.zeros(ROWS), greedy)
        if getattr(self, "tts", None) is not None:
            from models.common.sampling import SamplingParams
            from models.common.sampling.generator import format_sampling_params

            B = 32 * ROWS
            T_, K_, P_ = [0.0] * B, [1] * B, [1.0] * B
            for r in range(ROWS):
                T_[32 * r], K_[32 * r], P_[32 * r] = float(t[r]), int(k[r]), float(p[r])
            sp = format_sampling_params(SamplingParams(temperature=T_, top_k=K_, top_p=P_), B)
            self.tts.reset_params(torch.tensor(sp.top_k), torch.tensor(sp.top_p), torch.tensor(sp.temperature))
        self.variant = v
        return v
