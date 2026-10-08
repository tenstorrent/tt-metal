# SPDX-License-Identifier: Apache-2.0
"""Device-resident DFlash spec-decode round (no host work between rounds): drafter -> verify -> accept -> drafter-ring append -> state update, one trace, enqueue only.

Per-device state (fp32 TILE, shape [1,U,..]; mesh rows hold independent user groups, all 8 columns hold identical copies):
  pos [1,U,1,1]    position of the anchor token (= tokens already in the target KV)     alive [1,U,1,1]  1/0
  anchor [1,U,1,1] last committed token id (not yet fed to the target)                  cnt [1,U,1,1]    stream length (first token counted)
  out [1,U,MAXOUT,1] committed token stream                                             ring_pos [1,U,1024,1] position stored in each drafter ring slot (-1 empty)
Everything the old host driver computed per round is derived on device from these: window mask from ring_pos/pos, RoPE cos/sin from an embedding lookup of the positions, verify inputs
(tokens, positions, ordered cache-update indices, MoE live mask) assigned into the persistent inputs of the decode graph, accept (eq + triangular-matmul leading-ones count), bonus token,
EOS/max flags, ring scatter (one-hot masks from iota == position mod 1024), stream scatter. The host only enqueues N rounds and reads (alive, cnt, out) once per chunk.
"""
import time

import torch

import ttnn
from models.demos.blackhole.mimo_v26.tt.dflash import BLK, HD, KPAD, NEG, WIN, rope_cs

F32 = ttnn.float32


class SpecDevice:
    def __init__(self, model, dec, dr, U, max_pos, eos, max_new, maxout=None):
        self.m, self.mesh, self.dec, self.dr, self.U, self.max_pos, self.max_new = (
            model,
            model.mesh,
            dec,
            dr,
            U,
            max_pos,
            max_new,
        )
        self.eos = [float(e) for e in eos]
        self.B = U * BLK
        self.rows, self.cols = self.mesh.shape[0], self.mesh.shape[1]
        self.maxout = maxout or (-(-(max_new + 16) // 32) * 32)
        mesh = self.mesh
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.rowmap = ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, None))
        self.vocab_local = dec.vocab_local

        def const(t, dtype=F32, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t, device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
            )

        def state(t, dtype=F32):
            return ttnn.from_torch(
                t,
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.rowmap,
            )

        self.const, self.state_t = const, state
        z = lambda *s: torch.zeros(self.rows, *s)
        self.pos = state(z(U, 1, 1))
        self.alive = state(z(U, 1, 1))
        self.anchor = state(z(U, 1, 1))
        self.cnt = state(z(U, 1, 1))
        self.out = state(z(U, self.maxout, 1))
        self.rcount = state(z(U, 1, 1))  # alive rounds per user
        self.ring_pos = state(torch.full((self.rows, U, WIN, 1), -1.0))
        # constants
        self.c_iota8 = const(torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8))
        self.c_iota1024c = const(torch.arange(WIN, dtype=torch.float32).reshape(1, 1, WIN, 1))
        self.c_iotaout = const(torch.arange(self.maxout, dtype=torch.float32).reshape(1, 1, self.maxout, 1))
        self.c_tri7 = const(torch.triu(torch.ones(7, 7)).reshape(1, 1, 7, 7))
        self.c_offs = const(
            (torch.arange(self.cols, dtype=torch.float32) * self.vocab_local).reshape(1, 1, 1, self.cols)
        )
        self.c_rev = const(
            (self.vocab_local - torch.arange(self.vocab_local, dtype=torch.float32)).reshape(1, 1, 1, self.vocab_local)
        )  # for the lowest-index tie rule
        self.c_iotacols = const(torch.arange(self.cols, dtype=torch.float32).reshape(1, 1, 1, self.cols))
        E = torch.zeros(self.B, U)
        for u in range(U):
            E[u * BLK : (u + 1) * BLK, u] = 1.0
        self.c_E = const(E.reshape(1, 1, self.B, U))
        tail = torch.full((1, U, 1, KPAD - WIN), NEG)
        tail[..., :BLK] = 0
        self.c_tail = const(tail)
        self.c_zeros_live = const(torch.zeros(1, 1, 64 - self.B, 1), ttnn.bfloat16)
        ntab = max_pos + 16
        cos, sin = rope_cs(torch.arange(ntab))
        self.c_cos = const(cos.bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.c_sin = const(sin.bfloat16(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        self.c_m = []  # per block step t: [1,B] fp32 row, 1 where virtual row % 8 == t
        for t in range(BLK):
            m = torch.zeros(1, 1, 1, self.B)
            m[..., t::BLK] = 1.0
            self.c_m.append(const(m))
        self._tid = None
        self.times = []

    # ------------------------------------------------------------------ small helpers (fp32 control math)
    def rm(self, x):
        return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    def tl(self, x):
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def reshape_rm(self, x, shape, dtype=None):
        r = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), shape)
        return ttnn.to_layout(r, ttnn.TILE_LAYOUT)

    def global_argmax(self, vals, idx, lg=None):
        """Per-device local (max, argmax) of the vocab shard [1,1,N,1] -> global argmax fp32 [1,1,N,1]; lowest vocab id wins ties (lowest column first).
        With lg (the bf16 logits shard [1,1,N,V/8]) the LOCAL argmax is recomputed with an explicit lowest-index rule: ttnn.argmax on 8 rows breaks exact ties differently from the
        1-row plain decode (found at ISL 4096: 3 of 4 spec/plain divergences were exact logit ties that went to the higher id).
        """
        N = vals.shape[2]
        if lg is not None:
            m = ttnn.max(lg, dim=3, keepdim=True)  # [1,1,N,1] bf16
            hit = ttnn.typecast(ttnn.eq(lg, m), F32)
            sc = ttnn.max(ttnn.multiply(hit, self.c_rev), dim=3, keepdim=True)  # fp32: V/8 - first hit index
            vals = m
            idx_f = ttnn.subtract(float(self.vocab_local), sc)
        else:
            idx_f = ttnn.typecast(self.tl(idx), F32)
        vg = ttnn.all_gather(
            self.tl(ttnn.typecast(vals, F32)), dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [1,1,N,cols]
        ig = ttnn.all_gather(idx_f, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        col = ttnn.argmax(ttnn.to_layout(ttnn.typecast(vg, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT), dim=3, keepdim=True)
        colf = ttnn.typecast(self.tl(col), F32)  # [1,1,N,1]
        onehot = ttnn.eq(self.c_iotacols, colf)  # [1,1,N,cols]
        gid = ttnn.sum(ttnn.multiply(onehot, ttnn.add(ig, self.c_offs)), dim=3, keepdim=True)
        return gid

    # ------------------------------------------------------------------ one round (all device ops)
    def _round(self):
        U, B, dec, dr = self.U, self.B, self.dec, self.dr
        pos, alive, anchor, cnt = self.pos, self.alive, self.anchor, self.cnt
        posb = ttnn.add(pos, self.c_iota8)  # [1,U,1,8]
        # ---- rope tables for the block rows (drafter + append)
        idx = ttnn.typecast(posb, ttnn.uint32)
        idx = ttnn.reshape(ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT), (1, U * BLK))
        cos8 = ttnn.embedding(idx, self.c_cos, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1,U*8,128]
        sin8 = ttnn.embedding(idx, self.c_sin, layout=ttnn.ROW_MAJOR_LAYOUT)
        ck = ttnn.to_layout(ttnn.reshape(cos8, (U, 1, BLK, HD)), ttnn.TILE_LAYOUT)
        sk = ttnn.to_layout(ttnn.reshape(sin8, (U, 1, BLK, HD)), ttnn.TILE_LAYOUT)
        cq = ttnn.concat([ck] * 8, dim=2)  # rows (head, tok)
        sq = ttnn.concat([sk] * 8, dim=2)
        acos = ttnn.to_layout(ttnn.reshape(cos8, (1, 1, U * BLK, HD)), ttnn.TILE_LAYOUT)
        asin = ttnn.to_layout(ttnn.reshape(sin8, (1, 1, U * BLK, HD)), ttnn.TILE_LAYOUT)
        # ---- window mask from the ring position map
        ringrow = ttnn.transpose(self.ring_pos, -2, -1)  # [1,U,1,1024]
        lo = ttnn.subtract(pos, float(WIN))
        vis = ttnn.multiply(ttnn.multiply(ttnn.ge(ringrow, lo), ttnn.lt(ringrow, pos)), ttnn.ge(ringrow, 0.0))
        mrow = ttnn.multiply(ttnn.subtract(1.0, vis), NEG)
        mask = ttnn.concat([mrow, self.c_tail], dim=3)  # [1,U,1,KPAD]
        mask = ttnn.reshape(ttnn.typecast(mask, ttnn.bfloat16), (U, 1, 1, KPAD))
        # ---- drafter
        tok = ttnn.reshape(ttnn.to_layout(ttnn.typecast(anchor, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, U))
        y, dvals, didx = dr._draft_forward(inp=(tok, mask, cq, sq, ck, sk))
        ttnn.deallocate(y)
        dg = self.global_argmax(dvals, didx)  # [1,1,8U,1]
        drow = ttnn.reshape(
            ttnn.to_layout(ttnn.transpose(dg, -2, -1), ttnn.ROW_MAJOR_LAYOUT), (1, U, 1, BLK)
        )  # block row t=0 is the anchor slot (ignored)
        d7 = ttnn.slice(drow, [0, 0, 0, 1], [1, U, 1, BLK])  # RM [1,U,1,7]
        block = ttnn.concat([ttnn.to_layout(anchor, ttnn.ROW_MAJOR_LAYOUT), d7], dim=3)  # RM [1,U,1,8]
        block = ttnn.to_layout(block, ttnn.TILE_LAYOUT)
        # ---- verify inputs (dead rows: token 0 / position 0)
        tok_v = ttnn.multiply(block, alive)  # [1,U,1,8]
        pv = ttnn.multiply(posb, alive)
        ttnn.assign(
            ttnn.reshape(ttnn.to_layout(ttnn.typecast(tok_v, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, B)), dec.tok
        )
        pv_flat = ttnn.reshape(ttnn.to_layout(pv, ttnn.ROW_MAJOR_LAYOUT), (1, 1, 1, B))
        pv_t = ttnn.to_layout(pv_flat, ttnn.TILE_LAYOUT)
        ttnn.assign(
            ttnn.reshape(ttnn.to_layout(ttnn.typecast(pv_t, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT), (B,)), dec.pos["cur"]
        )
        ttnn.assign(
            ttnn.reshape(ttnn.to_layout(ttnn.typecast(pv_t, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, B)),
            dec.pos["idx"],
        )
        for t in range(BLK):  # ordered cache updates: index = position for the rows of token t, -1 for the others
            m = self.c_m[t]
            u_t = ttnn.add(ttnn.multiply(pv_t, m), ttnn.subtract(m, 1.0))
            ttnn.assign(
                ttnn.reshape(ttnn.to_layout(ttnn.typecast(u_t, ttnn.int32), ttnn.ROW_MAJOR_LAYOUT), (B,)), dec.upd[t]
            )
        alive_col = ttnn.reshape(ttnn.to_layout(alive, ttnn.ROW_MAJOR_LAYOUT), (1, 1, U, 1))
        live_v = ttnn.matmul(self.c_E, ttnn.to_layout(alive_col, ttnn.TILE_LAYOUT))  # [1,1,B,1]
        live = ttnn.concat(
            [
                ttnn.to_layout(ttnn.typecast(live_v, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT),
                ttnn.to_layout(self.c_zeros_live, ttnn.ROW_MAJOR_LAYOUT),
            ],
            dim=2,
        )
        ttnn.assign(ttnn.to_layout(live, ttnn.TILE_LAYOUT), dec.live.t)
        # ---- verify (the decode agent's graph; reads the persistent inputs assigned above)
        vvals, vidx, vlg, feat = dec._forward()
        ag = self.global_argmax(vvals, vidx, vlg)  # [1,1,B,1]
        arow = ttnn.reshape(ttnn.to_layout(ttnn.transpose(ag, -2, -1), ttnn.ROW_MAJOR_LAYOUT), (1, U, 1, BLK))
        arow = ttnn.to_layout(arow, ttnn.TILE_LAYOUT)  # target argmax after block position t
        # ---- accept
        ok = ttnn.eq(ttnn.slice(arow, [0, 0, 0, 0], [1, U, 1, 7]), ttnn.slice(block, [0, 0, 0, 1], [1, U, 1, BLK]))
        bad = ttnn.subtract(1.0, ok)
        cb = ttnn.matmul(bad, self.c_tri7)
        pref = ttnn.eq(cb, 0.0)
        k = ttnn.sum(pref, dim=3, keepdim=True)  # [1,U,1,1] accepted drafts
        valid = ttnn.multiply(ttnn.le(self.c_iota8, k), alive)  # rows 0..k of alive users
        onehot_k = ttnn.eq(self.c_iota8, k)
        bonus = ttnn.sum(ttnn.multiply(arow, onehot_k), dim=3, keepdim=True)
        iseos = ttnn.multiply(arow, 0.0)
        for e in self.eos:
            iseos = ttnn.add(iseos, ttnn.eq(arow, e))
        eos_hit = ttnn.gt(ttnn.sum(ttnn.multiply(iseos, valid), dim=3, keepdim=True), 0.0)
        kp1 = ttnn.multiply(ttnn.add(k, 1.0), alive)
        alive_n = ttnn.multiply(
            ttnn.multiply(alive, ttnn.subtract(1.0, eos_hit)),
            ttnn.multiply(
                ttnn.lt(ttnn.add(cnt, kp1), float(self.max_new)),
                ttnn.lt(ttnn.add(ttnn.add(pos, kp1), 9.0), float(self.max_pos)),
            ),
        )
        # ---- stream
        dest = ttnn.add(cnt, self.c_iota8)  # [1,U,1,8]
        S_out = ttnn.multiply(ttnn.eq(self.c_iotaout, dest), valid)  # [1,U,maxout,8]
        keep_out = ttnn.subtract(1.0, ttnn.sum(S_out, dim=3, keepdim=True))
        out_n = ttnn.add(ttnn.multiply(self.out, keep_out), ttnn.sum(ttnn.multiply(S_out, arow), dim=3, keepdim=True))
        # ---- drafter ring append of rows 0..k (device-selected)
        slot = ttnn.subtract(posb, ttnn.multiply(ttnn.floor(ttnn.multiply(posb, 1.0 / WIN)), float(WIN)))
        S = ttnn.multiply(ttnn.eq(self.c_iota1024c, slot), valid)  # [1,U,1024,8]
        keep = ttnn.subtract(1.0, ttnn.sum(S, dim=3, keepdim=True))  # [1,U,1024,1]
        ring_n = ttnn.add(ttnn.multiply(self.ring_pos, keep), ttnn.sum(ttnn.multiply(S, posb), dim=3, keepdim=True))
        S_b = ttnn.reshape(ttnn.typecast(S, ttnn.bfloat16), (U, 1, WIN, BLK))
        keep_b = ttnn.reshape(ttnn.typecast(keep, ttnn.bfloat16), (U, 1, WIN, 1))
        dr._append_forward(feat=feat, acos=acos, asin=asin, S=S_b, keep=keep_b)
        # ---- state update (assign in place)
        ttnn.assign(ring_n, self.ring_pos)
        ttnn.assign(ttnn.add(pos, kp1), pos)
        ttnn.assign(ttnn.add(ttnn.multiply(alive, bonus), ttnn.multiply(ttnn.subtract(1.0, alive), anchor)), anchor)
        ttnn.assign(ttnn.add(cnt, kp1), cnt)
        ttnn.assign(ttnn.add(self.rcount, alive), self.rcount)
        ttnn.assign(out_n, self.out)
        ttnn.assign(alive_n, alive)
        self.dbg = dict(block=block, arow=arow, k=k, drow=drow)
        return k

    # ------------------------------------------------------------------ host control (outside the inference loop)
    def init_state(self, pos, anchor, alive, first_tokens, ring_pos_host):
        """pos/anchor/alive/first_tokens [rows,U] torch; ring_pos_host [rows,U,1024] (drafter slot -> position, -1 empty)."""
        rows, U = pos.shape
        f = lambda t: t.float().reshape(rows, U, 1, 1)
        out = torch.zeros(rows, U, self.maxout, 1)
        out[:, :, 0, 0] = first_tokens.float()
        up = lambda host, dev: ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host, dtype=F32, layout=ttnn.TILE_LAYOUT, mesh_mapper=self.rowmap), dev
        )
        up(f(pos), self.pos)
        up(f(anchor), self.anchor)
        up(f(alive), self.alive)
        up(torch.ones(rows, U, 1, 1), self.cnt)
        up(torch.zeros(rows, U, 1, 1), self.rcount)
        up(out, self.out)
        up(ring_pos_host.float().reshape(rows, U, WIN, 1), self.ring_pos)
        ttnn.synchronize_device(self.mesh)

    def _snapshot(self):
        names = ["pos", "alive", "anchor", "cnt", "out", "ring_pos", "rcount"]
        snap = {n: ttnn.clone(getattr(self, n)) for n in names}
        snap["rk"] = [ttnn.clone(t) for t in self.dr.ring_k]
        snap["rv"] = [ttnn.clone(t) for t in self.dr.ring_v]
        return snap

    def _restore(self, snap):
        for n in ["pos", "alive", "anchor", "cnt", "out", "ring_pos", "rcount"]:
            ttnn.assign(snap[n], getattr(self, n))
        for a, b in zip(snap["rk"], self.dr.ring_k):
            ttnn.assign(a, b)
        for a, b in zip(snap["rv"], self.dr.ring_v):
            ttnn.assign(a, b)

    def prepare(self):
        """eager warm-up round (state restored afterwards), then trace capture of one round."""
        import os

        snap = self._snapshot()
        ttnn.synchronize_device(self.mesh)
        k = self._round()
        ttnn.synchronize_device(self.mesh)
        self._restore(snap)
        ttnn.synchronize_device(self.mesh)
        chk = os.environ.get("MIMO_CHECK_CAP")
        os.environ["MIMO_CHECK_CAP"] = "0"
        self._tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self._round()
        ttnn.end_trace_capture(self.mesh, self._tid, cq_id=0)
        if chk is None:
            os.environ.pop("MIMO_CHECK_CAP")
        else:
            os.environ["MIMO_CHECK_CAP"] = chk
        ttnn.synchronize_device(self.mesh)

    def read_state(self):
        cols = self.cols
        get = lambda t: torch.stack(
            [ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).float() for r in range(self.rows)]
        )
        return dict(
            alive=get(self.alive).reshape(self.rows, self.U),
            cnt=get(self.cnt).reshape(self.rows, self.U),
            out=get(self.out).reshape(self.rows, self.U, self.maxout),
            pos=get(self.pos).reshape(self.rows, self.U),
            anchor=get(self.anchor).reshape(self.rows, self.U),
            rcount=get(self.rcount).reshape(self.rows, self.U),
        )

    def host_dbg(self, t):
        cols = self.cols
        return torch.stack([ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).float() for r in range(self.rows)])

    def run(self, chunk=4, max_rounds=100000):
        """Enqueue rounds back to back (non-blocking); the host reads the state once per chunk. Returns (rounds enqueued, per-chunk wall s, per-chunk enqueue-only wall s)."""
        n = 0
        chunk_t, enq_t = [], []
        while n < max_rounds:
            t0 = time.perf_counter()
            for _ in range(chunk):
                ttnn.execute_trace(self.mesh, self._tid, cq_id=0, blocking=False)
            enq_t.append(time.perf_counter() - t0)
            n += chunk
            st = self.read_state()  # blocking read: one sync per chunk
            chunk_t.append(time.perf_counter() - t0)
            if not (st["alive"] > 0.5).any():
                break
        return n, chunk_t, enq_t

    def release(self):
        if self._tid is not None:
            ttnn.release_trace(self.mesh, self._tid)
            self._tid = None
