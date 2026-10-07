# SPDX-License-Identifier: Apache-2.0
"""DFlash speculative decoding driver (greedy, exact): round = drafter proposes 7 tokens, the target verifies the 8-token block, accept the longest prefix whose drafts
equal the target argmax, commit it plus the target's bonus token, refill the drafter context with the accepted rows' hidden states.

Per user (mesh row r, slot u): `pos` = position of the anchor token (the last committed token, not yet fed to the target), `anchor` = its id.
Round:  d1..d7 = drafter(anchor, ctx = target hidden states of positions < pos)
        block = [anchor, d1..d7] at positions pos..pos+7; target argmax after every block row p_0..p_7 (verify; FALLBACK implementation = 8 sequential plain decode steps
        that capture the 5 hidden states per step, KV of all 8 positions written, positions beyond the accepted ones are overwritten by the next round)
        k = number of leading j (1..7) with d_j == p_{j-1};  committed = d1..dk + p_k;  new anchor = p_k at pos+k+1;  hidden states of rows 0..k go to the drafter.
Ragged acceptance per user is handled with per-row/per-slot positions of the decode path and the `valid` mask of the drafter append.
"""
import os
import time

import torch

from models.demos.blackhole.mimo_v26.tt.decode import Decoder

CAP = (0, 11, 23, 35, 47)
BLK = 8


class SpecEngine:
    def __init__(self, model, pf, drafter, U, max_pos, eos, cap=CAP, trace=True, log=print):
        self.m, self.pf, self.dr, self.U, self.max_pos, self.eos, self.log = (
            model,
            pf,
            drafter,
            U,
            max_pos,
            set(eos),
            log,
        )
        self.rows = model.mesh.shape[0]
        self.dec = Decoder(model, trace=trace, U=U, block=BLK, capture=tuple(cap))
        self.trace = trace
        self.dr_mode = os.environ.get(
            "SPEC_DR_TRACE", "late"
        )  # 0 = eager drafter, 1 = traced from round 0, late = eager in round 0 then traced (verify trace captured first)
        self.dr_trace = trace and self.dr_mode == "1"
        self.dr.use_trace = self.dr_trace
        self.stats = dict(rounds=[], t_draft=[], t_verify=[], t_append=[])

    def place(self, n):
        return [(i % self.rows, i // self.rows) for i in range(n)]

    # ------------------------------------------------------------------ start: chunked prefill with captures -> drafter context
    def start(self, prompts, max_new, res=None):
        rows, U = self.rows, self.U
        n = len(prompts)
        self.place_ = self.place(n)
        per_row = [[] for _ in range(rows)]
        for i, p in enumerate(prompts):
            per_row[i % rows].append(list(p))
        t0 = time.time()
        if res is None:
            res = self.pf.prefill(per_row, capture_feats=True)
        ft = self.pf.first_tokens(res["last"])
        self.t_prefill = time.time() - t0
        self.pos = torch.zeros(rows, U, dtype=torch.long)
        self.anchor = torch.zeros(rows, U, dtype=torch.long)
        self.alive = torch.zeros(rows, U, dtype=torch.bool)
        self.out = [[] for _ in range(n)]
        self.max_new = max_new
        for i, (r, u) in enumerate(self.place_):
            t = ft[(r, u)][0]
            self.pos[r, u] = len(prompts[i])
            self.anchor[r, u] = t
            self.out[i].append(t)
            self.alive[r, u] = t not in self.eos and max_new > 1
        # drafter context from the prompt hidden states (last <= 1024 positions per user)
        dr = self.dr
        dr.ring_pos.fill_(-1)
        R = dr.R
        mx = max(v[1].shape[0] for v in res["feats"].values())
        for s in range(0, mx, R):
            f = torch.zeros(rows, U, R, 20480, dtype=torch.bfloat16)
            pos = torch.zeros(rows, U, R, dtype=torch.long)
            valid = torch.zeros(rows, U, R, dtype=torch.bool)
            for (r, u), (st, fe) in res["feats"].items():
                m_ = max(0, min(R, fe.shape[0] - s))
                if m_:
                    f[r, u, :m_] = fe[s : s + m_]
                    pos[r, u, :m_] = st + s + torch.arange(m_)
                    valid[r, u, :m_] = True
            dr.set_append(f, pos, valid)
            dr.append()
        return res

    # ------------------------------------------------------------------ rounds
    def round(self):
        rows, U = self.rows, self.U
        dr, dec = self.dr, self.dec
        alive = self.alive & (self.pos + BLK + 1 < self.max_pos)
        self.alive = alive
        t0 = time.perf_counter()
        dr.set_round(self.anchor, self.pos)
        t_set = time.perf_counter() - t0
        d, _ = dr.draft(use_trace=self.dr_trace)  # [rows,U,7]
        t_draft = time.perf_counter() - t0
        self.stats.setdefault("t_set", []).append(t_set)
        block = torch.cat([self.anchor[:, :, None], d], 2)  # [rows,U,8]
        t0 = time.perf_counter()
        dec.set_inputs(block, self.pos, ~alive)  # = Decoder.verify split into its two parts to time them
        t_in = time.perf_counter() - t0
        res_v = dec.run()
        post = res_v[0].reshape(
            rows, U, BLK
        )  # [rows,U,8] target argmax after every block row; dec.feat = captured hidden states (device)
        t_verify = time.perf_counter() - t0
        self.stats.setdefault("t_vin", []).append(t_in)
        k = Decoder.accept_prefix(post, block)  # [rows,U]
        t0 = time.perf_counter()
        valid = torch.zeros(rows, U, BLK, dtype=torch.bool)
        for r in range(rows):
            for u in range(U):
                if alive[r, u]:
                    valid[r, u, : int(k[r, u]) + 1] = True
        dr.set_append(None, self.pos[:, :, None] + torch.arange(BLK), valid)
        dr.feat_src = dec.feat
        dr.append()
        t_append = time.perf_counter() - t0
        self.last = dict(d=d.clone(), post=post.clone(), k=k.clone(), block=block.clone())
        acc = []
        for i, (r, u) in enumerate(self.place_):
            if not alive[r, u]:
                continue
            kk = int(k[r, u])
            committed = [int(x) for x in d[r, u, :kk]] + [int(post[r, u, kk])]
            acc.append(kk + 1)
            for t in committed:
                self.out[i].append(t)
                if t in self.eos or len(self.out[i]) >= self.max_new:
                    self.alive[r, u] = False
                    break
            self.pos[r, u] += kk + 1
            self.anchor[r, u] = post[r, u, kk]
        if self.trace and self.dr_mode == "late":
            self.dr_trace = True
            self.dr.use_trace = True
        self.stats["rounds"].append(acc)
        self.stats["t_draft"].append(t_draft)
        self.stats["t_verify"].append(t_verify)
        self.stats["t_append"].append(t_append)
        return acc

    def run(self, max_rounds=10000):
        n = 0
        while self.alive.any() and n < max_rounds:
            self.round()
            n += 1
        return self.out
