# SPDX-License-Identifier: Apache-2.0
"""Greedy generation on the 4x8 mesh: prefill (MiMoModel.prefill_first_token, or tt/prefill.py when it lands) + traced decode loop (tt/decode.py).

Users are spread over mesh rows (prompt i -> row i % rows, slot i // rows); up to 32 slots per row = 128 users. Finished users are marked dead (position/token
clamped to 0, MoE routing spread) so they can neither run past max_pos nor overflow dispatch buffers.
"""
import json
import time

import torch

from models.demos.blackhole.mimo_v26.tt.decode import SLOTS, Decoder

PAD = 151643


def load_eos(path="/mnt/tt-data/MiMo-V2.6-Flash-RL/generation_config.json"):
    e = json.load(open(path)).get("eos_token_id", [])
    return set(e if isinstance(e, list) else [e])


class Generator:
    def __init__(self, model, max_pos, trace=True, keep_logits=False, eos=None, prefill_group=8, log=print):
        self.model, self.max_pos, self.log = model, max_pos, log
        self.rows = model.mesh.shape[0]
        self.eos = set(eos) if eos is not None else load_eos()
        self.dec = Decoder(model, trace=trace, keep_logits=keep_logits)
        self.keep_logits = keep_logits
        self.prefill_group = prefill_group

    def place(self, n):
        return [(i % self.rows, i // self.rows) for i in range(n)]

    def prefill(self, prompts, S=None):
        """-> first-token ids [rows, U] (long) and logits [rows, U, V]; also fills the KV slots."""
        rows = self.rows
        n = len(prompts)
        U = -(-n // rows)
        assert U <= SLOTS
        maxlen = max(len(p) for p in prompts)
        S = S or max(256, -(-maxlen // 128) * 128)
        assert maxlen <= S and S + 8 <= self.max_pos
        ids = torch.zeros(rows, U, dtype=torch.long)
        lgs = []
        for g0 in range(0, U, self.prefill_group):
            Ug = min(self.prefill_group, U - g0)
            tokens = torch.full((rows, Ug * S), PAD, dtype=torch.int64)
            lens = torch.ones(rows, Ug, dtype=torch.int64)
            for i, p in enumerate(prompts):
                r, s = i % rows, i // rows
                if g0 <= s < g0 + Ug:
                    tokens[r, (s - g0) * S : (s - g0) * S + len(p)] = torch.tensor(p)
                    lens[r, s - g0] = len(p)
            gid, lg = self.model.prefill_first_token(tokens, lens, S, Ug, slot0=g0)
            ids[:, g0 : g0 + Ug] = gid
            if self.keep_logits:
                lgs.append(lg)
        return ids, (torch.cat(lgs, 1) if self.keep_logits else None)

    def generate(self, prompts, max_new, S=None, teacher=None):
        """prompts: list of token id lists. teacher: optional [n, max_new] tokens fed instead of the sampled ones (teacher forcing for oracle comparison).
        Returns dict(tokens [n][..], first_logits, step_logits (keep_logits), times)."""
        rows, n = self.rows, len(prompts)
        place = self.place(n)
        t0 = time.time()
        first, flg = self.prefill(prompts, S)
        t_pre = time.time() - t0
        cur = torch.zeros(rows, SLOTS, dtype=torch.long)
        pos = torch.zeros(rows, SLOTS, dtype=torch.long)
        alive = torch.zeros(rows, SLOTS, dtype=torch.bool)
        out = [[] for _ in range(n)]
        slog = [[] for _ in range(n)]
        for i, (r, s) in enumerate(place):
            tkn = int(first[r, s]) if teacher is None else int(teacher[i][0])
            out[i].append(int(first[r, s]))
            cur[r, s] = tkn
            pos[r, s] = len(prompts[i])
            alive[r, s] = int(first[r, s]) not in self.eos and max_new > 1
        times = []
        for step in range(max_new - 1):
            if not alive.any():
                break
            # clamp: a user whose next position would reach max_pos is finished
            alive &= pos < self.max_pos - 1
            self.dec.set_inputs(cur, pos, ~alive)
            t1 = time.perf_counter()
            res = self.dec.run()
            times.append(time.perf_counter() - t1)
            nxt = res[0]
            for i, (r, s) in enumerate(place):
                if not alive[r, s]:
                    continue
                t = int(nxt[r, s])
                out[i].append(t)
                if self.keep_logits:
                    slog[i].append(res[2][r, s].clone())
                pos[r, s] += 1
                cur[r, s] = t if teacher is None else int(teacher[i][min(step + 1, len(teacher[i]) - 1)])
                if t in self.eos or len(out[i]) >= max_new:
                    alive[r, s] = False
        return dict(tokens=out, first_logits=flg, step_logits=slog, times=times, prefill_s=t_pre, place=place)
