# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import time

import torch

import ttnn
from models.demos.llama31_8b_p100_v3.tt import ops_spec_rows
from models.ttt_compat.tt.generator import SamplingParams

TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_decode_table.json")
with open(TABLE_PATH) as _f:
    TABLE = json.load(_f)


def draft_config(kind):
    return TABLE["drafts"][kind]


def sampling_params():
    return SamplingParams(**TABLE["sampling"])


def block_room(position):
    size = TABLE["verify_block"]
    for block in TABLE["verify_blocks"]:
        if position < block["below"]:
            size = block["size"]
            break
    return size - 1 - position % size


class Context:
    def __init__(self, tokens):
        cfg = TABLE["lookup"]
        self.sizes = range(cfg["max_ngram"], cfg["min_ngram"] - 1, -1)
        self.tokens = []
        self.index = {n: {} for n in self.sizes}
        for t in tokens:
            self.append(t)

    def append(self, token):
        t = len(self.tokens)
        for n in self.sizes:
            if t >= n:
                self.index[n][tuple(self.tokens[t - n : t])] = t
        self.tokens.append(token)

    def lookup(self, k):
        for n in self.sizes:
            if len(self.tokens) < n:
                continue
            start = self.index[n].get(tuple(self.tokens[-n:]))
            if start is not None:
                return self.tokens[start : start + k]
        return []


class Draft:
    def start(self, prompt_tokens, prompt_len, first_token):
        raise NotImplementedError

    def propose(self, token, position, k):
        raise NotImplementedError

    def commit(self, position, accepted, verify_tokens):
        raise NotImplementedError


class SpecDecodeGenerator:
    def __init__(self, generator, model_args, mesh_device, kv_cache, page_table, draft, k):
        self.gen = generator
        self.args = model_args
        self.rows = model_args.max_batch_size
        assert self.rows in TABLE["row_counts"] and k + 1 <= self.rows
        self.offsets = TABLE["row_offsets"][: self.rows]
        self.k = k
        self.kv = kv_cache
        self.page_table = page_table
        self.draft = draft
        self.sp = sampling_params()
        self.fresh = True
        self.warm = False
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=tuple(TABLE["verify_sdpa"]["grid"]),
            exp_approx_mode=TABLE["verify_sdpa"]["exp_approx"],
            q_chunk_size=getattr(model_args, "sdpa_decode_q_chunk_size", 0),
            k_chunk_size=getattr(model_args, "sdpa_decode_k_chunk_size", 0),
        )
        self.rs = ops_spec_rows.RowState(mesh_device, self.rows, page_table, kv_cache, pc) if self.rows > 1 else None

    def _tok(self, out):
        t = out[0] if isinstance(out, tuple) else out
        return t.reshape(-1).to(torch.int64)

    def prefill(self, tokens, prompt_len):
        n = 1 if self.warm else self.rows
        self.warm = True
        out = self.gen.prefill_forward_text(
            tokens[0:1].repeat(n, 1),
            page_table=self.page_table[0:n],
            kv_cache=[self.kv],
            prompt_lens=[prompt_len] * n,
            sampling_params=self.sp,
            warmup_prefill=True,
            enable_trace=True,
        )
        self.fresh = True
        if isinstance(out, tuple):
            return int(self._tok(out)[0])
        return int(torch.argmax(out, dim=-1).reshape(-1)[0])

    def verify(self, tokens, positions):
        if self.rs is not None:
            self.rs.set(positions)
        out = self.gen.decode_forward(
            torch.tensor(tokens, dtype=torch.int64).reshape(self.rows, 1),
            torch.tensor(positions, dtype=torch.int64),
            page_table=self.page_table,
            kv_cache=[self.kv],
            enable_trace=True,
            sampling_params=self.sp,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=self.fresh,
        )
        self.fresh = False
        return self._tok(out)[: self.rows].tolist()

    def generate(self, prompt_tokens, prompt_len, max_new_tokens, stop_tokens=()):
        tp = time.perf_counter()
        c = self.prefill(prompt_tokens, prompt_len)
        t0 = time.perf_counter()
        prompt = prompt_tokens[0, :prompt_len].tolist()
        context = Context(prompt + [c])
        self.draft.start(prompt, prompt_len, c)
        out, q, rounds = [c], prompt_len, 0
        lookup_k = min(TABLE["lookup"]["k"], self.rows - 1)
        by_source = {"lookup": [0, 0], "draft": [0, 0]}
        by_position = {}
        while len(out) - 1 < max_new_tokens and out[-1] not in stop_tokens:
            room = block_room(q)
            found = context.lookup(min(lookup_k, room)) if self.k else []
            if found:
                self.draft.propose(c, q, 0)
                drafts, source = found, "lookup"
            else:
                drafts, source = list(self.draft.propose(c, q, min(self.k, room)))[: min(self.k, room)], "draft"
            m = len(drafts)
            vt = [c] + drafts + [0] * (self.rows - m - 1)
            vp = [q + self.offsets[i] for i in range(m + 1)] + [-1] * (self.rows - m - 1)
            g = self.verify(vt, vp)
            n = 0
            while n < m and drafts[n] == g[n]:
                n += 1
            self.draft.commit(q, n, g)
            by_source[source][0] += 1
            by_source[source][1] += n
            bucket = by_position.setdefault(q // 512 * 512, [0, 0])
            bucket[0] += 1
            bucket[1] += n
            for x in drafts[:n] + [g[n]]:
                out.append(x)
                context.append(x)
                if x in stop_tokens or len(out) - 1 >= max_new_tokens:
                    break
            q += n + 1
            c = g[n]
            rounds += 1
        return out, {
            "rounds": rounds,
            "prefill_s": t0 - tp,
            "decode_s": time.perf_counter() - t0,
            "by_source": by_source,
            "by_position": by_position,
        }
