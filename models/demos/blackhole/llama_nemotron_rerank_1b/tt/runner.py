# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Resident traced reranker for serving: traces per (batch, row length) bucket.

``score_ids`` takes any number of tokenized (query, passage) sequences and packs them into
rows: several pairs back to back in one row of S tokens (S a sequence bucket), at most 32
pairs per row. The device mask keeps each pair to its own tokens, so packing changes no
score, and the matmuls process far fewer padding tokens (MIRACL-en: 95 % of the token slots
are real tokens, against 67 % with one pair per row). ``_batch_plan`` splits the rows into
batch buckets at the least estimated cost (a batch that is not full is completed with repeats
of its first row, whose scores are dropped); the runner copies the host inputs into the
bucket's persistent device inputs and replays its trace. Each bucket has two traces: one with the mask, and one without it for
batches where every row is one pair of exactly S tokens (SDPA runs unmasked). A trace is
captured the first time it is used, or up front with ``warmup``.
"""

import time

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model import NemotronReranker
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import SEQ_BUCKETS

BATCH_BUCKETS = (1, 8, 32)
# Fixed cost of one trace launch in rows of 512 tokens: 1 x 512 takes 7.69 ms and 8 x 512 39.8 ms
# (demo/bench.py), i.e. ~4.6 ms per row plus ~3.1 ms per launch.
LAUNCH_COST_ROWS = 0.7


class TracedReranker:
    def __init__(self, model: NemotronReranker, max_seq_len=512, batch_buckets=BATCH_BUCKETS):
        self.model = model
        self.device = model.device
        self.seq_buckets = tuple(b for b in SEQ_BUCKETS if b <= max_seq_len)
        self.max_seq_len = self.seq_buckets[-1]
        self.batch_buckets = tuple(sorted(batch_buckets))
        self._traces = {}  # (B, S, masked) -> (trace_id, device inputs, output)
        # A replayed trace writes into the addresses its intermediates had at capture, which are
        # free afterwards. Every persistent tensor (RoPE tables) is therefore allocated before the
        # first capture, so none can land in a region a trace later overwrites. Per-request inputs
        # are rewritten before each replay and outputs read right after it, so they are safe.
        for S in self.seq_buckets:
            model.rot_mats(S)

    def _batch_plan(self, count):
        """Split ``count`` rows into batch buckets, largest first, at the least estimated cost: rows run
        (repeat rows are wasted work) plus LAUNCH_COST_ROWS per launch. E.g. with buckets 1 / 8 / 32:
        9 rows -> 8 + 1 (not one batch of 32: MIRACL, 25 documents per request, 58 vs 98 ms),
        31 rows -> 32, 3 rows -> 1 + 1 + 1."""
        # best[k] = (cost, buckets) covering at least k rows
        best = [(0.0, ())] + [None] * count
        for k in range(1, count + 1):
            for b in self.batch_buckets:
                cost, plan = best[max(0, k - b)]
                cand = (cost + b + LAUNCH_COST_ROWS, plan + (b,))
                if best[k] is None or cand[0] < best[k][0]:
                    best[k] = cand
        return sorted(best[count][1], reverse=True)

    def _trace(self, B, S, masked):
        key = (B, S, masked)
        if key not in self._traces:
            host, _ = self.model.prepare_host_inputs([[0] * S] * B, seq_len=S)
            dev_in = self.model.to_device(host)
            self.model.rot_mats(S)
            self.model.forward(*dev_in, masked=masked)  # compile
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            out = self.model.forward(*dev_in, masked=masked)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            self._traces[key] = (tid, dev_in, out)
        return self._traces[key]

    def warmup(self, shapes=None):
        """Capture both traces for ``shapes`` [(B, S), ...] (default: every batch x seq bucket)."""
        shapes = shapes or [(b, s) for s in self.seq_buckets for b in self.batch_buckets]
        t0 = time.perf_counter()
        for B, S in shapes:
            for masked in (True, False):
                self._trace(B, S, masked)
        ttnn.synchronize_device(self.device)
        return time.perf_counter() - t0

    @staticmethod
    def _pack(lens, S):
        """First-fit decreasing: pair indices into rows of at most S tokens and 32 pairs."""
        rows = []  # [tokens used, pair indices]
        for i in sorted(range(len(lens)), key=lambda i: -lens[i]):
            for row in rows:
                if row[0] + lens[i] <= S and len(row[1]) < 32:
                    row[0] += lens[i]
                    row[1].append(i)
                    break
            else:
                rows.append([lens[i], [i]])
        return [row[1] for row in rows]

    def _plan(self, lens):
        """(S, rows): the row length with the least estimated cost in token slots (rows run x S, plus
        LAUNCH_COST_ROWS x 512 per launch), the shorter S on a tie. E.g. 3 pairs of 100 tokens: 3 rows of
        128 run as 3 launches (384 + 3 x 358 slots, 3 x 6.3 ms), one row of 512 as one (512 + 358, 7.7 ms)."""
        best = None
        for S in self.seq_buckets:
            if S < max(lens):
                continue
            rows = self._pack(lens, S)
            plan = self._batch_plan(len(rows))
            cost = sum(plan) * S + len(plan) * LAUNCH_COST_ROWS * 512
            if best is None or cost < best[0]:
                best = (cost, S, rows)
        return best[1], best[2]

    def _run(self, batch_rows, S):
        B = len(batch_rows)
        host, lens = self.model.prepare_packed_inputs(batch_rows, S)
        tid, dev_in, out = self._trace(B, S, self.model.needs_mask(lens, S))
        for h, d in zip(host, dev_in):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
        return self.model.scores_from_output(out, lens)

    def score_ids(self, ids_list):
        """Token id lists (each <= max_seq_len) -> list of float logits, in input order."""
        if not ids_list:
            return []
        for i, ids in enumerate(ids_list):
            if not ids:
                raise ValueError(f"sequence {i} is empty")
            if len(ids) > self.max_seq_len:
                raise ValueError(f"sequence {i} has {len(ids)} tokens > max_seq_len {self.max_seq_len}")
        S, rows = self._plan([len(ids) for ids in ids_list])
        scores = [0.0] * len(ids_list)
        pos = 0
        for B in self._batch_plan(len(rows)):
            chunk = rows[pos : pos + B]
            pos += len(chunk)
            batch_rows = [[ids_list[i] for i in row] for row in chunk]
            batch_rows += [batch_rows[0]] * (B - len(batch_rows))
            # scores come back in row-major pair order; the repeated rows' scores are past the end
            for i, s in zip([i for row in chunk for i in row], self._run(batch_rows, S)):
                scores[i] = s
        return scores

    def close(self):
        for tid, dev_in, out in self._traces.values():
            ttnn.release_trace(self.device, tid)
        self._traces.clear()
