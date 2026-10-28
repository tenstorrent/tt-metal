# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Resident traced reranker for serving: one trace per (batch, seq_len) bucket.

``score_ids`` takes any number of tokenized (query, passage) sequences, groups them by
sequence-length bucket, packs each group into the largest batch bucket it fills (padding
the last batch with repeats whose scores are dropped), copies the host inputs into the
bucket's persistent device inputs and replays its trace. Each bucket has two traces: one
with the padding mask, and one without it for batches where every pair fills the bucket
(SDPA runs unmasked). A trace is captured the first time it is used, or up front with
``warmup``.
"""

import time

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model import NemotronReranker
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import SEQ_BUCKETS

BATCH_BUCKETS = (1, 8, 32)


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

    def _bucket_seq(self, n):
        return next(b for b in self.seq_buckets if b >= n)

    def _batch_plan(self, count):
        """Split ``count`` sequences into batch-bucket chunks: full largest buckets, then the
        smallest bucket that holds the remainder."""
        plan, left = [], count
        big = self.batch_buckets[-1]
        while left >= big:
            plan.append(big)
            left -= big
        if left:
            plan.append(next(b for b in self.batch_buckets if b >= left))
        return plan

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

    def _run(self, batch_ids, S):
        B = len(batch_ids)
        host, lens = self.model.prepare_host_inputs(batch_ids, seq_len=S)
        tid, dev_in, out = self._trace(B, S, self.model.needs_mask(lens, S))
        for h, d in zip(host, dev_in):
            ttnn.copy_host_to_device_tensor(h, d)
        ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
        return self.model.scores_from_output(out, lens)

    def score_ids(self, ids_list):
        """Token id lists (each <= max_seq_len) -> list of float logits, in input order."""
        scores = [0.0] * len(ids_list)
        groups = {}
        for i, ids in enumerate(ids_list):
            if not ids:
                raise ValueError(f"sequence {i} is empty")
            if len(ids) > self.max_seq_len:
                raise ValueError(f"sequence {i} has {len(ids)} tokens > max_seq_len {self.max_seq_len}")
            groups.setdefault(self._bucket_seq(len(ids)), []).append(i)
        for S, idx in groups.items():
            pos = 0
            for B in self._batch_plan(len(idx)):
                chunk = idx[pos : pos + B]
                pos += len(chunk)
                batch = [ids_list[i] for i in chunk]
                batch += [batch[0]] * (B - len(batch))
                for i, s in zip(chunk, self._run(batch, S)):
                    scores[i] = s
        return scores

    def close(self):
        for tid, dev_in, out in self._traces.values():
            ttnn.release_trace(self.device, tid)
        self._traces.clear()
