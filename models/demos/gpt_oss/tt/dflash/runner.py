# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Greedy batch-1 DFlash with the gpt-oss target (tt_transformers Generator) and the z-lab drafter.

Per round, all on device: the drafter's draft pass (dflash/fused_draft.py) writes [last token, 7 drafts] into the verify
pass's token input; the 8-row verify pass (dflash/verify.py) computes the target's argmax for every row, the acceptance,
and the next round's drafter / verify / context inputs; the context pass adds the accepted rows' target features to the
drafter's K / V cache. The host reads one [block, argmaxes] result per round. All three passes are traced after one eager
compile pass (capture_traces).
"""

import time

import torch
from loguru import logger

import ttnn

from .capture import TargetCapture

PROMPT_CHUNK = 256  # prompt-context rows per drafter call (the verify pass's persistent L1 buffers limit the matmul)


class DFlashRunner:
    def __init__(self, generator, model, page_table, kv_cache, draft, eos_ids=(200002, 200012)):
        from .fused_draft import FusedDraft
        from .verify import FusedVerify

        self.generator, self.model, self.page_table, self.kv_cache = generator, model, page_table, kv_cache
        self.draft = draft
        self.capture = TargetCapture(model, draft.target_layer_ids)
        self.eos_ids = set(eos_ids)
        self.V = model.vocab_size
        kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
        self.verify = FusedVerify(model, page_table, kv, rows=draft.block, capture_layers=draft.target_layer_ids)
        self.fused = FusedDraft(draft, self.verify)
        self.verify.set_context_buffer(self.fused.ctx_buf, self.fused.ctx_split)
        # the verify pass's argmax merge computes the acceptance and writes the next round's inputs
        self.verify.set_accept(
            draft.mask_token_id,
            [self.verify.pos, draft.t_toks, draft.t_pos, draft.t_rot, draft.t_bound, draft.t_cpos, draft.t_crot],
        )
        self.verify_trace = self.draft_trace = self.ctx_trace = None

    def compile(self):
        """One eager pass of each traced program (compiles them and allocates their cached buffers and CCL semaphores
        before any capture)."""
        self.fused.forward()
        self.verify.forward()
        self.fused.context()

    def capture_traces(self):
        """Capture the draft, verify and context traces (after one eager generate). A replayed trace writes to its
        intermediates' addresses, which are free again once the capture ends: a buffer allocated after the capture and
        alive at a replay can be overwritten, so every program is compiled first, and prefill re-captures the traces
        when a prompt compiled new programs (see prefill)."""
        mesh = self.model.mesh_device
        self.compile()
        traces = []
        for fn in (self.fused.forward, self.verify.forward, self.fused.context):
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            traces.append(tid)
        self.draft_trace, self.verify_trace, self.ctx_trace = traces

    def release_traces(self):
        mesh = self.model.mesh_device
        for tid in (self.draft_trace, self.verify_trace, self.ctx_trace):
            if tid is not None:
                ttnn.release_trace(mesh, tid)
        self.verify_trace = self.draft_trace = self.ctx_trace = None

    def set_draft_window(self, w):
        """The drafter's attention window (0: none, the checkpoint's setting); call with no traces captured."""
        self.draft.window = w or None
        self.fused.set_window(w)

    def prefill(self, prompt_ids):
        """The target's prefill + the drafter's prompt context; returns the target's next token. A prompt shape seen
        for the first time after capture_traces compiles new programs, whose persistent buffers (program-cache buffers,
        the all-reduce's global semaphores) can sit where the traces' intermediates were: a replay would overwrite
        them. The traces are then re-captured, now laid out around those buffers, and the prefill is repeated (the
        capture's eager passes write K / V rows)."""
        mesh = self.model.mesh_device
        n0 = mesh.num_program_cache_entries()
        first = self._prefill(prompt_ids)
        if self.verify_trace is not None and mesh.num_program_cache_entries() != n0:
            logger.warning(
                f"DFlash: prompt length {len(prompt_ids)} compiled new prefill programs after the trace capture; "
                "re-capturing the traces"
            )
            self.release_traces()
            self.capture_traces()
            first = self._prefill(prompt_ids)
        return first

    def _prefill(self, prompt_ids):
        L = len(prompt_ids)
        logits = self.generator.prefill_forward_text(
            torch.tensor([prompt_ids]),
            page_table=self.page_table,
            kv_cache=self.kv_cache,
            prompt_lens=[L],
            enable_trace=False,
            warmup_prefill=False,
        )
        self.draft.add_prompt_context(self.capture.prefill_features(L), chunk=PROMPT_CHUNK)
        return int(logits.reshape(-1)[: self.V].float().argmax())

    def generate(self, prompt_ids, max_new_tokens, first=None):
        """Returns (new tokens, per-round accepted counts, timings). first: the prefill's next token (prefill done).
        Eager until capture_traces, traced after."""
        L = len(prompt_ids)
        t0 = time.perf_counter()
        if first is None:
            first = self.prefill(prompt_ids)
        ids = list(prompt_ids) + [first]
        t_prefill = time.perf_counter() - t0
        if self.verify_trace is not None:
            accept, rounds = self._generate_traced(ids, L, max_new_tokens)
            return ids[L:], accept, dict(prefill=t_prefill, rounds=rounds)
        accept, rounds = [], []
        B = self.draft.block
        mesh = self.model.mesh_device
        while len(ids) - L < max_new_tokens and not (set(ids[L:]) & self.eos_ids):
            t = time.perf_counter()
            start = len(ids) - 1
            if not accept:  # later rounds: the previous verify pass wrote the inputs on device
                self.draft.set_draft_inputs(ids[-1], start)
                self.verify.set_positions(start)
            if self.verify_trace is None:
                self.fused.forward()
                self.verify.forward()
            else:
                ttnn.execute_trace(mesh, self.draft_trace, cq_id=0, blocking=False)
                ttnn.execute_trace(mesh, self.verify_trace, cq_id=0, blocking=False)
            t1 = time.perf_counter()
            block, post = self.verify.read_result()
            d = block[1:]
            t2 = time.perf_counter()
            k = 0
            while k < B - 1 and d[k] == post[k]:
                k += 1
            ids += d[:k] + [post[k]]
            if self.ctx_trace is None:  # the context positions are on device already
                self.fused.context()
            else:
                ttnn.execute_trace(mesh, self.ctx_trace, cq_id=0, blocking=False)
            t3 = time.perf_counter()
            accept.append(k + 1)
            rounds.append((t1 - t, t2 - t1, t3 - t2))
        return ids[L:], accept, dict(prefill=t_prefill, rounds=rounds)

    def _enqueue_round(self):
        """Queue one traced round: draft, verify, a non-blocking read of the result, context. Returns (host result,
        event recorded after the read)."""
        mesh = self.model.mesh_device
        ttnn.execute_trace(mesh, self.draft_trace, cq_id=0, blocking=False)
        ttnn.execute_trace(mesh, self.verify_trace, cq_id=0, blocking=False)
        host = self.verify.result.cpu(blocking=False)
        event = ttnn.record_event(mesh, 0)
        ttnn.execute_trace(mesh, self.ctx_trace, cq_id=0, blocking=False)
        return host, event

    def _generate_traced(self, ids, L, max_new_tokens):
        """Traced rounds, extends ids in place; returns (per-round accepted counts, per-round wall times). Every round's
        inputs are written on device by the previous verify pass, so the next round is queued before this round's
        result is read whenever this round cannot end the generation (it adds at most block tokens): the host's read
        round trip overlaps the next round instead of idling the device between rounds."""
        B = self.draft.block
        accept, rounds = [], []
        if len(ids) - L >= max_new_tokens or set(ids[L:]) & self.eos_ids:
            return accept, rounds
        start = len(ids) - 1
        self.draft.set_draft_inputs(ids[-1], start)
        self.verify.set_positions(start)
        t = time.perf_counter()
        pending = [self._enqueue_round()]
        while pending:
            if len(pending) == 1 and len(ids) - L + B < max_new_tokens:
                pending.append(self._enqueue_round())
            host, event = pending.pop(0)
            ttnn.event_synchronize(event)
            r = ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1).long().tolist()
            d, post = r[1:B], r[B : 2 * B]  # [block tokens, target argmaxes]
            k = 0
            while k < B - 1 and d[k] == post[k]:
                k += 1
            ids += d[:k] + [post[k]]
            accept.append(k + 1)
            now = time.perf_counter()
            rounds.append((now - t, 0.0, 0.0))
            t = now
            if len(ids) - L >= max_new_tokens or set(ids[L:]) & self.eos_ids:
                break
            if not pending:
                pending.append(self._enqueue_round())
        if pending:  # a queued round past the end (end of sequence): let it finish before returning
            ttnn.synchronize_device(self.model.mesh_device)
        return accept, rounds
