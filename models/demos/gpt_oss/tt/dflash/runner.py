# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Greedy batch-1 DFlash with the gpt-oss target (tt_transformers Generator) and the z-lab drafter.

Per round, all on device: the drafter's draft pass (dflash/fused_draft.py) writes [last token, 7 drafts] into the verify
pass's token input; the 8-row verify pass (dflash/verify.py) computes the target's argmax for every row, the acceptance,
and the next round's drafter / verify / context inputs; the context pass adds the accepted rows' target features to the
drafter's K / V cache. The host reads one [block, argmaxes] result per round. The three passes are captured as one
trace, replayed once per round, after one eager compile pass (capture_traces).
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
        self.round_trace = None

    def compile(self):
        """One eager pass of each traced program (compiles them and allocates their cached buffers and CCL semaphores
        before any capture)."""
        self.fused.forward()
        self.verify.forward()
        self.fused.context()

    def capture_traces(self):
        """Capture the round trace: the draft, verify and context passes in one trace (one trace instead of one per
        pass: each pass's first program then dispatches while the previous pass runs). A replayed trace writes to its
        intermediates' addresses, which are free again once the capture ends: a buffer allocated after the capture and
        alive at a replay can be overwritten, so every program is compiled first, and prefill re-captures the trace
        when a prompt compiled new programs (see prefill). A trace captured before is released first."""
        mesh = self.model.mesh_device
        self.release_traces()
        self.compile()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.fused.forward()
        self.verify.forward()
        self.fused.context()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        self.round_trace = tid

    def release_traces(self):
        """Release the round trace (after the device has finished every queued replay)."""
        if self.round_trace is not None:
            ttnn.synchronize_device(self.model.mesh_device)
            ttnn.release_trace(self.model.mesh_device, self.round_trace)
        self.round_trace = None

    def set_draft_window(self, w):
        """The drafter's attention window (0: none, the checkpoint's setting); call with no trace captured (a captured
        trace keeps the window it was captured with)."""
        assert self.round_trace is None, "set_draft_window: release_traces() first"
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
        if self.round_trace is not None and mesh.num_program_cache_entries() != n0:
            logger.warning(
                f"DFlash: prompt length {len(prompt_ids)} compiled new prefill programs after the trace capture; "
                "re-capturing the round trace"
            )
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

    def _bounded(self, toks, max_new_tokens):
        """The returned tokens: at most max_new_tokens, ending at the first EOS (a round appends its whole accepted
        block, which can pass either limit)."""
        toks = toks[: max(max_new_tokens, 0)]
        end = next((i + 1 for i, t in enumerate(toks) if t in self.eos_ids), len(toks))
        return toks[:end]

    def generate(self, prompt_ids, max_new_tokens, first=None):
        """Returns (new tokens, per-round accepted counts, timings): at most max_new_tokens tokens (the first one is the
        prefill's), up to and including the first EOS. first: the prefill's next token (prefill done). Eager until
        capture_traces, traced after."""
        L = len(prompt_ids)
        t0 = time.perf_counter()
        if first is None:
            first = self.prefill(prompt_ids)
        ids = list(prompt_ids) + [first]
        t_prefill = time.perf_counter() - t0
        if self.round_trace is not None:
            accept, rounds = self._generate_traced(ids, L, max_new_tokens)
            return self._bounded(ids[L:], max_new_tokens), accept, dict(prefill=t_prefill, rounds=rounds)
        accept, rounds = [], []
        B = self.draft.block
        while len(ids) - L < max_new_tokens and not (set(ids[L:]) & self.eos_ids):
            t = time.perf_counter()
            start = len(ids) - 1
            if not accept:  # later rounds: the previous verify pass wrote the inputs on device
                self.draft.set_draft_inputs(ids[-1], start)
                self.verify.set_positions(start)
            self.fused.forward()
            self.verify.forward()
            t1 = time.perf_counter()
            block, post = self.verify.read_result()
            d = block[1:]
            t2 = time.perf_counter()
            k = 0
            while k < B - 1 and d[k] == post[k]:
                k += 1
            ids += d[:k] + [post[k]]
            self.fused.context()  # the context positions are on device already
            t3 = time.perf_counter()
            accept.append(k + 1)
            rounds.append((t1 - t, t2 - t1, t3 - t2))
        return self._bounded(ids[L:], max_new_tokens), accept, dict(prefill=t_prefill, rounds=rounds)

    def _enqueue_round(self):
        """Queue one traced round (draft, verify, context) and a non-blocking read of its result. Returns (host result,
        event recorded after the read)."""
        mesh = self.model.mesh_device
        ttnn.execute_trace(mesh, self.round_trace, cq_id=0, blocking=False)
        host = self.verify.result.cpu(blocking=False)
        return host, ttnn.record_event(mesh, 0)

    def _generate_traced(self, ids, L, max_new_tokens):
        """Traced rounds, extends ids in place; returns (per-round accepted counts, per-round wall times). Every round's
        inputs are written on device by the previous verify pass, so the next round is queued before this round's
        result is read whenever this round cannot reach max_new_tokens (it adds at most block tokens): the host's read
        round trip overlaps the next round instead of idling the device between rounds. If an EOS ends the generation
        instead, the round queued behind it is waited for and its result dropped."""
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
            rounds.append((now - t, 0.0, 0.0))  # (the eager path's (passes, read, context) split; traced: all wall)
            t = now
            if len(ids) - L >= max_new_tokens or set(ids[L:]) & self.eos_ids:
                break
            if not pending:
                pending.append(self._enqueue_round())
        if pending:  # a queued round past the end (end of sequence): let it finish before returning
            ttnn.synchronize_device(self.model.mesh_device)
        return accept, rounds
