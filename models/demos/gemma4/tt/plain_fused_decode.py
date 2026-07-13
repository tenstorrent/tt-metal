# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fully fused plain greedy decode for Gemma4 (batch=1).

Mirrors the speculative fused iteration (``spec_decode.generate_fused``) but
without a drafter: one CCL-bearing metal trace per token runs

    embed → 60 layers → norm → lm_head (full-vocab AG) → argmax

and feeds the next token back on device. Host work per token is a single
uint32 readback + position+1 copy — not a 262k-vocab D2H + host argmax.

Enable with ``GEMMA4_FUSED_GREEDY=1``. Prefill must be eager (no prefill CCL
trace interleaved with this fused decode trace) — same contract as spec.
"""

from __future__ import annotations

import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.device_greedy_argmax import argmax_last, id_to_host, probe_device_argmax


def fused_greedy_enabled() -> bool:
    return os.environ.get("GEMMA4_FUSED_GREEDY", "0").lower() in ("1", "true", "yes")


class PlainFusedGreedyDecoder:
    """Batch=1 greedy decode via a single fused CCL trace per token."""

    def __init__(
        self,
        target_model,
        mesh_device,
        tt_kv_cache,
        page_table_torch,
        stop_tokens=None,
    ):
        self.target = target_model
        self.mesh_device = mesh_device
        # Peel ``from_pretrained`` model wrapper ``[[[k,v],...]]`` → ``[[k,v],...]``.
        if (
            isinstance(tt_kv_cache, (list, tuple))
            and tt_kv_cache
            and isinstance(tt_kv_cache[0], (list, tuple))
            and tt_kv_cache[0]
            and isinstance(tt_kv_cache[0][0], (list, tuple))
        ):
            tt_kv_cache = tt_kv_cache[0]
        self.tt_kv_cache = tt_kv_cache
        self.page_table_torch = page_table_torch
        self.stop_tokens = set(stop_tokens or [])
        if page_table_torch is None:
            raise ValueError("Plain fused greedy requires paged attention (page_table_torch is None).")
        if getattr(target_model, "hidden_size_per_layer_input", 0):
            raise ValueError("Plain fused greedy does not support E2B/E4B PLI models yet.")

        self._mapper = target_model._replicate_to_mesh_mapper()
        self._tp = target_model.mesh_config.tp if target_model.mesh_config else 1
        probe_device_argmax(mesh_device, self._mapper)

        _trace_env = os.environ.get("GEMMA4_FUSED_GREEDY_TRACE")
        self._use_trace = True if _trace_env is None else (_trace_env == "1")
        self._fused_trace = None
        self._last_setup_s = 0.0
        self._last_replay_s = 0.0

    def _sanitize_token_id(self, tok) -> int:
        tok = int(tok)
        vs = self.target.vocab_size
        if tok < 0 or tok >= vs:
            raise ValueError(f"invalid token id {tok} (vocab_size={vs})")
        return tok

    def _tokens_tensor(self, tokens):
        safe = [self._sanitize_token_id(t) for t in tokens]
        t = torch.tensor(safe, dtype=torch.int32).reshape(1, len(safe))
        return ttnn.from_torch(
            t,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=self._mapper,
        )

    def _pos_tensors(self, positions):
        batch = len(positions)
        pu = torch.zeros((1, 32), dtype=torch.int32)
        pu[0, :batch] = torch.tensor(positions, dtype=torch.int32)
        pos_uint32 = ttnn.from_torch(
            pu,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=self._mapper,
        )
        pos_int32 = ttnn.from_torch(
            torch.tensor(positions, dtype=torch.int32),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=self._mapper,
        )
        return pos_uint32, pos_int32

    def _page_table(self, batch: int = 1):
        user_row = self.page_table_torch[0:1] if self.page_table_torch.dim() > 1 else self.page_table_torch.unsqueeze(0)
        pt = user_row.repeat(batch, 1).to(torch.int32)
        return ttnn.from_torch(
            pt,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=self._mapper,
        )

    def _host_tokens(self, tokens):
        safe = [self._sanitize_token_id(t) for t in tokens]
        t = torch.tensor(safe, dtype=torch.int32).reshape(1, len(safe))
        return ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper)

    def _host_pos(self, positions):
        batch = len(positions)
        pu = torch.zeros((1, 32), dtype=torch.int32)
        pu[0, :batch] = torch.tensor(positions, dtype=torch.int32)
        pos_uint32 = ttnn.from_torch(pu, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=self._mapper)
        pos_int32 = ttnn.from_torch(
            torch.tensor(positions, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=self._mapper,
        )
        return pos_uint32, pos_int32

    def _argmax_last(self, logits, rows: int = 1):
        return argmax_last(
            logits,
            rows,
            mesh_device=self.mesh_device,
            mapper=self._mapper,
            tp=self._tp,
            vocab_size=self.target.vocab_size,
        )

    def _id_to_host(self, id_tt) -> int:
        return id_to_host(id_tt, tp=self._tp, sanitize=self._sanitize_token_id)

    def _begin_generation_guard(self, repetition_max_streak: int) -> None:
        from models.demos.gemma4.demo.sampling_utils import RepetitionStreakGuard

        self._streak_guard = RepetitionStreakGuard(repetition_max_streak)

    def _on_committed_token(self, tok: int, out: list[int], token_callback) -> bool:
        """Handle a newly sampled token. Return True when generation should stop.

        Stop tokens are NOT appended (matches the host decode loop in text_demo_v2).
        """
        if tok in self.stop_tokens:
            return True
        out.append(tok)
        if token_callback is not None:
            token_callback(tok)
        guard = getattr(self, "_streak_guard", None)
        if guard is not None and guard.observe(tok):
            return True
        return False

    def _plain_body(self, tr):
        """One greedy decode step over persistent buffers (capture + compile).

        Uses ``ttnn_verify_forward`` (batch=1) so lm_head always all-gathers the
        full vocab before argmax — same trick as the spec seed path.
        """
        logits, hidden = self.target.ttnn_verify_forward(
            x=tr["tok"],
            current_pos=tr["pos_u"],
            current_pos_cache=tr["pos_i"],
            page_table=tr["pt"],
            kv_cache=self.tt_kv_cache,
        )
        idx = self._argmax_last(logits, rows=1)
        logits.deallocate(True)
        if hidden is not None and hasattr(hidden, "deallocate"):
            hidden.deallocate(True)
        return idx

    def _bind_inputs(self, tr, cur_token: int, cur_pos: int) -> None:
        h_tok = self._host_tokens([cur_token])
        ttnn.copy_host_to_device_tensor(h_tok, tr["tok"])
        h_tok.deallocate(True)
        h_pu, h_pi = self._host_pos([cur_pos])
        ttnn.copy_host_to_device_tensor(h_pu, tr["pos_u"])
        ttnn.copy_host_to_device_tensor(h_pi, tr["pos_i"])
        h_pu.deallocate(True)
        h_pi.deallocate(True)

    def _capture_fused_trace(self, token: int, pos: int) -> None:
        logger.info("[plain-fused] capture: compile run")
        pu, pi = self._pos_tensors([pos])
        tr = {
            "tok": self._tokens_tensor([token]),
            "pos_u": pu,
            "pos_i": pi,
            "pt": self._page_table(1),
        }

        idx = self._plain_body(tr)
        ttnn.synchronize_device(self.mesh_device)
        idx.deallocate(True)

        logger.info("[plain-fused] capture: begin_trace_capture")
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        idx = self._plain_body(tr)
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        logger.info("[plain-fused] capture: DONE")
        tr["id"] = tid
        tr["idx"] = idx
        self._fused_trace = tr

    def release_fused_trace(self) -> None:
        """Release the fused decode trace + persistent buffers.

        Same CCL-trace teardown contract as ``SpeculativeDecoder.release_fused_trace``:
        must run before a prefill CCL trace or before recapture.
        """
        tr = self._fused_trace
        if tr is None:
            return
        tid = tr.get("id")
        if tid is not None:
            try:
                ttnn.release_trace(self.mesh_device, tid)
            except Exception:
                pass
        for key in ("tok", "pos_u", "pos_i", "pt", "idx"):
            t = tr.get(key)
            if t is not None and hasattr(t, "deallocate"):
                try:
                    t.deallocate(True)
                except Exception:
                    pass
        self._fused_trace = None

    def _step_eager(self, token: int, pos: int) -> int:
        tok_tt = self._tokens_tensor([token])
        pu, pi = self._pos_tensors([pos])
        pt = self._page_table(1)
        logits, hidden = self.target.ttnn_verify_forward(
            x=tok_tt,
            current_pos=pu,
            current_pos_cache=pi,
            page_table=pt,
            kv_cache=self.tt_kv_cache,
        )
        idx = self._argmax_last(logits, rows=1)
        logits.deallocate(True)
        if hidden is not None and hasattr(hidden, "deallocate"):
            hidden.deallocate(True)
        next_tok = self._id_to_host(idx)
        idx.deallocate(True)
        tok_tt.deallocate(True)
        pu.deallocate(True)
        pi.deallocate(True)
        pt.deallocate(True)
        return next_tok

    def generate(
        self,
        anchor_token: int,
        anchor_pos: int,
        max_new_tokens: int,
        token_callback=None,
        repetition_max_streak: int = 0,
    ) -> list[int]:
        """Generate greedy tokens starting AFTER the prefill-sampled first token.

        ``anchor_token`` / ``anchor_pos`` are the last committed token (typically
        the prefill-sampled next-token at ``prompt_len - 1`` position... wait:
        after prefill the demo appends the first generated token at position
        ``prompt_len``. Callers should pass that first generated token and its
        absolute position so this loop continues from there.
        """
        max_seq_len = getattr(self.target, "max_seq_len", None)
        if max_seq_len is not None:
            safe_new = max(0, max_seq_len - 1 - int(anchor_pos))
            if max_new_tokens > safe_new:
                logger.warning(
                    f"Clamping fused-greedy max_new_tokens {max_new_tokens} -> {safe_new} "
                    f"(max_seq_len={max_seq_len}, anchor_pos={anchor_pos})"
                )
                max_new_tokens = safe_new

        self._begin_generation_guard(repetition_max_streak)
        if self._use_trace:
            return self._generate_traced(anchor_token, anchor_pos, max_new_tokens, token_callback=token_callback)
        return self._generate_eager(anchor_token, anchor_pos, max_new_tokens, token_callback=token_callback)

    def _generate_eager(self, anchor_token, anchor_pos, max_new_tokens, token_callback=None) -> list[int]:
        out: list[int] = []
        cur_token, cur_pos = int(anchor_token), int(anchor_pos)
        t0 = time.perf_counter()
        while len(out) < max_new_tokens:
            next_tok = self._step_eager(cur_token, cur_pos)
            cur_pos += 1
            cur_token = next_tok
            if self._on_committed_token(next_tok, out, token_callback):
                break
        self._last_setup_s = 0.0
        self._last_replay_s = time.perf_counter() - t0
        return out

    def _generate_traced(self, anchor_token, anchor_pos, max_new_tokens, token_callback=None) -> list[int]:
        setup_t0 = time.perf_counter()
        reuse = self._fused_trace is not None
        if not reuse:
            self._capture_fused_trace(int(anchor_token), int(anchor_pos))
        tr = self._fused_trace
        assert tr is not None
        self._last_setup_s = time.perf_counter() - setup_t0

        out: list[int] = []
        cur_token, cur_pos = int(anchor_token), int(anchor_pos)
        first = not reuse
        replay_t0 = time.perf_counter()
        while len(out) < max_new_tokens:
            if not first:
                self._bind_inputs(tr, cur_token, cur_pos)
            first = False

            ttnn.execute_trace(self.mesh_device, tr["id"], cq_id=0, blocking=False)
            # Drain before host readback / next input copy (same as spec fused).
            ttnn.synchronize_device(self.mesh_device)

            next_tok = self._id_to_host(tr["idx"])
            cur_pos += 1
            cur_token = next_tok
            if self._on_committed_token(next_tok, out, token_callback):
                break

        self._last_replay_s = time.perf_counter() - replay_t0
        return out
