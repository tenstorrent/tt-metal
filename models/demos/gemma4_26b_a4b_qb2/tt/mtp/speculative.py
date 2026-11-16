# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy MTP speculative decoding, batch 1 (WS6).

Round r, anchor token ``a`` at position ``p`` (committed, its K/V not yet written) and seed hidden ``h`` = the
target's post-final-norm hidden at ``p - 1`` (the row that produced ``a``), as in transformers'
``SinglePositionMultiTokenCandidateGenerator``:

1. drafter: K steps at the fixed position ``p``: ``(a, h) -> d1``, ``(d1, h1) -> d2``, ... (argmax);
2. verify: one batched target decode of K+1 rows ``[a, d1..dK]`` at positions ``p..p+K`` on the user's page-table
   row (K/V written one row per program, row order), logits and post-norm hidden of every row;
3. accept the longest prefix with ``d(i+1) == argmax(row i)``; commit ``d1..dn`` and ``g = argmax(row n)``;
   next anchor ``g`` at ``p + n + 1``, next seed = hidden row ``n``.

Rejected rows leave K/V at positions beyond the committed length; the next verify rewrites them before any SDPA
reads them (SDPA of row i reads positions <= p'+i, all written by this verify or committed earlier).
"""

import os
import time
from types import SimpleNamespace

import torch

import ttnn
from models.demos.gemma4_26b_a4b_qb2.tt.mtp.argmax import DraftArgmax
from models.demos.gemma4_26b_a4b_qb2.tt.mtp.drafter import load_drafter, shared_layers
from models.demos.gemma4_26b_a4b_qb2.tt.mtp.lean_drafter import LeanDrafter

EOS = (1, 106)
# Verify K/V writes: "op" = gemma4 PagedUpdateRows (one program per tensor), "loop" = one paged_update_cache per row.
KV_WRITES = os.environ.get("GEMMA4_26B_MTP_KV_WRITES", "op")
# Draft-token argmax: "fast" = tt/mtp/argmax.py (~0.28 ms), "sampler" = the plain decode's sampler (~0.52 ms).
DRAFT_ARGMAX = os.environ.get("GEMMA4_26B_MTP_DRAFT_ARGMAX", "fast")


class SpeculativeDecoder:
    def __init__(self, generator, *, draft_len=5, drafter=None, drafter_cache_pos="hf", lean=True):
        self.gen = generator
        self.model = generator.model
        self.mesh = generator.mesh
        self.K = draft_len
        if drafter is None:
            drafter = LeanDrafter(self.model) if lean else load_drafter(self.model)
        self.drafter = drafter
        self.lean = isinstance(drafter, LeanDrafter)
        self.draft_argmax = (
            DraftArgmax(self.model, self.model.config.vocab_size // 4) if DRAFT_ARGMAX == "fast" else None
        )
        layers = shared_layers(self.model)
        self.layer_by_type = layers
        # "hf": the drafter attends positions <= p when slot p holds K/V (a rejected draft's), else <= p - 1.
        # "anchor": always <= p. "committed": always <= p - 1.
        self.drafter_cache_pos = drafter_cache_pos
        self.stats = {}

    # ---- small device inputs -------------------------------------------------------------------------------
    def _replicated(self, value, dtype):
        return ttnn.from_torch(
            value,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _pos_rows(self, positions):
        rows = torch.full((1, 32), -1, dtype=torch.int32)
        rows[0, : len(positions)] = torch.tensor(positions, dtype=torch.int32)
        return self._replicated(rows, ttnn.uint32), self._replicated(
            torch.tensor(positions, dtype=torch.int32), ttnn.int32
        )

    @staticmethod
    def _writes(rows, table):
        return SimpleNamespace(positions=rows, page_table=table) if KV_WRITES == "op" else rows

    def _row_writes(self, positions):
        out = []
        for i, position in enumerate(positions):
            value = torch.full((len(positions),), -1, dtype=torch.int32)
            value[i] = position
            out.append(self._replicated(value, ttnn.int32))
        return out

    def _eager_writes(self, positions, table_rows):
        if KV_WRITES == "op":
            rows = [self._replicated(torch.tensor([p], dtype=torch.int32), ttnn.int32) for p in positions]
            return self._writes(rows, ttnn.slice(table_rows, (0, 0), (1, table_rows.shape[1])))
        return self._row_writes(positions)

    def _tokens(self, ids):
        value = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
        value.flatten()[: len(ids)] = torch.tensor(ids, dtype=torch.int32)
        return self._replicated(value, ttnn.uint32)

    def _shared_kv(self, cache):
        return {kind: cache[number] for kind, number in self.layer_by_type.items()}

    # ---- untraced reference loop ----------------------------------------------------------------------------
    def _draft_eager(self, anchor, seed, position, written, cache, table_row):
        """K drafter steps; host argmax. Returns the draft ids."""
        if self.drafter_cache_pos == "anchor" or (self.drafter_cache_pos == "hf" and written):
            sdpa_pos = position
        else:
            sdpa_pos = position - 1
        pos_u = torch.zeros(1, 32, dtype=torch.int32)
        pos_u[0, 0] = position
        pos_u = self._replicated(pos_u, ttnn.uint32)
        pos_i = self._replicated(torch.tensor([sdpa_pos], dtype=torch.int32), ttnn.int32)
        ctx = self._draft_context(cache, table_row, pos_u, pos_i)
        drafts, token, hidden = [], anchor, seed
        for _ in range(self.K):
            token_tt = self._replicated(torch.tensor([[token]], dtype=torch.int32), ttnn.uint32)
            logits, hidden = self._draft_step(token_tt, hidden, ctx)
            row = self.gen._read_logits(logits)[..., :1, :].flatten()
            token = int(row.argmax())
            drafts.append(token)
        return drafts

    def _draft_context(self, cache, table, pos_u, pos_i):
        if self.lean:
            return {"cache": cache, "table": table, "sdpa_pos": pos_i, "rope": self.drafter.rope_rows(pos_u)}
        return {"shared": self._shared_kv(cache), "table": table, "pos_u": pos_u, "pos_i": pos_i}

    def _draft_step(self, token, hidden, ctx):
        """One drafter step; vocab-sharded logits (rows >= 1) and the next hidden."""
        if self.lean:
            return self.drafter.step(token, hidden, **ctx)
        return self.drafter.step(
            token, hidden, ctx["shared"], ctx["table"], ctx["pos_u"], ctx["pos_i"], gather_logits=False
        )

    def _verify_eager(self, ids, position, cache, table_rows):
        positions = [position + i for i in range(len(ids))]
        current, cache_pos = self._pos_rows(positions)
        logits, hidden = self.model.decode_forward(
            self._tokens(ids),
            current_pos=current,
            cache_pos=cache_pos,
            page_table=table_rows,
            kv_cache=cache,
            batch=len(ids),
            return_hidden=True,
            kv_write_rows=self._eager_writes(positions, table_rows),
        )
        top = self.gen._read_logits(logits)[..., : len(ids), :].reshape(len(ids), -1).argmax(-1).tolist()
        return [int(t) for t in top], hidden

    def generate(self, prompt_ids, max_new_tokens, *, stop_on_eos=False):
        """Untraced greedy speculative decode (correctness reference). Returns the generated ids."""
        gen, model, K = self.gen, self.model, self.K
        gen._release_trace()
        gen.reset()
        context = len(prompt_ids) + max_new_tokens + K + 2
        table = gen._standalone_cache(context)
        cache = gen.cache
        table_dev = model.upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        table_rows = model.upload(table.repeat(K + 1, 1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        ids = model.upload(torch.tensor(prompt_ids).reshape(1, 1, 1, -1).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        logits, seed = model.prefill_forward(ids, page_table=table_dev, kv_cache=cache, return_hidden=True)
        anchor = int(gen._read_logits(logits).flatten().argmax())
        out = [anchor]
        position = len(prompt_ids)
        written = False  # slot `position` holds no K/V yet
        accepted = []
        started = time.perf_counter()
        while len(out) < max_new_tokens and not (stop_on_eos and out[-1] in EOS):
            drafts = self._draft_eager(anchor, seed, position, written, cache, table_dev)
            top, hidden = self._verify_eager([anchor] + drafts, position, cache, table_rows)
            n = next((i for i in range(K) if drafts[i] != top[i]), K)
            accepted.append(n)
            out.extend(drafts[:n] + [top[n]])
            seed = ttnn.clone(hidden[:, :, n : n + 1, :])
            anchor, position, written = top[n], position + n + 1, n < K
        self.stats = {
            "rounds": len(accepted),
            "accepted_per_round": accepted,
            "mean_accepted": sum(accepted) / max(1, len(accepted)),
            "seconds": time.perf_counter() - started,
        }
        return out[:max_new_tokens]

    # ---- traced loop ------------------------------------------------------------------------------------------
    def _argmax(self, logits, rows):
        """Greedy ids of the first ``rows`` rows of vocab-sharded logits ``[1, 1, R, vocab / 4]``: the plain decode's
        sampler op (same tie rule as plain greedy). Returns ``[1, 1, 1, 32]`` uint32."""
        if logits.shape[-2] < 32:
            logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - logits.shape[-2]), (0, 0)], 0.0)
        ids, _ = self.gen.sampler.tt_sampling(logits)
        return ids

    def _round_body(self, s):
        """One speculative round as device ops over the persistent inputs in ``s``; returns ``(ids, vhidden)``:
        ids ``[1, 2K + 2]`` uint32 = anchor, d1..dK, target argmax of rows 0..K."""
        K = self.K
        tok, hidden = s["anchor"], s["seed"]
        drafts = []
        ctx = self._draft_context(s["cache"], s["table"], s["d_pos_u"], s["d_pos_i"])
        for _ in range(K):
            logits, hidden = self._draft_step(tok, hidden, ctx)
            if self.draft_argmax is not None:
                tok = self.draft_argmax(logits)
            else:
                ids = self._argmax(logits, 1)
                tok = ttnn.reshape(ttnn.slice(ids, (0, 0, 0, 0), (1, 1, 1, 1)), (1, 1))
            drafts.append(tok)
        verify_ids = ttnn.concat([s["anchor"]] + drafts, dim=1)  # [1, K + 1]
        logits, vhidden = self.model.decode_forward(
            verify_ids,
            current_pos=s["v_pos_u"],
            cache_pos=s["v_pos_i"],
            page_table=s["table_rows"],
            kv_cache=s["cache"],
            batch=K + 1,
            return_hidden=True,
            kv_write_rows=self._writes(s["row_writes"], s["table"]),
        )
        top = self._argmax(logits, K + 1)
        top = ttnn.reshape(ttnn.slice(top, (0, 0, 0, 0), (1, 1, 1, K + 1)), (1, K + 1))
        return ttnn.concat([verify_ids, top], dim=1), vhidden

    def _round_inputs(self, position, written):
        """Host values of the per-round position inputs."""
        K = self.K
        if self.drafter_cache_pos == "anchor" or (self.drafter_cache_pos == "hf" and written):
            sdpa_pos = position
        else:
            sdpa_pos = position - 1
        d_u = torch.zeros(1, 32, dtype=torch.int32)
        d_u[0, 0] = position
        positions = [position + i for i in range(K + 1)]
        v_u = torch.full((1, 32), -1, dtype=torch.int32)
        v_u[0, : K + 1] = torch.tensor(positions)
        rows = []
        for i in range(K + 1):
            value = torch.full((K + 1,), -1, dtype=torch.int32)
            value[i] = positions[i]
            rows.append(value)
        if KV_WRITES == "op":
            rows = [torch.tensor([p], dtype=torch.int32) for p in positions]
        return {
            "d_pos_u": d_u,
            "d_pos_i": torch.tensor([sdpa_pos], dtype=torch.int32),
            "v_pos_u": v_u,
            "v_pos_i": torch.tensor(positions, dtype=torch.int32),
            "row_writes": rows,
        }

    def _load_inputs(self, s, values):
        for name, value in values.items():
            if name == "row_writes":
                for target, row in zip(s[name], value):
                    self.gen._copy(row, target, "mtp_input_refreshes")
            else:
                self.gen._copy(value, s[name], "mtp_input_refreshes")

    def _set_anchor(self, s, token):
        self.gen._copy(torch.tensor([[token]], dtype=torch.int32), s["anchor"], "mtp_input_refreshes")

    def release(self):
        state = getattr(self, "_traced", None)
        if state is not None and state.get("trace") is not None:
            ttnn.release_trace(self.mesh, state["trace"])
        self._traced = None

    def generate_traced(self, prompt_ids, max_new_tokens, *, stop_on_eos=False):
        """Greedy speculative decode, one traced program per round (K drafter steps + verify + argmaxes).
        Per round the host reads 2K + 2 ids, picks the accepted prefix and refreshes the small inputs."""
        gen, model, K = self.gen, self.model, self.K
        gen._release_trace()
        gen.reset()
        context = len(prompt_ids) + max_new_tokens + K + 2
        table = gen._standalone_cache(context)
        cache = gen.cache
        s = getattr(self, "_traced", None)
        if s is not None and (s["cache"] is not cache or tuple(s["table"].shape) != tuple(table.shape)):
            self.release()
            s = None
        if s is None:
            up = model.upload
            s = {
                "cache": cache,
                "shared": self._shared_kv(cache),
                "table": up(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                "table_rows": up(table.repeat(K + 1, 1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                "anchor": up(torch.zeros(1, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "seed": up(torch.zeros(1, 1, 1, model.config.hidden_size)),
                "d_pos_u": up(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "d_pos_i": up(torch.zeros(1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                "v_pos_u": up(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "v_pos_i": up(torch.zeros(K + 1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                "row_writes": [
                    up(
                        torch.zeros(1 if KV_WRITES == "op" else K + 1, dtype=torch.int32),
                        ttnn.int32,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
                    for _ in range(K + 1)
                ],
                "trace": None,
            }
            self._traced = s
        else:
            gen._copy(table, s["table"], "mtp_input_refreshes")
            gen._copy(table.repeat(K + 1, 1), s["table_rows"], "mtp_input_refreshes")
        started = time.perf_counter()
        ids = model.upload(torch.tensor(prompt_ids).reshape(1, 1, 1, -1).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        logits, seed = model.prefill_forward(ids, page_table=s["table"], kv_cache=cache, return_hidden=True)
        anchor = int(ttnn.to_torch(ttnn.get_device_tensors(self._argmax(logits, 1))[0]).flatten()[0])
        ttnn.copy(seed, s["seed"])
        out = [anchor]
        position, written = len(prompt_ids), False
        self._set_anchor(s, anchor)
        self._load_inputs(s, self._round_inputs(position, written))
        if s["trace"] is None:
            capture_started = time.perf_counter()
            saved_seed = ttnn.clone(s["seed"])
            self._round_body(s)  # compile; the replay rewrites the same K/V rows
            ttnn.copy(saved_seed, s["seed"])
            ttnn.synchronize_device(self.mesh)
            s["trace"] = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            s["ids"], s["vhidden"] = self._round_body(s)
            ttnn.end_trace_capture(self.mesh, s["trace"], cq_id=0)
            ttnn.synchronize_device(self.mesh)
            self.capture_ms = (time.perf_counter() - capture_started) * 1000
        ttnn.synchronize_device(self.mesh)
        first_token_s = time.perf_counter() - started
        accepted = []
        decode_started = time.perf_counter()
        while len(out) < max_new_tokens and not (stop_on_eos and any(t in EOS for t in out)):
            ttnn.execute_trace(self.mesh, s["trace"], cq_id=0, blocking=False)
            row = ttnn.to_torch(ttnn.get_device_tensors(s["ids"])[0]).flatten().tolist()
            drafts, top = row[1 : K + 1], row[K + 1 : 2 * K + 2]
            n = next((i for i in range(K) if drafts[i] != top[i]), K)
            accepted.append(n)
            out.extend(drafts[:n] + [top[n]])
            ttnn.copy(s["vhidden"][:, :, n : n + 1, :], s["seed"])
            position, written = position + n + 1, n < K
            self._set_anchor(s, top[n])
            self._load_inputs(s, self._round_inputs(position, written))
        ttnn.synchronize_device(self.mesh)
        seconds = time.perf_counter() - decode_started
        if stop_on_eos:
            out = next((out[: i + 1] for i, t in enumerate(out) if t in EOS), out)
        produced = min(len(out), max_new_tokens) - 1
        self.stats = {
            "rounds": len(accepted),
            "accepted_per_round": accepted,
            "mean_accepted": sum(accepted) / max(1, len(accepted)),
            "decode_seconds": seconds,
            "tps": produced / seconds if seconds else None,
            "ms_per_round": seconds * 1000 / max(1, len(accepted)),
            "first_token_s": first_token_s,
        }
        return out[:max_new_tokens]

    def generate_plain_rows(self, prompt_ids, max_new_tokens):
        """Reference: plain greedy (one token per step) through the batched decode path the verify uses (2 rows, the
        second inactive). Returns the ids and the top-1 minus top-2 logit margin of every step."""
        gen, model = self.gen, self.model
        gen._release_trace()
        gen.reset()
        table = gen._standalone_cache(len(prompt_ids) + max_new_tokens + 2)
        cache = gen.cache
        table_dev = model.upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        table_rows = model.upload(table.repeat(2, 1), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        ids = model.upload(torch.tensor(prompt_ids).reshape(1, 1, 1, -1).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        logits = model.prefill_forward(ids, page_table=table_dev, kv_cache=cache)
        out = [int(gen._read_logits(logits).flatten().argmax())]
        margins = [None]
        position = len(prompt_ids)
        while len(out) < max_new_tokens:
            current, _ = self._pos_rows([position, -1])
            cache_pos = self._replicated(torch.tensor([position, -1], dtype=torch.int32), ttnn.int32)
            logits = model.decode_forward(
                self._tokens([out[-1], 0]),
                current_pos=current,
                cache_pos=cache_pos,
                page_table=table_rows,
                kv_cache=cache,
                batch=2,
            )
            row = gen._read_logits(logits)[..., :1, :].flatten()
            top2 = row.topk(2).values
            margins.append(float(top2[0] - top2[1]))
            out.append(int(row.argmax()))
            position += 1
        self.stats_rows = {"margins": margins}
        return out
