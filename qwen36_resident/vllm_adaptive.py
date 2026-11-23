# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DB3 phase 2 (P2-3): one served class for 1-32 greedy users, DFlash2 speculation while few users are live
(qwen36_resident/adaptive.py's two geometries in one process), vllm-tt-plugin's runner-model spec contract with
narrow decode (docs/SPEC_DECODE_CONTRACT.md).

Modes, chosen at propose time (the plugin cannot take a proposal back, so a mode change takes effect at the next
call): SPEC (live users <= 8): every live user is a user of the 8 x 4 verify-rows build; propose drafts K = 3 for each,
verify calls run the vrows build. PLAIN (more than 8 live users, until PLAIN_HOLD propose calls at <= SPEC_BACK live
users): every user is in the plain 32-row build; propose offers no drafts, so the runner sends ordinary [B, 1] calls,
served by the rows adapter (device greedy, the plain build). A switch to PLAIN moves the spec users at the next call
(a resolving verify: their commits from accepted_counts; or a narrow call); a switch back moves them at propose time
(state + option-B drafter refill) and drafts for them at once.

Every live request owns a plain row (the rows adapter's SlotMap; its KV pages are its block table in the shared pool);
a spec user index is extra (spec_of[plain row]). Prefill: every prompt through the rows adapter into its plain row, as
the batch profile prefills (the TT chunk core's chunks and gangs, batched PP passes, the host prefix cache). A user a
decode call showed greedy is promoted into the spec build at the next propose while in SPEC mode (state move +
option-B drafter refill from its token history: prompt + every committed token), and drafted for at once. A verify
call carrying rows of plain-build users (no drafts: column 0 only) runs a plain launch for them as well.

Sampling (phase 2b): only greedy requests speculate. A request is non-speculable when a decode call's sampling_params
show temperature != 0 or a penalty (the runner's own test): from then on the server is in PLAIN while it is live (no
drafts: ordinary calls, sampled on device or on the host by the runner as the batch profile does). A verify the runner
still sends (drafts proposed before it was seen, or a multi-token commit to resolve) commits column 0 for its rows: a
plain launch with the row's sampling parameters in the hub answers it (a sampled token where the contract names an
argmax; a spec user found sampled moves to the plain build first). Gap: a host-sampled verify step carries no sampling
parameters, so such a row gets the argmax there (temperature with top_k above max_device_top_k, penalties, min_p, in
the one transition step). The plugin refuses logprobs / structured output / logit_bias on a speculating launch.
"""
import os

import torch
from loguru import logger

from qwen36_resident import weights as Q
from qwen36_resident.model import TOP_K
from qwen36_resident.vllm_rows import CHUNK_UNIT, ResidentRowsAdapter, RowsEngine, expand_blocks, resident_capabilities

SPEC_USERS = 8
SPEC_K = 3
SPEC_BACK = int(os.environ.get("RESIDENT_ADAPTIVE_SPEC_BACK", "6"))  # back to SPEC at <= this many live users ...
PLAIN_HOLD = int(os.environ.get("RESIDENT_ADAPTIVE_PLAIN_HOLD", "32"))  # ... for this many propose calls in a row
PLAIN_ROWS = 32
# acceptance-aware policy: SPEC gives way to PLAIN when its measured tokens per user-second fall below a plain step's
# (x PAY_MARGIN) after PAY_MIN verify steps, for PAY_HOLD propose calls; plain step times measured per live count, a
# default model (ms) until then
PAY_MARGIN = float(os.environ.get("RESIDENT_ADAPTIVE_PAY_MARGIN", "1.05"))
PAY_MIN = int(os.environ.get("RESIDENT_ADAPTIVE_PAY_MIN", "16"))
PAY_HOLD = int(os.environ.get("RESIDENT_ADAPTIVE_PAY_HOLD", "512"))
PLAIN_MS_DEFAULT = lambda n: 16.5 + 0.5 * (n - 1)  # noqa: E731  (the batch card's 128 / 128 TPOT, c1 .. c8)
# a greedy user is promoted once it has generated min(window, its context) / PROMOTE_DIV tokens in the plain build: its
# refill re-prefills that many tokens (~0.17 ms each) and speculation gains ~6 ms a token at c1, so a short answer on a
# long context never pays a refill it cannot win back
PROMOTE_DIV = int(os.environ.get("RESIDENT_ADAPTIVE_PROMOTE_DIV", "32"))
# (log the per-step host / call times every 64 steps)
TIMING = os.environ.get("RESIDENT_ADAPTIVE_TIMING", "0") == "1"
# (0: synchronous serving, the first served version)
ASYNC = os.environ.get("RESIDENT_ADAPTIVE_ASYNC", "1") == "1"


class _PlainRowsEngine(RowsEngine):
    """the rows adapter's engine over the plain build (the PP prefill's batched pass runs with the drafter's tap writes
    sent to scratch: plain-build prompts need no drafter context)"""


class Qwen36ResidentAdaptiveForCausalLM(ResidentRowsAdapter):
    model_capabilities = {
        "supports_prefix_caching": False,
        # (async: plain narrow steps overlap as the batch profile's; speculative steps the runner drains before the
        # next, SPEC_DECODE_CONTRACT.md 4c; the verify's argmax block is a host tensor, valid through the readback)
        "supports_async_decode": ASYNC,
        "supports_async_spec_decode": ASYNC,
        "supports_sample_on_device": True,  # (the plain build's narrow calls: the hub's greedy)
        "max_device_top_k": TOP_K,
        "supports_device_penalties": False,
        "supports_spec_decode": True,
        "spec_requirements": ["device_propose"],
        "output_tokens_per_step": 1,
        # (the TT chunk core, as the batch profile: aligned exclusive chunks and prefill gangs, see resident_capabilities)
        **{k: v for k, v in resident_capabilities().items() if k.startswith(("supports_chunked", "tt_prefill"))},
    }

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if int(max_num_seqs) > PLAIN_ROWS:
            return SpecReject(reason=f"up to {PLAIN_ROWS} users (max_num_seqs {max_num_seqs})", supported_k=())
        if int(requested_k) < 1:
            return SpecReject(reason="no drafts requested", supported_k=(SPEC_K,))
        return SpecPlan(
            effective_k=min(SPEC_K, int(requested_k)),
            lanes_per_request=SPEC_K + 1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=True,
        )

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, **kwargs):
        from vllm.transformers_utils.config import uses_mrope

        path = Q.checkpoint_dir(
            os.environ.get("HF_MODEL") or hf_config._name_or_path, getattr(hf_config, "_commit_hash", None)
        )
        assert max_batch_size <= PLAIN_ROWS, f"{cls.__name__}: max_num_seqs <= {PLAIN_ROWS}"
        return cls(mesh_device, Q.Checkpoint(path), max_seq_len, max_batch_size, uses_mrope(hf_config))

    @classmethod
    def get_max_tokens_all_users(cls, max_model_len=None, max_num_seqs=None, **kwargs):
        if os.environ.get("RESIDENT_MAX_TOKENS_ALL_USERS"):
            return int(os.environ["RESIDENT_MAX_TOKENS_ALL_USERS"])
        return int(max_model_len) * int(max_num_seqs or 1)

    def __init__(self, mesh, ck, max_seq_len, max_batch, mrope):
        self.mesh, self.ck = mesh, ck
        super().__init__(None, ck.config["vocab_size"], max_seq_len, PLAIN_ROWS, CHUNK_UNIT, mrope, rows=PLAIN_ROWS)
        self.ae = self.sess = None
        self.spec_of = {}  # plain row (resident slot) -> spec user index
        self.history = {}  # resident slot -> every token so far (prompt + committed); [-1] is the next to run
        self.last_acc = {}  # resident slot -> tokens its last spec step committed (its next verify applies them)
        self.mode, self.low_calls = "SPEC", 0
        self.sampled = set()  # resident slots of non-speculable requests (temperature != 0 or a penalty)
        self.greedy_ok = set()  # resident slots a decode call showed greedy (promotable into the spec build)
        self.prompt_len = {}  # resident slot -> its prompt's length (the tokens generated since: history beyond it)
        # (the acceptance-aware policy's measurements: EMAs of tokens per spec user-step and of step times per kind)
        self._pay = dict(acc=None, spec_ms={}, plain_ms={}, verifies=0, last_t=None, kind=None, n=0, hold=0, excl=0.0)
        # the plain build's hub-fed inputs do not hold every live row's next token (a move into it, a step served by
        # the spec build): its next ordinary call rebuilds them from the history instead of a steady replay
        self._plain_stale = False
        self._t = {}  # (TIMING) name -> [total seconds, calls]

    def _lap(self, name, t0):
        import time

        if TIMING:
            acc = self._t.setdefault(name, [0.0, 0])
            acc[0] += time.perf_counter() - t0
            acc[1] += 1
            if name == "propose" and acc[1] % 64 == 0:
                logger.info(
                    "adaptive timing (mean ms): "
                    + ", ".join(f"{k} {v[0] / max(1, v[1]) * 1000:.2f}" for k, v in sorted(self._t.items()))
                    + f" (mode {self.mode})"
                )
                self._t = {}

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        from qwen36_resident.adaptive import AdaptiveEngine

        nb = int(kv_cache_shape[0])
        logger.info(f"Resident adaptive: {SPEC_USERS} x 4 verify rows + {PLAIN_ROWS} plain rows, KV pool {nb} blocks")
        self.ae = AdaptiveEngine(self.mesh, self.ck, self.max_seq_len + 2 * 4 + 64, nb)
        self.sess = self.ae.sess
        self.ae.plain_tables = self.tables  # (one list: the engine's moves and the rows adapter write the same tables)
        d = self.ae.d
        self.engine = _PlainRowsEngine(
            self.ae.plain, self.ae.pp, self.ae.emb, self.vocab, d.vocab_chip, PLAIN_ROWS, CHUNK_UNIT, self.mesh
        )
        return []

    def warmup_model_prefill(self, *args, **kwargs):
        pass

    def warmup_model_decode(self, *args, **kwargs):
        pass

    # ------------------------------------------------------------------ helpers
    def _free_spec(self):
        used = set(self.spec_of.values())
        return [u for u in range(SPEC_USERS) if u not in used]

    def _note(self, r, pos, tok):
        """the history of slot r: token tok at position pos (earlier entries kept, later dropped)"""
        h = self.history.setdefault(r, [])
        if pos < len(h):
            del h[pos:]
        if pos == len(h):
            h.append(int(tok))

    def _to_plain_all(self, acc_of=None):
        for r, u in sorted(self.spec_of.items()):
            acc = (acc_of or {}).get(r, self.last_acc.get(r, 1))
            self.ae.move_to_plain(u, r, self.tables[r], acc)
            self.last_acc[r] = 1
            self._plain_stale = True
        self.spec_of.clear()

    def _measure(self, spec_accepts):
        """update the policy's EMAs with the step just committed (its kind and live count from its decode call, its time
        since the previous propose); whether speculation pays: its tokens per user-ms >= a plain step's x PAY_MARGIN
        (True until PAY_MIN verify steps are in)"""
        import time

        pay, now, ema = self._pay, time.perf_counter(), lambda old, x, a=0.1: x if old is None else old + a * (x - old)
        # (a promotion's refill is no step's cost: excluded)
        dt_ms = None if pay["last_t"] is None else (now - pay["last_t"] - pay.get("excl", 0.0)) * 1000
        pay["last_t"], pay["excl"] = now, 0.0
        n = max(1, pay["n"])
        if dt_ms is not None and dt_ms < 5000:  # (an idle gap is no step)
            if pay["kind"] == "verify" and spec_accepts:
                pay["spec_ms"][n] = ema(pay["spec_ms"].get(n), dt_ms)
                pay["acc"] = ema(pay["acc"], sum(spec_accepts) / len(spec_accepts))
                pay["verifies"] += 1
            elif pay["kind"] == "plain":
                pay["plain_ms"][n] = ema(pay["plain_ms"].get(n), dt_ms)
        if pay["verifies"] < PAY_MIN or pay["acc"] is None or n not in pay["spec_ms"]:
            return True
        plain = pay["plain_ms"].get(n, PLAIN_MS_DEFAULT(n))
        return pay["acc"] / pay["spec_ms"][n] >= PAY_MARGIN / plain

    def _pay_line(self):
        pay, n = self._pay, max(1, self._pay["n"])
        return (
            f"{n} users: {pay['acc']:.2f} tokens per {pay['spec_ms'].get(n, 0):.1f} ms speculating vs 1 per "
            f"{pay['plain_ms'].get(n, PLAIN_MS_DEFAULT(n)):.1f} ms plain"
        )

    def _promote(self):
        """every live plain user a decode call showed greedy into a free spec user (its drafter window refilled from
        its history: option B)"""
        import time

        t0 = time.perf_counter()
        free, items = self._free_spec(), []
        for r in sorted(self.live):
            if r in self.spec_of or r not in self.greedy_ok or r in self.sampled or not free:
                continue
            h = self.history.get(r, [])
            if len(h) - self.prompt_len.get(r, 0) < min(self.ae.spec.window, len(h)) // PROMOTE_DIV:
                continue  # (not yet: its refill would cost more than a short answer gains)
            u = free.pop(0)
            items.append((r, u, self.tables[r], self.history[r]))
            self.spec_of[r], self.last_acc[r] = u, 1
        if items:  # (one batched refill for every user promoted now)
            self.ae.move_to_spec_many(items)
        self._pay["excl"] = self._pay.get("excl", 0.0) + time.perf_counter() - t0

    @staticmethod
    def _non_speculable(sp, i):
        """the runner's test (TTModelRunner._request_is_speculable): temperature != 0 or a penalty"""
        val = lambda name, default: (
            getattr(sp, name, default)
            if not isinstance(getattr(sp, name, default), (list, tuple))
            else getattr(sp, name)[i]
        )  # noqa: E731
        return (
            float(val("temperature", 0.0) or 0.0) != 0.0
            or float(val("presence_penalty", 0.0) or 0.0) != 0.0
            or float(val("frequency_penalty", 0.0) or 0.0) != 0.0
            or float(val("repetition_penalty", 1.0) or 1.0) != 1.0
        )

    def _plain_tokens(self, feed):
        """one plain launch of the slots in feed {slot: (token, position)} (the others held): their device greedy"""
        sitting = {s for s in self.live if s not in feed}
        self.engine.decode(feed, [0], preserve_launches={0} if sitting else ())
        self.engine.sync()
        if sitting:  # (live rows outside this launch: their fed inputs are not this step's)
            self._plain_stale = True
        self._fed_launches = [0] if self.engine.device_sampling else None
        return dict(zip(feed, self.engine.tokens_from(self.engine.read_tokens(list(feed), False))))

    # ------------------------------------------------------------------ prefill
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, start_pos=None, empty_slots=None, **kwargs):
        """every prompt through the rows adapter into its plain row, as the batch profile prefills (the TT chunk core's
        chunks and gangs, batched PP passes, the host prefix cache); a greedy user moves into the spec build at a later
        propose (_promote), once a decode call has shown its sampling parameters"""
        n = int(len(prompt_lens))
        if self.mode == "PLAIN" and not self.live:  # (an idle server starts speculating again)
            self.mode, self.low_calls = "SPEC", 0
        res = super().prefill_forward(
            tokens, page_table, kv_cache, prompt_lens, start_pos=start_pos, empty_slots=empty_slots, **kwargs
        )
        finals = kwargs.get("prefill_final_mask")
        for j in range(n):
            if finals is None or bool(finals[j]):  # (a prompt handed off: its tokens, for the drafter refill)
                r = self.slots.res_of[int(empty_slots[j])]
                self.history[r] = [int(t) for t in tokens[j, : int(prompt_lens[j])]]
                self.prompt_len[r] = int(prompt_lens[j])
        if len(self.live) > SPEC_USERS and self.mode == "SPEC":
            logger.info(f"adaptive: {len(self.live)} live users -> PLAIN")
            self.mode, self.low_calls = "PLAIN", 0
        return res

    # ------------------------------------------------------------------ decode
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        *args,
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        slot_remap=None,
        **kwargs,
    ):
        import time

        t0 = time.perf_counter()
        if TIMING and getattr(self, "_last_dec", None) is not None:
            self._lap("period", self._last_dec)  # (decode call to decode call: the step as the plugin runs it)
        self._last_dec = t0
        res_of = self.slots.remapped(slot_remap) if slot_remap is not None else dict(self.slots.res_of)
        B = int(tokens.shape[0])
        W = int(tokens.shape[1]) if tokens.dim() > 1 else 1
        pos = start_pos.reshape(B, -1)
        rows = {i: res_of[i] for i in range(B) if int(pos[i, 0]) >= 0 and res_of.get(i) in self.live}
        tok0 = tokens.reshape(B, -1)[:, 0].tolist()
        reload = bool(kwargs.get("reload_inputs", True))
        for i, r in rows.items():  # (a steady call's tokens / positions are a step behind: the history has them)
            if reload or spec_mode is not None:
                self._note(r, int(pos[i, 0]), tok0[i])
        sp = kwargs.get("sampling_params")
        if sp is not None:
            for i, r in rows.items():
                (self.sampled if self._non_speculable(sp, i) else self.greedy_ok).add(r)
        if (self.sampled & self.live or (sp is None and rows)) and self.mode == "SPEC":
            # (sampling_params absent: the runner fell back to host sampling for this step, which a live row needs —
            # penalties, a top_k past max_device_top_k: treat it as non-speculable traffic)
            logger.info("adaptive: a non-speculable request is live -> PLAIN")
            self.mode, self.low_calls = "PLAIN", 0
            self._host_steps = True
        elif sp is not None:
            self._host_steps = False
        # (the spec users' tables here; the plain rows' the rows adapter keeps — every row's 4,096-entry table converted
        # and compared each call cost several ms a step at 32 users)
        if page_table is not None and (spec_mode is not None or self.spec_of):
            changed = False
            for i, r in rows.items():
                if r not in self.spec_of and spec_mode is None:
                    continue
                tbl = expand_blocks(page_table[i].tolist(), self.pages_per_block)
                if tbl != self.tables[r]:
                    self.tables[r], changed = tbl, True
                    if r in self.spec_of:
                        self.ae._set_spec_table(self.spec_of[r], tbl)
            if changed:
                self._write_tables()
        self._pay["kind"] = "verify" if spec_mode is not None and self.spec_of else "plain"
        self._pay["n"] = len(rows)
        if spec_mode is None:
            # (PLAIN, or a host-sampled step — it wants logits, which the rows adapter returns: everyone to plain)
            if self.spec_of and (self.mode == "PLAIN" or sp is None):
                self.mode = "PLAIN"
                self._to_plain_all()
            if not self.spec_of:  # (every user plain: the rows adapter's own path)
                self._lap("decode_host", t0)
                t1 = time.perf_counter()
                if self._plain_stale and not reload:
                    # (a steady call after a move / a spec-served step: this step's inputs from the history, reloaded)
                    tokens, start_pos = tokens.clone().reshape(B, -1), start_pos.clone().reshape(B, -1)
                    for i, r in rows.items():
                        tokens[i, 0], start_pos[i, 0] = self.history[r][-1], len(self.history[r]) - 1
                    tokens, start_pos = tokens[:, :1], start_pos[:, 0]
                    kwargs = dict(kwargs, reload_inputs=True, reload_page_table=False)
                    kwargs["reload_sampling_params"] = kwargs.get("sampling_params") is not None
                out = super().decode_forward(tokens, start_pos, page_table, *args, slot_remap=slot_remap, **kwargs)
                if kwargs.get("reload_inputs", True):
                    self._plain_stale = False
                self._lap("decode_plain", t1)
                return out
            out = self._spec_row0(rows, tokens.reshape(B, -1)[:, 0], pos[:, 0], B)
            self._commit_remap(slot_remap, res_of)
            return out
        return self._verify(
            tokens, pos, rows, B, W, num_valid_drafts, accepted_counts, spec_mode, slot_remap, res_of, sp
        )

    def _commit_remap(self, slot_remap, res_of):
        if slot_remap is not None:
            remap = [int(j) for j in slot_remap]
            self._expect_moves = {j: i for i, j in enumerate(remap) if i != j}
            self.slots.res_of = res_of

    def _spec_row0(self, rows, tok0, pos0, B):
        """a narrow call while users are in the spec build (right after their start, before any drafts): a vrows step
        of row 0 (the drafts' rows run a dummy) for them, a plain launch for the plain ones; host token ids [B, 1]"""
        U, R = SPEC_USERS, 4
        toks, upos, act, of_u = [[0] * R for _ in range(U)], [0] * U, [False] * U, {}
        feed = {}
        for i, r in rows.items():
            if r in self.spec_of:
                u = self.spec_of[r]
                toks[u], upos[u], act[u], of_u[i] = [int(tok0[i])] + [0] * (R - 1), int(pos0[i]), True, u
            else:
                feed[r] = (int(tok0[i]), int(pos0[i]))
        res = torch.zeros(B, 1, dtype=torch.int32)
        self._plain_stale = True
        if of_u:
            g = self.sess.verify(toks, upos, act)
            for i, u in of_u.items():
                res[i, 0] = g[u][0]
                self.last_acc[rows[i]] = 1
        if feed:
            got = self._plain_tokens(feed)
            for i, r in rows.items():
                if r in got:
                    res[i, 0] = got[r]
        return res

    def _verify(
        self, tokens, pos, rows, B, W, num_valid_drafts, accepted_counts, spec_mode, slot_remap, res_of, sp=None
    ):
        from vllm_tt_plugin.spec_decode import VerifyOutput

        U, R = SPEC_USERS, 4
        tk = tokens.reshape(B, W)
        acc_of = {r: int(accepted_counts[i]) for i, r in rows.items()} if accepted_counts is not None else {}
        nvd = {i: int(num_valid_drafts[i]) for i in rows} if num_valid_drafts is not None else {}
        if self.mode == "PLAIN" and self.spec_of and not any(nvd.values()):
            self._to_plain_all(acc_of)  # (a resolving verify: the spec users' commits, then the plain build)
        for i, r in rows.items():  # (a spec user found non-speculable: to the plain build, it carries no drafts)
            if r in self.spec_of and r in self.sampled:
                assert nvd.get(i, 0) == 0, "drafts for a non-speculable request"
                self.ae.move_to_plain(self.spec_of.pop(r), r, self.tables[r], acc_of.get(r, 1))
                self.last_acc[r] = 1
        out = torch.zeros(B, W, dtype=torch.int32)
        toks, upos, act, of_u, feed = [[0] * R for _ in range(U)], [0] * U, [False] * U, {}, {}
        for i, r in rows.items():
            row = [int(t) for t in tk[i].tolist()]
            if r in self.spec_of:
                u = self.spec_of[r]
                row = [t if 0 <= t < self.vocab else 0 for t in row] + [0] * (R - W)
                toks[u], upos[u], act[u], of_u[i] = row[:R], int(pos[i, 0]), True, u
            else:
                assert nvd.get(i, 0) == 0, "a plain-geometry row with drafts"
                feed[r] = (row[0], int(pos[i, 0]))
        if of_u:
            g = self.sess.verify(toks, upos, act)
            for i, u in of_u.items():
                out[i] = torch.tensor(g[u][:W], dtype=torch.int32)
            self._plain_stale = True
        if feed:
            if sp is not None:  # (column 0 of a plain row: sampled with its parameters in the hub, greedy ones argmax)
                plain_rows = [i for i, r in rows.items() if r in feed]
                self._set_sampling(sp, plain_rows, [rows[i] for i in plain_rows])
            got = self._plain_tokens(feed)
            for i, r in rows.items():
                if r in got:
                    out[i, 0] = got[r]
        self._commit_remap(slot_remap, res_of)
        return VerifyOutput(spec_mode=spec_mode, argmax_ids=out)

    # ------------------------------------------------------------------ propose
    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        import time

        t0 = time.perf_counter()
        out = self._propose(num_drafts, committed_tokens, committed_positions, accepted_counts)
        self._lap("propose", t0)
        return out

    def _propose(self, num_drafts, committed_tokens, committed_positions, accepted_counts):
        from vllm_tt_plugin.spec_decode import DraftOutput

        B, K = int(committed_tokens.shape[0]), int(num_drafts)
        res_of = dict(self.slots.res_of)
        rows = {}
        for i in range(B):
            a = int(accepted_counts[i])
            p = int(committed_positions[i, a - 1])
            r = res_of.get(i)
            if p < 0 or r is None or r not in self.live:
                continue
            for c in range(a):
                self._note(r, int(committed_positions[i, c]), int(committed_tokens[i, c]))
            rows[i] = (r, a, int(committed_tokens[i, a - 1]), p)
            self.last_acc[r] = a
        live = len(self.live)
        pays = self._measure([a for r, a, _, _ in rows.values() if r in self.spec_of])
        # ---- the mode for the next step
        if self.sampled & self.live or getattr(self, "_host_steps", False):  # (non-speculable traffic: no drafts)
            self.mode, self.low_calls = "PLAIN", 0
        elif self.mode == "SPEC" and not pays:
            logger.info(f"adaptive: speculation does not pay ({self._pay_line()}) -> PLAIN for {PAY_HOLD} steps")
            self.mode, self.low_calls, self._pay["hold"], self._pay["verifies"] = "PLAIN", 0, PAY_HOLD, 0
        elif live > SPEC_USERS:
            if self.mode == "SPEC":
                logger.info(f"adaptive: {live} live users -> PLAIN")
            self.mode, self.low_calls = "PLAIN", 0
        elif self.mode == "PLAIN":
            self.low_calls = self.low_calls + 1 if live <= SPEC_BACK else 0
            if self.low_calls >= max(PLAIN_HOLD, self._pay["hold"]) and live > 0:
                self._pay["hold"] = 0
                logger.info(f"adaptive: {live} live users for {self.low_calls} steps -> SPEC")
                self.mode, self.low_calls = "SPEC", 0
        if self.mode == "SPEC":
            self._promote()  # (known-greedy plain users into the spec build: state moves + drafter refill; drafts below)
        ids = torch.zeros(B, K, dtype=torch.int32)
        valid = torch.zeros(B, dtype=torch.int32)
        if self.mode == "SPEC" and self.spec_of:
            U = SPEC_USERS
            anchors, positions, accepted, active, row_u = [0] * U, [0] * U, [1] * U, [False] * U, {}
            for i, (r, a, tok, p) in rows.items():
                if r in self.spec_of:
                    u = self.spec_of[r]
                    anchors[u], positions[u], accepted[u], active[u], row_u[i] = tok, p, a, True, u
            drafts = self.sess.propose(anchors, positions, accepted, active)
            for i, u in row_u.items():
                ids[i] = torch.tensor(drafts[u][:K], dtype=torch.int32)
                valid[i] = K
        return DraftOutput(draft_token_ids=ids, num_valid=valid)

    # ------------------------------------------------------------------ lifecycle
    def release_request(self, vslot):
        r = self.slots.res_of.get(int(vslot))
        super().release_request(vslot)
        if r is not None:
            if r in self.spec_of:
                u = self.spec_of.pop(r)
                self.ae._set_spec_table(u, [0])
            self.history.pop(r, None)
            self.prompt_len.pop(r, None)
            self.last_acc.pop(r, None)
            self.sampled.discard(r)
            self.greedy_ok.discard(r)
