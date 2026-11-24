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
from qwen36_resident.model import SPEC_SAMPLE, TOP_K
from qwen36_resident.adaptive import taps_usable, window_parts
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
# promotion gate: a move into the spec build refills the drafter window (a target pass over up to its 2,048 tokens)
# while every live user waits, so promote only when the expected saving pays for it: the candidates' expected remaining
# tokens x (plain step - spec step / PROMOTE_ACC) ms per token > refill ms x live users. Remaining tokens: an EMA of
# finished requests' output lengths (PROMOTE_OUT until one finishes) minus what each has generated (at least a quarter
# of it); refill ms per token measured (a default until then); spec step times measured per live count (a default
# model until then); acceptance for a request not yet speculating: PROMOTE_ACC tokens per verify
PROMOTE_GATE = os.environ.get("RESIDENT_ADAPTIVE_PROMOTE_GATE", "1") == "1"
PROMOTE_ACC = float(os.environ.get("RESIDENT_ADAPTIVE_PROMOTE_ACC", "2.7"))
PROMOTE_OUT = float(os.environ.get("RESIDENT_ADAPTIVE_PROMOTE_OUT", "256"))
REFILL_MS_FIXED, REFILL_MS_PER_TOKEN = 20.0, 0.13  # (defaults: tests/refill_batch.py, 2026-11-23)
SPEC_MS_DEFAULT = lambda n: 28.0 + 4.6 * (n - 1)  # noqa: E731  (served verify steps, rc-20261124j: 1 / 2 / 4 / 8
# users ~29 / 35 / 41 / 60 ms; 52.5 at 8 users let the gate promote 8 users before any 8-user step was measured)
# step times grow with the live users' total context (attention reads every row's KV): the estimates above are kept
# per live count at no context (a measured step's time minus its context's share, these ms per context token) and the
# decisions add the current context's share — a 250K request's ~44 ms verify steps kept as the 1-user estimate closed
# the promotion gate for the short chats after it (rc-20261124i natural c1: 61 vs 97 tok/s). (served, 2026-11-24:
# plain 16.5 / 17.3 / 23.3 ms at 128 / 32K / 250K context, verify ~28 / ~31 / ~44 ms)
PLAIN_CTX_MS = float(os.environ.get("RESIDENT_ADAPTIVE_PLAIN_CTX_MS", "2.7e-5"))
SPEC_CTX_MS = float(os.environ.get("RESIDENT_ADAPTIVE_SPEC_CTX_MS", "6.5e-5"))
# P2d: promotions fill the drafter from the prefill's taps when kept (RESIDENT_ADAPTIVE_P2D, default on); a fill's cost
P2D = os.environ.get("RESIDENT_ADAPTIVE_P2D", "1") == "1"
P2D_FILL_MS = 20.0
OUT_ALPHA = 0.05  # (the output-length EMA starts at PROMOTE_OUT: a few short requests, e.g. a warm-up, do not flip it)
OUTLIER = 3.0  # (a time sample over this x the estimate, or the default model's for the first one, is skipped: a
# first-time trace capture, a prefill's device time an async call left in a step)


def _store(d, k, v):
    """an estimate into its dict unless robust_ema skipped the first sample (None: the default model stays in use — a
    None stored there crashed the promotion gate served, 2026-11-24)"""
    if v is not None:
        d[k] = v


def robust_ema(old, x, default, a=0.1):
    """an EMA of times that skips outliers (OUTLIER) — one stray sample must not set the policy's estimates"""
    ref = default if old is None else old
    if ref is not None and x > OUTLIER * ref:
        return old
    return x if old is None else old + a * (x - old)


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
        # (narrow_decode_resolves_commits: this class records each verify's commits at propose time — a narrow call
        # right after a multi-token commit moves its spec users from those counts — so the plugin keeps a request
        # needing host-only sampling (logprobs, structured output, logit_bias, ...) out of every verify and serves it
        # instead of refusing it; plugin ed900d6+. Older plugins have no such field.)
        extra = {}
        fields = getattr(SpecPlan, "__dataclass_fields__", {})
        if "narrow_decode_resolves_commits" in fields:
            extra["narrow_decode_resolves_commits"] = True
        # (sampled speculation, RESIDENT_SPEC_SAMPLE: the verify rows draw with each request's parameters, so
        # temperature > 0 requests the device samples speculate; plugin 2d22bae+)
        if SPEC_SAMPLE and "verify_rows_sample" in fields:
            extra["verify_rows_sample"] = True
        return SpecPlan(
            effective_k=min(SPEC_K, int(requested_k)),
            lanes_per_request=SPEC_K + 1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=True,
            **extra,
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
        # decode calls and propose calls so far: under async scheduling an ordinary step N + 1 is submitted before step
        # N's propose, so the plain build is a token ahead of the history there; a promotion moves the plain state with
        # the history, so it waits for a propose with no step in flight (a step the runner does not overlap: after a
        # prefill / layout change, any speculative step)
        self._calls = [0, 0]
        self._spec_seed, self._spec_params = {}, None  # (sampled speculation: a request's seed, the uploaded params)
        # (the acceptance-aware policy's measurements: EMAs of tokens per spec user-step and of step times per kind)
        self._pay = self._new_pay()
        self.prompt_len = {}  # slot -> its prompt's length (generated tokens = history - prompt)
        self.taps_of = {}  # (P2d) slot -> its prefill's window taps (ResidentDFlashRows.capture_taps)
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
        if P2D:
            self.engine.on_taps, self.engine.entry_taps = self._on_taps, self._entry_taps
        return []

    def _entry_taps(self, parts, lo, H):
        """(P2d) the taps a prefix-cache entry at boundary H keeps: its window (H - window .. H) from the restored
        entry's parts and this pass's ring (from lo), or None when they do not cover it"""
        e = self.ae.spec
        start = max(0, H - e.window) // 64 * 64
        if lo <= start:
            return [e.capture_part(start, H)]
        old = window_parts(parts, start)
        if not old or old[0]["s0"] > start or old[-1]["hb"] != lo:
            return None
        return old + [e.capture_part(lo, H)]

    def _on_taps(self, r, lo, n, parts=None):
        """(P2d) a prompt just prefilled into plain row r: keep its window's taps (ResidentDFlashRows.capture_taps, ~34 ms)
        when the user may be promoted (a spec user free) and the pass covered the window — or, a prefix-cache hit
        (lo: the restored prefix), the entry's parts cover the window up to lo; a later promotion fills its drafter
        from them instead of re-prefilling the window (~0.36 s at 2K+ tokens)"""
        import time

        t0 = time.perf_counter()
        self.taps_of.pop(r, None)
        e = self.ae.spec
        start = max(0, n - e.window) // 64 * 64
        if len(self.live) >= SPEC_USERS:
            return
        if lo <= start:
            self.taps_of[r] = e.capture_taps(n)
        else:
            old = window_parts(parts, start)
            if not old or old[0]["s0"] > start or old[-1]["hb"] != lo:
                return
            rec = e.capture_taps(n, s0=lo)
            rec["parts"] = old + rec["parts"]
            self.taps_of[r] = rec
        self._lap("capture_taps", t0)

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
            self._moved_plain(r)
        self.spec_of.clear()

    def _moved_plain(self, r):
        """(Strata S3) the rows adapter's position tracking of a user just moved into its plain row: its state there
        consumed history[:-1] (verify steps do not update the tracking)"""
        h = self.history.get(r)
        if h:
            self.pos_of[r], self.dev_tok[r] = len(h) - 2, False
            self.seen.pop(r, None)

    @staticmethod
    def _new_pay():
        # (out: EMA of finished requests' output tokens; refill_tpt: EMA of refill ms per refilled token; gate: the
        # promotion gate's last verdict, for its log line)
        return dict(
            acc=None,
            spec_ms={},
            plain_ms={},
            verifies=0,
            last_t=None,
            kind=None,
            n=0,
            hold=0,
            excl=0.0,
            ctx=0,
            out=None,
            refill_tpt=None,
            gate=True,
        )

    def _note_output(self, n_out):
        """a finished request's output length, for the promotion gate's expected remaining tokens"""
        old = self._pay["out"] or PROMOTE_OUT
        self._pay["out"] = old + OUT_ALPHA * (n_out - old)

    def _step_ms(self, kind, n, ctx):
        """a step's estimated ms: the live count's estimate at no context + the context's share (PLAIN_CTX_MS /
        SPEC_CTX_MS)"""
        pay = self._pay
        if kind == "verify":
            return pay["spec_ms"].get(n, SPEC_MS_DEFAULT(n)) + SPEC_CTX_MS * ctx
        return pay["plain_ms"].get(n, PLAIN_MS_DEFAULT(n)) + PLAIN_CTX_MS * ctx

    def _promotion_pays(self, cands):
        """the promotion gate (PROMOTE_GATE): (verdict, its log line)"""
        pay, n = self._pay, max(1, len(self.live))
        ctx = sum(len(self.history.get(r) or ()) for r in self.live)
        plain, spec = self._step_ms("plain", n, ctx), self._step_ms("verify", n, ctx)
        save = plain - spec / PROMOTE_ACC  # (ms a token, per user)
        out = pay["out"] or PROMOTE_OUT
        left = [
            max(out - (len(self.history[r]) - self.prompt_len.get(r, len(self.history[r]))), out / 4) for r in cands
        ]
        window = getattr(getattr(self.ae, "spec", None), "window", 2048)
        # (P2d: a user with usable prefill taps costs a fill, ~15-20 ms; the others the window's PP pass)
        quick = [r for r in cands if taps_usable(self.taps_of.get(r), len(self.history[r]) - 1, window)]
        toks = sum(min(len(self.history[r]) - 1, window) for r in cands if r not in quick)
        refill = REFILL_MS_FIXED + (pay["refill_tpt"] or REFILL_MS_PER_TOKEN) * toks + P2D_FILL_MS * len(quick)
        gain, cost = sum(left) * save, refill * n
        line = (
            f"{len(cands)} of {n} users: ~{sum(left) / len(cands):.0f} tokens left each x {save:.1f} ms saved a token "
            f"= {gain:.0f} user-ms vs a {refill:.0f} ms refill ({toks} tokens) x {n} users = {cost:.0f}"
        )
        return gain > cost, line

    def _measure(self, spec_accepts):
        """update the policy's EMAs with the step just committed (its kind and live count from its decode call, its time
        since the previous propose); whether speculation pays: its tokens per user-ms >= a plain step's x PAY_MARGIN
        (True until PAY_MIN verify steps are in)"""
        import time

        pay, now = self._pay, time.perf_counter()
        ema = lambda old, x, a=0.1: x if old is None else old + a * (x - old)  # noqa: E731  (tokens: no outliers)
        # (a promotion's refill and a prefill between the steps are no step's cost: excluded)
        dt_ms = None if pay["last_t"] is None else (now - pay["last_t"] - pay.get("excl", 0.0)) * 1000
        pay["last_t"], pay["excl"] = now, 0.0
        n, ctx = max(1, pay["n"]), pay.get("ctx", 0)
        if pay.pop("skip", False):  # (a step across a prefill: no time sample; its acceptance still counts)
            dt_ms = -1.0
        if dt_ms is not None and dt_ms < 5000:  # (an idle gap is no step)
            # (the time at no context: the estimates' unit; a sample under 1 / OUTLIER of the default model is no
            # device step — the fake engine's microseconds — and is skipped like a high outlier)
            if pay["kind"] == "verify" and spec_accepts:
                x = dt_ms - SPEC_CTX_MS * ctx
                if x >= SPEC_MS_DEFAULT(n) / OUTLIER:
                    _store(pay["spec_ms"], n, robust_ema(pay["spec_ms"].get(n), x, SPEC_MS_DEFAULT(n)))
                pay["acc"] = ema(pay["acc"], sum(spec_accepts) / len(spec_accepts))
                pay["verifies"] += 1
            elif pay["kind"] == "plain":
                x = dt_ms - PLAIN_CTX_MS * ctx
                if x >= PLAIN_MS_DEFAULT(n) / OUTLIER:
                    _store(pay["plain_ms"], n, robust_ema(pay["plain_ms"].get(n), x, PLAIN_MS_DEFAULT(n)))
        # (before PAY_MIN verify steps: the acceptance at least PROMOTE_ACC — a probe; an unmeasured live count: the
        # default model's step time — users arriving together were promoted one by one at 1-4 live users and the
        # 8-user probe ran its PAY_MIN verify steps at ~60 ms before PLAIN: natural c8 41.7 vs 50.7 tok/s per user plain,
        # rc-20261124j; now the live count that does not pay goes PLAIN at once)
        probing = pay["verifies"] < PAY_MIN or pay["acc"] is None
        pay["acc_used"] = max(pay["acc"] or 0.0, PROMOTE_ACC) if probing else pay["acc"]
        return pay["acc_used"] / self._step_ms("verify", n, ctx) >= PAY_MARGIN / self._step_ms("plain", n, ctx)

    def _pay_line(self):
        pay, n, ctx = self._pay, max(1, self._pay["n"]), self._pay.get("ctx", 0)
        return (
            f"{n} users, {ctx} context tokens: {pay.get('acc_used') or 0:.2f} tokens per "
            f"{self._step_ms('verify', n, ctx):.1f} ms "
            f"speculating vs 1 per {self._step_ms('plain', n, ctx):.1f} ms plain"
        )

    def _promote(self):
        """every live plain user a decode call showed greedy into a free spec user (its drafter window refilled from
        its history: option B)"""
        import time

        t0 = time.perf_counter()
        free = self._free_spec()
        cands = [
            r for r in sorted(self.live) if r not in self.spec_of and r in self.greedy_ok and r not in self.sampled
        ]
        cands = cands[: len(free)]
        if not cands:
            return
        if PROMOTE_GATE:
            ok, line = self._promotion_pays(cands)
            if ok != self._pay["gate"]:
                logger.info(f"adaptive: promotion {'pays' if ok else 'would not pay'} ({line})")
                self._pay["gate"] = ok
            if not ok:
                return
        items = []
        for r in cands:
            u = free.pop(0)
            items.append((r, u, self.tables[r], self.history[r]))
            self.spec_of[r], self.last_acc[r] = u, 1
        window = getattr(getattr(self.ae, "spec", None), "window", 2048)
        taps = {j: self.taps_of.pop(j) for j, _, _, _ in items if j in self.taps_of}
        quick = {j for j, _, _, h in items if taps_usable(taps.get(j), len(h) - 1, window)}
        toks = sum(min(len(h) - 1, window) for j, _, _, h in items if j not in quick)
        self.ae.move_to_spec_many(items, taps=taps)  # (P2d fills, then one batched refill for the others)
        dt = time.perf_counter() - t0
        self._pay["excl"] += dt
        if toks:  # (the PP-pass refills' ms per token; the P2d fills' share taken out)
            tpt = max(0.0, dt * 1000 - REFILL_MS_FIXED - P2D_FILL_MS * len(quick)) / toks
            self._pay["refill_tpt"] = robust_ema(self._pay["refill_tpt"], tpt, REFILL_MS_PER_TOKEN, a=0.3)

    @staticmethod
    def _non_speculable(sp, i):
        """the runner's test (TTModelRunner._request_is_speculable): temperature != 0 or a penalty"""
        val = lambda name, default: (
            getattr(sp, name, default)
            if not isinstance(getattr(sp, name, default), (list, tuple))
            else getattr(sp, name)[i]
        )  # noqa: E731
        return (
            (float(val("temperature", 0.0) or 0.0) != 0.0 and not SPEC_SAMPLE)
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
        got = dict(zip(feed, self.engine.tokens_from(self.engine.read_tokens(list(feed), False))))
        for r, (_, p) in feed.items():  # (Strata S3: the rows adapter's position tracking and the device's draws)
            self.pos_of[r], self.dev_tok[r] = int(p), bool(self.engine.device_sampling)
            if getattr(self.engine, "swap", None) is not None:
                tail = self.seen.setdefault(r, {})
                tail[int(p) + 1] = got[r]
                for old in [k for k in tail if k < int(p) - 3]:
                    del tail[old]
        return got

    # ------------------------------------------------------------------ prefill
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, start_pos=None, empty_slots=None, **kwargs):
        """every prompt through the rows adapter into its plain row, as the batch profile prefills (the TT chunk core's
        chunks and gangs, batched PP passes, the host prefix cache); a greedy user moves into the spec build at a later
        propose (_promote), once a decode call has shown its sampling parameters"""
        import time

        t0 = time.perf_counter()
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
        # (a prefill between two decode steps is no step's cost either: excluded from the policy's step times, and the
        # step across it not sampled — its device time outlasts the call: plain estimates 21-22 ms vs 16.5-20 at 1-8
        # users let 6-7 users arriving together pass the step-time model, rc-20261124l)
        self._pay["excl"] = self._pay.get("excl", 0.0) + time.perf_counter() - t0
        self._pay["skip"] = True
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
        self._calls[0] += 1
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
        self._pay["ctx"] = sum(len(self.history.get(r) or ()) for r in rows.values())
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
            out = self._spec_row0(rows, tokens.reshape(B, -1)[:, 0], pos[:, 0], B, sp)
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

    def _spec_sampling(self, rows, sp):
        """(sampled speculation, RESIDENT_SPEC_SAMPLE) the spec build's draw parameters: each spec user's request's
        (temperature, top_k, top_p, seed) on its R rows (an unseeded request keeps one seed while it lives, so its rows
        and steps draw from one stream), greedy for idle users; uploaded when they change"""
        if not SPEC_SAMPLE or sp is None:
            return
        import numpy as np

        U, R = SPEC_USERS, 4
        per = [(0.0, 1, 1.0, 0)] * U
        for i, r in rows.items():
            u = self.spec_of.get(r)
            if u is None:
                continue
            seed = sp.seed[i] if getattr(sp, "seed", None) is not None else None
            if seed is None:
                seed = self._spec_seed.setdefault(r, int(np.random.randint(0, 2**31)))
            per[u] = (float(sp.temperature[i]), int(sp.top_k[i]), float(sp.top_p[i]), int(seed))
        if per != getattr(self, "_spec_params", None):
            self.ae.spec.m.set_sampling(rows=[p for p in per for _ in range(R)])
            self._spec_params = per

    def _spec_row0(self, rows, tok0, pos0, B, sp=None):
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
            self._spec_sampling(rows, sp)
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
                self._moved_plain(r)
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
            self._spec_sampling(rows, sp)
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
        self._calls[1] += 1
        if self.mode == "SPEC" and self._calls[0] == self._calls[1]:
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
    def preempt_request(self, vslot, token_ids=None):
        """(Strata S3) a preempted request's state to the swap store (ResidentRowsAdapter.preempt_request), keyed by this
        class's history (exact: every committed token, ahead of the runner's record): a spec user is moved back to its
        plain row first (its state there consumed history[:-1]; verify steps do not update the rows adapter's position
        tracking, so it is set here); its resume restores into a plain row (a later promotion may move it again)"""
        r = self.slots.res_of.get(int(vslot))
        h = self.history.get(r) if r is not None else None
        if r is not None and h and getattr(self.engine, "swap", None) is not None:
            if r in self.spec_of:
                u = self.spec_of.pop(r)
                self.ae.move_to_plain(u, r, self.tables[r], self.last_acc.get(r, 1))
                self.ae._set_spec_table(u, [0])
                self.last_acc[r], self._plain_stale = 1, True
                self._moved_plain(r)
            if token_ids is None or len(h) > len(token_ids):
                token_ids = list(h)
        super().preempt_request(vslot, token_ids)

    def release_request(self, vslot):
        r = self.slots.res_of.get(int(vslot))
        super().release_request(vslot)
        if r is not None:
            if r in self.spec_of:
                u = self.spec_of.pop(r)
                self.ae._set_spec_table(u, [0])
            self.taps_of.pop(r, None)
            h, plen = self.history.pop(r, None), self.prompt_len.pop(r, None)
            if h is not None and plen is not None and len(h) > plen:
                self._note_output(len(h) - plen)
            self.last_acc.pop(r, None)
            self.sampled.discard(r)
            self._spec_seed.pop(r, None)
            self.greedy_ok.discard(r)
