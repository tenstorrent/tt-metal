# SPDX-License-Identifier: Apache-2.0
"""Model-owned prefix hook of the Hy4 row-local prefix cache (HY4_PFX=1) for the vLLM TT plugin scheduler (program b45h, brief B45_1).

The plugin already has the policy: ``TTScheduler._hold_for_home_row`` (bounded wait of a request whose resident prefix sits on a mesh row with no free user, ``DSV41_HOME_ROW_HOLD_*``) and
``_match_waiting_to_free_rows`` (``DSV41_ROW_MATCH_*``). Both read the model through ``vllm_tt_plugin.prefix_hooks.peek_model_home`` -> ``hook.peek_home(...)``, which Hy4 did not provide, so they were
inert. This object answers ``peek_home`` over the engine's retained chains (``Hy4Engine.ret`` / ``ret_t`` / ``live``); it is pure bookkeeping (no device access, read only except the ``touch`` LRU refresh).

``plan_hit`` always returns (0, None): the hit itself stays inside ``prefill_forward`` (row-match admission ``Hy4Engine.assign_rows``), the plugin's own prefix-cache path remains off
(``--no-enable-prefix-caching``). ``cancel`` is a no-op (nothing is reserved). Registered by ``HYV4ForCausalLM.__init__`` only when HY4_PFX=1 and HY4_APC_HOOK=1 (default 0).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from models.demos.blackhole.hy4_preview.tt import hy4_sched_flags as SF


def fingerprint(tokens):
    """(length, 8-byte blake2b of the int32 bytes) of a prompt: the key that ties a scheduler request to the ``prefill_forward`` call that served it. The image plugin (a8962b0) passes ``request_ids`` to
    prefill_forward only under ``prefix_cache_hit_policy == "model"`` (model_runner.py:2683), which Hy4 does not declare (its own prefix cache is off), so the ids never arrive (checked in the pfxA / pfxB
    logs: 0.0 % usage hit although ``install_cached_tokens_report`` was installed); the prompt itself is available on both sides.
    """
    a = np.ascontiguousarray(np.asarray(tokens, dtype=np.int32))
    return (int(a.shape[0]), hashlib.blake2b(a.tobytes(), digest_size=8).hexdigest())


class Hy4ApcHook:
    wants_request_ids = True  # with plugin.patch (b45h): prefill_forward receives request_ids, so finished_ids can free the right row (without the patch the hook is exact one scheduler step late)

    def __init__(self, engine, rows=None):
        self.eng = engine
        self.rows = int(rows if rows is not None else len(engine.ret))
        self.rid_row = (
            {}
        )  # request id -> physical mesh row (filled by note_prefill when the plugin passes ``request_ids``)
        self.stats = dict(peeks=0, errors=0)
        self.cached = (
            {}
        )  # request id -> tokens of its prompt that were NOT recomputed (engine.last_cached), read by install_cached_tokens_report (only when the plugin passes request_ids)
        self.fp_done = (
            {}
        )  # fingerprint -> [(mesh row, cached tokens), ...] of prefill_forward calls not yet claimed by a scheduler request (oldest first)
        self.sched_peek = {}  # request id -> (fingerprint, peek_home hit at schedule time)
        self.report = dict(new=0, actual=0, peek=0, none=0, ridmap=0)

    # ---- plugin contract -------------------------------------------------------------------------------------------------------------------
    def plan_hit(self, request_id, prompt_token_ids, num_tokens, max_hit=None, **kw):
        return 0, None

    def cancel(self, request_id):
        return None

    def note_prefill(self, request_ids, phys_rows):
        """prefill_forward received ``request_ids`` (plugin policy "model") : remember which mesh row serves which request, so ``finished_ids`` frees the right row."""
        if not SF.enabled(
            "ridmap"
        ):  # flag ridmap (HY4_RIDMAP / live file, default off): the row map decides which row a finished request frees in free_per_row
            return
        for rid, r in zip(request_ids, phys_rows):
            self.rid_row[str(rid)] = int(r)
        if len(self.rid_row) > 4096:  # bounded: drop the oldest half
            for k in list(self.rid_row)[:2048]:
                del self.rid_row[k]

    def note_prefill_fp(self, prompts, phys_rows, cached):
        """prefill_forward without request_ids: remember (row, cached tokens) of each prompt by fingerprint; claimed by the scheduler wrapper in the same or the next ``update_from_output``."""
        for toks, r, c in zip(prompts, phys_rows, cached):
            self.fp_done.setdefault(fingerprint(toks), []).append((int(r), int(c)))
        if len(self.fp_done) > 512:  # bounded: drop the oldest keys
            for k in list(self.fp_done)[:256]:
                del self.fp_done[k]

    def note_scheduled(self, request_id, prompt_token_ids):
        """scheduler time (``schedule()`` returned the request as new): the read-only peek_home prediction of its cached tokens, kept as the fallback."""
        n = len(prompt_token_ids)
        fp = fingerprint(prompt_token_ids)
        try:
            hit = int(self.peek_home(request_id, prompt_token_ids, n)[0])
            if hit > 0 and getattr(self.eng, "pfx_redo", False):
                hit = max(
                    0, hit - int(self.eng.pf_rows)
                )  # the last chunk of a hit is recomputed (HY4_PFX_REDO): the engine reports last_cached = hit - 512
        except Exception:  # noqa: BLE001
            self.stats["errors"] += 1
            hit = 0
        self.sched_peek[str(request_id)] = (fp, hit)
        if len(self.sched_peek) > 4096:
            for k in list(self.sched_peek)[:2048]:
                del self.sched_peek[k]
        return fp, hit

    def claim(self, request_id, prompt_token_ids):
        """-> (cached tokens, source) for a new request whose prefill has run: 'actual' (engine.last_cached of the call that served this prompt; by request id when the plugin passed ids, else by fingerprint),
        'peek' (schedule-time prediction) or 'none'. Side effect (flag ridmap): rid_row[request] = the mesh row that served it.
        """
        rid = str(request_id)
        c = self.cached.pop(rid, None)
        if c is not None:
            return int(c), "actual"
        fp, hit = self.sched_peek.pop(rid, (None, 0))
        if fp is None:
            fp = fingerprint(prompt_token_ids)
        lst = self.fp_done.get(fp)
        if lst:
            row, c = lst.pop(0)
            if not lst:
                del self.fp_done[fp]
            if SF.enabled("ridmap"):
                self.rid_row[rid] = row
                self.report["ridmap"] += 1
            return int(c), "actual"
        return int(hit), ("peek" if hit else "none")

    def free_per_row(self, finished_ids=None):
        """1 when the row has no live request (a request the scheduler finished but the model was not told yet counts as gone), else 0."""
        gone = set()
        for rid in finished_ids or ():
            r = self.rid_row.get(str(rid))
            if r is not None:
                gone.add(r)
        return [1 if (not self.eng.live[r] or r in gone) else 0 for r in range(self.rows)]

    def peek_home(self, request_id, prompt_token_ids, num_tokens, finished_ids=None, touch=False):
        """-> (resident_hit_tokens, home_row, free_users_per_row). hit = chunk aligned, at least one token left (Hy4Engine.hit_len rule) on the row with the longest match; a tie prefers a free row.
        Hit 0 / home -1 when the cache is off (HY4_PFX_FILE switch) or nothing matches."""
        self.stats["peeks"] += 1
        free = self.free_per_row(finished_ids)
        eng = self.eng
        if not (eng.pfx and eng._pfx_on()):
            return 0, -1, free
        n = int(num_tokens)
        toks = np.asarray(prompt_token_ids[:n], dtype=np.int32)
        if len(toks) == 0:
            return 0, -1, free
        A = eng.pf_rows
        best, home = 0, -1
        for r in range(self.rows):
            h = (min(eng.match_len(r, toks), len(toks) - 1) // A) * A
            if h > best or (h == best and h > 0 and free[r] and not free[home]):
                best, home = h, r
        if touch and home >= 0 and best > 0:
            eng.ret_t[
                home
            ] = (
                time.time()
            )  # tip protection: a held prompt's chain is the last candidate for the LRU overwrite of a free row
        return int(best), int(home), free


def install_cached_tokens_report(hook, scheduler_cls=None):
    """usage.prompt_tokens_details.cached_tokens for the row-local cache (vLLM's own prefix cache is off, so it always said 0). Python only, idempotent, never raises into the scheduler.

    1. ``schedule()``: every request that is new in the step gets the read-only ``peek_home`` hit recorded (the scheduler-time value) and its prompt fingerprint.
    2. ``update_from_output()`` (the model has run the step's prefill by then): the prefill_stats of every new request get the engine's own number (tokens not recomputed, ``engine.last_cached`` of the call that
       served the same prompt, matched by fingerprint because the plugin does not pass request ids) or, when no call matches, the scheduler-time peek. vLLM then commits it (``take_prefill_stats``).
    Feature flag ``cached_tokens`` (HY4_CACHED_TOKENS / live file): off = both wrappers are pass-through.
    """
    if scheduler_cls is None:
        from vllm_tt_plugin.scheduler import TTScheduler as scheduler_cls
    if getattr(scheduler_cls, "_hy4_cached_patched", False):
        return False
    orig_sched = getattr(scheduler_cls, "schedule", None)
    orig = scheduler_cls.update_from_output

    def schedule(self, *a, **k):
        out = orig_sched(self, *a, **k)
        if SF.enabled("cached_tokens") or SF.enabled("ridmap"):
            try:
                for nr in getattr(out, "scheduled_new_reqs", ()) or ():
                    toks = getattr(nr, "prompt_token_ids", None)
                    if toks:
                        hook.note_scheduled(nr.req_id, toks)
            except Exception:  # noqa: BLE001
                hook.stats["errors"] += 1
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        try:
            if SF.enabled("cached_tokens") or SF.enabled("ridmap"):
                for nr in getattr(scheduler_output, "scheduled_new_reqs", ()) or ():
                    toks = getattr(nr, "prompt_token_ids", None)
                    c, src = hook.claim(nr.req_id, toks if toks is not None else ())
                    hook.report["new"] += 1
                    hook.report[src] += 1
                    req = self.requests.get(nr.req_id)
                    ps = getattr(req, "prefill_stats", None)
                    if SF.enabled("cached_tokens") and c and ps is not None:
                        c = max(0, min(int(c), int(ps.num_prompt_tokens)))
                        ps.num_cached_tokens = ps.num_local_cached_tokens = c
                        ps.num_computed_tokens = int(ps.num_prompt_tokens) - c
        except Exception:  # noqa: BLE001
            hook.stats["errors"] += 1
        return orig(self, scheduler_output, model_runner_output)

    if orig_sched is not None:
        scheduler_cls.schedule = schedule
    scheduler_cls.update_from_output = update_from_output
    scheduler_cls._hy4_cached_patched = True
    return True
