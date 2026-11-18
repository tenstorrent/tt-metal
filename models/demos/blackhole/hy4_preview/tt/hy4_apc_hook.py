# SPDX-License-Identifier: Apache-2.0
"""Model-owned prefix hook of the Hy4 row-local prefix cache (HY4_PFX=1) for the vLLM TT plugin scheduler (program b45h, brief B45_1).

The plugin already has the policy: ``TTScheduler._hold_for_home_row`` (bounded wait of a request whose resident prefix sits on a mesh row with no free user, ``DSV41_HOME_ROW_HOLD_*``) and
``_match_waiting_to_free_rows`` (``DSV41_ROW_MATCH_*``). Both read the model through ``vllm_tt_plugin.prefix_hooks.peek_model_home`` -> ``hook.peek_home(...)``, which Hy4 did not provide, so they were
inert. This object answers ``peek_home`` over the engine's retained chains (``Hy4Engine.ret`` / ``ret_t`` / ``live``); it is pure bookkeeping (no device access, read only except the ``touch`` LRU refresh).

``plan_hit`` always returns (0, None): the hit itself stays inside ``prefill_forward`` (row-match admission ``Hy4Engine.assign_rows``), the plugin's own prefix-cache path remains off
(``--no-enable-prefix-caching``). ``cancel`` is a no-op (nothing is reserved). Registered by ``HYV4ForCausalLM.__init__`` only when HY4_PFX=1 and HY4_APC_HOOK=1 (default 0).
"""
from __future__ import annotations

import time

import numpy as np


class Hy4ApcHook:
    wants_request_ids = True  # with plugin.patch (b45h): prefill_forward receives request_ids, so finished_ids can free the right row (without the patch the hook is exact one scheduler step late)

    wants_request_ids = True  # with plugin.patch (b45h) prefill_forward receives request_ids, so finished_ids frees the right row; without it the hook is one scheduler step late

    def __init__(self, engine, rows=None):
        self.eng = engine
        self.rows = int(rows if rows is not None else len(engine.ret))
        self.rid_row = (
            {}
        )  # request id -> physical mesh row (filled by note_prefill when the plugin passes ``request_ids``)
        self.stats = dict(peeks=0, errors=0)

    # ---- plugin contract -------------------------------------------------------------------------------------------------------------------
    def plan_hit(self, request_id, prompt_token_ids, num_tokens, max_hit=None, **kw):
        return 0, None

    def cancel(self, request_id):
        return None

    def note_prefill(self, request_ids, phys_rows):
        """prefill_forward received ``request_ids`` (plugin policy "model") : remember which mesh row serves which request, so ``finished_ids`` frees the right row."""
        for rid, r in zip(request_ids, phys_rows):
            self.rid_row[str(rid)] = int(r)
        if len(self.rid_row) > 4096:  # bounded: drop the oldest half
            for k in list(self.rid_row)[:2048]:
                del self.rid_row[k]

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
