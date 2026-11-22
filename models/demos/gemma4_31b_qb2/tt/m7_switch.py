# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""M7's batch-level speculate / plain switch (bench/resident/m7/M7_REBASE_v2_ADDENDUM_2.md §1, M7_BUILD_PREREG_v1 §3 and
§2.2): one decision for every request running together, from measured step times and recent acceptance. Host logic
only; the M7 class feeds it each step it ran and asks it which mode the next step takes.

- G = T * s / it: T is the mean over live users of each user's tokens per SPEC iteration (its accepted drafts + 1) over
  its last `window` SPEC iterations; s is the PLAIN step's wall time at the current (R, live count), measured on PLAIN
  steps (seeded from a table); it is the SPEC iteration's wall time, measured.
- SPEC -> PLAIN when G <= `leave` over the last `window` iterations (outside a dwell).
- PLAIN -> SPEC after n_r PLAIN steps. That opens a dwell of max(32, J) SPEC iterations, judged at its end on the
  iterations J - 15 .. J (prereg v3 §5.2: a holed ring gets J iterations to heal). A failed dwell (G <= `leave`) goes
  back to PLAIN and doubles n_r up to `n_r_cap`; a passed one resets n_r to `n_r0`. reentry=False never leaves PLAIN
  (a block whose cap is below 256, or c1 with J8 > 32: prereg v3 §5.3).
- A band change resets n_r to `n_r0` and starts its PLAIN count over. Membership changes do nothing (no probe storm).
- Steps the runner forces to an ordinary decode (a penalized or host-sampled row) are counted apart; they are neither
  SPEC iterations nor the switch's PLAIN steps.
- `pin` ("spec" / "plain", from GEMMA4_M7_SWITCH_PIN, gates only) fixes the mode; it is reported in every decision.
J, the n_r cap and reentry are frozen from H5's hole-length runs (prereg v3 §5.2-5.3) and passed in.
"""

import os
from collections import Counter, deque

SPEC, PLAIN = "spec", "plain"


class TSwitch:
    def __init__(self, *, s_table=None, pin=None, n_r0=256, n_r_cap=4096, J=16, reentry=True, window=16, leave=1.02,
                 s_ewma=0.1):
        pin = pin if pin is not None else (os.environ.get("GEMMA4_M7_SWITCH_PIN") or None)
        if pin not in (None, SPEC, PLAIN):
            raise ValueError(f"GEMMA4_M7_SWITCH_PIN must be spec or plain, got {pin!r}")
        if J not in (8, 16, 32, 64) or n_r0 > n_r_cap or window < 1:
            raise ValueError(f"switch parameters: J {J} (8 / 16 / 32 / 64), n_r {n_r0}..{n_r_cap}, window {window}")
        self.pin, self.n_r0, self.n_r_cap, self.reentry = pin, n_r0, n_r_cap, reentry
        self.dwell, self.judged = max(32, J), (max(0, J - 16), J)  # iterations J - 15 .. J, 0-based
        self.window, self.leave, self.s_ewma = window, leave, s_ewma
        self.s = dict(s_table or {})  # (R, live) -> PLAIN step seconds
        self.mode = pin or SPEC
        self.band_id, self.n_r, self.plain_steps = None, n_r0, 0
        self.dwelling, self.dwell_iters = False, []  # SPEC iterations of the current dwell: (G inputs)
        self.its = deque(maxlen=window)  # (wall seconds, R, live) of the last SPEC iterations
        self.tokens = {}  # user -> deque of tokens per SPEC iteration
        self.counters = Counter()

    # ------------------------------------------------------------------ inputs

    def band(self, band_id):
        if band_id != self.band_id:
            if self.band_id is not None:
                self.counters["band_changes"] += 1
            self.band_id, self.n_r, self.plain_steps = band_id, self.n_r0, 0

    def plain_step(self, wall_s, R, live):
        """a PLAIN step ran (the switch's own mode); returns the next step's mode"""
        key = (R, live)
        self.s[key] = wall_s if key not in self.s else (1 - self.s_ewma) * self.s[key] + self.s_ewma * wall_s
        self.counters["plain_steps"] += 1
        if self.pin:
            return self.mode
        self.plain_steps += 1
        if self.mode == PLAIN and self.reentry and self.plain_steps >= self.n_r:
            self._enter()
        return self.mode

    def spec_iteration(self, wall_s, accepted, R, live):
        """a SPEC iteration ran; accepted = {user: drafts accepted in it}; returns the next step's mode"""
        self.counters["spec_iterations"] += 1
        self.its.append((wall_s, R, live))
        for user, a in accepted.items():
            self.tokens.setdefault(user, deque(maxlen=self.window)).append(a + 1)
        for user in [u for u in self.tokens if u not in accepted]:
            del self.tokens[user]  # a user no longer live leaves the mean
        if self.pin:
            return self.mode
        if self.dwelling:
            self.dwell_iters.append(self.gain(R, live))
            if len(self.dwell_iters) >= self.dwell:
                judged = [g for g in self.dwell_iters[self.judged[0]: self.judged[1]] if g is not None]
                passed = bool(judged) and sum(judged) / len(judged) > self.leave
                self.dwelling, self.dwell_iters = False, []
                if passed:
                    self.n_r = self.n_r0
                    self.counters["dwells_passed"] += 1
                else:
                    self.n_r = min(2 * self.n_r, self.n_r_cap)
                    self.counters["dwells_failed"] += 1
                    self._leave()
            return self.mode
        g = self.gain(R, live)
        if g is not None and len(self.its) >= self.window and g <= self.leave:
            self._leave()
        return self.mode

    def forced_step(self):
        self.counters["forced_ordinary_steps"] += 1

    # ------------------------------------------------------------------ the gain

    def T(self):
        means = [sum(d) / len(d) for d in self.tokens.values() if d]
        return sum(means) / len(means) if means else None

    def gain(self, R, live):
        """G = T * s / it at (R, live), or None while any term is unknown"""
        t, s = self.T(), self.s.get((R, live))
        its = [w for w, r, n in self.its if (r, n) == (R, live)]
        if t is None or s is None or not its:
            return None
        return t * s / (sum(its) / len(its))

    # ------------------------------------------------------------------ transitions

    def _leave(self):
        self.mode, self.plain_steps = PLAIN, 0
        self.counters["to_plain"] += 1

    def _enter(self):
        self.mode, self.dwelling, self.dwell_iters = SPEC, True, []
        self.tokens.clear()  # the dwell is judged on its own iterations, not on the SPEC period before PLAIN
        self.its.clear()
        self.counters["to_spec"] += 1

    def state(self):
        return {"mode": self.mode, "pin": self.pin, "band": self.band_id, "n_r": self.n_r, "dwelling": self.dwelling,
                "plain_steps": self.plain_steps, "reentry": self.reentry, **self.counters}
