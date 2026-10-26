# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive verification length for the vLLM adapter (``DSV41_VLLM_SPEC_ADAPT``).

The demo's ``tt.spec_model.AdaptiveSpec`` owns one resident traced ``SpecRunner`` per candidate k (shared drafter, rings and pool) and picks the runner of a round from the draft head's confidence.
The vLLM adapter drives the rounds from the plugin: ``propose_draft_tokens`` hands out the drafts of the last round together with ``num_valid`` (how many of them the plugin will verify), the
following ``_spec_verify`` runs the block the plugin built. ``AdaptiveVerify`` wraps an ``AdaptiveSpec`` for that protocol:

  * ``choose`` (called once per ``propose_draft_tokens``): the k of the next round, batch-uniform (the device block has one width), from the confidence of the drafts of the offered rows and the
    profiled round time of every runner (``E[tokens](k) / round_ms(k)``, the demo's rule);
  * ``runner_for`` (called by ``_spec_verify``): the runner of the block width the plugin sent, which verifies ``k + 1`` columns and returns ``argmax_ids`` of that width;
  * everything the adapter used from a fixed-k ``SpecRunner`` (``k``, ``release``, ``seed``, ``seed_from_prefill``, ``taps_ready``, ``d_final``, ``conf_final``, ``dev_first``, ``samp``) is forwarded to
    the seed runner (k = smallest candidate) or to every runner (``release``).
"""

import os

import torch


class AdaptiveVerify:
    def __init__(self, ad, log=print):
        self.ad = ad
        self.runners = dict(ad.runners)
        self.ks = sorted(self.runners)
        self.k = self.ks[-1]  # the plugin's block width (num_speculative_tokens) is the largest candidate
        self.n = self.k + 1
        self.log = log
        self.k_hist = {}
        self.ema = float(os.environ.get("DSV41_SPEC_TIME_EMA", "0.2"))
        # a k that does not run is not measured: its estimate goes stale (a startup or prefill-contaminated measurement can then pin the choice to the other k for the whole run). ``probe_every`` bounds the
        # staleness: a k not run for that many rounds is run once (a probe round, one in ``probe_every`` at worst) and its fresh measurement REPLACES the stale estimate. 0 = no probes (the EMA only).
        self.probe_every = int(os.environ.get("DSV41_SPEC_PROBE_EVERY", "128"))
        self.since = {k: 0 for k in self.ks}  # rounds since the runner of k last ran
        self.probes = 0
        self._cur = None
        self._calib_off = False
        self.forced_k = int(
            os.environ.get("DSV41_VLLM_SPEC_FORCED_K", "0")
        )  # runner of a step without any draft (default: the smallest candidate; a debugging knob)

    # ---- runners ------------------------------------------------------------------------------------------------------------------------
    def all_runners(self):
        return [self.runners[k] for k in self.ks]

    @property
    def seedr(self):
        return self.ad.seedr

    @property
    def d_final(self):
        return self.seedr.d_final

    @property
    def conf_final(self):
        return self.seedr.conf_final

    @property
    def dev_first(self):
        return self.seedr.dev_first

    @property
    def samp(self):
        return self.runners[self.k].samp

    @property
    def times(self):
        return self.ad.times

    def release(self):
        self.ad.release()

    def prepare(self, base):
        """compile pass of EVERY runner, before the first trace of any of them exists (a tensor allocated after a captured trace can land on its scratch memory)."""
        for r in self.all_runners():
            r.prepare(torch.zeros(r.B, r.n, dtype=torch.long), base)

    # ---- seeding (forwarded to the seed runner; its _capture_all captures / calibrates every sibling) -----------------------------------------
    def taps_ready(self, lens, users=None):
        return self.seedr.taps_ready(lens, users)

    def seed(self, tokens, lens, first):
        out = self.seedr.seed(tokens, lens, first)
        self._after_seed()
        return out

    def seed_from_prefill(self, lens, first, users=None, draft=True):
        out = self.seedr.seed_from_prefill(lens, first, users, draft=draft)
        self._after_seed()
        return out

    def _after_seed(self):
        """round times of the runners (the calibration of the first capture), kept; later captures (after every prefill) do not calibrate again."""
        for k, r in self.runners.items():
            if k not in self.ad.times and r.round_ms is not None:
                self.ad.times[k] = r.round_ms
        for kv in os.environ.get("DSV41_SPEC_TIMES", "").split(","):
            if kv:
                k, v = kv.split(":")
                self.ad.times[int(k)] = float(v)
        if not self._calib_off and all(k in self.ad.times for k in self.ks):
            self._calib_off = True
            os.environ["DSV41_SPEC_CALIB"] = "0"
            self.log(f"spec adapt: round-time table (ms) {dict(sorted(self.ad.times.items()))}")

    # ---- scheduler ---------------------------------------------------------------------------------------------------------------------------
    def candidates(self, max_pos=None):
        """The candidate k's: those whose runner has a position bucket covering ``max_pos`` (a runner with a shorter ladder replays its whole-slab traces beyond it, which cost far more) when
        at least one does."""
        if max_pos is None:
            return list(self.ks)
        cov = [k for k in self.ks if self.runners[k].bucket_for(int(max_pos)) is not None]
        return cov if cov else list(self.ks)

    def choose(self, conf, active=None, max_pos=None):
        """k of the next round for the rows ``conf`` [R, 5] (draft acceptance probabilities of the offered rows); ``active`` [R] bool (default all); ``max_pos`` the highest position the block reaches."""
        cands = self.candidates(max_pos)
        if len(cands) == 1:
            return cands[0]
        if self.ad.policy.startswith("k"):
            return int(self.ad.policy[1:])
        if any(k not in self.ad.times for k in cands):
            return cands[-1]  # no round times yet: the fixed-k behaviour
        if active is None:
            active = torch.ones(conf.shape[0], dtype=torch.bool)
        if not bool(active.any()):
            return cands[0]
        if self.probe_every > 0:
            stale = [k for k in cands if self.since.get(k, 0) >= self.probe_every]
            if stale:
                self.probes += 1
                return max(stale, key=lambda k: self.since[k])
        E = self.ad.expected_tokens(conf, active)
        score = {k: E[k] / self.ad.times[k] for k in cands}
        best = max(score, key=score.get)
        cur = self._cur
        if cur is not None and cur in score and score[best] < score[cur] * (1 + self.ad.hyst):
            best = cur
        self._cur = best
        return best

    def runner_for(self, nv_max):
        """(k, runner) verifying the block of a step whose largest valid draft count is ``nv_max`` (0: no row carries a draft: the cheapest runner; a count that is not a candidate: the next larger)."""
        if nv_max <= 0:
            k = self.forced_k if self.forced_k in self.runners else self.ks[0]
        else:
            k = next((q for q in self.ks if q >= nv_max), self.ks[-1])
        return k, self.runners[k]

    def note_round(self, k, ms):
        """measured round time of runner k (ms): EMA into the profile."""
        t = self.ad.times
        stale = self.probe_every > 0 and self.since.get(k, 0) >= self.probe_every
        t[k] = (1.0 - self.ema) * t[k] + self.ema * ms if (k in t and not stale) else ms
        for q in self.ks:
            self.since[q] = 0 if q == k else self.since.get(q, 0) + 1
        self.k_hist[k] = self.k_hist.get(k, 0) + 1
