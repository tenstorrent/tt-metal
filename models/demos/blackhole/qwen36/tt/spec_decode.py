# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding for Qwen3.6-27B using the built-in MTP drafter head.

SpeculativeDecoder.generate() is a thin client of MTPSpecEngine (tt/spec_engine.py): the engine owns
device buffers, the verify trace and MTP KV upkeep; this module owns acceptance (greedy, or exact
rejection sampling from tt/spec_sampling.py), per-user stop/freeze bookkeeping and statistics.

Per iteration, user u verifies [last, d_0..d_{K-1}] at pos..pos+K (B users x T = K+1 rows, user-major,
B*T <= 32). Accepted drafts plus the correction/bonus token are committed; the correction becomes the
next `last`. The first verify after prefill is the seed: row 0 holds the first generated token
(accepted_counts = 1, other rows filler) and nothing is counted. A finished user is FROZEN: it keeps
replaying its own slots with no stats and no sampler draws. TP (P150x4) only; B == max_batch_size.
Drafts are the device argmax under sampling too, so the accept test is u < p(d)."""
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.spec_engine import EAGER_RESEED_PROMPT_LEN, MAX_SPEC_ROWS, get_spec_engine
from models.demos.blackhole.qwen36.tt.spec_sampling import SpecSampler, SpecSamplingParams

__all__ = ["SpeculativeDecoder", "EAGER_RESEED_PROMPT_LEN", "MAX_SPEC_ROWS"]


class SpeculativeDecoder:
    """MTP speculative decode over B users, greedy or sampling.

    Greedy (``sampling=None``) reproduces the plain-decode greedy trajectory exactly. A
    ``SpecSamplingParams`` switches acceptance to exact speculative rejection sampling over the verify
    logits. One shared ``SpecSampler`` serves the LIVE users in order 0..B-1, so a seeded run is
    reproducible for a fixed batch composition. Default demo decode path (QWEN36_SPEC=0 opts out).
    Env knobs: QWEN36_SPEC_DRAFT_LEN (K), QWEN36_SPEC_TIMING, QWEN36_SPEC_CHECK_ANCHOR (engine).
    """

    # Field order of the QWEN36_SPEC_TIMING lines; engine.propose is timed as one phase ("draft"), so anchor/reseed are 0.
    _TPHASES = ("draft", "verify", "readback", "accept", "anchor", "reseed", "other", "total")
    # Tags timing lines so a warmup generate() and the timed one differ.
    _gen_calls = 0

    def __init__(
        self,
        model,
        page_tables,
        draft_len=None,
        stop_tokens=None,
        sampling: SpecSamplingParams | None = None,
    ):
        assert model.mtp is not None, "model has no MTP head (has_mtp / mtp.* weights?)"
        assert model.num_devices > 1, "SpeculativeDecoder is TP-only for now"
        self.model = model
        self.mesh = model.mesh_device
        self.args = model.args
        self.vocab = model.args.vocab_size
        # T = K+1 must be in ``TPAttention._SPEC_SDPA_L1_FIT`` ({4, 8, 12}) to take the fused SDPA plan;
        # any other T uses the per-row SDPA path.
        self.K = int(draft_len if draft_len is not None else os.environ.get("QWEN36_SPEC_DRAFT_LEN", 3))
        # torch int32 [B, num_blocks], one row per user; a [nb] vector is the B == 1 case.
        pt = page_tables if isinstance(page_tables, torch.Tensor) else torch.as_tensor(page_tables)
        if pt.dim() == 1:
            pt = pt.reshape(1, -1)
        assert pt.dim() == 2, f"page_tables must be [B, num_blocks], got {tuple(pt.shape)}"
        self.page_tables = pt.to(torch.int32).contiguous()
        self.B = int(self.page_tables.shape[0])
        assert self.B == self.args.max_batch_size, (
            f"page_tables has {self.B} rows but the model was built for max_batch_size="
            f"{self.args.max_batch_size}; the verify trace's row count is baked in at capture"
        )
        assert self.B * (self.K + 1) <= MAX_SPEC_ROWS, (
            f"B={self.B} users x T={self.K + 1} rows = {self.B * (self.K + 1)} > {MAX_SPEC_ROWS}: the verify "
            f"is one decode tile. Lower K (see the auto-K-by-batch policy) or lower the batch."
        )
        self._timing = bool(int(os.environ.get("QWEN36_SPEC_TIMING", "0")))
        self._tsum = {}  # phase -> summed seconds
        self._tn = 0  # iterations folded into _tsum
        self.stop_tokens = set(stop_tokens or [])
        self.mtp = model.mtp
        self.sampler = SpecSampler(sampling, self.vocab) if sampling is not None else None
        self.read_verify_logits = self.sampler is not None  # only the sampling accept step needs full logits
        # Mean target probability of the drafts the sampler evaluated (see stats()).
        self._p_draft_sum = 0.0
        self._p_draft_n = 0
        self.total_drafted = 0
        self.total_accepted = 0  # accepted DRAFT tokens (excludes the mandatory correction/bonus)
        self.iters = 0  # loop iterations (the seed verify is not counted)
        # Live (not FROZEN) user-iterations; denominator of every rate.
        self.user_iters = 0
        self.accept_hist = [0] * (self.K + 1)  # how often exactly j drafts were accepted
        self.depth_hits = [0] * self.K  # depth_hits[j] = user-iterations that accepted draft j
        self.zero_accept = 0  # user-iterations with no accepted draft (still worth 1 token)
        self.mtp_extra_steps = 0  # drafter forwards spent on KV maintenance (reseed)
        self.prefill_time = 0.0  # set by generate(): TTFT (prefill + MTP warm + seed verify)
        self.decode_time = 0.0  # set by generate(): spec-loop wall-clock after the seed

        # Shared per model; rebuilt if these tables are wider than it was prepared for (cannot be clipped)
        # or its trace died with the KV caches.
        eng = get_spec_engine(model, self.B, self.K)
        stale = eng.prepared and (
            self.page_tables.shape[1] > eng.nb or (eng.captured and getattr(model, "_vfy_trace_id", None) is None)
        )
        if stale:
            eng.shutdown()
            eng = get_spec_engine(model, self.B, self.K)
        self.engine = eng
        eng.prepare(self.page_tables)
        eng.capture()

    # --------------------------------------------------------------------- #
    # Accept
    # --------------------------------------------------------------------- #
    def _accept_greedy(self, drafts, verify_ids):
        """Greedy acceptance of one user's matching prefix; returns the number of accepted drafts.

        ``verify_ids`` is that user's [T] argmax ids: verify_ids[j] is the base's token after row j, i.e.
        the one drafts[j] must equal. The correction verify_ids[m] becomes the next `last`.
        """
        K = len(drafts)
        m = K
        for j in range(K):
            if drafts[j] != verify_ids[j]:
                m = j
                break
        self.accept_hist[m] += 1
        for j in range(m):
            self.depth_hits[j] += 1
        if m == 0:
            self.zero_accept += 1
        return m

    def _accept_sample(self, drafts, vlogits, penalize_base=None):
        """Exact speculative rejection sampling over ONE user's verify logits; returns (m, token).

        ``vlogits`` is that user's [T, vocab] verify logits. Drafts are argmax (a delta proposal), so
        the accept test is ``u_j < p_j(d_j)``. ``next_token`` is the recovered token (m < K) or the bonus
        token (m == K) and becomes the next `last`. ``penalize_base`` is the presence-penalty set
        (``generated_so_far ∪ {last}``, None without presence penalty); the sampler adds ``drafts[:j]``
        per row. See spec_sampling.py."""
        m, next_tok, p_draft = self.sampler.accept(vlogits, drafts, penalize_base)
        self.accept_hist[m] += 1
        for j in range(m):
            self.depth_hits[j] += 1
        if m == 0:
            self.zero_accept += 1
        self._p_draft_sum += sum(p_draft)
        self._p_draft_n += len(p_draft)
        return m, next_tok

    def _pick_token(self, logits_row, penalize=None):
        """One token from a host 1-D logits row: argmax (greedy) or a sampler draw.

        Used at the two seed sites (prefill's first token, and the seed verify's row 0), where there is
        no draft to accept. The row may be padded past the vocab, so it is sliced first.

        ``penalize`` is the presence-penalty set: None for the first token, ``{first}`` for the seed
        verify's token. Unused when greedy.
        """
        row = logits_row.reshape(-1)[: self.vocab]
        if self.sampler is None:
            return int(row.float().argmax())
        return self.sampler.pick(row.float(), penalize)

    # --------------------------------------------------------------------- #
    # Per-iteration timing (QWEN36_SPEC_TIMING=1)
    # --------------------------------------------------------------------- #
    def _tick(self):
        """Fence the device, then take a host timestamp (dispatch is async; unfenced phases measure ~0)."""
        ttnn.synchronize_device(self.mesh)
        return time.perf_counter()

    def _log_iter_timing(self, row):
        """Log one iteration's breakdown and fold it into the mean (first 2 iterations excluded)."""
        row["other"] = row["total"] - sum(v for k, v in row.items() if k != "total")
        logger.info(
            f"[SPEC_TIMING] iter={self.iters} "
            + " ".join(f"{k}={row[k] * 1e3:.2f}" for k in self._TPHASES)
            + f" gen={self._gen_id}"
        )
        if self.iters >= 2:  # skip the 2 warmup iterations
            for k, v in row.items():
                self._tsum[k] = self._tsum.get(k, 0.0) + v
            self._tn += 1

    def _log_mean_timing(self):
        """Log the mean-over-iterations breakdown gathered under QWEN36_SPEC_TIMING=1."""
        if not self._timing or not self._tn:
            return
        mean = {k: self._tsum.get(k, 0.0) / self._tn for k in self._TPHASES}
        cpi = self.accept_rate() + 1.0  # per LIVE user
        tot = cpi * self.B  # tokens the batch commits per iteration with every user still live
        # Frozen users commit nothing at the same iteration cost, so the delivered rate uses the mean live users.
        live_mean = self.user_iters / max(1, self.iters)
        dlv = cpi * live_mean
        logger.info(
            f"[SPEC_TIMING] MEAN gen={self._gen_id} iters={self._tn} (excl 2 warmup) users={self.B} "
            + " ".join(f"{k}={mean[k] * 1e3:.2f}" for k in self._TPHASES)
            + f" committed_per_iter={cpi:.3f} tokens_per_iter_total={tot:.3f} "
            + f"tok_s={tot / max(mean['total'], 1e-9):.2f} "
            + f"live_users={live_mean:.3f} tokens_per_iter_delivered={dlv:.3f} "
            + f"delivered_tok_s={dlv / max(mean['total'], 1e-9):.2f}"
        )

    # --------------------------------------------------------------------- #
    # Generate
    # --------------------------------------------------------------------- #
    def generate(self, prompts, max_new_tokens):
        """Speculative generation over the batch; returns the generated ids (prompt excluded).

        ``prompts`` is one ``list[int]`` (B == 1; returns a flat list) or B ``list[list[int]]`` (returns
        B lists); lengths may differ. Sets self.prefill_time (TTFT: prefill + MTP warm + seed verify)
        and self.decode_time (the rest).
        """
        engine = self.engine
        B, K = self.B, self.K
        T = K + 1
        assert len(prompts) > 0, "generate() needs at least one prompt"
        _p0 = prompts[0]
        flat_in = not (isinstance(_p0, (list, tuple)) or (isinstance(_p0, torch.Tensor) and _p0.ndim >= 1))
        if flat_in:
            assert B == 1, f"a flat list[int] prompt needs B == 1, but this decoder carries {B} users"
            prompt_lists = [[int(t) for t in prompts]]
        else:
            prompt_lists = [[int(t) for t in pr] for pr in prompts]
        assert len(prompt_lists) == B, f"expected {B} prompts (one per page-table row), got {len(prompt_lists)}"
        Tp = [len(pr) for pr in prompt_lists]
        SpeculativeDecoder._gen_calls += 1
        self._gen_id = SpeculativeDecoder._gen_calls
        _t_start = time.perf_counter()
        _extra0 = engine.mtp_extra_steps

        if self.sampler is None:
            _samp = "sampling=greedy"
        else:
            _sp = self.sampler.params
            _samp = (
                f"sampling=temp={_sp.temperature} top_k={_sp.top_k} top_p={_sp.top_p} "
                f"presence={_sp.presence_penalty} seed={self.sampler.seed}"
            )
        eager_reseed = B == 1 and Tp[0] > engine.eager_reseed_len
        logger.info(
            f"[spec] gen={self._gen_id} users={B} T_prompt={Tp} K={K} rows={B * T} "
            f"reseed={'eager' if eager_reseed else 'batched'} max_new={max_new_tokens} {_samp}"
        )

        # Capacity: bounded by the speculative high-water slot (verify writes T slots per iteration);
        # the last live iteration starts at Tp + max_new - 2 and writes T more. max(1, ...) covers
        # max_new_tokens <= 1 (the seed verify still writes T slots).
        block_size = int(self.engine.block_size)
        nb = self.page_tables.shape[-1]
        _cap = nb * block_size
        for u in range(B):
            _hi = Tp[u] + K + max(1, max_new_tokens - 1)
            assert _hi < _cap, (
                f"user {u}: high-water slot {_hi} (prompt={Tp[u]}, max_new={max_new_tokens}, K={K}, "
                f"users={B}) does not fit the paged KV: {nb} blocks x {block_size} = {_cap} slots"
            )

        # last[u]: last committed token (KV not yet written) at pos[u]; counts[u]: tokens the previous verify committed.
        out = [[] for _ in range(B)]
        done = [False] * B
        self._out_sets = [set() for _ in range(B)]
        out_sets = self._out_sets
        first = []
        for u in range(B):
            logits = engine.prefill(u, prompt_lists[u], self.page_tables[u : u + 1])
            f = self._pick_token(logits, None)  # output is still empty: no presence penalty
            first.append(f)
            out[u].append(f)
            out_sets[u].add(f)
            # `first` can already be a stop token or satisfy max_new_tokens == 1.
            done[u] = f in self.stop_tokens or len(out[u]) >= max_new_tokens
        last = list(first)
        pos = list(Tp)
        counts = [1] * B
        drafts = [[] for _ in range(B)]  # offered drafts per user (none for the seed verify)
        nvalid = [0] * B
        seed = True
        _t_decode = None  # set once the seed verify (and its propose) is done

        while any(not d for d in done):
            live = [not d for d in done]
            n_live = sum(live)
            _tm = self._tick() if self._timing else 0.0
            tokens, positions = [], []
            for u in range(B):
                n = nvalid[u] if live[u] else 0
                tokens.append([last[u]] + drafts[u][:n] + [-1] * (K - n))
                positions.append([pos[u] + j for j in range(n + 1)] + [-1] * (K - n))
            res = engine.decode_forward(
                tokens,
                positions,
                num_valid_drafts=[nvalid[u] if live[u] else 0 for u in range(B)],
                accepted_counts=counts,
                page_table=self.page_tables,
                spec_mode="logits" if self.read_verify_logits else "argmax_ids",
            )
            ids, vlogits = res.argmax_ids, res.logits
            _t_verify = time.perf_counter() if self._timing else 0.0

            ctok = [[-1] * T for _ in range(B)]
            cpos = [[-1] * T for _ in range(B)]
            new_counts = [1] * B
            for u in range(B):
                if not live[u]:
                    # FROZEN: replay [last] at its own slot.
                    ctok[u][0], cpos[u][0] = last[u], pos[u]
                    continue
                if seed:
                    # Seed: row 0 is the base's prediction after `first`; nothing to accept.
                    m = 0
                    nd = 0
                    corr = (
                        int(ids[u, 0])
                        if self.sampler is None
                        else self._pick_token(vlogits[u, 0], torch.tensor([first[u]], dtype=torch.int64))
                    )
                else:
                    nd = nvalid[u]
                    if self.sampler is None:
                        m = self._accept_greedy(drafts[u][:nd], ids[u].tolist())
                        corr = int(ids[u, m])
                    else:
                        # Presence-penalty set for the first row: output so far plus `last`.
                        penalize_base = (
                            torch.tensor(sorted(out_sets[u] | {last[u]}), dtype=torch.int64)
                            if self.sampler.params.presence_penalty > 0
                            else None
                        )
                        m, corr = self._accept_sample(drafts[u][:nd], vlogits[u][: nd + 1], penalize_base)
                    self.total_accepted += m  # full m even if a stop truncates below
                new = drafts[u][:m] + [corr]
                # Drafts may be accepted past a stop token: emit only through the first one.
                stop_i = next((i for i, t in enumerate(new) if t in self.stop_tokens), None)
                if stop_i is not None:
                    new = new[: stop_i + 1]
                out[u].extend(new)
                out_sets[u].update(new)
                ctok[u][: len(new)] = new
                cpos[u][: len(new)] = [pos[u] + 1 + i for i in range(len(new))]
                new_counts[u] = len(new)
                if stop_i is not None or len(out[u]) >= max_new_tokens:
                    # FREEZE at the pre-commit last/pos.
                    done[u] = True
                    ctok[u] = [last[u]] + [-1] * (T - 1)
                    cpos[u] = [pos[u]] + [-1] * (T - 1)
                    new_counts[u] = 1
                else:
                    last[u] = new[-1]
                    pos[u] += len(new)
            counts = new_counts
            _t_accept = time.perf_counter() if self._timing else 0.0
            if not seed:
                self.iters += 1
                self.user_iters += n_live
                self.total_drafted += sum(nvalid[u] for u in range(B) if live[u])
            if all(done):
                if seed:
                    ttnn.synchronize_device(self.mesh)
                    self.prefill_time = time.perf_counter() - _t_start
                    _t_decode = time.perf_counter()
                break
            d = engine.propose_draft_tokens(K, ctok, cpos, counts, self.page_tables)
            drafts, nvalid = d.draft_token_ids.tolist(), d.num_valid.tolist()
            if self._timing and not seed:
                _t_end = self._tick()
                _dev, _rd = engine.last_verify_timing or (0.0, 0.0)
                self._log_iter_timing(
                    {
                        "draft": _t_end - _t_accept,  # engine.propose: anchor + reseed + draft
                        "verify": _dev,
                        "readback": _rd,
                        "accept": _t_accept - _t_verify,
                        "anchor": 0.0,
                        "reseed": 0.0,
                        "total": _t_end - _tm,
                    }
                )
            if seed:
                ttnn.synchronize_device(self.mesh)
                self.prefill_time = time.perf_counter() - _t_start  # TTFT: prefill + MTP warm + seed
                _t_decode = time.perf_counter()
                seed = False
        if _t_decode is None:  # every user finished at prefill: no verify ran
            self.prefill_time = time.perf_counter() - _t_start
            self.decode_time = 0.0
        else:
            ttnn.synchronize_device(self.mesh)
            self.decode_time = time.perf_counter() - _t_decode
        for u in range(B):
            engine.release(u)
        self.mtp_extra_steps += engine.mtp_extra_steps - _extra0
        self._log_mean_timing()
        res = [o[:max_new_tokens] for o in out]
        return res[0] if flat_in else res

    def accept_rate(self):
        """Mean accepted DRAFT tokens per USER per iteration (0..K); tokens/iter/user is this + 1."""
        return self.total_accepted / max(1, self.user_iters)

    def stats(self):
        """Acceptance breakdown. Mean acceptance alone cannot distinguish 'the drafter is weak at
        depth 3' from 'the first draft keeps aborting the iteration', which need different fixes.

        Every rate is per USER-ITERATION (there is one accept decision per user per loop iteration),
        so the numbers are comparable across batch sizes; ``tokens_per_iter_total`` is what the batch
        commits per iteration and is the number throughput scales with.

        FINISHED USERS DO NOT COUNT. A user that has stopped is FROZEN: its T rows still replay
        (the row count is baked into the trace) but its accept result is discarded, so it feeds
        neither ``user_iters`` nor the histograms nor ``total_drafted`` / ``total_accepted``. Every
        rate below is therefore over USEFUL work only, whatever order the users finish in.
        ``iters`` stays the raw loop-iteration count, and ``mean_live_users`` (= user_iters / iters)
        says how full the batch was on average. So ``tokens_per_iter_total`` (= B x
        committed_per_iter) is the rate with the batch FULL — the number throughput scales with —
        while ``tokens_per_iter_delivered`` (= mean_live_users x committed_per_iter) is what the run
        actually emitted per iteration. The two are equal when no user finishes early."""
        n = max(1, self.user_iters)
        return {
            "iters": self.iters,
            "n_users": self.B,
            "K": self.K,
            "accept_rate": self.accept_rate(),
            "committed_per_iter": self.accept_rate() + 1.0,
            "tokens_per_iter_total": self.B * (self.accept_rate() + 1.0),
            # Mean users still generating per iteration (== B until the first user finishes) and
            # the delivered rate that follows from it; see the docstring.
            "mean_live_users": self.user_iters / max(1, self.iters),
            "tokens_per_iter_delivered": (self.user_iters / max(1, self.iters)) * (self.accept_rate() + 1.0),
            # depth_rate[j] = P(draft j accepted), cumulative: [0] >= [1] >= ...
            "depth_rate": [h / n for h in self.depth_hits],
            # conditional[j] = P(draft j accepted | drafts 0..j-1 accepted) — isolates per-depth
            # drafter quality from the compounding of earlier rejections.
            "conditional": [self.depth_hits[j] / max(1, self.depth_hits[j - 1] if j else n) for j in range(self.K)],
            "hist": list(self.accept_hist),
            "zero_accept_rate": self.zero_accept / n,
            "mtp_extra_steps": self.mtp_extra_steps,
            # Sampling only: mean target probability of the drafts the sampler evaluated. A draft is
            # accepted with exactly that probability, so it is the per-draft acceptance odds.
            "mean_draft_target_prob": (
                (self._p_draft_sum / max(1, self._p_draft_n)) if self.sampler is not None else None
            ),
        }

    def log_stats(self, prefix="spec"):
        s = self.stats()
        logger.info(
            f"[{prefix}] {s['iters']} iters, users={s['n_users']}, K={s['K']}: "
            f"accept={s['accept_rate']:.2f}/{s['K']} -> {s['committed_per_iter']:.2f} committed tokens/iter/user, "
            f"{s['tokens_per_iter_total']:.2f} tokens/iter total"
        )
        if self.sampler is not None:
            sp = self.sampler.params
            logger.info(
                f"[{prefix}] sampling: temp={sp.temperature} top_k={sp.top_k} top_p={sp.top_p} "
                f"presence={sp.presence_penalty} "
                f"seed={self.sampler.seed} mean p_target(draft)={s['mean_draft_target_prob']:.3f}"
            )
        logger.info(f"[{prefix}] per-depth accept   : {[f'{x:.2f}' for x in s['depth_rate']]}")
        logger.info(f"[{prefix}] conditional accept : {[f'{x:.2f}' for x in s['conditional']]}")
        logger.info(f"[{prefix}] accepted histogram : {s['hist']} (index j = exactly j drafts accepted)")
        logger.info(f"[{prefix}] zero-accept iters  : {s['zero_accept_rate']:.1%} (still commit the pending token)")
        logger.info(f"[{prefix}] MTP reseed forwards: {s['mtp_extra_steps']}")
        return s
