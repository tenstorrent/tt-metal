# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact speculative (rejection) sampling for the MTP drafter's argmax drafts — host math, no ttnn.
Sampling counterpart to greedy accept in tt/spec_decode.py. Drafter proposes K tokens by argmax
(deterministic); verify returns T=K+1 target rows (row j for draft j's position, row K bonus);
this emits one token plus an accept length from the sampling-modified target. accept(): draw
u=rand(K) once (RNG stream independent of rejection depth); for j=0..K-1, p_j=dist(logits[j])
(temp, then top-k, then top-p) and accept d_j iff u[j]<p_j(d_j); first rejection at m samples a
recovered token from p_m with d_m zeroed and renormalized; if m=K sample a bonus from p_K. Rows
past the decision are never touched (lazy dist()). Identical to vLLM V1 RejectionSampler when
draft_probs is None. With q=delta_d, accept prob is p(d) and the residual is p with d zeroed, so
P(x)=p(x) exactly (even if p(d)=0); chaining row by row makes the whole sequence lossless. presence_penalty subtracts pp from logits of tokens already in the OUTPUT (generated only, not prompt) before temp/top-k/top-p; pp==0 is a no-op. Row j's set is generated_so_far ∪ {pending} ∪ drafts[:j] (bonus: drafts[:K]); caller passes penalize_base and accept adds drafts[:j]. Drafter is unpenalized (demoted tokens accepted less); losslessness holds because accept/recovery run on the penalized p. Top-p keep: keep[i] <=> cumsum[i]-prob[i] < top_p (top-1 always kept; prefix of sorted order; top_p==1 / top_k==0 disable). dist() returns sparse (idx, probs); top_k>0: one torch.topk then ops on <=top_k; top_k==0 and top_p==1: dense softmax over arange(vocab); top_k==0 and top_p<1: lse+topk(2048) is exact if prefix mass reaches top_p, else full sort (_topp_full_sorts). All draws go through self.gen (CPU torch.Generator from self.seed). temperature==0 is the caller's greedy path; SpecSamplingParams rejects it."""

from dataclasses import dataclass

import torch
from loguru import logger

# Size of the descending prefix used to make top-p exact without sorting the full row.
_TOPP_PREFIX = 2048


@dataclass(frozen=True)
class SpecSamplingParams:
    """Sampling knobs for one request (mirrors the vLLM subset the demo exposes)."""

    temperature: float  # must be > 0; temperature == 0 is the caller's greedy path
    top_k: int = 0  # 0 disables top-k, else keep the top_k largest logits
    top_p: float = 1.0  # 1.0 disables top-p, must be in (0, 1]
    # Subtracted from the logit of every token already in the OUTPUT, before temperature; 0 disables
    # it (see "Presence penalty" in the module docstring).
    presence_penalty: float = 0.0
    seed: int | None = None  # None -> SpecSampler draws a fresh seed once and records it

    def __post_init__(self):
        assert self.temperature > 0, f"temperature must be > 0 (0 is the caller's greedy path), got {self.temperature}"
        assert self.top_k >= 0, f"top_k must be >= 0 (0 disables top-k), got {self.top_k}"
        assert 0.0 < self.top_p <= 1.0, f"top_p must be in (0, 1], got {self.top_p}"
        assert self.presence_penalty >= 0, f"presence_penalty must be >= 0 (0 disables it), got {self.presence_penalty}"


class SpecSampler:
    """Exact speculative sampling over verify logits, for deterministic (argmax) drafts."""

    def __init__(self, params: SpecSamplingParams, vocab_size: int):
        assert vocab_size >= 1, f"vocab_size must be >= 1, got {vocab_size}"
        self.params = params
        self.vocab_size = int(vocab_size)
        seed = params.seed
        if seed is None:
            # torch.Generator.seed() draws a non-deterministic seed (random_device / clock)
            # without disturbing the global RNG. Recorded so a run can be replayed.
            seed = int(torch.Generator().seed())
        self.seed = int(seed)
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(self.seed)
        # No truncation at all -> the support is the whole vocabulary, in vocabulary order.
        self._dense = params.top_k == 0 and params.top_p == 1.0
        # Test/observability hook: how many times the exact-but-slow full sort was needed.
        self._topp_full_sorts = 0
        logger.info(
            f"SpecSampler: vocab={self.vocab_size} temperature={params.temperature} "
            f"top_k={params.top_k} top_p={params.top_p} seed={self.seed}"
        )

    # ---------------------------------------------------------------- distribution

    def dist(self, logits: torch.Tensor, penalize: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sampling-modified target distribution for one row of logits.

        ``logits`` is 1-D of length vocab_size (bf16 converted with .float()). ``penalize`` is an
        optional int64 1-D of UNIQUE token ids already in the output; when given and
        presence_penalty > 0 those logits are lowered by presence_penalty BEFORE temperature
        (ignored when the request has no presence penalty, so that path is bit-identical to a call
        without it). Returns ``(idx, probs)``: int64 and float32, same length n>=1, the SUPPORT of
        the distribution; probs sums to 1. Sorted descending when top_k>0 or top_p<1; for
        top_k==0 and top_p==1.0 it is the dense softmax over arange(vocab_size), unsorted.
        """
        params = self.params
        penalizing = penalize is not None and params.presence_penalty > 0
        if logits.dtype != torch.float32:
            logits = logits.float()  # .float() already copied, so the penalty can go in place
        elif penalizing:
            # A float32 row belongs to the CALLER (the verify block is read back once and its rows
            # are handed out as views), so never subtract in place: copy exactly once.
            logits = logits.clone()
        assert (
            logits.dim() == 1 and logits.shape[0] == self.vocab_size
        ), f"expected 1-D logits of length {self.vocab_size}, got {tuple(logits.shape)}"
        if penalizing:
            logits[penalize] -= params.presence_penalty

        # Dividing by exactly 1.0 is the identity in IEEE-754, so skip the O(vocab) pass.
        z = logits if params.temperature == 1.0 else logits / params.temperature

        # (1) temperature done. (2) top-k: one O(vocab) topk, everything else is <= top_k.
        if params.top_k > 0:
            vals, idx = torch.topk(z, min(params.top_k, self.vocab_size), sorted=True)
            probs = torch.softmax(vals, 0)  # (3) renormalized softmax over the survivors
            if params.top_p < 1.0:  # (4) + (5) top-p on the descending survivors
                n_keep, _ = self._top_p_keep(probs)
                return self._truncate(idx, probs, n_keep)
            return idx, probs

        # No top-k. Nothing to truncate at all: one dense softmax, vocabulary order.
        if params.top_p == 1.0:
            return torch.arange(self.vocab_size, dtype=torch.int64), torch.softmax(z, 0)

        # No top-k, top_p < 1: exp(vals - logsumexp) are the exact full-softmax probs of
        # the top 2048 tokens, so top-p is exact whenever its support fits in that prefix.
        lse = torch.logsumexp(z, 0)
        vals, idx = torch.topk(z, min(_TOPP_PREFIX, self.vocab_size), sorted=True)
        probs = torch.exp(vals - lse)
        n_keep, before_last = self._top_p_keep(probs)
        if before_last >= params.top_p:
            # The last prefix entry is already past top_p -> the keep set is a strict
            # prefix of the 2048 and nothing outside the prefix could have been kept.
            return self._truncate(idx, probs, n_keep)

        # Flat row: the top-p support runs past the prefix. Exact slow path.
        self._topp_full_sorts += 1
        full = torch.softmax(z, 0)
        sorted_probs, sorted_idx = torch.sort(full, descending=True)
        n_keep, _ = self._top_p_keep(sorted_probs)
        return self._truncate(sorted_idx, sorted_probs, n_keep)

    def _top_p_keep(self, probs: torch.Tensor) -> tuple[int, float]:
        """Top-p keep count for descending ``probs``, plus the prefix mass of the last entry.

        Keeps token ``i`` iff ``cumsum[i] - prob[i] < top_p``; since ``probs`` is
        descending that prefix mass is non-decreasing, so the keep set is a prefix.
        """
        cum = torch.cumsum(probs, 0)
        before = cum - probs
        n_keep = int((before < self.params.top_p).sum())
        # The top-1 token is always kept (prefix mass 0 < top_p); guard float paranoia.
        return max(n_keep, 1), float(before[-1])

    @staticmethod
    def _truncate(idx: torch.Tensor, probs: torch.Tensor, n_keep: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the first ``n_keep`` entries and renormalize them to sum to 1."""
        if n_keep < probs.numel():
            idx, probs = idx[:n_keep], probs[:n_keep]
        return idx, probs / probs.sum()

    # ---------------------------------------------------------------- support lookup

    def _support_pos(self, idx: torch.Tensor, token: int) -> int | None:
        """Position of ``token`` inside a support, or None if it was truncated away."""
        if self._dense and idx.numel() == self.vocab_size:
            return token if 0 <= token < self.vocab_size else None
        hits = (idx == token).nonzero()
        return None if hits.numel() == 0 else int(hits[0])

    def prob_of(self, dist: tuple[torch.Tensor, torch.Tensor], token: int) -> float:
        """Probability of ``token`` under a TARGET ``dist``, or 0.0 if outside its support."""
        idx, probs = dist
        pos = self._support_pos(idx, int(token))
        return 0.0 if pos is None else float(probs[pos])

    # ---------------------------------------------------------------- sampling

    def pick(self, logits: torch.Tensor, penalize: torch.Tensor | None = None) -> int:
        """Sample one token id from ``dist(logits, penalize)``."""
        idx, probs = self.dist(logits, penalize)
        pos = int(torch.multinomial(probs, 1, generator=self.gen)[0])
        return int(idx[pos])

    def _recover(self, dist: tuple[torch.Tensor, torch.Tensor], draft: int) -> int:
        """Sample from ``dist`` with the draft token's mass removed (rejection residual)."""
        idx, probs = dist
        pos = self._support_pos(idx, draft)
        residual = probs.clone()
        if pos is not None:
            residual[pos] = 0.0
        total = float(residual.sum())
        if total <= 0.0:
            # Unreachable in exact arithmetic: a rejection implies p(draft) < 1, so some
            # mass survives. Only float error can land here; degrade, never crash.
            if pos is None:
                return int(idx[int(probs.argmax())])
            other = probs.clone()
            other[pos] = -1.0
            best = int(other.argmax())
            return int(draft) if best == pos else int(idx[best])
        residual /= total
        return int(idx[int(torch.multinomial(residual, 1, generator=self.gen)[0])])

    def _penalize_row(self, penalize_base: torch.Tensor | None, drafts: list[int], j: int) -> torch.Tensor | None:
        """Presence-penalty token set for verify row ``j``: ``penalize_base ∪ drafts[:j]``.

        Row ``j``'s target distribution follows ``drafts[:j]``, so those tokens are part of
        "the output so far" for it (row ``K``, the bonus row, follows all ``K`` drafts).
        ``None`` when the request has no presence penalty or the caller passed no base set —
        which is what keeps ``presence_penalty == 0`` bit-identical to the unpenalized path.
        """
        if penalize_base is None or self.params.presence_penalty <= 0:
            return None
        if j == 0:  # drafts[:0] is empty, so the base set is already the answer
            return penalize_base
        return torch.cat([penalize_base, torch.tensor(drafts[:j], dtype=torch.int64)]).unique()

    def accept(
        self, logits: torch.Tensor, drafts: list[int], penalize_base: torch.Tensor | None = None
    ) -> tuple[int, int, list[float]]:
        """Exact speculative sampling over one verify block.

        ``logits`` [T, vocab_size], T==len(drafts)+1; bf16 converted PER ROW inside dist(), so a
        rejection at row 0 casts one row instead of all T. Row j is the distribution for the
        position draft j predicts; row K is the bonus row. ``drafts`` are the K deterministic
        drafted ids. ``penalize_base`` is UNIQUE token ids already in the output at the start of
        this window (generated_so_far ∪ {pending}); row j is penalized on penalize_base ∪
        drafts[:j]; ignored when the request has no presence penalty. Returns (m, next_token,
        p_draft): m accepted drafts (0<=m<=K), next_token recovered from row m (m<K) or bonus from
        row K (m==K), p_draft the target probability of each draft actually evaluated (length min(m+1, K))."""
        num_drafts = len(drafts)
        assert logits.dim() == 2, f"expected [T, vocab] verify logits, got {tuple(logits.shape)}"
        assert logits.shape[0] == num_drafts + 1, (
            f"verify logits must have T == len(drafts) + 1 = {num_drafts + 1} rows, " f"got {tuple(logits.shape)}"
        )

        # One draw per draft, always, so the RNG stream per call is independent of where
        # the rejection lands.
        uniforms = torch.rand(num_drafts, generator=self.gen).tolist()
        p_draft: list[float] = []
        for j in range(num_drafts):
            # lazy: rows past the decision row are never built
            row = self.dist(logits[j], self._penalize_row(penalize_base, drafts, j))
            p_d = self.prob_of(row, drafts[j])
            p_draft.append(p_d)
            if not (uniforms[j] < p_d):
                return j, self._recover(row, int(drafts[j])), p_draft
        bonus = self.pick(logits[num_drafts], self._penalize_row(penalize_base, drafts, num_drafts))
        return num_drafts, bonus, p_draft
