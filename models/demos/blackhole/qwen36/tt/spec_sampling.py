# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact speculative rejection sampling for the MTP drafter's argmax drafts.
Host fp32; optional device top-k via accept_support. With a delta proposal, P(x) = p(x).
"""

from dataclasses import dataclass

import torch
from loguru import logger

# Descending prefix that makes top-p exact without sorting the full row.
_TOPP_PREFIX = 2048


@dataclass(frozen=True)
class SpecSamplingParams:
    temperature: float  # > 0; 0 is the caller's greedy path
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables; must be in (0, 1]
    presence_penalty: float = 0.0  # subtracted from logits of output tokens, before temperature; 0 disables
    seed: int | None = None  # None -> sampler draws and records one seed

    def __post_init__(self):
        assert self.temperature > 0, f"temperature must be > 0 (0 is the caller's greedy path), got {self.temperature}"
        assert self.top_k >= 0, f"top_k must be >= 0 (0 disables top-k), got {self.top_k}"
        assert 0.0 < self.top_p <= 1.0, f"top_p must be in (0, 1], got {self.top_p}"
        assert self.presence_penalty >= 0, f"presence_penalty must be >= 0 (0 disables it), got {self.presence_penalty}"


class SpecSampler:
    def __init__(self, params: SpecSamplingParams, vocab_size: int):
        assert vocab_size >= 1, f"vocab_size must be >= 1, got {vocab_size}"
        self.params = params
        self.vocab_size = int(vocab_size)
        seed = params.seed
        if seed is None:
            # Does not touch the global RNG; recorded so a run can be replayed.
            seed = int(torch.Generator().seed())
        self.seed = int(seed)
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(self.seed)
        # No truncation: support is the whole vocabulary, in vocabulary order.
        self._dense = params.top_k == 0 and params.top_p == 1.0
        # How many times the exact full sort was needed.
        self._topp_full_sorts = 0
        logger.info(
            f"SpecSampler: vocab={self.vocab_size} temperature={params.temperature} "
            f"top_k={params.top_k} top_p={params.top_p} seed={self.seed}"
        )

    def dist(self, logits: torch.Tensor, penalize: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(idx, probs) support for one row; presence penalty applies before temperature."""
        params = self.params
        penalizing = penalize is not None and params.presence_penalty > 0
        if logits.dtype != torch.float32:
            logits = logits.float()  # .float() already copied, so the penalty can go in place
        elif penalizing:
            # Caller-owned view: copy once, never subtract in place.
            logits = logits.clone()
        assert (
            logits.dim() == 1 and logits.shape[0] == self.vocab_size
        ), f"expected 1-D logits of length {self.vocab_size}, got {tuple(logits.shape)}"
        if penalizing:
            logits[penalize] -= params.presence_penalty

        # Divide by 1.0 is an IEEE-754 identity; skip it.
        z = logits if params.temperature == 1.0 else logits / params.temperature

        if params.top_k > 0:
            vals, idx = torch.topk(z, min(params.top_k, self.vocab_size), sorted=True)
            probs = torch.softmax(vals, 0)
            if params.top_p < 1.0:
                n_keep, _ = self._top_p_keep(probs)
                return self._truncate(idx, probs, n_keep)
            return idx, probs

        if params.top_p == 1.0:
            return torch.arange(self.vocab_size, dtype=torch.int64), torch.softmax(z, 0)

        # Prefix softmax is exact while the top-p support fits in _TOPP_PREFIX.
        lse = torch.logsumexp(z, 0)
        vals, idx = torch.topk(z, min(_TOPP_PREFIX, self.vocab_size), sorted=True)
        probs = torch.exp(vals - lse)
        n_keep, before_last = self._top_p_keep(probs)
        if before_last >= params.top_p:
            # Prefix already covers top_p, so nothing outside it would be kept.
            return self._truncate(idx, probs, n_keep)

        # Top-p support runs past the prefix.
        self._topp_full_sorts += 1
        full = torch.softmax(z, 0)
        sorted_probs, sorted_idx = torch.sort(full, descending=True)
        n_keep, _ = self._top_p_keep(sorted_probs)
        return self._truncate(sorted_idx, sorted_probs, n_keep)

    def _top_p_keep(self, probs: torch.Tensor) -> tuple[int, float]:
        """Keep i iff cumsum[i] - prob[i] < top_p; returns (n_keep, last prefix mass)."""
        cum = torch.cumsum(probs, 0)
        before = cum - probs
        n_keep = int((before < self.params.top_p).sum())
        # Top-1 is always kept (prefix mass 0).
        return max(n_keep, 1), float(before[-1])

    @staticmethod
    def _truncate(idx: torch.Tensor, probs: torch.Tensor, n_keep: int) -> tuple[torch.Tensor, torch.Tensor]:
        if n_keep < probs.numel():
            idx, probs = idx[:n_keep], probs[:n_keep]
        return idx, probs / probs.sum()

    def _support_pos(self, idx: torch.Tensor, token: int) -> int | None:
        """Index of token in the support, or None if it was truncated away."""
        if self._dense and idx.numel() == self.vocab_size:
            return token if 0 <= token < self.vocab_size else None
        hits = (idx == token).nonzero()
        return None if hits.numel() == 0 else int(hits[0])

    def prob_of(self, dist: tuple[torch.Tensor, torch.Tensor], token: int) -> float:
        """Probability of token, or 0.0 outside the support."""
        idx, probs = dist
        pos = self._support_pos(idx, int(token))
        return 0.0 if pos is None else float(probs[pos])

    def pick(self, logits: torch.Tensor, penalize: torch.Tensor | None = None) -> int:
        return self._pick_dist(self.dist(logits, penalize))

    def _pick_dist(self, dist: tuple[torch.Tensor, torch.Tensor]) -> int:
        idx, probs = dist
        return int(idx[int(torch.multinomial(probs, 1, generator=self.gen)[0])])

    def _recover(self, dist: tuple[torch.Tensor, torch.Tensor], draft: int) -> int:
        """Sample from dist with the draft token's mass removed."""
        idx, probs = dist
        pos = self._support_pos(idx, draft)
        residual = probs.clone()
        if pos is not None:
            residual[pos] = 0.0
        total = float(residual.sum())
        if total <= 0.0:
            # Unreachable in exact arithmetic; float error only. Degrade, do not crash.
            if pos is None:
                return int(idx[int(probs.argmax())])
            other = probs.clone()
            other[pos] = -1.0
            best = int(other.argmax())
            return int(draft) if best == pos else int(idx[best])
        residual /= total
        return int(idx[int(torch.multinomial(residual, 1, generator=self.gen)[0])])

    def _penalize_row(self, penalize_base: torch.Tensor | None, drafts: list[int], j: int) -> torch.Tensor | None:
        """Row j penalty ids: penalize_base plus drafts[:j]; None when the penalty is off."""
        if penalize_base is None or self.params.presence_penalty <= 0:
            return None
        if j == 0:
            return penalize_base
        return torch.cat([penalize_base, torch.tensor(drafts[:j], dtype=torch.int64)]).unique()

    def accept(
        self, logits: torch.Tensor, drafts: list[int], penalize_base: torch.Tensor | None = None
    ) -> tuple[int, int, list[float]]:
        """T=K+1 rows; accept while u < p(d). Returns (m, recovered or bonus token, evaluated draft probs)."""
        num_drafts = len(drafts)
        assert logits.dim() == 2, f"expected [T, vocab] verify logits, got {tuple(logits.shape)}"
        assert logits.shape[0] == num_drafts + 1, (
            f"verify logits must have T == len(drafts) + 1 = {num_drafts + 1} rows, " f"got {tuple(logits.shape)}"
        )

        return self._accept_rows(
            num_drafts, drafts, lambda j: self.dist(logits[j], self._penalize_row(penalize_base, drafts, j))
        )

    def _accept_rows(self, num_drafts: int, drafts: list[int], row_dist) -> tuple[int, int, list[float]]:
        """One uniform per draft, so the RNG stream does not depend on rejection depth."""
        uniforms = torch.rand(num_drafts, generator=self.gen).tolist()
        p_draft: list[float] = []
        for j in range(num_drafts):
            # Rows past the decision row are never built.
            row = row_dist(j)
            p_d = self.prob_of(row, drafts[j])
            p_draft.append(p_d)
            if not (uniforms[j] < p_d):
                return j, self._recover(row, int(drafts[j])), p_draft
        return num_drafts, self._pick_dist(row_dist(num_drafts)), p_draft

    def dist_from_support(self, idx: torch.Tensor, vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """dist() for a device top-k support; temperature commutes with the selection, presence penalty does not."""
        params = self.params
        z = vals if params.temperature == 1.0 else vals / params.temperature
        probs = torch.softmax(z, 0)
        if params.top_p < 1.0:
            n_keep, _ = self._top_p_keep(probs)
            return self._truncate(idx, probs, n_keep)
        return idx, probs

    def accept_support(
        self, support: tuple[torch.Tensor, torch.Tensor], drafts: list[int]
    ) -> tuple[int, int, list[float]]:
        """accept() over a device (idx, vals) support; requires top_k > 0 and no presence penalty."""
        idx, vals = support
        num_drafts = len(drafts)
        assert self.params.top_k > 0, "device support requires top_k > 0"
        assert self.params.presence_penalty <= 0, "presence penalty must use the host path"
        assert idx.shape[0] == num_drafts + 1, f"support needs T == {num_drafts + 1} rows, got {tuple(idx.shape)}"
        return self._accept_rows(num_drafts, drafts, lambda j: self.dist_from_support(idx[j], vals[j]))
