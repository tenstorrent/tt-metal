# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host side of the pup's greedy speculative decode (``PUP_SPEC=1``, default off): exact acceptance and the loop.

Torch only (no ttnn, no device): the device engine (``models/demos/blackhole/qwen36/tt/spec_pup.PupSpecEngine``) is
reached through the structural :class:`SpecEngine` protocol, so the CPU unit tests drive the same loop with a fake.

Exactness contract (README.md, "Upsampler: speculative decode"): every committed token equals
``upsampler.presence_greedy(row, generated_so_far, p)`` on the verify row that predicts it, i.e. the same fp32
penalty subtraction, the same set semantics (each generated id once, prompt excluded, the penalty applied only when
``p > 0``) and the same lowest-index tie-break (``torch.argmax`` returns the first maximal index). The device hands
back, per verify row, the per-device top-N ``(value, global id)`` of the bf16 logits plus a bound ``thresh`` that
every non-returned logit is ``<=`` to. :func:`certified_argmax` decides from the candidates only when that is provably
the full-row answer (``best > thresh``), else the caller reads the full row (``read_rows``) and runs
:func:`penalized_argmax` on it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Sequence

import torch

TopN = tuple  # (vals fp32 [R, C], ids int64 [R, C], thresh fp32 [R])


class SpecEngine(Protocol):
    """The subset of ``PupSpecEngine`` the loop uses (see the design, section 6.2)."""

    K: int

    def seed(self, first_token: int) -> torch.Tensor:
        ...

    def verify(self, tokens: Sequence[int], start_pos: int, accepted_prev: int, num_valid: int) -> TopN:
        ...

    def read_rows(self, rows: Sequence[int]) -> torch.Tensor:
        ...

    def propose(self, committed: Sequence[int], first_pos: int) -> list[int]:
        ...


class Drafter(Protocol):
    k: int

    def start(self, prompt_ids: Sequence[int]) -> None:
        ...

    def propose(self, context_ids: Sequence[int], committed: Sequence[int], first_pos: int) -> list[int]:
        ...


def _penalty_ids(penalize: Iterable[int]) -> torch.Tensor:
    return torch.tensor(sorted(set(int(t) for t in penalize)), dtype=torch.long)


def penalized_argmax(row: torch.Tensor, penalize: Iterable[int], penalty: float) -> int:
    """``upsampler.presence_greedy`` with the penalty SET given directly: the same ops in the same order (fp32 copy,
    ``v[sorted ids] -= float(p)`` only when ``p > 0`` and the set is non-empty, ``torch.argmax``). A NaN anywhere in
    the row raises ``RuntimeError`` (the non-spec loop raises on a NaN row before sampling)."""
    v = row.detach().reshape(-1).float().clone()
    if torch.isnan(v).any():
        raise RuntimeError("NaN in the verify logits")
    penalize = set(int(t) for t in penalize)
    if penalty > 0 and penalize:
        v[_penalty_ids(penalize)] -= float(penalty)
    return int(torch.argmax(v).item())


def certified_argmax(
    vals: torch.Tensor, ids: torch.Tensor, thresh: float | torch.Tensor, penalize: Iterable[int], penalty: float
) -> int | None:
    """The full-row ``presence_greedy`` answer from one row's top-N candidates, or ``None`` when the candidates cannot
    prove it.

    ``vals`` fp32 [C] are exact casts of the bf16 logits of the unique token ``ids`` int64 [C]; every other token has
    a logit ``<= thresh``. After the penalty (``x - p`` in fp32 round-to-nearest-even never exceeds ``x`` for
    ``p >= 0``; the penalty is skipped for ``p <= 0`` exactly as ``presence_greedy`` skips it) every non-candidate
    still has ``y <= thresh``. So when the best penalised candidate is ``> thresh`` the maximum and ALL its ties are
    candidates, and the lowest id among them is the full row's lowest-index argmax. NaN among the candidates raises.
    """
    v = vals.detach().reshape(-1).float().clone()
    i = ids.detach().reshape(-1).to(torch.long)
    if v.numel() == 0:
        return None
    if torch.isnan(v).any():
        raise RuntimeError("NaN in the verify logits")
    penalize = set(int(t) for t in penalize)
    if penalty > 0 and penalize:
        hit = torch.isin(i, _penalty_ids(penalize))
        v[hit] -= float(penalty)
    best = v.max()
    t = torch.as_tensor(thresh, dtype=torch.float32).reshape(())
    if torch.isnan(t) or not bool(best > t):
        return None
    return int(i[v == best].min().item())


def reference_topn(rows: torch.Tensor, n_dev: int, top_n: int, generator: torch.Generator | None = None) -> TopN:
    """The CPU model of the device top-N contract (tests, and the device probe's check): ``rows`` [R, V] logit rows
    (bf16-representable), the vocab split into ``n_dev`` equal contiguous shards, per shard the ``top_n`` largest
    ``(value, global id)``. ``generator`` permutes each shard before ``topk`` so the choice among tied values is
    arbitrary (the device kernel promises nothing about it). ``thresh[j] = max_d min(vals_d[j])``."""
    rows = rows.detach().float()
    if rows.dim() == 1:
        rows = rows.unsqueeze(0)
    R, V = rows.shape
    if V % n_dev:
        raise ValueError(f"vocab {V} does not split over {n_dev} devices")
    shard = V // n_dev
    if not 1 <= top_n <= shard:
        raise ValueError(f"top_n {top_n} outside [1, {shard}]")
    vals, ids, mins = [], [], []
    for d in range(n_dev):
        x = rows[:, d * shard : (d + 1) * shard]
        perm = torch.randperm(shard, generator=generator) if generator is not None else torch.arange(shard)
        v, i = torch.topk(x[:, perm], top_n, dim=-1)
        vals.append(v)
        ids.append(perm[i] + d * shard)
        mins.append(v.min(dim=-1).values)
    thresh = torch.stack(mins, dim=-1).max(dim=-1).values
    return torch.cat(vals, dim=-1), torch.cat(ids, dim=-1).to(torch.int64), thresh


@dataclass
class BlockResult:
    m: int
    """Accepted drafts (0..num_valid)."""
    bonus: int
    """The target's token at row ``m`` (may be a stop id)."""
    rows_evaluated: int
    fallback_rows: int


def accept_block(
    topn: TopN,
    drafts: Sequence[int],
    num_valid: int,
    generated_set: set,
    penalty: float,
    read_rows: Callable[[Sequence[int]], torch.Tensor],
    stop_ids: Sequence[int] = (),
) -> BlockResult:
    """Greedy acceptance over one verify block: row ``j`` (``0 <= j <= num_valid``) predicts the token after
    ``[pending] + drafts[:j]`` with the penalty set ``generated_set ∪ drafts[:j]`` (``generated_set`` already holds
    ``pending``). Rows are decided lazily, in order: ``m`` = the first ``j < num_valid`` whose target ``w_j`` differs
    from ``drafts[j]`` (or whose ``w_j`` is a stop id: the stream ends there, so later rows are never needed), else
    ``num_valid``; ``bonus = w_m``. A row the candidates cannot certify is read in full (``read_rows([j])``)."""
    vals, ids, thresh = topn
    drafts = [int(d) for d in drafts]
    if not 0 <= num_valid <= len(drafts):
        raise ValueError(f"num_valid {num_valid} outside [0, {len(drafts)}]")
    stops = set(int(s) for s in stop_ids)
    penalize = set(int(t) for t in generated_set)
    fallback = 0
    for j in range(num_valid + 1):
        if j > 0:
            penalize.add(drafts[j - 1])
        w = certified_argmax(vals[j], ids[j], thresh[j], penalize, penalty)
        if w is None:
            full = read_rows([j])
            w = penalized_argmax(full.reshape(-1), penalize, penalty)
            fallback += 1
        if j == num_valid or drafts[j] != w or w in stops:
            return BlockResult(m=j, bonus=w, rows_evaluated=j + 1, fallback_rows=fallback)
    raise AssertionError("unreachable")


@dataclass
class SpecStats:
    """Per-request counters of :func:`run_spec_greedy` (host wall times in ms)."""

    k: int = 0
    iterations: int = 0
    """Verify calls."""
    tokens: int = 0
    """Generated tokens (stop id excluded), the prefill's first token included."""
    verify_tokens: int = 0
    """Tokens committed by verify calls (the prefill token and the seed token excluded)."""
    drafted: int = 0
    """Draft tokens sent to verify (sum of num_valid)."""
    accepted: int = 0
    empty_draft_steps: int = 0
    accept_hist: dict = field(default_factory=dict)
    """{m: count} over verify calls."""
    rows_evaluated: int = 0
    fallback_rows: int = 0
    seed_ms: float = 0.0
    verify_ms: float = 0.0
    """Sum of host wall time inside ``engine.verify`` (device verify + candidate readback)."""
    accept_ms: float = 0.0
    """Sum of host acceptance time (including fallback full-row reads)."""
    draft_ms: float = 0.0
    """Sum of drafter time (``engine.propose`` for MTP, the host lookup for n-gram)."""
    loop_s: float = 0.0
    """Wall time of the verify iterations (after the seed)."""
    engine_timing: dict = field(default_factory=dict)
    """Sums of ``engine.last_timing`` values (``PUP_SPEC_TIMING=1``), see :func:`_add_engine_timing`."""

    def summary(self) -> dict:
        it = max(1, self.iterations)
        return {
            "k": self.k,
            "iterations": self.iterations,
            "tokens": self.tokens,
            "tokens_per_iter": (self.verify_tokens / self.iterations) if self.iterations else 0.0,
            "accepted_per_step": self.accepted / it if self.iterations else 0.0,
            "draft_precision": (self.accepted / self.drafted) if self.drafted else 0.0,
            "empty_draft_steps": self.empty_draft_steps,
            "accept_hist": {int(k): int(v) for k, v in sorted(self.accept_hist.items())},
            "rows_evaluated": self.rows_evaluated,
            "fallback_rows": self.fallback_rows,
            "fallback_rate": (self.fallback_rows / self.rows_evaluated) if self.rows_evaluated else 0.0,
            "seed_ms": self.seed_ms,
            "verify_ms": self.verify_ms / it if self.iterations else 0.0,
            "accept_ms": self.accept_ms / it if self.iterations else 0.0,
            "draft_ms": self.draft_ms / it if self.iterations else 0.0,
            "iter_ms": 1000.0 * self.loop_s / it if self.iterations else 0.0,
            "ms_per_token": (1000.0 * self.loop_s / self.verify_tokens) if self.verify_tokens else 0.0,
            "engine_timing_mean": {k: v / it for k, v in self.engine_timing.items()} if self.iterations else {},
        }


def _add_engine_timing(stats: SpecStats, engine, phase: str) -> None:
    """``engine.last_timing`` after the drafter (``phase="draft"``: the ``propose*`` keys only) or after the verify +
    acceptance (``phase="verify"``: every other key). ``PupSpecEngine.verify`` replaces the dict, ``read_rows`` and
    ``propose`` add to it, so the split counts each key once per iteration."""
    t = getattr(engine, "last_timing", None)
    if not t:
        return
    for key, value in dict(t).items():
        if not isinstance(value, (int, float)) or key.startswith("propose") != (phase == "draft"):
            continue
        stats.engine_timing[key] = stats.engine_timing.get(key, 0.0) + float(value)


def run_spec_greedy(
    engine: SpecEngine,
    drafter: Drafter,
    *,
    prompt_ids: Sequence[int],
    first_token: int,
    prompt_len: int,
    penalty: float,
    stop_ids: Sequence[int],
    max_new_tokens: int,
) -> tuple[list[int], str, SpecStats]:
    """Greedy speculative decode equal, token for token, to the non-spec host loop of
    ``Flux3KleinUpsampler.generate_ids`` (``presence_greedy`` per step, stop id ends the stream and is never appended,
    ``max_new_tokens`` = the request's ``new_token_budget``).

    ``first_token`` is ``presence_greedy(prefill_row, [], p)`` (the caller ran the prefill); ``prompt_len`` = Tp, the
    position ``first_token`` is processed at by the seed. Positions: ``seed(first)`` commits ``nxt`` (position Tp+1);
    each verify processes ``[pending, d_0..d_{K-1}]`` at ``start_pos..start_pos+K`` and commits ``drafts[:m] + [bonus]``;
    the next verify starts at ``start_pos + m + 1`` with ``accepted_prev = m + 1``. The drafter's ``propose`` runs
    exactly once after every seed / verify that does not end the request (the MTP engine's KV upkeep).
    Returns ``(generated, finish, stats)``: ``finish`` is ``"stop"`` or ``"length"``.
    """
    K = int(engine.K)
    stops = set(int(s) for s in stop_ids)
    stats = SpecStats(k=K)
    prompt = [int(t) for t in prompt_ids]
    generated: list[int] = []
    max_new_tokens = int(max_new_tokens)
    first_token = int(first_token)

    if first_token in stops:
        return generated, "stop", stats
    if max_new_tokens <= 0:
        return generated, "length", stats
    generated.append(first_token)
    stats.tokens = 1
    if len(generated) >= max_new_tokens:
        return generated, "length", stats

    # Seed: first_token at position Tp; the full row predicting Tp+1 (always read in full).
    t0 = time.perf_counter()
    row = engine.seed(first_token)
    nxt = penalized_argmax(row, generated, penalty)
    stats.seed_ms = 1000.0 * (time.perf_counter() - t0)
    if nxt in stops:
        return generated, "stop", stats
    generated.append(nxt)
    stats.tokens = len(generated)
    if len(generated) >= max_new_tokens:
        return generated, "length", stats
    generated_set = set(generated)

    drafter.start(prompt)
    committed = [nxt]
    pending, start_pos, accepted_prev = nxt, int(prompt_len) + 1, 1
    t_loop = time.perf_counter()
    finish = "length"
    while True:
        # Drafts continuing `committed` (committed[-1] == pending sits at start_pos, committed[0] at
        # start_pos - len(committed) + 1).
        t0 = time.perf_counter()
        drafts = [int(d) for d in drafter.propose(prompt + generated, committed, start_pos - len(committed) + 1)]
        stats.draft_ms += 1000.0 * (time.perf_counter() - t0)
        _add_engine_timing(stats, engine, "draft")
        remaining = max_new_tokens - len(generated)  # >= 1 here
        num_valid = min(len(drafts), K, remaining - 1)  # m + 1 <= remaining: never verify past the budget
        drafts = drafts[:num_valid]
        tokens = [pending] + drafts + [pending] * (K - num_valid)

        t0 = time.perf_counter()
        topn = engine.verify(tokens, start_pos, accepted_prev, num_valid)
        t1 = time.perf_counter()
        res = accept_block(topn, drafts, num_valid, generated_set, penalty, engine.read_rows, stops)
        t2 = time.perf_counter()
        _add_engine_timing(stats, engine, "verify")
        stats.verify_ms += 1000.0 * (t1 - t0)
        stats.accept_ms += 1000.0 * (t2 - t1)
        stats.iterations += 1
        stats.drafted += num_valid
        stats.accepted += res.m
        stats.empty_draft_steps += num_valid == 0
        stats.accept_hist[res.m] = stats.accept_hist.get(res.m, 0) + 1
        stats.rows_evaluated += res.rows_evaluated
        stats.fallback_rows += res.fallback_rows

        committed = drafts[: res.m] + [res.bonus]
        ended = False
        for tok in committed:
            if tok in stops:
                finish, ended = "stop", True
                break
            generated.append(tok)
            generated_set.add(tok)
            stats.verify_tokens += 1
            if len(generated) >= max_new_tokens:
                finish, ended = "length", True
                break
        if ended:
            break
        pending = res.bonus
        start_pos += res.m + 1
        accepted_prev = res.m + 1
    stats.loop_s = time.perf_counter() - t_loop
    stats.tokens = len(generated)
    return generated, finish, stats


__all__ = [
    "BlockResult",
    "Drafter",
    "SpecEngine",
    "SpecStats",
    "TopN",
    "accept_block",
    "certified_argmax",
    "penalized_argmax",
    "reference_topn",
    "run_spec_greedy",
]
