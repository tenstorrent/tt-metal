# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness-first host sampling for TP4 Qwen3.8 logits.

The released LM head owns four *different*, contiguous vocabulary row ranges.
This module therefore never reads coordinate zero as though the four local
``[62080]`` tensors were replicas.  It first validates the explicit
:class:`Qwen38ShardedLogits` ownership metadata and TTNN tensor topology, then
performs an explicit four-way vocabulary all-gather.  The replicated result is
read back from every coordinate and checked for exact agreement before host
sampling.  This is deliberately a correctness path, not a performance claim.

The input logit tensor remains borrowed.  The replicated all-gather result is
the sampler's only TTNN allocation.  It is tracked before any validation or
readback that can fail, and release is retry-safe: an owner remains pending
until ``Tensor.is_allocated()`` proves it was released.  A failed cleanup must
be drained by :meth:`Qwen38TTNNHostSampler.release_owned_tensors` (or the next
sample call) before another allocation is made.

Penalties follow the OpenAI additive semantics (``presence_penalty`` once per
seen token, ``frequency_penalty`` per occurrence) over the request's own output,
``token_history[prompt_tokens:]`` (the vLLM rule: the prompt's words are not
penalized), and then the transformers ``repetition_penalty`` rule (positive
logits divided, negative multiplied) over the prompt and the output.
Temperature is applied next, then the transformers warper order: stable
global-index top-k, shifted nucleus top-p, min-p.  One uniform draw from a CPU
``torch.Generator`` picks the token by inverse CDF, so the same draw gives the
same token whether the filters ran over the full vocabulary or over the
candidate row; a chat request passes its own generator with the same initial
seed to advance one reproducible stream across tokens.  ``temperature=0`` is
the exact greedy baseline and retains the lowest global token ID on ties.

The candidate row (:class:`Qwen38CandidateRow`, produced by the optional TAIL
epilogue ``Qwen38TTNNLMHead.sampling_candidates``) holds every shard's top-k
logits and ids and nothing else.  With ``top_k`` at most k the filters see
exactly the vector the full-vocabulary sampler sees whenever every token that
could be kept was read: the guard ``kept minimum > max over shards of the
shard's k-th value`` (after the penalties, on the temperature-scaled values)
proves it, because an unread token's value is at most its shard's k-th value
and the penalties only lower logits.  Top-p and min-p act on the softmax of the
kept top-k vector (the transformers warper order), so they need no
full-vocabulary normalizer and are exact over the row.  When the guard fails, or
``top_k`` is 0 (the nucleus may extend past the row), or a penalty would raise
logits, :class:`Qwen38CandidateFallback` tells the caller to take the
full-vocabulary path for that step.  Only the reported log-probabilities depend
on the normalizer: the row sampler reports them relative to the read candidates
(``Qwen38CandidateSample.logprob``), the full-vocabulary sampler exactly.

Within-shard ties at the k-th value: the device top-k and ``torch.topk`` may keep
different members of a tie group at the boundary.  The values are still bitwise
the same multiset, the guard is unchanged (the k-th value is the floor either
way), and :meth:`Qwen38CandidateRow.agreement` accepts id sets that differ only
among candidates at the k-th value while reporting how many such ties there are.

:func:`candidate_distribution` and :func:`full_distribution` are the two samplers
stopped before their draw: the kept tokens and their probabilities as one
:class:`Qwen38RowDistribution`, the object the speculative acceptance of the MTP
verify rows (``ttnn/speculative_sampling.py``) reads and conditions.  The same
``_penalize`` and ``_filter`` run in all four functions.  :func:`candidate_distributions`
is :func:`candidate_distribution` over the k + 1 rows of one MTP pass in one batched
pass (:class:`Qwen38CandidateRows` parses them at once): the same operations along the
last dimension of ``[rows, candidates]`` tensors, bitwise per row, so the pass loop's
decision costs a few dozen tensor operations instead of a few dozen per row.
"""

from __future__ import annotations

import array
import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import InitVar, dataclass
from enum import Enum
from numbers import Real
from typing import Any, Literal

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import Qwen38BuildProvenance, Qwen38LiveBuildIdentity
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    MESH_SHAPE,
    TP_SIZE,
    Qwen38MeshContract,
    TensorPlacement,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    LOCAL_VOCAB_SIZE,
    SAMPLING_CANDIDATE_ROW_SHAPE,
    SAMPLING_CANDIDATES_PER_DEVICE,
    VOCAB_SIZE,
    Qwen38ShardedLogits,
    Qwen38TTNNLMHead,
)

TP_AXIS = 1
EXPECTED_VOCAB_RANGES = tuple(
    (coordinate * LOCAL_VOCAB_SIZE, (coordinate + 1) * LOCAL_VOCAB_SIZE) for coordinate in range(TP_SIZE)
)
MAX_SEED = (1 << 63) - 1
# The candidate row's per-shard k; a request's top_k above it is refused (never a silent fallback).
CANDIDATE_TOP_K_LIMIT = SAMPLING_CANDIDATES_PER_DEVICE
MAX_TOP_LOGPROBS = 20
Clock = Callable[[], int]


class Qwen38SamplingProfile(str, Enum):
    """Inspectable parameter source for one sampling request."""

    THINKING = "thinking"
    NON_THINKING = "non_thinking"
    GREEDY = "greedy"
    CUSTOM = "custom"


@dataclass(frozen=True)
class Qwen38SamplingParameters:
    """One deterministic host-sampling policy.

    Use :meth:`official_thinking` or :meth:`official_non_thinking` for the
    released model-card defaults.  ``custom`` is explicit so a caller cannot
    accidentally report modified parameters as an official profile.
    """

    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    seed: int
    profile: Qwen38SamplingProfile = Qwen38SamplingProfile.CUSTOM
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        temperature = _finite_real(self.temperature, label="temperature")
        top_p = _finite_real(self.top_p, label="top_p")
        presence_penalty = _finite_real(self.presence_penalty, label="presence_penalty")
        min_p = _finite_real(self.min_p, label="min_p")
        frequency_penalty = _finite_real(self.frequency_penalty, label="frequency_penalty")
        repetition_penalty = _finite_real(self.repetition_penalty, label="repetition_penalty")
        if temperature < 0:
            raise ValueError(f"temperature must be nonnegative, got {temperature}")
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0,1], got {top_p}")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or not 0 <= self.top_k <= VOCAB_SIZE:
            raise ValueError(f"top_k must be an integer in [0,{VOCAB_SIZE}], got {self.top_k!r}")
        if not -2 <= presence_penalty <= 2:
            raise ValueError(f"presence_penalty must be in [-2,2], got {presence_penalty}")
        if not 0 <= min_p <= 1:
            raise ValueError(f"min_p must be in [0,1], got {min_p}")
        if not -2 <= frequency_penalty <= 2:
            raise ValueError(f"frequency_penalty must be in [-2,2], got {frequency_penalty}")
        if repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be positive, got {repetition_penalty}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= MAX_SEED:
            raise ValueError(f"seed must be an integer in [0,{MAX_SEED}], got {self.seed!r}")
        if not isinstance(self.profile, Qwen38SamplingProfile):
            raise TypeError("profile must be a Qwen38SamplingProfile")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "presence_penalty", presence_penalty)
        object.__setattr__(self, "min_p", min_p)
        object.__setattr__(self, "frequency_penalty", frequency_penalty)
        object.__setattr__(self, "repetition_penalty", repetition_penalty)

        expected: tuple[float, float, int, float] | None
        if self.profile is Qwen38SamplingProfile.THINKING:
            expected = (1.0, 0.95, 20, 0.0)
        elif self.profile is Qwen38SamplingProfile.NON_THINKING:
            expected = (0.7, 0.8, 20, 1.5)
        elif self.profile is Qwen38SamplingProfile.GREEDY:
            expected = (0.0, 1.0, 0, 0.0)
        else:
            expected = None
        actual = (temperature, top_p, self.top_k, presence_penalty)
        if expected is not None and (
            actual != expected or (min_p, frequency_penalty, repetition_penalty) != (0.0, 0.0, 1.0)
        ):
            raise ValueError(
                f"{self.profile.value} profile requires {expected} and no min_p/frequency/repetition, got {actual}"
            )

    @property
    def penalizes(self) -> bool:
        return self.presence_penalty != 0 or self.frequency_penalty != 0 or self.repetition_penalty != 1

    @property
    def raises_logits(self) -> bool:
        """A penalty that can lift a token: the candidate guard cannot bound unread tokens then."""

        return self.presence_penalty < 0 or self.frequency_penalty < 0 or self.repetition_penalty < 1

    @classmethod
    def official_thinking(cls, *, seed: int) -> "Qwen38SamplingParameters":
        return cls(
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            presence_penalty=0.0,
            seed=seed,
            profile=Qwen38SamplingProfile.THINKING,
        )

    @classmethod
    def official_non_thinking(cls, *, seed: int) -> "Qwen38SamplingParameters":
        return cls(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            presence_penalty=1.5,
            seed=seed,
            profile=Qwen38SamplingProfile.NON_THINKING,
        )

    @classmethod
    def greedy(cls) -> "Qwen38SamplingParameters":
        return cls(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            presence_penalty=0.0,
            seed=0,
            profile=Qwen38SamplingProfile.GREEDY,
        )


@dataclass(frozen=True)
class Qwen38SamplingTiming:
    """Host wall-clock domains; readback synchronizes, but none is a device-profiler metric."""

    full_logit_gather_readback_ns: int
    host_filter_sample_ns: int
    end_to_end_ns: int

    @property
    def validation_and_packaging_ns(self) -> int:
        return self.end_to_end_ns - self.full_logit_gather_readback_ns - self.host_filter_sample_ns


@dataclass(frozen=True)
class Qwen38SampledTokens:
    """CPU tokens plus the exact launch and policy identity that produced them."""

    token_ids: torch.Tensor
    eos_mask: torch.Tensor
    parameters: Qwen38SamplingParameters
    timing: Qwen38SamplingTiming
    source_identity_key: str
    source_provenance_key: str
    rng_mode: Literal["per-call-seed", "request-generator"]
    readback: Literal["explicit-tp4-full-logit-host-gather"] = "explicit-tp4-full-logit-host-gather"
    borrowed_logits_ownership: Literal["caller-retained"] = "caller-retained"
    owned_ttnn_tensors_after_return: Literal[0] = 0

    @property
    def rows(self) -> int:
        return int(self.token_ids.shape[2])

    @property
    def any_eos(self) -> bool:
        return bool(torch.any(self.eos_mask))


class Qwen38SamplingError(RuntimeError):
    """Base exception for the strict host sampler."""


class Qwen38SamplingCleanupError(Qwen38SamplingError):
    """At least one task-owned gather tensor still needs a release retry."""

    def __init__(
        self,
        operation: str,
        cleanup_errors: Sequence[BaseException],
        *,
        primary_error: BaseException | None = None,
    ) -> None:
        self.operation = operation
        self.cleanup_errors = tuple(cleanup_errors)
        self.primary_error = primary_error
        cleanup = "; ".join(f"{type(error).__name__}: {error}" for error in self.cleanup_errors)
        primary = "" if primary_error is None else f" after {type(primary_error).__name__}: {primary_error}"
        super().__init__(f"sampling cleanup failed during {operation}{primary}: {cleanup}; retry release before reuse")


@dataclass
class _OwnedTensor:
    tensor: Any
    label: str


@dataclass(frozen=True)
class _RuntimeOps:
    all_gather: Callable[..., Any]
    make_concat_composer: Callable[[Any, int], Any]
    to_torch: Callable[..., torch.Tensor]
    deallocate: Callable[[Any], None]


def _default_runtime_ops() -> _RuntimeOps:
    return _RuntimeOps(
        all_gather=ttnn.all_gather,
        make_concat_composer=lambda mesh, dim: ttnn.ConcatMeshToTensor(mesh, dim=dim),
        to_torch=ttnn.to_torch,
        deallocate=ttnn.deallocate,
    )


def _finite_real(value: Real, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _require_key(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase 64-hex, got {value!r}")
    return value


def _duration(start: int, end: int, *, label: str) -> int:
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise RuntimeError(f"{label} clock must return integer nanoseconds")
    if end < start:
        raise RuntimeError(f"{label} monotonic clock moved backwards: {start} -> {end}")
    return end - start


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _collective_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if callable(name):
        name = name()
    if not isinstance(name, str) or not name:
        raise ValueError(f"collective topology has no stable name: {name!r}")
    return name


def _normalize_eos_ids(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("eos_token_ids must be a sequence of integers")
    result = tuple(values)
    if not result:
        raise ValueError("at least one pinned EOS token ID is required")
    if len(set(result)) != len(result):
        raise ValueError(f"EOS token IDs must be unique, got {result}")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < VOCAB_SIZE for value in result):
        raise ValueError(f"EOS token IDs must be integers in [0,{VOCAB_SIZE}), got {result}")
    return result


def _normalize_one_history(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of token IDs")
    result = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError(f"{label} must contain only integer token IDs")
    if any(not 0 <= value < VOCAB_SIZE for value in result):
        raise ValueError(f"{label} contains a token outside [0,{VOCAB_SIZE})")
    return result


def _normalize_histories(
    token_histories: Sequence[int] | Sequence[Sequence[int]],
    *,
    rows: int,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(token_histories, (str, bytes)) or not isinstance(token_histories, Sequence):
        raise TypeError("token_histories must be a token sequence or one sequence per logit row")
    raw = tuple(token_histories)
    if rows == 1 and (not raw or all(isinstance(value, int) and not isinstance(value, bool) for value in raw)):
        return (_normalize_one_history(raw, label="token history"),)
    if len(raw) != rows:
        raise ValueError(f"expected exactly {rows} token histories, got {len(raw)}")
    return tuple(_normalize_one_history(history, label=f"token history row {row}") for row, history in enumerate(raw))


def _validate_host_logits(logits: torch.Tensor) -> tuple[torch.Tensor, int]:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("host logits must be a torch.Tensor")
    if logits.device.type != "cpu" or not logits.dtype.is_floating_point:
        raise ValueError("host logits must be a CPU floating-point tensor")
    if logits.ndim != 4 or logits.shape[:2] != (1, 1) or logits.shape[2] <= 0 or logits.shape[3] != VOCAB_SIZE:
        raise ValueError(f"host logits must be true-global-B1 [1,1,rows,{VOCAB_SIZE}], got {tuple(logits.shape)}")
    values = logits.detach().to(dtype=torch.float32).reshape(int(logits.shape[2]), VOCAB_SIZE).contiguous()
    if not bool(torch.all(torch.isfinite(values))):
        raise ValueError("host logits contain NaN or infinity")
    return values, int(logits.shape[2])


_MASK64 = (1 << 64) - 1
UNIFORM_BITS = 24  # u = n * 2**-24: exact in fp32, the draw the device sampler consumes


class UniformStream:
    """splitmix64 per request: ``state += gamma; z = mix(state); u = (z >> 40) * 2**-24`` (exact in fp32).

    The draw of the device sampler (``ttnn/device_sampler.py``), written to the device once per step; given to the
    host samplers in place of a ``torch.Generator`` it makes a host fallback step continue the same sequence.
    Reproducible from any language: ten lines of 64-bit integer arithmetic.
    """

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MASK64:
            raise ValueError(f"seed must be an integer in [0, 2**64), got {seed!r}")
        self.seed = seed
        self.state = seed
        self.draws = 0

    def initial_seed(self) -> int:
        return self.seed

    def next_bits(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z ^= z >> 31
        self.draws += 1
        return z >> (64 - UNIFORM_BITS)

    def next_uniform(self) -> float:
        """The next draw in [0, 1) as a float fp32 holds exactly."""

        return self.next_bits() / float(1 << UNIFORM_BITS)


def _validate_generator(generator: torch.Generator | UniformStream, *, seed: int) -> None:
    if isinstance(generator, UniformStream):
        if generator.seed != seed:
            raise ValueError(f"uniform stream seed {generator.seed} differs from parameters {seed}")
        return
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or a UniformStream")
    if torch.device(generator.device).type != "cpu":
        raise ValueError("sampling generator must be a CPU generator")
    if generator.initial_seed() != seed:
        raise ValueError(f"sampling generator initial seed {generator.initial_seed()} differs from parameters {seed}")


# --- the filter core shared by the full-vocabulary and the candidate samplers ---------------------
#
# Every function below works on one row: fp32 ``scores`` over tokens ``ids`` (int64, ascending, so
# the stable sort breaks ties on the lowest global id the way torch.argmax does).  The same fp32
# operations run whether ``ids`` is the whole vocabulary or the candidate set, which is what makes
# the candidate sampler bitwise equal to the full one on the kept vector.


def _penalize(
    scores: torch.Tensor,
    ids: torch.Tensor | None,
    history: tuple[int, ...],
    p: Qwen38SamplingParameters,
    *,
    prompt_tokens: int = 0,
):
    """OpenAI presence and frequency penalties over the output, ``history[prompt_tokens:]``, then the transformers
    repetition rule over the whole history (prompt and output)."""

    if isinstance(prompt_tokens, bool) or type(prompt_tokens) is not int or not 0 <= prompt_tokens <= len(history):
        raise ValueError(f"prompt_tokens must be an integer in [0, {len(history)}], got {prompt_tokens!r}")
    if not history or not p.penalizes:
        return scores
    if (p.presence_penalty != 0 or p.frequency_penalty != 0) and prompt_tokens < len(history):
        positions, counts = _seen(ids, history[prompt_tokens:])
        if p.presence_penalty != 0:
            scores[positions] -= p.presence_penalty
        if p.frequency_penalty != 0:
            scores[positions] -= counts * p.frequency_penalty
    if p.repetition_penalty != 1:
        positions, _counts = _seen(ids, history)
        hit_scores = scores[positions]
        scores[positions] = torch.where(
            hit_scores < 0, hit_scores * p.repetition_penalty, hit_scores / p.repetition_penalty
        )
    return scores


def _seen(ids: torch.Tensor | None, tokens: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """The row positions of ``tokens`` (the ids themselves over the full vocabulary) and their occurrence counts."""

    counts = Counter(tokens)
    seen = torch.tensor(sorted(counts), dtype=torch.int64)
    seen_counts = torch.tensor([counts[int(token)] for token in seen.tolist()], dtype=torch.float32)
    if ids is None:
        return seen, seen_counts
    hit = ids.reshape(-1, 1) == seen.reshape(1, -1)
    positions, seen_columns = torch.nonzero(hit, as_tuple=True)
    return positions, seen_counts[seen_columns]


def _filter(scores: torch.Tensor, p: Qwen38SamplingParameters) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Temperature, top-k, shifted top-p, min-p (the transformers warper order) over one penalized row.

    Returns ``(positions, scaled, probabilities)``: the kept positions in descending
    order (top-k prefix of the stable sort), their temperature-scaled logits, and
    their probabilities with the nucleus / min-p cut zeroed and the rest renormalized.
    """

    scaled = scores / p.temperature
    if not bool(torch.all(torch.isfinite(scaled))):
        raise ValueError("temperature scaling produced non-finite logits")
    positions = torch.argsort(scaled, descending=True, stable=True)
    if p.top_k:
        positions = positions[: p.top_k]
    ordered = scaled[positions]
    probabilities = torch.softmax(ordered, dim=-1)
    if p.top_p < 1:
        remove = torch.cumsum(probabilities, dim=-1) > p.top_p
        remove[1:] = remove[:-1].clone()  # keep the token that crosses the boundary
        remove[0] = False
        probabilities = probabilities.masked_fill(remove, 0.0)
        probabilities = probabilities / probabilities.sum()
    if p.min_p > 0:
        remove = probabilities < p.min_p * probabilities[0]  # position 0 holds the maximum
        remove[0] = False
        probabilities = probabilities.masked_fill(remove, 0.0)
        probabilities = probabilities / probabilities.sum()
    if not bool(torch.all(torch.isfinite(probabilities))) or bool(probabilities.sum() <= 0):
        raise RuntimeError("the filters removed every finite candidate")
    return positions, ordered, probabilities


def _inverse_cdf(probabilities: torch.Tensor, uniform: torch.Tensor) -> int:
    """The first kept position whose cumulative probability exceeds ``uniform`` (the last positive one if none)."""

    cumulative = torch.cumsum(probabilities, dim=-1)
    index = int(torch.searchsorted(cumulative, uniform.reshape(1), right=True).item())
    last_positive = int(torch.nonzero(probabilities > 0).max().item())
    return min(index, last_positive)


def _draw(
    generator: torch.Generator | UniformStream | None, p: Qwen38SamplingParameters
) -> tuple[torch.Tensor, torch.Generator | UniformStream]:
    """One fp32 uniform: the request's torch generator (or a one-shot one from the seed), or the device sampler's
    splitmix64 stream (the same ``u`` the device would have consumed on this step)."""

    request_generator = generator
    if request_generator is None:
        request_generator = torch.Generator(device="cpu")
        request_generator.manual_seed(p.seed)
    if isinstance(request_generator, UniformStream):
        return torch.tensor(request_generator.next_uniform(), dtype=torch.float32), request_generator
    return torch.rand((), generator=request_generator, dtype=torch.float32), request_generator


class Qwen38CandidateFallback(Qwen38SamplingError):
    """The candidate row cannot prove the step exact: sample this step over the full vocabulary."""


@dataclass(frozen=True)
class Qwen38RowDistribution:
    """The post-processor distribution of one logits row: the kept tokens in the filter's order (descending
    scaled score, ties by the lowest id) and their probabilities (zero where the nucleus or min-p cut removed a
    token).  Both samplers draw from exactly this (``tokens[_inverse_cdf(probabilities, u)]``); the speculative
    acceptance reads ``probability`` and conditions with ``without``.  The dtype is the caller's (fp32 from the
    samplers; the algebra tests use float64).  ``checked=False`` skips the validation: for
    :func:`candidate_distributions`, which verified the invariants over every row of a pass at once."""

    tokens: torch.Tensor  # int64 [n]
    probabilities: torch.Tensor  # floating [n], summing to 1
    checked: InitVar[bool] = True

    def __post_init__(self, checked: bool) -> None:
        if not checked:
            return
        tokens, probabilities = self.tokens, self.probabilities
        if (
            not isinstance(tokens, torch.Tensor)
            or tokens.dtype != torch.int64
            or tokens.ndim != 1
            or not tokens.numel()
        ):
            raise ValueError(f"tokens must be a nonempty int64 vector, got {getattr(tokens, 'shape', None)}")
        if (
            not isinstance(probabilities, torch.Tensor)
            or not probabilities.dtype.is_floating_point
            or tuple(probabilities.shape) != tuple(tokens.shape)
        ):
            raise ValueError(f"probabilities must be a floating vector of {tokens.numel()}, got {probabilities}")
        if bool(torch.any(probabilities < 0)) or not bool(torch.all(torch.isfinite(probabilities))):
            raise ValueError("probabilities must be finite and nonnegative")
        if not bool(probabilities.sum() > 0):
            raise ValueError("a distribution needs positive mass")
        if tokens.unique().numel() != tokens.numel():
            raise ValueError("a distribution names every token once")

    def probability(self, token: int) -> float:
        """``p(token)``: 0 for a token outside the kept set."""

        hits = torch.nonzero(self.tokens == int(token))
        return 0.0 if hits.numel() == 0 else float(self.probabilities[int(hits[0])])

    def draw(self, uniform: float) -> int:
        """The token at ``uniform`` under the inverse CDF (the samplers' rule, in this distribution's dtype)."""

        return int(self.tokens[_inverse_cdf(self.probabilities, torch.tensor(uniform, dtype=self.probabilities.dtype))])

    def without(self, token: int) -> "Qwen38RowDistribution":
        """This distribution conditioned on ``token`` not being drawn: its mass zeroed, the rest renormalised
        (the speculative rejection distribution ``norm(max(0, p - delta(token)))``)."""

        remaining = self.probabilities * (self.tokens != int(token)).to(self.probabilities.dtype)
        total = remaining.sum()
        if not bool(total > 0):
            raise ValueError(f"no probability mass remains without token {token}")
        return Qwen38RowDistribution(self.tokens, remaining / total, False)  # the same tokens, a scaled sub-vector


@dataclass(frozen=True)
class Qwen38CandidateSample:
    """One sampled token with its raw (pre-penalty, pre-temperature) log-probability.

    ``logprob`` and ``top_logprobs`` are log-softmax values of the raw logits.  From
    :func:`sample_full_vocabulary` the softmax runs over the whole vocabulary; from
    :func:`sample_candidates` it runs over the read candidates (the epilogue reads no
    normalizer), so the row's values exceed the full ones by ``-log`` of the probability
    mass the read candidates hold: a few hundredths of a nat for a peaked distribution,
    never negative.
    """

    token_id: int
    logprob: float
    top_logprobs: tuple[tuple[int, float], ...]
    kept: int  # tokens with nonzero probability after the filters
    uniform: float | None  # the inverse-CDF draw; None for temperature 0


def _top_logprobs(values: torch.Tensor, ids: torch.Tensor, count: int, log_normalizer: float):
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= MAX_TOP_LOGPROBS:
        raise ValueError(f"top_logprobs must be an integer in [0,{MAX_TOP_LOGPROBS}], got {count!r}")
    if count == 0:
        return ()
    order = torch.argsort(values, descending=True, stable=True)[:count]
    return tuple((int(ids[i]), float(values[i]) - log_normalizer) for i in order.tolist())


def sample_full_vocabulary(
    logits: torch.Tensor,
    parameters: Qwen38SamplingParameters,
    *,
    token_history: Sequence[int] = (),
    prompt_tokens: int = 0,
    generator: torch.Generator | None = None,
    top_logprobs: int = 0,
) -> Qwen38CandidateSample:
    """The reference: one fp32 ``[VOCAB_SIZE]`` row, every filter over the whole vocabulary.

    ``token_history`` is the prompt followed by the request's output so far; its first
    ``prompt_tokens`` are exempt from the presence and frequency penalties.
    """

    if not isinstance(parameters, Qwen38SamplingParameters):
        raise TypeError("parameters must be Qwen38SamplingParameters")
    if generator is not None:
        _validate_generator(generator, seed=parameters.seed)
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (VOCAB_SIZE,) or logits.device.type != "cpu":
        raise ValueError(
            f"full-vocabulary logits must be a CPU [{VOCAB_SIZE}] row, got {getattr(logits, 'shape', None)}"
        )
    raw = logits.detach().to(torch.float32)
    if not bool(torch.all(torch.isfinite(raw))):
        raise ValueError("host logits contain NaN or infinity")
    history = _normalize_one_history(token_history, label="token history")
    log_normalizer = float(torch.logsumexp(raw.double(), dim=0))
    ids = torch.arange(VOCAB_SIZE, dtype=torch.int64)
    scores = _penalize(raw.clone(), None, history, parameters, prompt_tokens=prompt_tokens)
    if parameters.temperature == 0:
        token = int(torch.argmax(scores))  # the first (lowest-id) maximum
        return Qwen38CandidateSample(
            token, float(raw[token]) - log_normalizer, _top_logprobs(raw, ids, top_logprobs, log_normalizer), 1, None
        )
    positions, _scaled, probabilities = _filter(scores, parameters)
    uniform, _generator = _draw(generator, parameters)
    token = int(positions[_inverse_cdf(probabilities, uniform)])
    return Qwen38CandidateSample(
        token,
        float(raw[token]) - log_normalizer,
        _top_logprobs(raw, ids, top_logprobs, log_normalizer),
        int((probabilities > 0).sum()),
        float(uniform),
    )


def full_distribution(
    logits: torch.Tensor,
    parameters: Qwen38SamplingParameters,
    *,
    token_history: Sequence[int] = (),
    prompt_tokens: int = 0,
) -> Qwen38RowDistribution:
    """:func:`sample_full_vocabulary` stopped before its draw: the reference distribution of one row."""

    if not isinstance(parameters, Qwen38SamplingParameters):
        raise TypeError("parameters must be Qwen38SamplingParameters")
    if parameters.temperature == 0:
        raise ValueError("a distribution needs temperature > 0 (temperature 0 is the argmax)")
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (VOCAB_SIZE,) or logits.device.type != "cpu":
        raise ValueError(
            f"full-vocabulary logits must be a CPU [{VOCAB_SIZE}] row, got {getattr(logits, 'shape', None)}"
        )
    raw = logits.detach().to(torch.float32)
    if not bool(torch.all(torch.isfinite(raw))):
        raise ValueError("host logits contain NaN or infinity")
    history = _normalize_one_history(token_history, label="token history")
    scores = _penalize(raw.clone(), None, history, parameters, prompt_tokens=prompt_tokens)
    positions, _scaled, probabilities = _filter(scores, parameters)
    return Qwen38RowDistribution(positions, probabilities)  # over the whole vocabulary a position is its id


def sample_host_logits(
    logits: torch.Tensor,
    parameters: Qwen38SamplingParameters,
    *,
    token_histories: Sequence[int] | Sequence[Sequence[int]] = (),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample CPU full-vocabulary logits ``[1,1,rows,VOCAB_SIZE]`` row by row; ``int64 [1,1,rows]`` token IDs.

    The numerical contract of the device owner and of the candidate sampler.  If
    ``generator`` is omitted, the parameter seed starts a one-shot stream; a
    request-scoped CPU generator advances across calls and is rolled back if
    sampling raises.
    """

    if not isinstance(parameters, Qwen38SamplingParameters):
        raise TypeError("parameters must be Qwen38SamplingParameters")
    if generator is not None:
        _validate_generator(generator, seed=parameters.seed)
    scores, rows = _validate_host_logits(logits)
    histories = _normalize_histories(token_histories, rows=rows)
    request_generator = generator
    if request_generator is None and parameters.temperature != 0:
        request_generator = torch.Generator(device="cpu")
        request_generator.manual_seed(parameters.seed)
    original_state = generator.get_state().clone() if generator is not None else None
    try:
        tokens = []
        for row, history in enumerate(histories):
            penalized = _penalize(scores[row], None, history, parameters)
            if parameters.temperature == 0:
                tokens.append(int(torch.argmax(penalized)))
                continue
            positions, _scaled, probabilities = _filter(penalized, parameters)
            uniform, _generator = _draw(request_generator, parameters)
            tokens.append(int(positions[_inverse_cdf(probabilities, uniform)]))
    except BaseException:
        if original_state is not None:
            generator.set_state(original_state)
        raise
    return torch.tensor(tokens, dtype=torch.int64).reshape(1, 1, rows)


# --- the candidate row: the TAIL epilogue's readback, parsed and sampled on the host ----------------


@dataclass(frozen=True)
class Qwen38CandidateRow:
    """The parsed ``SAMPLING_CANDIDATE_ROW_SHAPE`` readback: per shard the top-k values and global ids, descending.

    The row is ``[TP_SIZE, 2, k]`` flattened: shard d's k values then its k ids.
    """

    values: torch.Tensor  # fp32 [TP_SIZE, k]
    ids: torch.Tensor  # int64 [TP_SIZE, k]

    @classmethod
    def from_host_row(cls, row: torch.Tensor) -> "Qwen38CandidateRow":
        if (
            not isinstance(row, torch.Tensor)
            or tuple(row.shape) != SAMPLING_CANDIDATE_ROW_SHAPE
            or row.dtype != torch.float32
        ):
            raise ValueError(
                f"candidate row must be fp32 {SAMPLING_CANDIDATE_ROW_SHAPE}, got "
                f"{getattr(row, 'dtype', None)} {tuple(getattr(row, 'shape', ()))}"
            )
        if not bool(torch.all(torch.isfinite(row))):
            raise ValueError("candidate row contains NaN or infinity")
        k = SAMPLING_CANDIDATES_PER_DEVICE
        packs = row.reshape(TP_SIZE, 2, k)
        values = packs[:, 0].clone()
        ids_fp32 = packs[:, 1]
        ids = ids_fp32.to(torch.int64)
        if not torch.equal(ids.to(torch.float32), ids_fp32):
            raise ValueError("candidate ids are not integers")
        starts = torch.tensor([start for start, _ in EXPECTED_VOCAB_RANGES], dtype=torch.int64).reshape(TP_SIZE, 1)
        if bool(torch.any(ids < starts)) or bool(torch.any(ids >= starts + LOCAL_VOCAB_SIZE)):
            raise ValueError(f"candidate ids leave their shards: {ids.tolist()}")
        if any(len(set(shard)) != k for shard in ids.tolist()):
            raise ValueError("a shard's candidate ids repeat")
        if bool(torch.any(values[:, 1:] > values[:, :-1])):
            raise ValueError(f"candidate values are not descending per shard: {values.tolist()}")
        return cls(values, ids)

    @property
    def log_normalizer(self) -> float:
        """log sum exp over the read candidates (the row carries no full-vocabulary normalizer)."""

        return float(torch.logsumexp(self.values.double().reshape(-1), dim=0))

    @property
    def shard_floor(self) -> torch.Tensor:
        """The largest of the shards' k-th values: every unread token's raw logit is at most this."""

        return self.values[:, -1].max()

    @classmethod
    def emulate(cls, logits: torch.Tensor) -> "Qwen38CandidateRow":
        """What the device epilogue produces for one bf16 ``[VOCAB_SIZE]`` row (``torch.topk`` per shard)."""

        if tuple(logits.shape) != (VOCAB_SIZE,) or logits.dtype != torch.bfloat16:
            raise ValueError(f"emulation needs a bf16 [{VOCAB_SIZE}] row, got {logits.dtype} {tuple(logits.shape)}")
        shards = logits.to(torch.float32).reshape(TP_SIZE, LOCAL_VOCAB_SIZE)
        values, local = torch.topk(shards, SAMPLING_CANDIDATES_PER_DEVICE, dim=-1, sorted=True)
        starts = torch.tensor([start for start, _ in EXPECTED_VOCAB_RANGES], dtype=torch.int64).reshape(TP_SIZE, 1)
        return cls(values.contiguous(), (local + starts).contiguous())

    def to_host_row(self) -> torch.Tensor:
        return torch.stack([self.values, self.ids.to(torch.float32)], dim=1).reshape(SAMPLING_CANDIDATE_ROW_SHAPE)

    def agreement(self, truth: "Qwen38CandidateRow") -> dict[str, Any]:
        """Per shard against the emulation: values bitwise; id sets equal, or equal up to boundary ties.

        Two top-k selections of one shard may keep different members of a tie group at
        the k-th value.  ``ids_equal_up_to_boundary_ties`` is true when every id one row
        has and the other lacks carries the k-th value in its row; ``boundary_ties`` is the
        number of read candidates at the k-th value (1 when there is no tie), and
        ``ids_exchanged`` how many ids the two rows chose differently.
        """

        report: dict[str, list] = {
            "values_bitwise": [],
            "ids_equal": [],
            "ids_equal_up_to_boundary_ties": [],
            "boundary_ties": [],
            "ids_exchanged": [],
        }
        for shard in range(TP_SIZE):
            mine, theirs = self.values[shard], truth.values[shard]
            my_value = dict(zip(self.ids[shard].tolist(), mine.tolist()))
            their_value = dict(zip(truth.ids[shard].tolist(), theirs.tolist()))
            only_mine = set(my_value) - set(their_value)
            only_theirs = set(their_value) - set(my_value)
            kth = float(mine[-1])
            values_bitwise = bool(torch.equal(mine, theirs))
            report["values_bitwise"].append(values_bitwise)
            report["ids_equal"].append(not only_mine and not only_theirs)
            report["ids_equal_up_to_boundary_ties"].append(
                values_bitwise
                and all(my_value[i] == kth for i in only_mine)
                and all(their_value[i] == kth for i in only_theirs)
            )
            report["boundary_ties"].append(int((mine == kth).sum()))
            report["ids_exchanged"].append(len(only_mine))
        return report


def sample_candidates(
    row: Qwen38CandidateRow,
    parameters: Qwen38SamplingParameters,
    *,
    token_history: Sequence[int] = (),
    prompt_tokens: int = 0,
    generator: torch.Generator | None = None,
    top_logprobs: int = 0,
) -> Qwen38CandidateSample:
    """Sample from the candidate row exactly as :func:`sample_full_vocabulary` would, or raise the fallback.

    Refuses ``top_k`` above ``CANDIDATE_TOP_K_LIMIT`` (a request-time error, never a
    fallback).  Raises :class:`Qwen38CandidateFallback` before any draw when
    ``top_k`` is 0, a penalty raises logits, or the kept minimum does not exceed the
    shard floor; the generator is untouched then, so the fallback sampler continues
    the same stream.  ``token_history`` and ``prompt_tokens`` as for the reference.
    """

    if not isinstance(row, Qwen38CandidateRow):
        raise TypeError("row must be a Qwen38CandidateRow")
    if not isinstance(parameters, Qwen38SamplingParameters):
        raise TypeError("parameters must be Qwen38SamplingParameters")
    if generator is not None:
        _validate_generator(generator, seed=parameters.seed)
    history = _normalize_one_history(token_history, label="token history")
    ids, raw, scores, floor = _candidate_scores(row, parameters, history, prompt_tokens)
    log_normalizer = row.log_normalizer
    if parameters.temperature == 0:
        position = int(torch.argmax(scores))
        if not bool(scores[position] > floor):
            raise Qwen38CandidateFallback(
                f"greedy candidate {float(scores[position])} does not exceed the shard floor {float(floor)}"
            )
        token = int(ids[position])
        return Qwen38CandidateSample(
            token, float(raw[position]) - log_normalizer, _top_logprobs(raw, ids, top_logprobs, log_normalizer), 1, None
        )
    positions, probabilities = _kept_candidates(scores, parameters, floor)
    uniform, _generator = _draw(generator, parameters)
    position = int(positions[_inverse_cdf(probabilities, uniform)])
    return Qwen38CandidateSample(
        int(ids[position]),
        float(raw[position]) - log_normalizer,
        _top_logprobs(raw, ids, top_logprobs, log_normalizer),
        int((probabilities > 0).sum()),
        float(uniform),
    )


def _candidate_scores(
    row: Qwen38CandidateRow, parameters: Qwen38SamplingParameters, history: tuple[int, ...], prompt_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The read candidates in ascending global id order (the stable sort then breaks ties on the lowest id, as the
    full row does), their raw and penalized scores and the shard floor; ``top_k`` above the row's k is refused, and
    ``top_k`` 0 or a boosting penalty over a history is the fallback (before any draw)."""

    if parameters.top_k > CANDIDATE_TOP_K_LIMIT:
        raise Qwen38SamplingError(f"top_k {parameters.top_k} exceeds the candidate limit {CANDIDATE_TOP_K_LIMIT}")
    if parameters.top_k == 0:
        raise Qwen38CandidateFallback("top_k 0: the nucleus may extend past the candidate row")
    if history and parameters.raises_logits:
        raise Qwen38CandidateFallback("a penalty raises logits: unread tokens are unbounded")
    ids, order = torch.sort(row.ids.reshape(-1))
    raw = row.values.reshape(-1)[order]
    scores = _penalize(raw.clone(), ids, history, parameters, prompt_tokens=prompt_tokens)
    return ids, raw, scores, row.shard_floor


def _kept_candidates(
    scores: torch.Tensor, parameters: Qwen38SamplingParameters, floor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The filters over the penalized candidates, then the guard: the kept minimum must exceed the scaled shard
    floor, else an unread token could have been kept and the step falls back."""

    positions, scaled, probabilities = _filter(scores, parameters)
    if not bool(scaled[-1] > floor / parameters.temperature):
        raise Qwen38CandidateFallback(
            f"kept minimum {float(scaled[-1])} does not exceed the scaled shard floor {float(floor / parameters.temperature)}"
        )
    return positions, probabilities


def candidate_distribution(
    row: Qwen38CandidateRow,
    parameters: Qwen38SamplingParameters,
    *,
    token_history: Sequence[int] = (),
    prompt_tokens: int = 0,
) -> Qwen38RowDistribution:
    """:func:`sample_candidates` stopped before its draw: the row's distribution, exact under the same guard (the
    same :class:`Qwen38CandidateFallback` where that sampler would fall back)."""

    if not isinstance(row, Qwen38CandidateRow):
        raise TypeError("row must be a Qwen38CandidateRow")
    if not isinstance(parameters, Qwen38SamplingParameters):
        raise TypeError("parameters must be Qwen38SamplingParameters")
    if parameters.temperature == 0:
        raise ValueError("a distribution needs temperature > 0 (temperature 0 is the argmax)")
    history = _normalize_one_history(token_history, label="token history")
    ids, _raw, scores, floor = _candidate_scores(row, parameters, history, prompt_tokens)
    positions, probabilities = _kept_candidates(scores, parameters, floor)
    return Qwen38RowDistribution(ids[positions], probabilities)


# --- the verify rows of one MTP pass: the candidate sampler's distributions in one batched pass -----------------
#
# ``candidate_distributions`` is ``candidate_distribution`` for the k + 1 rows of a pass, computed together: every
# operation of ``_penalize`` and ``_filter`` runs along the last dimension of a ``[rows, candidates]`` tensor.  On the
# CPU each of those kernels (the stable sort, softmax, the cumulative sum, the elementwise updates) works one row at a
# time, so the rows come out bitwise as the per-row function computes them; the one reduction whose 1-D and last-dim
# kernels differ (``sum``) is taken row by row.  The history is counted in C once for every row instead of walked
# in Python once per row.  ``tests/test_candidate_distributions_no_device.py`` holds the bitwise proof.


@dataclass(frozen=True)
class Qwen38CandidateRows:
    """The candidate rows of one verify pass parsed at once: row j is ``Qwen38CandidateRow(values[j], ids[j])`` (fp32
    ``[rows, TP_SIZE, k]``, int64 ``[rows, TP_SIZE, k]``); :meth:`from_host_rows` runs
    :meth:`Qwen38CandidateRow.from_host_row`'s checks over every row in a few tensor operations."""

    values: torch.Tensor
    ids: torch.Tensor

    @classmethod
    def from_host_rows(cls, rows: torch.Tensor) -> "Qwen38CandidateRows":
        """``rows`` fp32 ``[rows, 2 * TP_SIZE * k]``, row j flattened as :meth:`Qwen38CandidateRow.from_host_row` reads it."""

        width = SAMPLING_CANDIDATE_ROW_SHAPE[-1]
        if (
            not isinstance(rows, torch.Tensor)
            or rows.ndim != 2
            or rows.shape[0] == 0
            or rows.shape[1] != width
            or rows.dtype != torch.float32
        ):
            raise ValueError(
                f"candidate rows must be fp32 [rows, {width}], got "
                f"{getattr(rows, 'dtype', None)} {tuple(getattr(rows, 'shape', ()))}"
            )
        if not bool(torch.all(torch.isfinite(rows))):
            raise ValueError("candidate rows contain NaN or infinity")
        k = SAMPLING_CANDIDATES_PER_DEVICE
        packs = rows.reshape(rows.shape[0], TP_SIZE, 2, k)
        values = packs[:, :, 0].contiguous()
        ids_fp32 = packs[:, :, 1]
        ids = ids_fp32.to(torch.int64).contiguous()
        if not torch.equal(ids.to(torch.float32), ids_fp32):
            raise ValueError("candidate ids are not integers")
        starts = torch.tensor([start for start, _ in EXPECTED_VOCAB_RANGES], dtype=torch.int64).reshape(1, TP_SIZE, 1)
        if bool(torch.any(ids < starts)) or bool(torch.any(ids >= starts + LOCAL_VOCAB_SIZE)):
            raise ValueError("candidate ids leave their shards")
        ordered = torch.sort(ids, dim=-1).values
        if bool(torch.any(ordered[..., 1:] == ordered[..., :-1])):
            raise ValueError("a shard's candidate ids repeat")
        if bool(torch.any(values[..., 1:] > values[..., :-1])):
            raise ValueError("candidate values are not descending per shard")
        return cls(values, ids)

    @property
    def rows(self) -> int:
        return int(self.values.shape[0])

    def row(self, index: int) -> Qwen38CandidateRow:
        return Qwen38CandidateRow(self.values[index], self.ids[index])

    @property
    def shard_floors(self) -> torch.Tensor:
        """Every row's :attr:`Qwen38CandidateRow.shard_floor`: ``[rows]``."""

        return self.values[:, :, -1].max(dim=-1).values


@dataclass(frozen=True)
class Qwen38CandidateDistributions:
    """The rows of one pass after the processors: row j's kept tokens ``tokens[j]`` (descending scaled score, ties by
    the lowest id) and probabilities ``probabilities[j]``, or in ``outcomes[j]`` what :func:`candidate_distribution`
    raises for it (the :class:`Qwen38CandidateFallback` of a row the guard cannot bound, a ``ValueError`` or
    ``RuntimeError`` of the filters).  :meth:`row` answers like the per-row function, so a caller that asks row by row
    (the acceptance never asks past its first rejection) sees the fallbacks the per-row function would have raised."""

    tokens: torch.Tensor  # int64 [rows, kept]
    probabilities: torch.Tensor  # floating [rows, kept]
    outcomes: tuple[Exception | None, ...]

    @property
    def rows(self) -> int:
        return len(self.outcomes)

    def row(self, index: int) -> Qwen38RowDistribution:
        """Row ``index``'s distribution, or the exception :func:`candidate_distribution` raises for that row."""

        error = self.outcomes[index]
        if error is not None:
            raise error
        return Qwen38RowDistribution(self.tokens[index], self.probabilities[index], False)


def _history_tensor(values: Sequence[int], *, label: str) -> torch.Tensor:
    """:func:`_normalize_one_history` for the long committed stream, in C: ``array.array`` refuses a non-integer element
    and the range check runs vectorised; int64 ``[len]``.  The relaxation against the per-row check: any ``__index__``
    object (a bool, a numpy int, a 0-dim tensor) passes as its integer value, and an int past 64 bits is a
    ``TypeError`` rather than the range ``ValueError``; the session's committed stream holds Python ints."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence of token IDs")
    try:
        packed = array.array("q", values)
    except (TypeError, OverflowError) as error:
        raise TypeError(f"{label} must contain only integer token IDs") from error
    if not packed:
        return torch.empty(0, dtype=torch.int64)
    history = torch.frombuffer(packed, dtype=torch.int64)
    low, high = torch.aminmax(history)
    if int(low) < 0 or int(high) >= VOCAB_SIZE:
        raise ValueError(f"{label} contains a token outside [0,{VOCAB_SIZE})")
    return history


def _occurrences(unique: torch.Tensor, inverse: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """How often each candidate (``unique[inverse]``: every row's ids) occurs in ``tokens``; int64 of ``inverse``'s
    shape.  Each token finds its slot among the sorted unique ids, the slots are counted."""

    if not tokens.numel():
        return torch.zeros_like(inverse)
    slots = torch.searchsorted(unique, tokens)
    matched = unique[slots.clamp(max=unique.numel() - 1)] == tokens
    buckets = torch.where(matched, slots, unique.numel())
    return torch.bincount(buckets, minlength=unique.numel() + 1)[: unique.numel()][inverse]


def _penalize_rows(
    scores: torch.Tensor,
    ids: torch.Tensor,
    committed: torch.Tensor,
    pass_tokens: tuple[int, ...],
    p: Qwen38SamplingParameters,
    *,
    prompt_tokens: int,
) -> torch.Tensor:
    """:func:`_penalize` over every row at once, row j's history ``committed`` then ``pass_tokens[:j + 1]``: the
    per-row function's operations on the whole tensor, kept where the row's history holds the candidate.  The
    frequency count is fp32 whatever the scores' dtype, as ``_seen`` builds it."""

    rows = scores.shape[0]
    unique, inverse = torch.unique(ids, return_inverse=True)
    split = min(prompt_tokens, committed.numel())  # the committed stream's prompt part ends here
    index = torch.arange(rows)
    counted = index.reshape(-1, 1) >= index.reshape(1, -1)  # row j's history holds pass token i <= j
    in_output = (
        index >= prompt_tokens - committed.numel()
    )  # pass token i lies in the output unless the prompt covers it
    hits = ids.unsqueeze(-1) == torch.tensor(pass_tokens, dtype=torch.int64).reshape(1, 1, rows)  # [rows, n, rows]
    output_counts = _occurrences(unique, inverse, committed[split:]) + (hits & (counted & in_output).unsqueeze(1)).sum(
        -1
    )
    if p.presence_penalty != 0 or p.frequency_penalty != 0:
        seen = output_counts > 0
        if p.presence_penalty != 0:
            scores = torch.where(seen, scores - p.presence_penalty, scores)
        if p.frequency_penalty != 0:
            scores = torch.where(seen, scores - output_counts.to(torch.float32) * p.frequency_penalty, scores)
    if p.repetition_penalty != 1:
        prompt_counts = _occurrences(unique, inverse, committed[:split]) + (
            hits & (counted & ~in_output).unsqueeze(1)
        ).sum(-1)
        seen = (prompt_counts + output_counts) > 0
        scores = torch.where(
            seen, torch.where(scores < 0, scores * p.repetition_penalty, scores / p.repetition_penalty), scores
        )
    return scores


def _row_sums(values: torch.Tensor) -> torch.Tensor:
    """Each row's ``.sum()`` as :func:`_filter` takes it (the full reduction of one vector; its kernel and the
    last-dimension reduction's need not agree bitwise), ``[rows, 1]``."""

    return torch.stack([row.sum() for row in values]).reshape(-1, 1)


def _every_row(rows: Qwen38CandidateRows, error: type[Exception], message: str) -> Qwen38CandidateDistributions:
    """Every row of the pass raises ``error(message)`` (one instance per row) and holds no tokens."""

    empty = torch.empty((rows.rows, 0), dtype=torch.int64)
    return Qwen38CandidateDistributions(
        empty, empty.to(rows.values.dtype), tuple(error(message) for _ in range(rows.rows))
    )


def candidate_distributions(
    rows: Qwen38CandidateRows,
    parameters: Qwen38SamplingParameters,
    *,
    token_history: Sequence[int] = (),
    row_tokens: Sequence[int],
    prompt_tokens: int = 0,
) -> Qwen38CandidateDistributions:
    """:func:`candidate_distribution` over the ``rows`` of one MTP pass in one batched pass, bitwise: row j's history is
    ``token_history + row_tokens[:j + 1]`` (the committed stream, then the pass's ``[t_P, d_1 .. d_j]``), its first
    ``prompt_tokens`` exempt from the additive penalties.  The checks and fallbacks are the per-row function's: the
    refusals (``top_k`` above the row's k, temperature 0) are raised here, ``top_k`` 0 and a boosting penalty make
    every row fall back, the guard and the filters' errors are recorded per row
    (:attr:`Qwen38CandidateDistributions.outcomes`)."""

    if not isinstance(rows, Qwen38CandidateRows):
        raise TypeError("rows must be Qwen38CandidateRows")
    if not isinstance(parameters, Qwen38SamplingParameters):
        raise TypeError("parameters must be Qwen38SamplingParameters")
    if parameters.temperature == 0:
        raise ValueError("a distribution needs temperature > 0 (temperature 0 is the argmax)")
    count = rows.rows
    pass_tokens = _normalize_one_history(row_tokens, label="row tokens")
    if len(pass_tokens) != count:
        raise ValueError(f"expected one token per row ({count}), got {len(pass_tokens)}")
    committed = _history_tensor(token_history, label="token history")
    if parameters.top_k > CANDIDATE_TOP_K_LIMIT:
        raise Qwen38SamplingError(f"top_k {parameters.top_k} exceeds the candidate limit {CANDIDATE_TOP_K_LIMIT}")
    if parameters.top_k == 0:
        return _every_row(rows, Qwen38CandidateFallback, "top_k 0: the nucleus may extend past the candidate row")
    if parameters.raises_logits:  # every row's history holds at least its own token
        return _every_row(rows, Qwen38CandidateFallback, "a penalty raises logits: unread tokens are unbounded")
    shortest = committed.numel() + 1  # row 0's history, the shortest
    if isinstance(prompt_tokens, bool) or type(prompt_tokens) is not int or not 0 <= prompt_tokens <= shortest:
        raise ValueError(f"prompt_tokens must be an integer in [0, {shortest}], got {prompt_tokens!r}")

    ids, order = torch.sort(rows.ids.reshape(count, -1), dim=-1)
    scores = torch.gather(rows.values.reshape(count, -1), 1, order)
    if parameters.penalizes:
        scores = _penalize_rows(scores, ids, committed, pass_tokens, parameters, prompt_tokens=prompt_tokens)
    # _filter along the last dimension
    scaled = scores / parameters.temperature
    finite = torch.isfinite(scaled).all(dim=-1)
    positions = torch.argsort(scaled, dim=-1, descending=True, stable=True)[:, : parameters.top_k]
    ordered = torch.gather(scaled, 1, positions)
    probabilities = torch.softmax(ordered, dim=-1)
    if parameters.top_p < 1:
        remove = torch.cumsum(probabilities, dim=-1) > parameters.top_p
        # keep the token that crosses the boundary: the mask shifts one column right
        remove = torch.cat([torch.zeros((count, 1), dtype=torch.bool), remove[:, :-1]], dim=1)
        probabilities = probabilities.masked_fill(remove, 0.0)
        probabilities = probabilities / _row_sums(probabilities)
    if parameters.min_p > 0:
        remove = probabilities < parameters.min_p * probabilities[:, :1]  # column 0 holds each row's maximum
        remove[:, 0] = False
        probabilities = probabilities.masked_fill(remove, 0.0)
        probabilities = probabilities / _row_sums(probabilities)
    positive = torch.isfinite(probabilities).all(dim=-1) & (probabilities.sum(dim=-1) > 0)
    # _kept_candidates' guard per row
    floors = rows.shard_floors / parameters.temperature
    kept_minimum = ordered[:, -1]
    exact = kept_minimum > floors
    outcomes: list[Exception | None] = []
    for row, (is_finite, is_positive, is_exact) in enumerate(zip(finite.tolist(), positive.tolist(), exact.tolist())):
        if not is_finite:
            outcomes.append(ValueError("temperature scaling produced non-finite logits"))
        elif not is_positive:
            outcomes.append(RuntimeError("the filters removed every finite candidate"))
        elif not is_exact:
            outcomes.append(
                Qwen38CandidateFallback(
                    f"kept minimum {float(kept_minimum[row])} does not exceed the scaled shard floor {float(floors[row])}"
                )
            )
        else:
            outcomes.append(None)
    return Qwen38CandidateDistributions(torch.gather(ids, 1, positions), probabilities, tuple(outcomes))


class Qwen38TTNNHostSampler:
    """Provenance-bound TP4 sampler for one already-open, qualified mesh.

    Construction and calls never discover or open hardware.  The supplied LM
    head and live identity must already belong to the task-owned target.
    """

    def __init__(
        self,
        lm_head: Qwen38TTNNLMHead,
        live_identity: Qwen38LiveBuildIdentity,
        *,
        expected_provenance: Qwen38BuildProvenance,
        expected_identity_key: str,
        eos_token_ids: Sequence[int],
        _runtime_ops: _RuntimeOps | None = None,
        clock_ns: Clock = time.perf_counter_ns,
    ) -> None:
        if type(lm_head) is not Qwen38TTNNLMHead:
            raise TypeError("host sampler requires the exact Qwen38TTNNLMHead owner")
        if type(live_identity) is not Qwen38LiveBuildIdentity:
            raise TypeError("host sampler requires the exact Qwen38LiveBuildIdentity")
        if type(expected_provenance) is not Qwen38BuildProvenance:
            raise TypeError("expected_provenance must be a validated Qwen38BuildProvenance")
        identity_key = _require_key(expected_identity_key, label="expected_identity_key")
        if live_identity.provenance != expected_provenance:
            raise ValueError("sampler live identity and independently supplied provenance differ")
        if live_identity.key != identity_key:
            raise ValueError(f"sampler live identity {live_identity.key} differs from expected {identity_key}")
        if tuple(live_identity.mesh_shape) != MESH_SHAPE:
            raise ValueError(f"sampler requires logical mesh {MESH_SHAPE}, got {live_identity.mesh_shape}")
        if type(lm_head.mesh_contract) is not Qwen38MeshContract:
            raise TypeError("LM head does not carry the exact Qwen38MeshContract")
        if lm_head.mesh_contract.physical_ids != live_identity.physical_ids:
            raise ValueError("LM-head physical order differs from the live build identity")
        lm_head.mesh_contract.validate_mesh(lm_head.mesh_device)
        if _collective_name(lm_head.collective_topology) != live_identity.collective_topology:
            raise ValueError("LM-head collective topology differs from the live build identity")
        if tuple(lm_head.weights.vocab_ranges) != EXPECTED_VOCAB_RANGES:
            raise ValueError("LM head does not own the exact four contiguous Qwen3.8 vocabulary ranges")
        if _runtime_ops is not None and type(_runtime_ops) is not _RuntimeOps:
            raise TypeError("_runtime_ops must be the exact internal runtime operation bundle")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")

        self.lm_head = lm_head
        self.mesh_device = lm_head.mesh_device
        self.mesh_contract = lm_head.mesh_contract
        self.live_identity = live_identity
        self.provenance = expected_provenance
        self.identity_key = identity_key
        self.eos_token_ids = _normalize_eos_ids(eos_token_ids)
        self._runtime_ops = _runtime_ops or _default_runtime_ops()
        self._clock_ns = clock_ns
        self._owned: dict[int, _OwnedTensor] = {}
        self._closed = False

    @property
    def pending_owned_tensor_count(self) -> int:
        return len(self._owned)

    @property
    def closed(self) -> bool:
        return self._closed

    def _validate_logits(self, logits: Qwen38ShardedLogits) -> int:
        if type(logits) is not Qwen38ShardedLogits:
            raise TypeError("sampler requires Qwen38ShardedLogits with explicit vocabulary ownership")
        if tuple(logits.vocab_ranges) != EXPECTED_VOCAB_RANGES:
            raise ValueError("logit ranges must be four distinct contiguous vocabulary shards in mesh order")
        shape = tuple(logits.global_shape)
        if (
            len(shape) != 4
            or shape[:2] != (1, 1)
            or isinstance(shape[2], bool)
            or not isinstance(shape[2], int)
            or shape[2] <= 0
            or shape[3] != VOCAB_SIZE
            or logits.shard_dim != 3
        ):
            raise ValueError(f"sharded logits have invalid global shape/axis: {shape}, dim={logits.shard_dim}")
        rows = shape[2]
        if _shape(logits.tensor) != (1, 1, rows, LOCAL_VOCAB_SIZE):
            raise ValueError("local logits do not contain exactly one quarter of the vocabulary")
        if logits.tensor.dtype != ttnn.bfloat16 or logits.tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError("local logits must be TILE BF16")
        self.mesh_contract.validate_tensor(
            logits.tensor,
            placement=TensorPlacement.VOCAB_SHARDED,
            shard_dim=3,
        )
        tensor_mesh = logits.tensor.device()
        if (
            tensor_mesh is None
            or _shape(tensor_mesh) != MESH_SHAPE
            or tuple(int(value) for value in tensor_mesh.get_device_ids()) != self.live_identity.physical_ids
        ):
            raise RuntimeError("logit tensor is not resident on the provenance-bound physical mesh")
        return rows

    def _register_owned(self, tensor: Any, *, label: str, borrowed: Any) -> int:
        if tensor is borrowed:
            raise RuntimeError(f"{label} unexpectedly aliases the borrowed sharded logits")
        if not callable(getattr(tensor, "is_allocated", None)):
            raise TypeError(f"owned {label} has no retry-safe is_allocated() query")
        key = id(tensor)
        if key in self._owned:
            raise RuntimeError(f"owned {label} identity is already registered")
        self._owned[key] = _OwnedTensor(tensor=tensor, label=label)
        return key

    def _release_one(self, key: int) -> None:
        owner = self._owned[key]
        if not bool(owner.tensor.is_allocated()):
            # A prior deallocation may have succeeded and then raised.  Proving
            # the allocation is gone makes this retry safe without double-free.
            del self._owned[key]
            return
        self._runtime_ops.deallocate(owner.tensor)
        if bool(owner.tensor.is_allocated()):
            raise RuntimeError(f"deallocate returned but owned {owner.label} remains allocated")
        del self._owned[key]

    def release_owned_tensors(self) -> None:
        """Retry every pending task-owned release; successful calls are idempotent."""

        errors = []
        for key in tuple(self._owned):
            try:
                self._release_one(key)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise Qwen38SamplingCleanupError("release_owned_tensors", errors)

    def close(self) -> None:
        """Release pending gathers and make this owner permanently unusable."""

        if self._closed:
            return
        self.release_owned_tensors()
        self._closed = True

    def _ensure_ready(self) -> None:
        if self._closed:
            raise Qwen38SamplingError("sampler is closed")
        if self._owned:
            self.release_owned_tensors()

    def _read_full_logits(self, logits: Qwen38ShardedLogits, *, rows: int) -> torch.Tensor:
        gathered = self._runtime_ops.all_gather(
            logits.tensor,
            dim=3,
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.lm_head.collective_topology,
        )
        key = self._register_owned(gathered, label="replicated full-logit gather", borrowed=logits.tensor)
        primary_error: BaseException | None = None
        host: torch.Tensor | None = None
        try:
            self.mesh_contract.validate_tensor(gathered, placement=TensorPlacement.REPLICATED)
            gathered_mesh = gathered.device()
            if (
                gathered_mesh is None
                or _shape(gathered_mesh) != MESH_SHAPE
                or tuple(int(value) for value in gathered_mesh.get_device_ids()) != self.live_identity.physical_ids
            ):
                raise RuntimeError("full-logit gather moved outside the provenance-bound physical mesh")
            if (
                _shape(gathered) != (1, 1, rows, VOCAB_SIZE)
                or gathered.dtype != ttnn.bfloat16
                or gathered.layout != ttnn.TILE_LAYOUT
            ):
                raise RuntimeError("full-logit gather did not produce replicated TILE BF16 global vocabulary")
            composer = self._runtime_ops.make_concat_composer(self.mesh_device, 0)
            replicas = self._runtime_ops.to_torch(gathered, mesh_composer=composer)
            expected = (TP_SIZE, 1, rows, VOCAB_SIZE)
            if (
                not isinstance(replicas, torch.Tensor)
                or replicas.device.type != "cpu"
                or tuple(replicas.shape) != expected
            ):
                actual = (
                    None if not isinstance(replicas, torch.Tensor) else (replicas.device.type, tuple(replicas.shape))
                )
                raise RuntimeError(f"full-logit host readback must be CPU {expected}, got {actual}")
            first = replicas[0:1]
            for coordinate in range(1, TP_SIZE):
                if not torch.equal(first, replicas[coordinate : coordinate + 1]):
                    raise RuntimeError(f"replicated full-logit readback differs at mesh coordinate {coordinate}")
            host = first.contiguous()
        except BaseException as error:
            primary_error = error

        try:
            self._release_one(key)
        except BaseException as cleanup_error:
            raise Qwen38SamplingCleanupError(
                "full-logit readback",
                (cleanup_error,),
                primary_error=primary_error,
            ) from cleanup_error
        if primary_error is not None:
            raise primary_error
        if host is None:
            raise AssertionError("successful full-logit readback did not retain a CPU tensor")
        return host

    def sample(
        self,
        logits: Qwen38ShardedLogits,
        parameters: Qwen38SamplingParameters,
        *,
        token_histories: Sequence[int] | Sequence[Sequence[int]] = (),
        source_identity_key: str,
        generator: torch.Generator | None = None,
    ) -> Qwen38SampledTokens:
        """Gather, read back, and sample one exact TP4 logit object."""

        request_start = self._clock_ns()
        self._ensure_ready()
        declared_identity = _require_key(source_identity_key, label="source_identity_key")
        if declared_identity != self.identity_key:
            raise ValueError("logit source identity differs from the sampler's pinned live target")
        if not isinstance(parameters, Qwen38SamplingParameters):
            raise TypeError("parameters must be Qwen38SamplingParameters")
        if generator is not None:
            _validate_generator(generator, seed=parameters.seed)
        rows = self._validate_logits(logits)
        # Validate histories before allocating the full-logit gather.
        histories = _normalize_histories(token_histories, rows=rows)
        readback_start = self._clock_ns()
        host_logits = self._read_full_logits(logits, rows=rows)
        readback_end = self._clock_ns()
        original_generator_state = generator.get_state().clone() if generator is not None else None
        try:
            token_ids = sample_host_logits(
                host_logits,
                parameters,
                token_histories=histories,
                generator=generator,
            )
            sampling_end = self._clock_ns()
            eos_ids = torch.tensor(self.eos_token_ids, dtype=torch.int64)
            eos_mask = torch.isin(token_ids, eos_ids)
            request_end = self._clock_ns()
            timing = Qwen38SamplingTiming(
                full_logit_gather_readback_ns=_duration(
                    readback_start,
                    readback_end,
                    label="full-logit gather/readback",
                ),
                host_filter_sample_ns=_duration(readback_end, sampling_end, label="host filter/sample"),
                end_to_end_ns=_duration(request_start, request_end, label="sampling end-to-end"),
            )
            if timing.validation_and_packaging_ns < 0:
                raise RuntimeError("sampling timing domains overlap or exceed the end-to-end boundary")
            return Qwen38SampledTokens(
                token_ids=token_ids,
                eos_mask=eos_mask,
                parameters=parameters,
                timing=timing,
                source_identity_key=self.identity_key,
                source_provenance_key=self.provenance.key,
                rng_mode="per-call-seed" if generator is None else "request-generator",
            )
        except BaseException:
            if original_generator_state is not None:
                generator.set_state(original_generator_state)
            raise


__all__ = [
    "CANDIDATE_TOP_K_LIMIT",
    "EXPECTED_VOCAB_RANGES",
    "MAX_TOP_LOGPROBS",
    "Qwen38CandidateDistributions",
    "Qwen38CandidateFallback",
    "Qwen38CandidateRow",
    "Qwen38CandidateRows",
    "Qwen38CandidateSample",
    "Qwen38RowDistribution",
    "Qwen38SampledTokens",
    "Qwen38SamplingCleanupError",
    "Qwen38SamplingError",
    "Qwen38SamplingParameters",
    "Qwen38SamplingProfile",
    "Qwen38SamplingTiming",
    "Qwen38TTNNHostSampler",
    "UNIFORM_BITS",
    "UniformStream",
    "candidate_distribution",
    "candidate_distributions",
    "full_distribution",
    "sample_candidates",
    "sample_full_vocabulary",
    "sample_host_logits",
]
