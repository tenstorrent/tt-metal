# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU compatibility sampling; never used by measured device decode.

One HostSampler owns one request, so moving slots means moving its object.
Filtering uses installed HF processors. Ordering, generated-only frequency /
presence / bad-word history, and raw logprobs follow vLLM v0.31.0's sampler.
The CPU exponential draw follows vLLM's native algorithm; cross-device RNG
bit parity is not promised. The future serving adapter should normally return
raw logits to the plugin's own compatibility sampler instead of sampling twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Mapping

import torch
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    NoBadWordsLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


@dataclass(frozen=True)
class HostSamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    # None=no logprobs; 0=sampled only; -1=full vocabulary.
    logprobs: int | None = None
    bad_words_token_ids: tuple[tuple[int, ...], ...] = ()
    logit_bias: Mapping[int, float] = field(default_factory=dict)
    allowed_token_ids: tuple[int, ...] | None = None
    min_tokens: int = 0
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ("temperature", "top_p", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0 or self.repetition_penalty <= 0:
            raise ValueError("temperature must be nonnegative and repetition_penalty positive")
        if not 0 <= self.min_p <= 1 or not 0 <= self.top_p <= 1:
            raise ValueError("min_p and top_p must lie in [0,1]")
        if self.top_p == 0 and self.temperature >= 1e-5 and self.top_k != 1:
            raise ValueError("top_p=0 is accepted only for a greedy configuration")
        if not isinstance(self.top_k, int) or self.top_k < -1:
            raise ValueError("top_k must be -1, 0, or a positive integer")
        if not isinstance(self.min_tokens, int) or self.min_tokens < 0:
            raise ValueError("min_tokens must be a nonnegative integer")
        if self.logprobs is not None and (not isinstance(self.logprobs, int) or self.logprobs < -1):
            raise ValueError("logprobs must be None, -1, or a nonnegative integer")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an integer or None")
        if self.allowed_token_ids is not None and not self.allowed_token_ids:
            raise ValueError("allowed_token_ids cannot be empty")
        if any(not sequence for sequence in self.bad_words_token_ids):
            raise ValueError("bad-word token sequences cannot be empty")
        if any(not math.isfinite(float(value)) for value in self.logit_bias.values()):
            raise ValueError("logit biases must be finite")
        # Detach mutable caller collections from request state.
        object.__setattr__(self, "bad_words_token_ids", tuple(tuple(s) for s in self.bad_words_token_ids))
        object.__setattr__(
            self, "logit_bias", {int(k): max(-100.0, min(100.0, float(v))) for k, v in self.logit_bias.items()}
        )
        object.__setattr__(self, "eos_token_ids", tuple(self.eos_token_ids))
        if self.allowed_token_ids is not None:
            object.__setattr__(self, "allowed_token_ids", tuple(self.allowed_token_ids))

    @classmethod
    def from_object(cls, params, *, row=0, **overrides):
        """Select scalar/per-row common sampling fields without importing TTNN.

        Token lists, bad-word sequences and bias maps are request-wide fields.
        String bad words must be tokenized by the caller with the HF tokenizer.
        Unknown explicit overrides fail at construction rather than being ignored.
        """
        if row < 0:
            raise ValueError("row must be nonnegative")
        values = {}
        per_row = {
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "logprobs",
        }

        def get(name, default=None):
            value = params.get(name, default) if isinstance(params, Mapping) else getattr(params, name, default)
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            if name in per_row or name in {"enable_log_probs", "num_logprobs"}:
                if isinstance(value, (list, tuple)):
                    if row >= len(value):
                        raise ValueError(f"No sampling value for {name} row {row}")
                    value = value[row]
            return value

        for item in fields(cls):
            value = get(item.name)
            if value is not None:
                values[item.name] = value
        if "logprobs" not in values and get("enable_log_probs", False):
            values["logprobs"] = int(get("num_logprobs", 0))
        for unsupported in ("logits_processors", "structured_outputs"):
            if get(unsupported):
                raise ValueError(
                    f"{unsupported} requires the serving plugin; use allowed_token_mask for a prepared grammar"
                )
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class HostSample:
    token_id: int
    sampled_logprob: float | None = None
    selected_token_rank: int | None = None
    top_logprobs: dict[int, float] | None = None
    full_logprobs: torch.Tensor | None = None


class HostSampler:
    """Per-request RNG and token history. No device access or global RNG mutation."""

    def __init__(self, params=None, *, prompt_token_ids=(), output_token_ids=()):
        self.params = params if isinstance(params, HostSamplingParams) else HostSamplingParams.from_object(params or {})
        self.prompt_token_ids = [int(x) for x in prompt_token_ids]
        self.output_token_ids = [int(x) for x in output_token_ids]
        self.generator = torch.Generator(device="cpu")
        if self.params.seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(self.params.seed % (2**64))
        self._bad_words = (
            NoBadWordsLogitsProcessor([list(s) for s in self.params.bad_words_token_ids], eos_token_id=None)
            if self.params.bad_words_token_ids
            else None
        )

    def append_token(self, token_id):
        """Commit caller-selected feedback when sample(commit=False) is used."""
        token_id = int(token_id)
        if token_id < 0:
            raise ValueError("Feedback token must be nonnegative")
        self.output_token_ids.append(token_id)

    def _processed_logits(self, logits, allowed_token_mask=None):
        if logits.device.type != "cpu" or logits.numel() != logits.shape[-1]:
            raise ValueError("HostSampler requires one CPU vocabulary row")
        raw = logits.detach().float().reshape(1, -1).clone()
        if torch.isnan(raw).any() or torch.isposinf(raw).any() or not torch.isfinite(raw).any():
            raise ValueError("Logits must contain a finite candidate and no NaN/+inf")
        scores, p = raw.clone(), self.params
        vocab = scores.shape[-1]
        ids = self.prompt_token_ids + self.output_token_ids
        controls = list(p.logit_bias) + list(p.eos_token_ids) + [x for s in p.bad_words_token_ids for x in s]
        if p.allowed_token_ids is not None:
            controls += list(p.allowed_token_ids)
        if any(not isinstance(x, int) or not 0 <= x < vocab for x in ids + controls):
            raise ValueError("History or control token lies outside the vocabulary")
        if p.allowed_token_ids is not None:
            allow = torch.zeros(vocab, dtype=torch.bool)
            allow[list(p.allowed_token_ids)] = True
            scores[:, ~allow] = -torch.inf
        if allowed_token_mask is not None:
            if (
                allowed_token_mask.device.type != "cpu"
                or allowed_token_mask.dtype != torch.bool
                or allowed_token_mask.numel() != vocab
            ):
                raise ValueError("allowed_token_mask must be a CPU bool vocabulary vector")
            scores[:, ~allowed_token_mask.reshape(-1)] = -torch.inf
        generated = torch.tensor([self.output_token_ids], dtype=torch.long)
        if self._bad_words is not None:
            scores = self._bad_words(generated, scores)
        if len(self.output_token_ids) < p.min_tokens:
            scores[:, list(p.eos_token_ids)] = -torch.inf
        for token_id, bias in p.logit_bias.items():
            scores[:, token_id] += bias
        if p.repetition_penalty != 1.0 and ids:
            scores = RepetitionPenaltyLogitsProcessor(float(p.repetition_penalty))(torch.tensor([ids]), scores)
        if self.output_token_ids:
            counts = torch.bincount(generated.flatten(), minlength=vocab).float()
            scores -= p.frequency_penalty * counts + p.presence_penalty * (counts > 0)
        if p.temperature >= 1e-5:
            scores = TemperatureLogitsWarper(float(p.temperature))(generated, scores)
            if p.min_p:
                scores = MinPLogitsWarper(float(p.min_p))(generated, scores)
            if p.top_k > 0:
                scores = TopKLogitsWarper(p.top_k)(generated, scores)
            if p.top_p < 1:
                scores = TopPLogitsWarper(float(p.top_p))(generated, scores)
        if not torch.isfinite(scores).any() or torch.isnan(scores).any() or torch.isposinf(scores).any():
            raise ValueError("Sampling controls leave no valid finite distribution")
        return raw[0], scores[0]

    def sample(self, logits, *, commit=True, allowed_token_mask=None):
        """Return one token and optional raw logprobs, never modify input logits.

        Each call consumes one RNG draw when stochastic, even if commit=False.
        That option lets teacher forcing append its actual feedback token instead.
        A prepared grammar can supply an allow-mask; grammar compilation and stop
        matching remain caller/plugin responsibilities.
        """
        raw, scores = self._processed_logits(logits, allowed_token_mask)
        if self.params.temperature < 1e-5:
            token = int(scores.argmax())
        else:
            noise = torch.empty_like(scores).exponential_(generator=self.generator)
            token = int((scores.softmax(-1) / noise).argmax())
        result = HostSample(token)
        if self.params.logprobs is not None:
            raw_logprobs = raw.log_softmax(-1)
            count = self.params.logprobs
            full = raw_logprobs.clone() if count == -1 else None
            top = None
            if count >= 0:
                values, indices = raw_logprobs.topk(min(count, raw.numel()))
                top = dict(zip(indices.tolist(), values.tolist(), strict=True))
                top[token] = float(raw_logprobs[token])
            result = HostSample(token, float(raw_logprobs[token]), int((raw >= raw[token]).sum()), top, full)
        if commit:
            self.append_token(token)
        return result
