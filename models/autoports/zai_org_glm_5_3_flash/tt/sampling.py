# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sampler contract for GLM-5.3-Flash: host reference + the TTNN op-level plan Stage 6 implements (no device code here).

Host reference (used by the generator's explicit host-sampling compatibility mode and by the tests as the oracle):
  * `SamplingParams` (temperature / top_k / top_p / seed; temperature 0 or top_k 1 = greedy) and the checkpoint
    defaults (`generation_config.json`: temperature 1.0, top_p 0.95, no top_k).
  * `greedy(logits)`: argmax, ties -> lowest token id (the device tie-break of `models/common/sampling`).
  * `warp(logits, params)`: transformers' logits-warper order (temperature -> top-k -> top-p, filter value -inf,
    at least one token kept) - `TemperatureLogitsWarper`, `TopKLogitsWarper`, `TopPLogitsWarper`.
  * `HostSampler`: per-slot seeded `torch.Generator`s; `sample(logits [B, V], params per row)`.
  * `candidate_sample_reference`: the split top-k candidate algorithm of the chosen device path (per-shard local
    top-`max_top_k`, gather of the candidates, k/p/temperature inside the candidate set), used to prove on host that
    it is exactly greedy for k = 1 and to quantify where it differs from full-vocab sampling.

TTNN plan (decision record; see `SAMPLER_PLAN` and $GLM53_WS/doc/ttnn/06a-full-model-prep/contract.md section 7):
  Chosen: `models/common/sampling/` (`SamplingGenerator` wrapping `TTSampling`).
  Rejected: `models/common/modules/sampling/sampling_1d.py` (`Sampling1D`).
  Model facts: vocab 154,880 = 4,840 tiles (untied lm_head [4096, 154880] bfp8, logits bf16 or fp32), decode batch
  1..32, first 8-chip placement = layer pipeline over eight 1x1 submeshes (doc/DECISIONS.md, Stage 0b): final
  mean-of-streams + RMSNorm + lm_head + sampler on ONE chip (the last stage) with full-vocab logits; possible later
  TP8 vocab-sharded head (plan C2: 154,880 -> 155,648 padded, 19,456 columns/chip) on a 1x8 view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

VOCAB_SIZE = 154880
TILE = 32
MAX_TOP_K = 32  # device candidate count per shard / per vocab split (ttnn.sampling policy in both common samplers)
SAMPLING_MAX_USERS = 32  # ttnn.sampling: one core per user, 1..32 users per call
TOPK_MAX_WIDTH = 64 * 1024  # models/common/sampling/tt_sampling.py TOPK_MAX_WIDTH (policy, not an op limit)


@dataclass(frozen=True)
class SamplingParams:
    """Per-request sampling parameters. temperature == 0 (or top_k == 1) is greedy; top_k 0/None = no top-k;
    top_p 1.0 = no nucleus truncation; seed None = unseeded (device PRNG / fresh host generator)."""

    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(f"temperature {self.temperature} < 0")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError(f"top_k {self.top_k} < 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p {self.top_p} outside (0, 1]")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0 or self.top_k == 1

    @property
    def device_mode(self) -> str:
        """Trace key of the device sampler: 'greedy' (semantically greedy split path k=1, p=0, temp=1) or 'sample'."""
        return "greedy" if self.is_greedy else "sample"

    def device_values(self, max_top_k: int = MAX_TOP_K) -> tuple[int, float, float]:
        """(k, p, temperature) as the device sampler's persistent buffers take them. Greedy -> (1, 0.0, 1.0)
        (models/common/sampling greedy row convention). top_k is clamped to [1, max_top_k]: requests with
        top_k == 0 or > max_top_k are approximated by the top-`max_top_k` candidates (see SAMPLER_PLAN)."""
        if self.is_greedy:
            return 1, 0.0, 1.0
        k = max_top_k if not self.top_k else min(self.top_k, max_top_k)
        return k, float(self.top_p), float(self.temperature)

    @classmethod
    def from_generation_config(cls, generation_config: dict, seed: int | None = None) -> "SamplingParams":
        return cls(
            temperature=float(generation_config.get("temperature", 1.0)),
            top_k=generation_config.get("top_k"),
            top_p=float(generation_config.get("top_p", 1.0)),
            seed=seed,
        )


GREEDY = SamplingParams()
CHECKPOINT_DEFAULT = SamplingParams(temperature=1.0, top_k=None, top_p=0.95)


def mask_padded_vocab(logits: torch.Tensor, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Logits wider than the model vocabulary (TP-padded lm_head columns) -> padding columns at -inf."""
    if logits.shape[-1] == vocab_size:
        return logits
    if logits.shape[-1] < vocab_size:
        raise ValueError(f"logits width {logits.shape[-1]} < vocab {vocab_size}")
    out = logits.clone()
    out[..., vocab_size:] = float("-inf")
    return out


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """argmax over the last dim; ties resolve to the lowest token id (torch.argmax returns the first maximum)."""
    return torch.argmax(logits, dim=-1)


def warp(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """transformers' warper order on one row or a batch: temperature, top-k, top-p (filter -inf, keep >= 1)."""
    x = logits.float()
    if params.temperature not in (0.0, 1.0):
        x = x / params.temperature
    if params.top_k:
        k = min(int(params.top_k), x.shape[-1])
        kth = torch.topk(x, k, dim=-1).values[..., -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    if params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=False, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove_sorted = cum <= (1 - params.top_p)
        remove_sorted[..., -1:] = False  # min_tokens_to_keep = 1
        remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
        x = x.masked_fill(remove, float("-inf"))
    return x


def probabilities(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Sampling distribution of the host reference (one-hot at the argmax for greedy)."""
    if params.is_greedy:
        out = torch.zeros_like(logits, dtype=torch.float32)
        out.scatter_(-1, greedy(logits).unsqueeze(-1), 1.0)
        return out
    return warp(logits, params).softmax(dim=-1)


class HostSampler:
    """Host reference sampler with per-slot seeded generators (compatibility mode / test oracle).
    A slot's generator is (re)seeded at `reset_slot` (request boundary); unseeded requests draw from a generator
    seeded from the slot index and a per-sampler counter, so runs are reproducible given the same request order."""

    def __init__(self, max_batch: int, base_seed: int = 0):
        self.max_batch = max_batch
        self.base_seed = base_seed
        self._gens = [torch.Generator().manual_seed(base_seed + i) for i in range(max_batch)]
        self._counter = 0

    def reset_slot(self, slot: int, seed: int | None):
        self._counter += 1
        s = seed if seed is not None else (self.base_seed * 1_000_003 + 7919 * self._counter + slot)
        self._gens[slot] = torch.Generator().manual_seed(int(s))

    def state(self):
        return [g.get_state() for g in self._gens], self._counter

    def restore(self, state):
        states, self._counter = state
        for g, s in zip(self._gens, states):
            g.set_state(s)

    def sample(self, logits: torch.Tensor, params: Sequence[SamplingParams], slots: Sequence[int]) -> torch.Tensor:
        """logits [B, V] (rows in `slots` order) -> tokens [B] int64."""
        if logits.dim() != 2 or logits.shape[0] != len(params) or len(params) != len(slots):
            raise ValueError("logits [B, V] with one params entry and one slot per row")
        out = torch.empty(logits.shape[0], dtype=torch.int64)
        for r, (p, s) in enumerate(zip(params, slots)):
            if p.is_greedy:
                out[r] = greedy(logits[r])
            else:
                probs = warp(logits[r], p).softmax(dim=-1)
                out[r] = torch.multinomial(probs, 1, generator=self._gens[s])[0]
        return out


# ------------------------------------------------------------------------------------------- device algorithm model
def vocab_splits_single_device(padded_vocab: int, max_width: int = TOPK_MAX_WIDTH) -> int | None:
    """models/common/sampling TTSampling.num_single_device_vocab_splits: fewest power-of-two tile-aligned chunks of
    <= 64K columns (154,880 -> 4 chunks of 38,720 = 1,210 tiles)."""
    n = 2
    while padded_vocab // n > max_width:
        n *= 2
    if padded_vocab % n or (padded_vocab // n) % TILE:
        return None
    return n


def tp_padded_vocab(vocab: int, n_devices: int, tile: int = TILE) -> int:
    """tt_transformers compute_padded_vocab_size: nearest multiple of tile x devices (154,880 is already one for 8)."""
    q = tile * n_devices
    return ((vocab + q - 1) // q) * q


def candidate_sample_reference(
    logits: torch.Tensor,
    params: SamplingParams,
    *,
    n_shards: int,
    max_top_k: int = MAX_TOP_K,
    generator: torch.Generator | None = None,
    vocab_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Host model of the chosen device path for one row: split the (padded) vocab into `n_shards` equal chunks
    (vocab splits on one chip, or TP shards), take each chunk's local top-`max_top_k` (values, global ids), concatenate
    the candidates (n_shards x max_top_k, a power-of-two tile count as ttnn.sampling requires), then apply
    k / p / temperature inside the candidate set and sample (greedy rows: lowest-id argmax among tied maxima).
    Returns (token, candidate_ids, candidate_probs)."""
    x = logits.float().reshape(-1)
    if vocab_size is not None and x.numel() > vocab_size:
        x = mask_padded_vocab(x, vocab_size)
    if x.numel() % n_shards:
        raise ValueError(f"width {x.numel()} not divisible into {n_shards} shards")
    w = x.numel() // n_shards
    vals, ids = [], []
    for s in range(n_shards):
        v, i = torch.topk(x[s * w : (s + 1) * w], min(max_top_k, w))
        vals.append(v)
        ids.append(i + s * w)
    cv, ci = torch.cat(vals), torch.cat(ids)
    k, p, temp = params.device_values(max_top_k)
    if params.is_greedy:
        best = cv.max()
        tok = ci[cv == best].min()
        probs = (ci == tok).float()
        return tok, ci, probs
    order = torch.argsort(cv, descending=True, stable=True)
    cv, ci = cv[order][:k], ci[order][:k]
    sub = SamplingParams(temperature=temp, top_k=None, top_p=p)
    probs = warp(cv, sub).softmax(dim=-1)
    tok = ci[torch.multinomial(probs, 1, generator=generator)[0]]
    return tok, ci, probs


def nucleus_size(logits: torch.Tensor, top_p: float, temperature: float = 1.0) -> torch.Tensor:
    """Tokens kept by top-p (after temperature) per row - how many candidates full-vocab nucleus sampling needs."""
    w = warp(logits, SamplingParams(temperature=temperature, top_k=None, top_p=top_p))
    return torch.isfinite(w).sum(dim=-1)


# ------------------------------------------------------------------------------------------- decision record
SAMPLER_PLAN = {
    "chosen": "models/common/sampling (SamplingGenerator -> TTSampling)",
    "rejected": "models/common/modules/sampling/sampling_1d.py (Sampling1D)",
    "model_contract": {
        "vocab_size": VOCAB_SIZE,
        "vocab_tiles": VOCAB_SIZE // TILE,
        "logits_dtype": "bf16 (sampler typecasts fp32 to bf16 on the top-k path)",
        "decode_batch": "1..32 (ttnn.sampling: 1..32 users per call; TTSampling pads max_batch to 32 rows)",
        "placement_first": "layer pipeline over 8 x 1x1 submeshes; norm + lm_head + sampler on the last stage (one chip, full vocab)",
        "placement_later": "TP8 vocab-sharded lm_head on a 1x8 view (154,880 divides by 8 x 32: 19,360 cols/chip; plan C2 pads to 155,648 = 19,456/chip)",
    },
    "single_chip_path": {
        "vocab_splits": vocab_splits_single_device(VOCAB_SIZE),
        "split_width": VOCAB_SIZE // vocab_splits_single_device(VOCAB_SIZE),
        "candidates": vocab_splits_single_device(VOCAB_SIZE) * MAX_TOP_K,
        "blackhole_topk": "one full-row ttnn.topk(k=128) routed to topk_large_indices (width <= 2^19, k multiple of 16)",
        "greedy": "top-k path with k=1/p=0/temp=1 and the lowest-id tie-break (force_argmax optional, chunked untilize)",
    },
    "tp8_path": "local top-32 per chip -> all-gather of [1,1,32,32] values + indices -> ttnn.sampling over 256 candidates (8 tiles); 1x8 view only (2x4 cannot split the vocab over both axes in TTSampling)",
    "trace": "decode trace up to sampler-ready logits (32 rows, DRAM interleaved) + TTSampling trace (or its eager forward inside the caller's trace) with tt_out_tok = the persistent decode token input",
    "approximation": "requests with top_k 0 (the checkpoint default: temperature 1.0, top_p 0.95, no top_k) or top_k > 32 are sampled from the top-32 candidates per split (128 on one chip, 256 on TP8); exact for greedy and for top_k <= 32",
    "open_items": [
        "token feedback across the pipeline: tt_out_tok lives on the last-stage chip; the next step's token input is consumed by the embedding on stage 0 - needs a device-side hop (ring closure) or a replicated embedding on the last stage that forwards streams, not a host round trip",
        "explicit per-request seeds disable TTSampling's own trace (models/common/sampling/generator.py) - run it eagerly inside the model's decode trace or keep seeded requests on the sampled (untraced) variant",
        "TTPenalties allocates ~99 MB of [32, padded_vocab] int32 buffers on the terminal chip even when penalties are unused",
        "no logprobs on one chip (LogProbsCalculator needs 8 or 32 devices)",
    ],
}
