# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter over the traced chain: one resident slot, host sampling from full-vocabulary logits, no MTP.

vllm-tt-plugin constructs ``Qwen38ForCausalLM`` through ``initialize_vllm_model`` on the mesh it opened.  The chain
is built as the chat server builds it (``sampling=True``, so the full-vocabulary gather compiles in the warm pass)
and is complete when ``Qwen38TracedChain.open`` returns: both warmups are checks plus one smoke pass.
``prefill_forward`` resets the state, runs the chunk driver over all prompt tokens but the last, teacher-forces the
last one and reads its full logits; ``decode_forward`` teacher-forces the token vLLM sampled and reads the next row.
Both return CPU fp32 ``[1, 1, VOCAB_SIZE]`` with the LM head's padding rows at ``-inf``.  vLLM's page table and KV
cache are accepted and ignored: the QSA caches are the chain's resident ones, masked by position.  Every
configuration the adapter cannot serve is refused before the device is touched.

Under the plugin's device-sampling contract (``--additional-config '{"tt": {"sample_on_device_mode": "decode_only"}}'``)
a decode step arrives with ``sampling_params`` (the plugin's ``TTSamplingParams`` as per-row lists) and must return the
token instead of the row.  A plain greedy row (temperature 0, no penalty) returns the device's greedy
token from the token row, for decode and for prefill, without the full-logits gather (a row that asks for the sampled token's logprob keeps
its token and also reads its full row).  Any other row: the adapter reads the full row and samples on the host with vLLM's sampler semantics
(``vllm/v1/sample/sampler.py``: penalties, temperature, top-k, top-p, exponential-noise draw) computed over the row's
top candidates wherever that is exact, and over the full row otherwise. vLLM's own ``Sampler`` sorts the full
248,320-wide row. The reduced route preserves its distribution; the token stream for a given seed differs because
the noise is drawn over the candidates. Historical fork sampling timings are not measurements of this port.
The plugin decides host or device per step before the call (min_p, logit_bias, bad_words, allowed_token_ids,
min_tokens, structured output and top-N logprobs stay on its host sampler) and a step without ``sampling_params`` returns
the row as before.  At 8 or 32 devices the plugin keeps a row with ``logprobs=0`` (the sampled token's logprob) on device
sampling and expects ``(tokens, logprobs)`` back whenever any row of the step asks for it: the adapter reads those rows,
returns the raw logprob (``log_softmax`` of the row, padding at ``-inf``, before temperature, penalties, top-k and
top-p; vLLM's ``raw_logprobs``) and never changes the drawn token.  Prefill sampling (mode ``"all"``) is refused at warmup: prefill returns the row.

The prefill form is the environment's (``prefill_form``): ``QWEN38_PREFILL_SLAB`` (default 2048) opens the chain
with a prefill slab of that many rows ahead of the 128-row and 32-row chunks; ``0`` or ``off`` serves without the
slab, where ``QWEN38_LONG_CHUNKS=1`` still turns the 128-row chunks on.  The slab's resident dense weights are
admitted against the free DRAM when the chain opens; a refusal ends the start-up naming the knob, never a silent
fallback.  ``QWEN38_PREFILL_PLAN`` (``cost``, the default, or ``legacy``; anything else is refused by name) picks how a
prompt's rows are covered by those forms: ``cost`` the cheapest sequence of passes with one padded last pass, ``legacy``
the slabs, then 128-row chunks, then 32-row chunks and a padded 32-row tail (``tools/qwen38_prefill_driver``).  One log
line after the open states the form the chain opened in and the plan.

Batch mode (``max_num_seqs`` 2..32, P2a): one plain-lane-body trace with B = ``max_num_seqs`` lanes, residue-free, over
a paged KV pool that IS vLLM's block pool (page size = block size, page id = vLLM block id; vLLM never hands out block
0, which is the pool's scratch page).  ``initialize_vllm_model`` builds the weights and components but does not open the
chain; ``allocate_kv_cache`` reads ``num_blocks`` and ``block_size`` from the plugin's shape, opens the chain with the
lane decoder's ``prepare`` as its warm hook (the lane tensors must exist before the chain captures) and captures the
lane trace.  ``prefill_forward`` runs the B=1 chain prefill per row and imports it into the lane that owns
``empty_slots[i]``; ``decode_forward`` replays the lane trace once for all live rows.  A host map slot -> lane follows
the plugin's ``slot_remap`` (no device moves: the lanes stay where they are); with decode buckets on (32 lanes) a new
request takes the lowest free lane, swapping the slot -> lane map, so live requests stay in the low lanes.  A seeded host draw uses a generator
seeded from (seed, position), so it does not depend on ``reset_sampling_state`` or on the other lanes.  ``max_num_seqs`` 1
keeps the single-slot path above unchanged.

vLLM imports this module in its registry inspection, frontend, engine and worker processes: no device is opened
here and ``sys.argv`` is not read.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy
import torch
from loguru import logger
from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING

import ttnn
from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.checkpoint import PINNED_CHECKPOINT_REVISION, checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import (
    MODE,
    construct_live_decode_diagnostic,
    prepare_live_decode_diagnostic,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_batch_decode import (
    WARM_TOKEN_ID,
    Qwen38PlainLaneDecoder,
    pack_buckets_from_env,
    pack_dram_bytes,
    prefill_pack_enabled,
    row_read_bucket,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    CHUNK_PREFILL_MIN_ROWS,
    RESIDUE_CLASSES,
    SEED_TOKEN_ID,
    WARM_CHUNK_TOKEN_IDS,
    Qwen38ChatChainError,
    Qwen38TracedChain,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_pack_plan import pack_force_enabled, plan_prefill_groups
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_prefill_driver import prefill_plan
from models.demos.blackhole.qwen38_flash_next.tools.runtime_admission import (
    RuntimeAdmissionError,
    git_identity,
    sha256_of,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_QSA_CACHE_CAPACITIES, Qwen38ResidentContext
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    DEFAULT_SLAB_ROWS,
    FABRIC_CONFIG,
    MESH_SHAPE,
    SLAB_ROWS_RULE,
    TP_SIZE,
    is_slab_rows,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import LOCAL_VOCAB_SIZE, SAMPLING_CANDIDATES_PER_DEVICE
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging

OPTIMIZATIONS = (None, "performance", "accuracy")  # the plugin's tt.optimizations values; no effect on this chain
# The warmup smoke's prompt: one full 32-row chunk, a 7-row padded tail, the hand-off, one forced step, one gather.
WARMUP_PROMPT_TOKEN_IDS = tuple(WARM_CHUNK_TOKEN_IDS[index % len(WARM_CHUNK_TOKEN_IDS)] for index in range(40))
PROTOCOL_SHIM_MESSAGE = "Qwen38ForCausalLM runs on the device through prefill_forward and decode_forward"
HF_REPO_ID = "Qwen/Qwen3.8-Flash-Next"
SAMPLING_EPS = 1e-5  # vLLM's _SAMPLING_EPS: a temperature below it is greedy
REDUCED_CANDIDATES = 1024  # the top candidates the host sampler works on in place of the full row
PREFILL_SLAB_VARIABLE = "QWEN38_PREFILL_SLAB"  # the slab's rows; default DEFAULT_SLAB_ROWS; "0" / "off": no slab
LONG_CHUNKS_VARIABLE = "QWEN38_LONG_CHUNKS"  # "1": the 128-row chunks without a slab (a slab implies them)
PREFILL_SLAB_OFF = ("0", "off")
BATCH_MAX_LANES = 32  # batch mode: max_num_seqs 2..32 (one 32-row tile)
MAX_TOKENS_ALL_USERS_VARIABLE = "QWEN38_MAX_TOKENS_ALL_USERS"  # the KV token pool the plugin sizes its blocks from
PLE_EARLY_DECODE_VARIABLE = (
    "QWEN38_PLE_EARLY"  # "1": B=1 decode does the PLE lookup + host row before HEAD (default off)
)
TOKEN_FEEDBACK_VARIABLE = "QWEN38_TOKEN_FEEDBACK"  # B=1 decode skips the token-row write when the device row already holds the token (default on since x53: -0.35..-0.62 ms; "0" off)
HOST_FAST_VARIABLE = "QWEN38_HOST_FAST"  # "1": B=1 decode skips redundant per-step host work (default off)
BATCH_CANDIDATES_VARIABLE = "QWEN38_BATCH_CANDIDATES"  # "1" (default) / "0": candidate rows for the sampled lanes
BATCH_NORMALIZER_VARIABLE = (
    "QWEN38_BATCH_NORMALIZER"  # "1" (default) / "0": the device normalizer for top-k-off sampled rows
)
BATCH_ROW_READ_VARIABLE = "QWEN38_BATCH_ROW_READ"  # "1" (default) / "0": a fallback step reads only its rows' logits
DECODE_BUCKETS_VARIABLE = (
    "QWEN38_DECODE_BUCKETS"  # unset: 1,2,4,8,16,32 (32 lanes); "off" / "0": one full-width trace; or a list
)
DEFAULT_DECODE_BUCKETS = (1, 2, 4, 8, 16, 32)
DECODE_COMPACT_VARIABLE = (
    "QWEN38_DECODE_COMPACT"  # "1" (default) / "0": move a high live lane down so a step replays a narrower bucket
)
DECODE_BUCKETS_OFF = ("0", "off")
CANDIDATE_FALLBACK_REASONS = ("top_k", "penalty", "guard", "nonfinite", "nucleus")
# The host branches of ``sample_from_row`` (the keys of a ``branch_counts`` dict): greedy; top_k <= the candidates; the
# nucleus over the top 1024 values; the nucleus over the argsorted full row; the whole row by inverse CDF (top-k off,
# top_p >= 1); the top-k threshold over the full row (1024 < top_k < vocab).
HOST_DRAW_BRANCHES = (
    "greedy",
    "top_k",
    "nucleus_1024",
    "nucleus_full",
    "inverse_cdf",
    "top_k_threshold",
)  # fmt: skip
DECODE_LOG_INTERVAL = 256  # batch mode: one adapter log line per this many decode calls
BATCH_LANE_INDEXER_FORM = "wide"
KV_BYTES_PER_TOKEN = 12 * 512 * 2  # 12 QSA layers x one 512-wide bf16 row, per device
BATCH_MIN_BLOCK_SIZE = 32  # the QSA page scatter works in 32-row slabs
BATCH_MAX_BLOCK_SIZE = 4096
# The DRAM admission's refusal of the slab's resident prefill weights (ttnn/prefill_dense.py, admit_prefill_dense_dram).
PREFILL_DENSE_REFUSAL_MARK = "refused: the resident prefill weights need"


@dataclass(frozen=True)
class Qwen38HostSamplingPolicy:
    """One request's sampling parameters as the plugin's ``TTSamplingParams`` row carries them: the plugin has
    normalized ``top_k`` (``<= 0`` or ``>= vocab`` to the vocabulary size, ``top_p`` to 0 when ``top_k`` is 1) and
    turned the seed sentinel into ``None``."""

    temperature: float
    top_k: int
    top_p: float
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    logprobs: bool = False  # the row asks for the sampled token's raw logprob (``logprobs=0``); never changes the token

    @property
    def greedy(self) -> bool:
        return self.temperature < SAMPLING_EPS

    @property
    def penalized(self) -> bool:
        return self.presence_penalty != 0.0 or self.frequency_penalty != 0.0 or self.repetition_penalty != 1.0

    @classmethod
    def from_contract(cls, sampling_params: Any, row: int | None = None) -> "Qwen38HostSamplingPolicy":
        """The one slot's row of the plugin's per-row lists (tensors and scalars accepted); refused before the device
        is touched when it holds more than one row or asks for top-N logprobs (``num_logprobs`` above 0 with
        ``enable_log_probs``: the plugin keeps such a step on its host sampler, so it is never sent here).  Batch mode
        passes ``row``: that row of the padded per-row lists (a scalar applies to every row)."""

        def one(name: str, default: Any = None) -> Any:
            value = getattr(sampling_params, name, default)
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                if row is None:
                    if len(value) != 1:
                        raise ValueError(
                            f"one resident slot: sampling_params.{name} must hold one row, got {len(value)}"
                        )
                    value = value[0]
                else:
                    if not 0 <= row < len(value):
                        raise ValueError(f"sampling_params.{name} holds {len(value)} rows, row {row} is missing")
                    value = value[row]
            return value

        logprobs = bool(one("enable_log_probs", False))
        if logprobs and int(one("num_logprobs", 0) or 0) > 0:
            raise ValueError(
                f"num_logprobs {one('num_logprobs')}: top-N logprobs are not produced under device sampling "
                "(the plugin keeps them on its host sampler); only the sampled token's logprob (0) is"
            )
        temperature, top_k, top_p = float(one("temperature")), int(one("top_k")), float(one("top_p"))
        if temperature < 0.0 or not 0.0 <= top_p <= 1.0:
            raise ValueError(f"sampling_params out of range: temperature {temperature}, top_p {top_p}")
        seed = one("seed", None)
        return cls(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=float(one("presence_penalty", 0.0)),
            frequency_penalty=float(one("frequency_penalty", 0.0)),
            repetition_penalty=float(one("repetition_penalty", 1.0)),
            seed=None if seed is None else int(seed),
            logprobs=logprobs,
        )


def logprobs_requested(sampling_params: Any) -> bool:
    """The plugin's own test for a tuple return: ANY entry of the step's ``enable_log_probs`` (the whole padded list,
    padding rows included) is true."""

    value = getattr(sampling_params, "enable_log_probs", False)
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    return any(value) if isinstance(value, (list, tuple)) else bool(value)


def raw_logprob(row: torch.Tensor, token: int) -> float:
    """``log_softmax(row)[token]`` of one raw fp32 row (the LM head's padding at ``-inf``): vLLM's ``raw_logprobs``, before
    temperature, penalties, top-k and top-p."""

    return float(row[token] - torch.logsumexp(row, 0))


def penalty_history(prompt_tokens: Any, output_tokens: Any, row: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Row ``row``'s (the one slot's: 0) prompt and output token ids out of the contract's ``[rows, L]`` int32
    histories (``-1`` pads the rows and the batch), sent by the plugin whenever a penalty is active."""

    if prompt_tokens is None or output_tokens is None:
        raise ValueError("a penalty is active but the plugin sent no prompt_tokens / output_tokens history")

    def ids(history: Any) -> torch.Tensor:
        history = torch.as_tensor(history)
        if history.dim() != 2 or history.shape[0] < row + 1:
            raise ValueError(f"token history must be [rows, L] with row {row}, got {tuple(history.shape)}")
        ids_row = history[row].to(torch.long)
        return ids_row[(ids_row >= 0) & (ids_row < VOCAB_SIZE)]

    return ids(prompt_tokens), ids(output_tokens)


def device_greedy_contract(sampling_params: Any, row: int | None = None) -> bool:
    """True when the contract is plain greedy for the one row: temperature below the epsilon (the policy's own
    ``greedy``) and no penalty active (presence 0, frequency 0, repetition 1).  Such a row's token is the device's greedy
    resolve, so the full logits need not leave the device for the token.  A logprobs request does not change that (it
    reads the row besides)."""

    if sampling_params is None:
        return False
    policy = Qwen38HostSamplingPolicy.from_contract(sampling_params, row)
    return policy.greedy and not policy.penalized


def greedy_batch_sampling_params(rows: int) -> SimpleNamespace:
    """The plugin's ``TTSamplingParams`` (as lists) for ``rows`` plain greedy rows: the warmup's and the tests' input."""

    return SimpleNamespace(
        temperature=[0.0] * rows,
        top_k=[1] * rows,
        top_p=[0.0] * rows,
        presence_penalty=[0.0] * rows,
        frequency_penalty=[0.0] * rows,
        repetition_penalty=[1.0] * rows,
        seed=[None] * rows,
        num_logprobs=[0] * rows,
        enable_log_probs=[False] * rows,
    )


def apply_penalties(
    row: torch.Tensor, prompt_ids: torch.Tensor, output_ids: torch.Tensor, policy: Qwen38HostSamplingPolicy
) -> torch.Tensor:
    """vLLM's rule (``apply_penalties``, vllm/model_executor/layers/utils.py) on one fp32 row, in place: the
    repetition penalty divides the positive logits of tokens seen in the prompt or the output and multiplies the
    negative ones; the frequency penalty subtracts penalty x count over the output; the presence penalty subtracts
    itself once for every token seen in the output."""

    counts = torch.bincount(output_ids, minlength=row.numel())
    seen = counts > 0
    if policy.repetition_penalty != 1.0:  # vLLM's expressions, so the row is bitwise its torch path's
        penalized = seen.clone()
        penalized[prompt_ids] = True
        penalty = torch.tensor(policy.repetition_penalty, dtype=row.dtype)
        row[penalized] *= torch.where(row[penalized] > 0, 1.0 / penalty, penalty)
    if policy.frequency_penalty != 0.0:
        row -= policy.frequency_penalty * counts
    if policy.presence_penalty != 0.0:
        row -= policy.presence_penalty * seen
    return row


def sample_from_row(
    row: torch.Tensor,
    policy: Qwen38HostSamplingPolicy,
    generator: torch.Generator | None,
    *,
    candidates: int = REDUCED_CANDIDATES,
    branch_counts: dict[str, int] | None = None,
) -> int:
    """vLLM's sampler over one fp32 row (``Sampler.sample`` + ``apply_top_k_top_p_pytorch`` + ``random_sample``):
    greedy below the temperature epsilon; else temperature, the top-k set (every value equal to the k-th kept), the
    nucleus (a token is kept while the probability mass ranked above it is below top_p; the first always), then the
    exponential-noise argmax over the kept set.  Computed on the row's top ``candidates`` values when that is exact: the
    top-k set fits in them, or top-k is off and their mass over the full row's softmax reaches top_p (they hold the
    nucleus).  When it does not, the full row ordered by ``numpy.argsort(-row)`` (default kind, not stable,
    deterministic for the same input) gives the nucleus.  Top-k off with top_p at 1 keeps the whole row: the draw is the
    inverse CDF of its softmax (float64 cumulative sum, one uniform variate), never a token of zero probability.  A
    top-k wider than the candidates takes vLLM's full-row threshold.  Same distribution as vLLM's sampler; the noise is
    drawn over the kept set, so a seed does not reproduce vLLM's token stream.
    ``branch_counts`` (optional): the branch that drew the row (``HOST_DRAW_BRANCHES``) is counted in it."""

    def count(branch: str) -> None:
        if branch_counts is not None:
            branch_counts[branch] = branch_counts.get(branch, 0) + 1

    if policy.greedy:
        count("greedy")
        return int(row.argmax())
    vocab = row.numel()
    top_k = policy.top_k if 0 < policy.top_k < vocab else vocab
    top_p = policy.top_p
    if top_k <= candidates:
        count("top_k")
        values, indices = row.topk(top_k)
        extra = int((row >= values[-1]).sum()) - top_k
        if extra > 0:
            values, indices = row.topk(top_k + extra)
        probs = torch.softmax(values / policy.temperature, 0)
    elif top_k == vocab:
        scaled = row / policy.temperature
        if top_p >= 1.0:
            count("inverse_cdf")
            return _sample_whole_row(scaled, generator)
        lse = torch.logsumexp(scaled, 0)
        values, indices = row.topk(candidates)
        probs = torch.exp(values / policy.temperature - lse)
        if float(probs.sum()) >= top_p:
            count(f"nucleus_{candidates}")
        else:  # the nucleus is wider than the candidates: the whole row, ordered by numpy's (unstable) argsort
            count("nucleus_full")
            indices = torch.from_numpy(numpy.argsort(-row.cpu().numpy()))
            values = row[indices]
            probs = torch.exp(values / policy.temperature - lse)
    else:
        count("top_k_threshold")
        return _sample_full_row(row / policy.temperature, top_k, top_p, generator)
    if top_p < 1.0:
        keep = torch.cumsum(probs, 0) - probs < top_p
        keep[0] = True
        probs, indices = probs[keep], indices[keep]
    noise = torch.empty_like(probs).exponential_(generator=generator)
    return int(indices[(probs / noise).argmax()])


def decode_compact_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """``QWEN38_DECODE_COMPACT``: "1" or unset = a step first moves live lanes down into free low lanes when that lets it
    replay a narrower decode bucket (W5), "0" = lanes stay where they were admitted; anything else is refused.  Only used with
    decode buckets on."""

    value = (os.environ if environ is None else environ).get(DECODE_COMPACT_VARIABLE)
    if value is None:
        return True
    if value not in ("0", "1"):
        raise ValueError(f"{DECODE_COMPACT_VARIABLE} must be 0 or 1, got {value!r}")
    return value == "1"


def batch_candidates_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """``QWEN38_BATCH_CANDIDATES``: "1" or unset = the sampled lanes draw from the device candidate rows (P3g-1), "0" = every
    sampled row reads the full gathered logits (the P2 path); anything else is refused."""

    value = (os.environ if environ is None else environ).get(BATCH_CANDIDATES_VARIABLE)
    if value is None:
        return True
    if value not in ("0", "1"):
        raise ValueError(f"{BATCH_CANDIDATES_VARIABLE} must be 0 or 1, got {value!r}")
    return value == "1"


def batch_normalizer_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """``QWEN38_BATCH_NORMALIZER``: "1" or unset = a sampled row with top-k off draws from the candidate rows with the device
    normalizer (P3g-2), "0" = such rows read the full gathered logits (the P3g-1 path); anything else is refused.  It only
    matters while ``QWEN38_BATCH_CANDIDATES`` is on."""

    value = (os.environ if environ is None else environ).get(BATCH_NORMALIZER_VARIABLE)
    if value is None:
        return True
    if value not in ("0", "1"):
        raise ValueError(f"{BATCH_NORMALIZER_VARIABLE} must be 0 or 1, got {value!r}")
    return value == "1"


def batch_row_read_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """``QWEN38_BATCH_ROW_READ``: "1" or unset = a step with fallback rows reads only those lanes' logits rows from the
    device (P3g-3), "0" = it reads the full gathered logits of every lane (the P3g-2 path); anything else is refused.
    It
    only matters while ``QWEN38_BATCH_CANDIDATES`` is on."""

    value = (os.environ if environ is None else environ).get(BATCH_ROW_READ_VARIABLE)
    if value is None:
        return True
    if value not in ("0", "1"):
        raise ValueError(f"{BATCH_ROW_READ_VARIABLE} must be 0 or 1, got {value!r}")
    return value == "1"


def decode_buckets_from_env(lanes: int, environ: Mapping[str, str] | None = None) -> tuple[int, ...]:
    """``QWEN38_DECODE_BUCKETS``: the decode-width buckets of the lane decoder.  Unset or empty = ``(1, 2, 4, 8, 16, 32)``
    with 32 lanes, else the one full-width trace ``(lanes,)``; "off" or "0" = ``(lanes,)``; otherwise a comma list of
    distinct widths in 1..lanes that contains ``lanes``.  More than one bucket needs 32 lanes; anything else is refused.
    """

    value = (os.environ if environ is None else environ).get(DECODE_BUCKETS_VARIABLE)
    if value is None or not value.strip():
        return DEFAULT_DECODE_BUCKETS if lanes == 32 else (lanes,)
    value = value.strip()
    if value.lower() in DECODE_BUCKETS_OFF:
        return (lanes,)
    try:
        buckets = tuple(int(part) for part in value.split(","))
    except ValueError:
        raise ValueError(f"{DECODE_BUCKETS_VARIABLE} must be off, 0 or a comma list of widths, got {value!r}") from None
    if list(buckets) != sorted(set(buckets)) or any(not 1 <= width <= lanes for width in buckets):
        raise ValueError(f"{DECODE_BUCKETS_VARIABLE} must be sorted distinct widths in 1..{lanes}, got {value!r}")
    if buckets[-1] != lanes:
        raise ValueError(f"{DECODE_BUCKETS_VARIABLE} must contain the full width {lanes}, got {value!r}")
    if len(buckets) > 1 and lanes != 32:
        raise ValueError(f"{DECODE_BUCKETS_VARIABLE}: more than one bucket needs 32 lanes, got {lanes} lanes")
    return buckets


class CandidateFallback(Exception):
    """A lane's candidate row cannot give :func:`sample_from_row`'s law exactly: the row draws over the full logits.
    ``reason`` is one of ``CANDIDATE_FALLBACK_REASONS``; raised before any draw, so the generator is untouched."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def top_k_is_off(policy: Qwen38HostSamplingPolicy) -> bool:
    """``sample_from_row``'s top-k-off test: ``top_k`` <= 0 or at least the vocabulary size (the plugin sends the size)."""

    return not 0 < policy.top_k < VOCAB_SIZE


def candidate_route(policy: Qwen38HostSamplingPolicy, normalizer: bool = False) -> str | None:
    """The fallback reason fixed by the policy alone, else ``None`` (the row may draw from candidates): ``top_k`` outside
    1 .. 32 (the plugin turns top_k <= 0 into the vocabulary size), or a penalty that raises logits (unread tokens would
    become unbounded).  ``normalizer``: the device normalizer is available, so a sampled row with top-k off may draw from the
    candidates (:func:`normalizer_law`) unless it is penalized (the penalties change the normalizer: reason ``penalty``) or
    has ``top_p`` >= 1 (the whole row is the nucleus: reason ``top_k``)."""

    if not 0 < policy.top_k <= SAMPLING_CANDIDATES_PER_DEVICE:
        if not (normalizer and top_k_is_off(policy) and not policy.greedy):
            return "top_k"
        if policy.penalized:
            return "penalty"
        return "top_k" if policy.top_p >= 1.0 else None
    if policy.presence_penalty < 0 or policy.frequency_penalty < 0 or policy.repetition_penalty < 1.0:
        return "penalty"
    return None


def needs_normalizer(policy: Qwen38HostSamplingPolicy) -> bool:
    """True for a row :func:`candidate_route` (with the normalizer) lets through with top-k off: it needs the full-row
    normalizer of its lane."""

    return candidate_route(policy, normalizer=True) is None and not 0 < policy.top_k <= SAMPLING_CANDIDATES_PER_DEVICE


def parse_candidate_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The device candidate rows fp32 ``[n, 256]`` to ``(values [n, 4, 32] fp32, ids [n, 4, 32] int64, valid [n] bool)``;
    a row is valid when every lane is finite, the ids are integers inside their shard and a shard's ids do not repeat.
    """

    count = rows.shape[0]
    k = SAMPLING_CANDIDATES_PER_DEVICE
    packs = rows.reshape(count, TP_SIZE, 2, k)
    values, ids_fp32 = packs[:, :, 0], packs[:, :, 1]
    finite = torch.isfinite(rows).all(dim=1)
    ids = torch.where(torch.isfinite(ids_fp32), ids_fp32, torch.zeros(())).to(torch.int64)
    starts = torch.arange(TP_SIZE, dtype=torch.int64).reshape(1, TP_SIZE, 1) * LOCAL_VOCAB_SIZE
    valid = finite & (ids.to(torch.float32) == ids_fp32).all(dim=(1, 2))
    valid &= ((ids >= starts) & (ids < starts + LOCAL_VOCAB_SIZE)).all(dim=(1, 2))
    ordered = torch.sort(ids, dim=-1).values
    valid &= ~(ordered[..., 1:] == ordered[..., :-1]).any(dim=(1, 2))
    return values, ids, valid


def candidate_law(
    values: torch.Tensor,
    ids: torch.Tensor,
    policy: Qwen38HostSamplingPolicy,
    history: tuple[torch.Tensor, torch.Tensor] | None = None,
    lse: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`sample_from_row`'s law for one lane from its candidate row (``values`` fp32 ``[4, 32]``, ``ids`` int64
    ``[4, 32]``, each shard descending): ``(probabilities, token ids)`` of the kept set in the order the noise argmax
    reads it.  The same top-k set (with every value tied to the k-th), temperature softmax and nucleus as the full-row
    code, over 128 candidates; the penalties of :func:`apply_penalties` on the candidates only.

    Raises :class:`CandidateFallback` (before any draw) where the row cannot be exact: ``top_k`` outside 1 .. 32 or a
    penalty that raises logits (:func:`candidate_route`), a non-finite candidate, or the guard: the k-th kept score must
    exceed the shard floor (the largest of the shards' 32nd values), else an unread token could belong to the kept set.
    A penalty only lowers a score here, so every unread token stays at or under the floor.

    ``lse`` (the lane's full-row ``logsumexp(row / T)``, the device normalizer's four shard values combined): a top-k-off
    row (:func:`normalizer_law`) is drawn with it; without it such a row raises ``top_k``."""

    reason = candidate_route(policy, normalizer=lse is not None)
    if reason is None and lse is not None and not 0 < policy.top_k <= SAMPLING_CANDIDATES_PER_DEVICE:
        return normalizer_law(values, ids, policy, lse)
    if reason is not None:
        raise CandidateFallback(reason, f"top_k {policy.top_k}" if reason == "top_k" else "penalty raises logits")
    flat_values, flat_ids = values.reshape(-1), ids.reshape(-1)
    if not bool(torch.isfinite(flat_values).all()):
        raise CandidateFallback("nonfinite", "candidate row holds a non-finite value")
    floor = values[:, -1].max()
    scores = flat_values.clone()
    scores[flat_ids >= TOKENIZER_SIZE] = float("-inf")  # the LM head's padding columns: masked on the full row too
    if policy.penalized:
        if history is None:
            raise ValueError("a penalized policy needs the row's prompt and output history")
        prompt_ids, output_ids = history
        counts = (output_ids.reshape(-1, 1) == flat_ids.reshape(1, -1)).sum(dim=0)
        seen = counts > 0
        if policy.repetition_penalty != 1.0:  # apply_penalties' expressions on the candidates
            penalized = seen | (prompt_ids.reshape(-1, 1) == flat_ids.reshape(1, -1)).any(dim=0)
            penalty = torch.tensor(policy.repetition_penalty, dtype=scores.dtype)
            scores[penalized] *= torch.where(scores[penalized] > 0, 1.0 / penalty, penalty)
        if policy.frequency_penalty != 0.0:
            scores -= policy.frequency_penalty * counts
        if policy.presence_penalty != 0.0:
            scores -= policy.presence_penalty * seen
    if policy.greedy:  # a penalized greedy row: the penalized argmax, the lowest id among ties (torch.argmax's rule)
        best = scores.max()
        if not bool(best > floor):
            raise CandidateFallback(
                "guard", f"greedy score {float(best)} does not exceed the shard floor {float(floor)}"
            )
        return torch.ones(1), flat_ids[scores == best].min().reshape(1)
    top_k = policy.top_k
    top, positions = scores.topk(top_k)
    extra = int((scores >= top[-1]).sum()) - top_k
    if extra > 0:
        top, positions = scores.topk(top_k + extra)
    if not bool(top[-1] > floor):
        raise CandidateFallback("guard", f"k-th score {float(top[-1])} does not exceed the shard floor {float(floor)}")
    probs = torch.softmax(top / policy.temperature, 0)
    indices = flat_ids[positions]
    if policy.top_p < 1.0:
        keep = torch.cumsum(probs, 0) - probs < policy.top_p
        keep[0] = True
        probs, indices = probs[keep], indices[keep]
    return probs, indices


def combine_shard_normalizers(shard_lse: torch.Tensor) -> float:
    """The lane's full-row ``logsumexp(row / T)`` from the device's four shard values (fp32 ``[4]``): their logsumexp."""

    return float(torch.logsumexp(shard_lse.to(torch.float32), 0))


def normalizer_law(
    values: torch.Tensor, ids: torch.Tensor, policy: Qwen38HostSamplingPolicy, lse: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`sample_from_row`'s top-k-off law for one lane from its candidate row and the full-row normalizer: the same
    kept set (every token whose probability mass ranked above it is below ``top_p``, the first always) and the same
    probabilities ``exp(value / T - lse)`` over the full row, in the order the noise argmax reads them.

    Exactness guard: every kept candidate must lie strictly above the shard floor (the largest of the four shards' 32nd
    values).  A token missing from the candidates is at or under its shard's 32nd value, so every token above the last kept
    one is then in the candidates, with the full row's cumulative mass; otherwise (``CandidateFallback("nucleus")``) the
    nucleus may reach unread tokens, and the row draws over the full logits.  Requires 0 < ``top_p`` < 1 and no penalty
    (:func:`candidate_route`).  Raises ``nonfinite`` for a non-finite candidate or normalizer."""

    flat_values, flat_ids = values.reshape(-1), ids.reshape(-1)
    if not bool(torch.isfinite(flat_values).all()) or not math.isfinite(lse):
        raise CandidateFallback("nonfinite", "candidate row or normalizer holds a non-finite value")
    floor = values[:, -1].max()
    scores = flat_values.clone()
    scores[flat_ids >= TOKENIZER_SIZE] = float("-inf")  # the LM head's padding columns, left out of the normalizer too
    order = torch.argsort(scores, descending=True, stable=True)
    scores, indices = scores[order], flat_ids[order]
    probs = torch.exp(scores / policy.temperature - lse)
    keep = torch.cumsum(probs, 0) - probs < policy.top_p
    keep[0] = True
    last = int(keep.nonzero().max())
    if not bool(scores[last] > floor):
        raise CandidateFallback(
            "nucleus", f"the nucleus reaches {float(scores[last])}, not above the shard floor {float(floor)}"
        )
    return probs[keep], indices[keep]


def sample_from_candidates(
    values: torch.Tensor,
    ids: torch.Tensor,
    policy: Qwen38HostSamplingPolicy,
    generator: torch.Generator | None,
    history: tuple[torch.Tensor, torch.Tensor] | None = None,
    lse: float | None = None,
) -> int:
    """One lane's token from its candidate row: :func:`candidate_law`, then the exponential-noise argmax of
    :func:`sample_from_row` with the same generator, so a seeded row draws the token the full-row path draws for the same
    logits (up to the order of exactly tied values).  A greedy row reaches here only with a penalty (its penalized
    argmax)."""

    probs, indices = candidate_law(values, ids, policy, history, lse)
    noise = torch.empty_like(probs).exponential_(generator=generator)
    return int(indices[(probs / noise).argmax()])


def position_generator(seed: int | None, position: int) -> torch.Generator | None:
    """Batch mode's draw generator: seeded from the request's seed and the position of the token that produced the row,
    so a draw does not depend on the plugin's layout resets (``reset_sampling_state`` fires for every row whenever a
    request joins or leaves) or on the other lanes: alone or concurrent, a seeded request draws the same tokens.
    ``None`` seed: the global RNG."""

    if seed is None:
        return None
    return torch.Generator().manual_seed((int(seed) * 0x9E3779B97F4A7C15 + int(position)) % (1 << 63))


def _sample_full_row(scaled: torch.Tensor, top_k: int, top_p: float, generator: torch.Generator | None) -> int:
    """vLLM's full-row algorithm on the temperature-scaled row: the top-k threshold, a sort for the nucleus when
    top_p < 1, softmax, exponential noise, argmax.  Reached when top-k is wider than the candidates but on (the
    top-k-off rows take :func:`sample_from_row`'s nucleus branches or :func:`_sample_whole_row`)."""

    if top_k < scaled.numel():
        scaled = scaled.masked_fill(scaled < scaled.topk(top_k).values[-1], float("-inf"))
    if top_p < 1.0:
        ordered, order = scaled.sort(descending=True)
        probs = torch.softmax(ordered, 0)
        keep = torch.cumsum(probs, 0) - probs < top_p
        keep[0] = True
        scaled = scaled.clone()
        scaled[order[~keep]] = float("-inf")
    probs = torch.softmax(scaled, 0)
    noise = torch.empty_like(probs).exponential_(generator=generator)
    return int((probs / noise).argmax())


def _sample_whole_row(scaled: torch.Tensor, generator: torch.Generator | None) -> int:
    """One draw from the softmax of the whole temperature-scaled row (top-k off, top_p at 1) by inverse CDF: the float64
    cumulative sum of the probabilities, one float64 uniform variate scaled to its total, the first index whose
    cumulative mass exceeds it.  The index is clamped to the last token of positive probability, so a masked column
    (``-inf``) is
    never drawn.  One pass over the row instead of 248k exponential variates."""

    probs = torch.softmax(scaled, 0)
    cdf = probs.double().cumsum(0)
    uniform = torch.rand(1, generator=generator, dtype=torch.float64) * cdf[-1]
    last = int((probs > 0).nonzero()[-1])
    return min(int(torch.searchsorted(cdf, uniform, right=True)), last)


def prefill_form(environ: Mapping[str, str] | None = None) -> tuple[bool, int | None]:
    """The prefill form the chain is opened with, as ``(long_chunks, slab_rows)``, from the environment (``os.environ``
    by default).  ``QWEN38_PREFILL_SLAB`` is the slab's row count, a multiple of 128 in 256..2048 (default 2048, the
    measured form; the MoE serves multiples of 512); ``0`` or ``off`` serves without the slab.  A slab implies the 128-row chunks; without one,
    ``QWEN38_LONG_CHUNKS=1`` turns the 128-row chunks on alone.  Any other value is refused by name."""

    environ = os.environ if environ is None else environ
    raw = environ.get(PREFILL_SLAB_VARIABLE)
    value = "" if raw is None else raw.strip().lower()
    if value == "":
        slab_rows: int | None = DEFAULT_SLAB_ROWS
    elif value in PREFILL_SLAB_OFF:
        slab_rows = None
    else:
        rows = int(value) if value.isascii() and value.isdigit() else None
        if rows is None or not is_slab_rows(rows):
            raise ValueError(
                f"{PREFILL_SLAB_VARIABLE} must be the slab's rows, {SLAB_ROWS_RULE}, or 0 / off for no slab, "
                f"got {raw!r}"
            )
        slab_rows = rows
    long_chunks = slab_rows is not None or environ.get(LONG_CHUNKS_VARIABLE, "").strip() == "1"
    return long_chunks, slab_rows


def describe_prefill_form(long_chunks: bool, slab_rows: int | None, plan: str | None = None) -> str:
    """The chunk kinds a prompt runs through, largest first (the chunk driver's order), and with ``plan`` the plan mode
    that covers a prompt's rows with them (``QWEN38_PREFILL_PLAN``)."""

    parts = [f"{slab_rows}-row slabs"] if slab_rows is not None else []
    if long_chunks:
        parts.append("128-row chunks")
    form = ", then ".join(parts + ["32-row chunks"])
    return form if plan is None else f"{form}; plan: {plan}"


def prefill_dense_refusal(error: BaseException) -> ValueError | None:
    """The DRAM admission's refusal of the slab's resident prefill weights carried by ``error``: the error itself, or
    one level down as its ``__cause__`` (a resident build failure whose owner cleanup also failed arrives wrapped, with
    the primary error as the cause: ttnn/builder.py); None when ``error`` is something else.  The admission raises
    inside the target build when the weights, the context-scaled state, the slab's working set and the margin exceed
    the free DRAM."""

    for candidate in (error, error.__cause__):
        if isinstance(candidate, ValueError) and PREFILL_DENSE_REFUSAL_MARK in str(candidate):
            return candidate
    return None


def is_prefill_dense_refusal(error: BaseException) -> bool:
    return prefill_dense_refusal(error) is not None


# Paged pool budget.  M1 (2026-10-24, QB2, context 32768, DRAM per device, ``test_vllm_batch_contract_device.py`` ids
# m1_b8 / m1_b32): 13,368,200,192 B free when the lane decoder's ``prepare`` starts (chain built, no lanes yet); the lane
# states and what the lane capture holds then took 1,045,793,792 B at B=8 and 3,093,709,824 B at B=32 (each beside a
# 0.75 GiB pool), i.e. 363.2 MB + 85.3 MB per lane (rounded up here).
DRAM_CHAIN_FREE_32K = 13_368_200_192  # bytes free per device when the lane decoder's prepare starts
LANE_FIXED_BASE_32K = 364_000_000  # bytes: the lane states' intercept at context 32768
LANE_FIXED_PER_LANE_32K = 85_400_000  # bytes: and per lane
# Each decode bucket R other than the full one holds lane-state views and a lane trace: a fixed part and a part per lane.
# Measured at TP=8, context 32768 (test_batch_buckets_device.py step-times run): bucket views 5.7 / 7.7 / 11.6 / 21.5 /
# 41.2 MB and lane traces 9.9 / 10.0 / 10.3 / 10.8 / 11.9 MB for R = 1 / 2 / 4 / 8 / 16, plus margin.
BUCKET_FIXED_BYTES_32K = 16_000_000  # bytes per extra bucket
BUCKET_PER_LANE_BYTES_32K = 2_600_000  # bytes per lane of an extra bucket
CONTEXT_BYTES_PER_TOKEN = 13 * 1024  # per extra allocated-context token: the chain's B=1 KV and index caches
CONTEXT_BYTES_PER_TOKEN_PER_LANE = 768  # and each lane's compressed index cache
POOL_MARGIN_BYTES = 1 << 30  # transients: the full-logits gather, prefill scratch, fragmentation
# The pool admission refuses with half the margin: the plugin adds ``max(block_size, 1) * max_num_seqs`` tokens (and a
# rounding block) to the budget the cap returned, which the cap's full margin covers.
POOL_ADMISSION_MARGIN_BYTES = POOL_MARGIN_BYTES // 2


def bucket_dram_bytes(buckets: Sequence[int], lanes: int) -> int:
    """DRAM per device of the decode buckets narrower than the full width ``lanes``: ``BUCKET_FIXED_BYTES_32K +
    R * BUCKET_PER_LANE_BYTES_32K`` each."""

    return sum(BUCKET_FIXED_BYTES_32K + r * BUCKET_PER_LANE_BYTES_32K for r in buckets if r < lanes)


def batch_pool_token_cap(max_model_len: int, max_num_seqs: int, buckets: Sequence[int] = ()) -> int:
    """Tokens the paged KV pool may hold in batch mode: ``(free - lanes - margin) // KV_BYTES_PER_TOKEN`` where
    ``free`` is the DRAM free per device after the chain build at the allocated context, and ``lanes`` the lane states
    (GDN, ring, PLE and compressed index caches) of ``max_num_seqs`` lanes.  Both are measured at context 32768 and
    scaled by ``CONTEXT_BYTES_PER_TOKEN`` (and per lane) for the other contexts.  The decode ``buckets`` narrower than
    ``max_num_seqs`` add ``bucket_dram_bytes``."""

    context = resident_allocated_context(max_model_len)
    extra = context - 32768
    free = DRAM_CHAIN_FREE_32K - extra * CONTEXT_BYTES_PER_TOKEN
    lanes = LANE_FIXED_BASE_32K + max_num_seqs * (LANE_FIXED_PER_LANE_32K + extra * CONTEXT_BYTES_PER_TOKEN_PER_LANE)
    lanes += bucket_dram_bytes(buckets, max_num_seqs)
    if prefill_pack_enabled(lanes=max_num_seqs):  # the pack's persistent state and bucket traces take DRAM too
        lanes += pack_dram_bytes(pack_buckets_from_env(lanes=max_num_seqs), context)
    return max(0, (free - lanes - POOL_MARGIN_BYTES) // KV_BYTES_PER_TOKEN)


def check_block_size(block_size: Any) -> int:
    """Batch mode's page size is vLLM's block size: a power of two >= 32 (the QSA page scatter works in 32-row slabs),
    at most ``BATCH_MAX_BLOCK_SIZE``; anything else is refused naming the knob (``--block-size``)."""

    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError(f"vLLM block_size must be an int, got {block_size!r}")
    if block_size < BATCH_MIN_BLOCK_SIZE or block_size & (block_size - 1) or block_size > BATCH_MAX_BLOCK_SIZE:
        raise ValueError(
            f"batch mode serves a paged KV pool whose page is vLLM's block: --block-size must be a power of two in "
            f"{BATCH_MIN_BLOCK_SIZE}..{BATCH_MAX_BLOCK_SIZE}, got {block_size}"
        )
    return block_size


def batch_max_blocks(allocated_context: int, block_size: int) -> int:
    """The lane page table's width: ``ceil(allocated_context / block_size)`` rounded up to a multiple of 8."""

    return -(-(-(-allocated_context // block_size)) // 8) * 8


def resident_allocated_context(max_model_len: int) -> int:
    """The smallest resident QSA cache capacity whose context limit holds ``max_model_len`` (ValueError if none)."""

    limits = {capacity: Qwen38ResidentContext(capacity).context_limit for capacity in RESIDENT_QSA_CACHE_CAPACITIES}
    fitting = [capacity for capacity, limit in limits.items() if limit >= max_model_len]
    if not fitting:
        raise ValueError(f"max_model_len {max_model_len} exceeds every resident context limit {list(limits.values())}")
    return fitting[0]


def parse_slot_remap(slot_remap: Any, lanes: int) -> list[int]:
    """The plugin's ``slot_remap`` as a list: ``None`` is the identity; else an absolute permutation of all ``lanes``
    state slots (row ``i`` reads the state now in slot ``remap[i]``)."""

    if slot_remap is None:
        return list(range(lanes))
    remap = [int(slot) for slot in torch.as_tensor(slot_remap).reshape(-1).tolist()]
    if len(remap) != lanes or sorted(remap) != list(range(lanes)):
        raise ValueError(f"slot_remap must be a permutation of the {lanes} state slots, got {remap}")
    return remap


@dataclass
class Qwen38VLLMSlot:
    """Host mirror of the one resident slot: the tokens the device consumed (their count is the device position),
    the PLE n-gram context, whether a chain call failed (the device state is not trusted afterwards), and under the
    device-sampling contract the request's sampling policy and its draw generator (``None``: the global RNG)."""

    committed: list[int] = field(default_factory=list)
    ple_context: tuple[int, int] | None = None
    failed: bool = False
    policy: Qwen38HostSamplingPolicy | None = None
    generator: torch.Generator | None = None


class Qwen38ForCausalLM:
    """vLLM adapter over the traced chain: one resident slot, host sampling from full-vocabulary logits, no MTP.
    ``supports_sample_on_device`` admits the plugin's ``sample_on_device_mode`` (``decode_only``): the sampled decode
    steps then return the token the adapter draws on the host from the same row (module docstring)."""

    model_capabilities = {
        "fabric_config": dict(FABRIC_CONFIG),
        "supports_device_penalties": True,
        "supports_chunked_prefill": False,
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
    }
    decode_input_update_contract = 1

    def __init__(
        self,
        chain: Any,
        *,
        allocated_context: int,
        batch_lanes: int = 1,
        chain_opener: Callable[[Callable[[Any], None] | None], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.chain = chain  # batch mode: None until allocate_kv_cache opens it
        self.allocated_context = allocated_context
        self.context_limit = Qwen38ResidentContext(allocated_context).context_limit
        self.slot = Qwen38VLLMSlot()
        self.trace_mode_warned = False
        self._warned_padded_greedy = False
        # batch mode (batch_lanes > 1); single-slot mode never reads these
        self.lanes = batch_lanes
        self.batch_mode = batch_lanes > 1
        self.chain_opener = chain_opener
        self.decoder: Qwen38PlainLaneDecoder | None = None
        self.block_size: int | None = None
        self.num_blocks: int | None = None
        self.lane_of_slot: list[int] = list(range(batch_lanes))  # slot -> lane (the lane holding that slot's state)
        # decode-width buckets (QWEN38_DECODE_BUCKETS; refused at start when bad): a step replays the smallest bucket that
        # holds the highest live lane, so a new request takes the lowest free lane (_take_lane)
        self.decode_buckets: tuple[int, ...] = (
            decode_buckets_from_env(batch_lanes) if batch_lanes > 1 else (batch_lanes,)
        )
        # W5: with buckets on, a step first moves a live lane that keeps the bucket wide down to a free low lane
        self.decode_compact = decode_compact_enabled() if len(self.decode_buckets) > 1 else False
        self.lane_moves = 0
        self.lane_slots: list[Qwen38VLLMSlot] = [Qwen38VLLMSlot() for _ in range(batch_lanes)]  # per-lane policy/RNG
        self.batch_failed = False
        self.last_prefill_groups: list = []  # the groups of the last packed-mode prefill call (pack on)
        self._pending_note: dict[int, int] | None = None  # the moves decode_forward applied, not yet noted
        self.decode_calls = 0  # batch mode: decode calls so far (the periodic log line)
        # P3g-1: sampled lanes draw from the device candidate rows (QWEN38_BATCH_CANDIDATES, default on); refused at start
        self.batch_candidates = batch_candidates_enabled() if batch_lanes > 1 else False
        # P3g-2: top-k-off sampled rows draw from the candidate rows with the device normalizer (QWEN38_BATCH_NORMALIZER,
        # default on; refused at start when bad); only with the candidate rows on
        batch_normalizer = batch_normalizer_enabled() if batch_lanes > 1 else False
        self.batch_normalizer = bool(self.batch_candidates and batch_normalizer)
        # P3g-3: a step with fallback rows reads only those lanes' logits rows (QWEN38_BATCH_ROW_READ, default on;
        # refused at start when bad); only with the candidate rows on
        batch_row_read = batch_row_read_enabled() if batch_lanes > 1 else False
        self.batch_row_read = bool(self.batch_candidates and batch_row_read)
        # candidate-path counters: rows drawn from candidates, rows sent to the full-logits path per reason, the steps
        # that paid the full gather for them
        self.candidate_counters: dict[str, int] = {
            "candidate_steps": 0,
            "candidate_rows": 0,
            "fallback_steps": 0,
            "normalizer_steps": 0,  # steps that replayed the candidate + normalizer trace
            "normalizer_rows": 0,  # rows drawn with the device normalizer (top-k off)
            "row_read_steps": 0,  # fallback steps that read only their rows' logits (P3g-3)
            "row_read_rows": 0,  # lanes those reads returned
            "full_read_steps": 0,  # fallback steps that read the full gather (knob off, or too many rows)
            "logprobs_rows": 0,  # rows that returned a raw logprob of their sampled token
            **{f"fallback_{reason}": 0 for reason in CANDIDATE_FALLBACK_REASONS},
        }
        self.host_branch_counts: dict[str, int] = {
            branch: 0 for branch in HOST_DRAW_BRANCHES
        }  # P3g-3: per fallback row
        self.decode_stats: deque[dict[str, float]] = deque(maxlen=4096)  # per decode call: total/step/gather seconds
        self.dram: dict[
            str, dict[str, int]
        ] = {}  # free DRAM per device: "chain_only" (lanes not built), "after_capture"

    # -- vLLM's text-generation protocol: inspected by the registry, never called on the TT path ---------------------

    def embed_input_ids(self, input_ids: Any) -> Any:
        raise NotImplementedError(PROTOCOL_SHIM_MESSAGE)

    def forward(self, input_ids: Any, positions: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(PROTOCOL_SHIM_MESSAGE)

    def compute_logits(self, hidden_states: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(PROTOCOL_SHIM_MESSAGE)

    # -- construction ------------------------------------------------------------------------------------------------

    @classmethod
    def get_max_tokens_all_users(
        cls,
        *,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs: Any,
    ) -> int:
        """The KV token pool the plugin sizes its block table from.  ``QWEN38_MAX_TOKENS_ALL_USERS`` (a positive int)
        overrides it in both modes.  Single-slot mode (``max_num_seqs`` 1): one context per sequence,
        ``max_model_len * max_num_seqs``.  Batch mode: ``min(max_model_len * max_num_seqs, cap)``, where ``cap`` is the
        paged pool's DRAM budget (``batch_pool_token_cap``): the free DRAM per device after the chain build, less the
        lane states of ``max_num_seqs`` lanes and a margin, over the pool's 12,288 bytes per token per device."""

        tokens = int(max_model_len) * int(max_num_seqs)
        override = os.environ.get(MAX_TOKENS_ALL_USERS_VARIABLE, "").strip()
        if override:
            if not override.isascii() or not override.isdigit() or int(override) < 1:
                raise ValueError(f"{MAX_TOKENS_ALL_USERS_VARIABLE} must be a positive integer, got {override!r}")
            return int(override)
        if int(max_num_seqs) <= 1:
            return tokens
        return min(
            tokens,
            batch_pool_token_cap(int(max_model_len), int(max_num_seqs), decode_buckets_from_env(int(max_num_seqs))),
        )

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: Any,
        mesh_device: Any,
        max_batch_size: int,
        max_seq_len: int,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **kwargs: Any,
    ) -> "Qwen38ForCausalLM":
        """The chain on the mesh vLLM opened, constructed as the chat server constructs it.

        The cache identity is the extension digest and a 40-hex sha: ``QWEN38_TT_METAL_SHA``, else the imported ttnn's
        checkout head, else the digest's prefix (an image without ``.git`` or ``git``); the chat server's runtime
        admission (a ttnn built from this checkout) is not applied, vLLM's python environment is trusted as given.
        The checkpoint is the ``--model`` directory, else ``HF_MODEL`` (a directory or a repo id), else ``--model`` as a
        repo id, else ``Qwen/Qwen3.8-Flash-Next``; a repo id resolves to the pinned revision in the local Hugging Face
        hub cache.
        The prefill form is ``prefill_form()``'s (the slab rows and the 128-row chunks); the slab's DRAM admission
        refusal is re-raised naming ``QWEN38_PREFILL_SLAB``.  ``max_batch_size`` 1: the chain is open on return.
        ``max_batch_size`` 2..32 (batch mode): the construction is done and the chain is NOT open; ``allocate_kv_cache``
        opens it (the lane tensors need the pool's size)."""

        if not TRACE_ALLOC_TRACKING:
            raise ValueError("the trace chain needs TT_METAL_TRACE_ALLOC_TRACKING=1 exported before ttnn is imported")
        if not 1 <= max_batch_size <= BATCH_MAX_LANES or tt_data_parallel != 1:
            raise ValueError(
                f"the Qwen3.8 adapter serves one resident slot (max_num_seqs 1) or a batch of lanes (max_num_seqs "
                f"2..{BATCH_MAX_LANES}) and no data parallelism, got batch {max_batch_size}, DP {tt_data_parallel}"
            )
        if optimizations not in OPTIMIZATIONS:
            raise ValueError(f"optimizations must be one of {OPTIMIZATIONS}, got {optimizations!r}")
        long_chunks, slab_rows = prefill_form()  # refused by name before any library call
        plan = prefill_plan()  # ditto: QWEN38_PREFILL_PLAN
        shape = tuple(int(extent) for extent in mesh_device.shape)
        if shape != MESH_SHAPE or mesh_device.get_num_devices() != TP_SIZE:
            raise ValueError(
                f"Qwen3.8 needs a {MESH_SHAPE} mesh of {TP_SIZE} devices, "
                f"got {shape} with {mesh_device.get_num_devices()}"
            )
        if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
            raise ValueError(f"Qwen3.8 needs a Blackhole mesh, got {mesh_device.arch()}")
        allocated_context = resident_allocated_context(max_seq_len)
        physical_ids = tuple(int(device_id) for device_id in mesh_device.get_device_ids())
        # The checkpoint: --model when vLLM was given a directory, else HF_MODEL (a directory or a repo id), else
        # --model as a repo id, else the default repo; a repo id resolves to the pinned revision in the hub cache.
        name_or_path = getattr(hf_config, "_name_or_path", None)
        if not (name_or_path and Path(name_or_path).is_dir()):
            name_or_path = os.environ.get("HF_MODEL") or name_or_path or HF_REPO_ID
        try:
            checkpoint_root = checkpoint_directory({"HF_MODEL": name_or_path}, revision=PINNED_CHECKPOINT_REVISION)
        except FileNotFoundError as error:
            raise ValueError(f"{error} (vLLM's --model directory, else HF_MODEL, names the checkpoint)") from error
        cache_root = os.environ.get("TT_CACHE_PATH")
        if not cache_root:
            raise ValueError(
                "TT_CACHE_PATH must name the cache root: <root>/caches/<label>/{components,model-io} and "
                "<root>/caches/bf4-experts, the chat server launcher's layout"
            )
        caches = Path(cache_root) / "caches"
        label = f"c{allocated_context}-vllm"
        corpus_root = os.environ.get("QWEN38_BF4_CORPUS")
        corpus_verification = os.environ.get("QWEN38_BF4_CORPUS_VERIFICATION")
        extension = Path(ttnn._ttnn.__file__).resolve()
        tt_metal_sha = os.environ.get("QWEN38_TT_METAL_SHA")
        if tt_metal_sha and (len(tt_metal_sha) != 40 or any(c not in "0123456789abcdef" for c in tt_metal_sha)):
            raise ValueError(f"QWEN38_TT_METAL_SHA must be lowercase 40-hex, got {tt_metal_sha!r}")
        runtime_sha256 = sha256_of(extension)
        # The identity's sha without git: the launcher's pin, else the checkout head, else the extension digest (an
        # image with no .git or git binary); the digest already keys the identity, so each build keeps one namespace.
        if tt_metal_sha:
            source = "QWEN38_TT_METAL_SHA"
        else:
            try:
                tt_metal_sha, source = git_identity(Path(ttnn.__file__).resolve().parents[2])["head"], "checkout head"
            except (RuntimeAdmissionError, FileNotFoundError):
                tt_metal_sha, source = runtime_sha256[:40], "runtime extension digest"
        logger.info(f"tt_metal_sha {tt_metal_sha} ({source})")
        # The library's construction gate is read at call time; set here so vLLM's launch needs no export.
        os.environ.setdefault("QWEN38_HARDWARE_MODE", MODE)
        prepared = prepare_live_decode_diagnostic(
            checkpoint_root=checkpoint_root,
            component_cache_root=caches / label / "components",
            routed_bf4_scratch_root=caches / "bf4-experts",
            model_io_cache_root=caches / label / "model-io",
            tt_metal_sha=tt_metal_sha,
            runtime_extension=extension,
            runtime_sha256=runtime_sha256,
            allocated_context=allocated_context,
            physical_ids=physical_ids,
            bf4_corpus_root=None if corpus_root is None else Path(corpus_root),
            bf4_corpus_verification=None if corpus_verification is None else Path(corpus_verification),
            marker=logger.info,
        )
        construction = construct_live_decode_diagnostic(
            prepared,
            mesh_device=mesh_device,
            collective_topology=ttnn.Topology.Linear,
            marker=logger.info,
            stage_missing_bf4=os.environ.get("QWEN38_VLLM_ALLOW_BF4_CONVERSION") == "1",
        )

        def open_chain(warm_hook: Callable[[Any], None] | None) -> Any:
            options = {} if warm_hook is None else {"warm_hook": warm_hook}
            if os.environ.get("QWEN38_GREEDY_TAIL", "0") == "1" and max_batch_size == 1:
                # the single-slot host sampler draws from the full-vocabulary row, never the candidate row: the TAIL
                # traces skip the candidate chain (tools/qwen38_sampling_step.SKIP_CANDIDATE_ROW)
                from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step

                qwen38_sampling_step.SKIP_CANDIDATE_ROW = True
            try:
                chain = Qwen38TracedChain.open(
                    construction,
                    marker=logger.info,
                    chunked_prefill=True,
                    sampling=True,
                    long_chunks=long_chunks,
                    slab_rows=slab_rows,
                    **options,
                )
            except Exception as error:
                refusal = None if slab_rows is None else prefill_dense_refusal(error)
                if refusal is None:
                    raise
                detail = str(error) if refusal is error else f"{error} (cause: {refusal})"
                raise ValueError(
                    f"the {slab_rows}-row prefill slab does not fit beside the resident build at context "
                    f"{allocated_context}: {detail}; {PREFILL_SLAB_VARIABLE}=0 serves this context without the slab "
                    f"({LONG_CHUNKS_VARIABLE}=1 keeps the 128-row chunks)"
                ) from error
            logger.info(
                f"prefill form: {describe_prefill_form(long_chunks, slab_rows, plan)}"
            )  # the form the chain opened in
            return chain

        if max_batch_size > 1:
            # Batch mode: the chain opens in allocate_kv_cache, where the pool's size is known (see the module docstring).
            return cls(None, allocated_context=allocated_context, batch_lanes=max_batch_size, chain_opener=open_chain)
        return cls(open_chain(None), allocated_context=allocated_context)

    def allocate_kv_cache(self, kv_cache_shape: Any, dtype: Any, num_layers: int) -> None:
        """Single-slot mode: vLLM's paged KV is not used, the chain allocated the resident QSA caches at open, sized by
        the context.  Batch mode: the shape's ``num_blocks`` (``shape[0]``) and ``block_size`` (``shape[2]``) size the
        lane pool (the head count, head size, dtype and layer count of vLLM's spec are ignored: the pool's rows are the
        QSA's 512-wide bf16 rows); the chain opens here with the lane decoder's ``prepare`` as its warm hook, then the
        lane trace is captured.  Returns None (the pool is the decoder's; vLLM's ``kv_cache`` argument is not read)."""

        if self.batch_mode:
            shape = tuple(int(extent) for extent in kv_cache_shape)
            if len(shape) != 4:
                raise ValueError(f"kv_cache_shape must be (num_blocks, heads, block_size, head_size), got {shape}")
            self._open_batch(num_blocks=shape[0], block_size=shape[2])
            return None
        logger.info(
            f"vLLM KV cache {tuple(kv_cache_shape)} {dtype} x {num_layers} layers not allocated: "
            f"resident QSA caches at context {self.allocated_context}"
        )

    # -- the two forward calls -------------------------------------------------------------------------------------

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,
        page_table: Any,
        kv_cache: Any,
        enable_trace: bool,
        prompt_lens: Any,
        start_pos: Any,
        empty_slots: Any,
        sampling_params: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The prompt into the reset state; returns the last position's logits (or, for a plain greedy
        ``sampling_params``, the device's greedy token as int32 ``[1, 1]``) and the zero mrope delta."""

        if self.batch_mode:
            return self._prefill_batch(
                tokens=tokens,
                page_table=page_table,
                enable_trace=enable_trace,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                empty_slots=empty_slots,
                sampling_params=sampling_params,
            )
        greedy_token = sampling_params is not None
        if greedy_token and not device_greedy_contract(sampling_params):
            raise ValueError("prefill returns logits: sample_on_device_mode must be decode_only, not all")
        if tokens.shape[0] != 1:
            raise ValueError(f"one resident slot: tokens must be [1, P], got {tuple(tokens.shape)}")
        if int(start_pos[0]) != 0:
            raise ValueError(
                f"prefill must start at position 0 (no prefix caching, no chunking), got {int(start_pos[0])}"
            )
        length = int(prompt_lens[0])
        if not 1 <= length <= min(self.context_limit, tokens.shape[1]):
            raise ValueError(
                f"prompt length {length} must be in [1, {self.context_limit}] and within tokens {tuple(tokens.shape)}"
            )
        if list(empty_slots) != [0]:
            raise ValueError(f"one resident slot: empty_slots must be [0], got {list(empty_slots)}")
        ids = tokens[0, :length].tolist()
        if min(ids) < 0 or max(ids) >= TOKENIZER_SIZE:
            raise ValueError(f"prompt ids must be in [0, {TOKENIZER_SIZE}), got {min(ids)}..{max(ids)}")
        self._require_serviceable(enable_trace, recovering=True)
        out = self._prefill_one(ids, greedy_token=greedy_token)
        return out, torch.zeros(1, dtype=torch.long)

    def _prefill_one(self, ids: list[int], *, greedy_token: bool) -> torch.Tensor:
        """The B=1 chain prefill of ``ids`` into the reset state (the single-slot path, and batch mode's per-row
        prefill before its import); returns the device greedy token ``[1, 1]`` int32 or the masked full logits
        ``[1, 1, VOCAB_SIZE]``.  The state it leaves (``self.slot``, the chain's generic state) is the one request's."""

        head = ids[:-1]
        try:
            self.chain.reset_and_seed(SEED_TOKEN_ID)
            self.slot = Qwen38VLLMSlot()
            if len(head) >= CHUNK_PREFILL_MIN_ROWS:

                def forced_step(token_id: int, _context: tuple[int, int] | None) -> tuple[int, int] | None:
                    self._step(token_id)  # the driver's alignment steps; none from position 0
                    return self.slot.ple_context

                result = self.chain.chunk_prefill(head, start_position=0, ple_context=None, forced_step=forced_step)
                if result.stopped is not None or result.position != len(head):
                    raise Qwen38ChatChainError(
                        f"chunk prefill ended at position {result.position} (stopped {result.stopped!r}), "
                        f"expected {len(head)}"
                    )
                self.slot.committed = list(head)
                self.slot.ple_context = result.ple_context
            else:
                with self.chain.loop_guard():
                    for token_id in head:
                        self._step(token_id)
            # The last prompt token is the first decode replay: teacher-forced under the guard, like every decode step.
            with self.chain.loop_guard():
                residue = self._step(ids[-1])
                out = self._greedy_token(residue) if greedy_token else self._read_logits(residue)
        except BaseException:
            self.slot.failed = True
            raise
        return out

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: Any,
        page_table: Any,
        kv_cache: Any,
        enable_trace: bool,
        read_from_device: bool,
        rope_deltas_all_users: Any = None,
        slot_remap: Any = None,
        reload_inputs: bool = True,
        reload_page_table: bool = False,
        reload_sampling_params: bool = False,
        reset_sampling_state: bool = False,
        sampling_params: Any = None,
        prompt_tokens: Any = None,
        output_tokens: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """One teacher-forced step with the token vLLM sampled; returns the next position's logits, or, when the
        plugin sends ``sampling_params`` (its device-sampling contract), the next token as int32 ``[1, 1]``: the device's greedy
        token when the row is plain greedy (no penalty), else drawn on the host from the full row with
        vLLM's sampler semantics.  A row with ``enable_log_probs`` keeps the token its route gives and returns
        ``(token [1, 1] int32, logprob [1] float32)``: the raw logprob of that token (``log_softmax`` of the full row,
        before temperature, penalties, top-k and top-p), read from the full row besides the device's greedy token."""

        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract 1: reset_batch is not accepted")
        if reload_inputs is False:
            raise ValueError("host tokens are authoritative on this adapter: reload_inputs must be True")
        if self.batch_mode:
            return self._decode_batch(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                enable_trace=enable_trace,
                slot_remap=slot_remap,
                reset_sampling_state=reset_sampling_state,
                sampling_params=sampling_params,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        if slot_remap is not None and list(slot_remap) != [0]:
            raise ValueError(f"one resident slot: slot_remap must be None or [0], got {list(slot_remap)}")
        if tokens.shape[0] != 1:
            raise ValueError(f"one resident slot: tokens must be [1, 1], got {tuple(tokens.shape)}")
        token_id = int(tokens[0, 0])
        if not 0 <= token_id < TOKENIZER_SIZE:
            raise ValueError(f"decode token must be in [0, {TOKENIZER_SIZE}), got {token_id}")
        policy = history = None
        host_fast = os.environ.get(HOST_FAST_VARIABLE) == "1"
        if host_fast:  # one parse of the contract: device_greedy_contract's rule on the policy this step needs anyway
            greedy_token = False
            if sampling_params is not None:
                policy = Qwen38HostSamplingPolicy.from_contract(sampling_params)
                greedy_token = policy.greedy and not policy.penalized
        else:
            greedy_token = device_greedy_contract(sampling_params)
            if sampling_params is not None:
                policy = Qwen38HostSamplingPolicy.from_contract(sampling_params)
        if policy is not None and policy.penalized:
            history = penalty_history(prompt_tokens, output_tokens)
        self._require_serviceable(enable_trace)
        position = int(start_pos[0])
        if position != len(self.slot.committed):
            raise Qwen38ChatChainError(f"runner position {position} vs device position {len(self.slot.committed)}")
        if policy is not None:
            self._adopt_policy(policy, reseed=reset_sampling_state)
        logprobs = policy is not None and policy.logprobs
        try:
            with nullcontext() if host_fast else self.chain.loop_guard():  # the guard only rebinds ttnn names
                residue = self._step(token_id)
                if greedy_token:
                    token = self._greedy_token(residue)
                    if not logprobs:
                        return token
                logits = self._read_logits(residue)
        except BaseException:
            self.slot.failed = True
            raise
        if policy is None:
            return logits
        row = logits[0, 0]
        raw = row.clone() if logprobs and history is not None else row  # the penalties act in place
        if not greedy_token:
            if history is not None:
                apply_penalties(row, *history, policy)
            token = torch.tensor([[sample_from_row(row, policy, self.slot.generator)]], dtype=torch.int32)
        if not logprobs:
            return token
        return token, torch.tensor([raw_logprob(raw, int(token))], dtype=torch.float32)

    def _adopt_policy(
        self, policy: Qwen38HostSamplingPolicy, *, reseed: bool, slot: Qwen38VLLMSlot | None = None
    ) -> None:
        """The request's policy, logged when it changes; the draw generator is seeded from ``seed`` for a new request
        (the prefill cleared the slot), a new seed, or the plugin's ``reset_sampling_state``.  ``slot``: batch mode's
        per-lane slot (default: the one resident slot)."""

        slot = self.slot if slot is None else slot
        if policy != slot.policy:
            logger.info(f"host sampler under the device-sampling contract: {policy}")
            reseed = reseed or slot.policy is None or policy.seed != slot.policy.seed
            slot.policy = policy
        if reseed or slot.generator is None:
            slot.generator = None if policy.seed is None else torch.Generator().manual_seed(policy.seed)

    # -- warmup and release ------------------------------------------------------------------------------------------

    def warmup_model_prefill(
        self, *, enable_trace: bool, kv_cache: Any, can_sample_on_device: bool, **kwargs: Any
    ) -> None:
        """Nothing to compile or capture (``open`` did both): checks only, in both phases.  The plugin's flag here is
        ``sample_on_device_mode == "all"`` (prefill would have to return tokens): refused, prefill returns the row."""

        if can_sample_on_device and not self.batch_mode:
            raise ValueError("prefill returns logits: sample_on_device_mode must be decode_only, not all")
        if self.chain is None:
            raise Qwen38ChatChainError("batch mode: allocate_kv_cache opens the chain, call it before warmup")
        if self.chain.closed or not self.chain.misses_forbidden:
            raise Qwen38ChatChainError("the chain must be open with its miss guard closed before warmup")

    def warmup_model_decode(
        self,
        *,
        enable_trace: bool,
        kv_cache: Any,
        max_batch_size: int,
        num_blocks: int,
        can_sample_on_device: bool,
        **kwargs: Any,
    ) -> None:
        """Phase 1 (``enable_trace`` False): checks.  Phase 2: both adapter paths once, under the closed miss guard,
        in the vLLM process, before the first request.  ``can_sample_on_device`` (mode ``all`` or ``decode_only``) is
        served: the sampled decode steps draw on the host from the row the greedy path reads."""

        self.warmup_model_prefill(enable_trace=enable_trace, kv_cache=kv_cache, can_sample_on_device=False)
        if self.batch_mode:
            if max_batch_size != self.lanes:
                raise ValueError(f"batch mode: max_batch_size must be the {self.lanes} lanes, got {max_batch_size}")
            if enable_trace:
                self._warmup_batch()
            return
        if max_batch_size != 1:
            raise ValueError(f"one resident slot: max_batch_size must be 1, got {max_batch_size}")
        if not enable_trace:
            return
        prompt = torch.tensor([WARMUP_PROMPT_TOKEN_IDS], dtype=torch.int32)
        logits, _rope_deltas = self.prefill_forward(
            tokens=prompt,
            page_table=None,
            kv_cache=kv_cache,
            enable_trace=True,
            prompt_lens=[prompt.shape[1]],
            start_pos=[0],
            empty_slots=[0],
        )
        self.decode_forward(
            tokens=torch.tensor([[int(logits.argmax())]], dtype=torch.int32),
            start_pos=[prompt.shape[1]],
            page_table=None,
            kv_cache=kv_cache,
            enable_trace=True,
            read_from_device=False,
        )
        try:
            self.chain.reset_and_seed(SEED_TOKEN_ID)
        except BaseException:
            self.slot.failed = True
            raise
        self.slot = Qwen38VLLMSlot()

    def release_request(self, slot: int) -> None:
        """The request's slot is free; the device is untouched (the next prefill resets it)."""

        if self.batch_mode:
            self._release_lane_of_slot(slot)
            return
        if slot != 0:
            raise ValueError(f"one resident slot: slot must be 0, got {slot}")
        self.slot.committed = []
        self.slot.ple_context = None

    def release_persistent_capture(self) -> None:
        """Worker shutdown while the mesh is open: the chain's release order, skipped after a failed chain call.  Batch
        mode: the lane decoder (lane trace, pagers, lane state) first, then the chain."""

        if self.slot.failed or self.batch_failed:
            logger.warning("a chain call failed; the traces and states are left to the mesh close")
            return
        if self.batch_mode:
            if self.decoder is not None:
                self.decoder.release()
            if self.chain is None:
                return  # allocate_kv_cache never ran: nothing is captured
        self.chain.close()

    def note_state_slots_moved(self, moves: Mapping[int, int]) -> None:
        """The plugin's ``{old_slot: new_slot}`` after it committed a slot movement (batch mode: the slot -> lane map;
        single-slot mode: nothing to do).  ``decode_forward`` already applied the permutation its ``slot_remap`` carried,
        and the plugin sends the same permutation here once the decode was accepted: that call changes nothing.  A
        permutation ``decode_forward`` did not carry is applied here, once (the lanes never move on the device)."""

        if not self.batch_mode:
            return
        moves = {int(old): int(new) for old, new in dict(moves).items()}
        pending, self._pending_note = self._pending_note, None
        if not moves or moves == pending:
            return
        if set(moves) != set(moves.values()) or any(not 0 <= slot < self.lanes for slot in moves):
            raise ValueError(f"note_state_slots_moved: {moves} is not a permutation of state slots")
        lane_of_slot = list(self.lane_of_slot)
        for old, new in moves.items():
            lane_of_slot[new] = self.lane_of_slot[old]
        self.lane_of_slot = lane_of_slot

    # -- batch mode (max_num_seqs 2..32) -----------------------------------------------------------------------------

    def _open_batch(self, *, num_blocks: int, block_size: int) -> None:
        """Open the chain with the lane decoder's prepare as its warm hook and capture the lane trace.  ``num_blocks``
        pages of ``block_size`` rows form the pool; page id = vLLM block id, page 0 (vLLM's null block) the scratch."""

        if self.decoder is not None:
            raise RuntimeError("the lane pool is already allocated: allocate_kv_cache runs once")
        if self.chain_opener is None:
            raise RuntimeError("batch mode needs the chain opener initialize_vllm_model hands in")
        block_size = check_block_size(block_size)
        if isinstance(num_blocks, bool) or not isinstance(num_blocks, int) or num_blocks < 2:
            raise ValueError(f"vLLM num_blocks must be an int >= 2 (block 0 is the scratch), got {num_blocks!r}")
        paging = Qwen38LanePaging(
            page_size=block_size,
            num_pages=num_blocks,
            max_blocks=batch_max_blocks(self.allocated_context, block_size),
        ).validate(self.allocated_context)
        decoder = Qwen38PlainLaneDecoder(
            self.lanes,
            paging=paging,
            indexer_form=BATCH_LANE_INDEXER_FORM,
            pack=prefill_pack_enabled(lanes=self.lanes),  # on with >= 8 lanes (P3e); QWEN38_PREFILL_PACK wins
            candidates=self.batch_candidates,
            normalizer=self.batch_normalizer,
            row_read=self.batch_row_read,
            buckets=self.decode_buckets,
        )
        decoder.warm_logits_gather = True  # host-sampled rows and host logits read the lane logits under the miss guard
        self.decoder, self.block_size, self.num_blocks = decoder, block_size, num_blocks
        logger.info(
            f"batch mode: {self.lanes} lanes, KV pool {num_blocks} blocks x {block_size} rows = "
            f"{paging.pool_rows} rows ({paging.pool_rows * KV_BYTES_PER_TOKEN / 2**30:.2f} GiB per device), page table "
            f"{paging.max_blocks} blocks per lane, context {self.allocated_context}, indexer {decoder.indexer_form}, "
            f"decode buckets {self.decode_buckets}"
        )

        def warm_hook(chain: Any) -> None:
            self._admit_pool(chain, paging)
            decoder.prepare(chain)

        self.chain = self.chain_opener(warm_hook)
        decoder.capture(self)
        self.dram["after_capture"] = self._dram_free(self.chain.mesh)
        logger.info(
            f"batch mode ready: free DRAM per device {self.dram['after_capture']['free_bytes'] / 2**30:.2f} GiB after "
            f"the lane capture (chain only: {self.dram.get('chain_only', {}).get('free_bytes', 0) / 2**30:.2f} GiB)"
        )

    @staticmethod
    def _dram_free(mesh: Any) -> dict[str, int]:
        """The mesh allocator's DRAM view (one virtual allocator, every card alike), per device, in bytes."""

        view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        banks = int(view.num_banks)
        return {
            "banks": banks,
            "free_bytes": banks * int(view.total_bytes_free_per_bank),
            "allocated_bytes": banks * int(view.total_bytes_allocated_per_bank),
            "largest_contiguous_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
        }

    def _admit_pool(self, chain: Any, paging: Qwen38LanePaging) -> None:
        """Refuse a pool the free DRAM cannot hold (the chain is built, the lanes are not yet), naming the knob."""

        self.dram["chain_only"] = self._dram_free(chain.mesh)
        free = self.dram["chain_only"]["free_bytes"]
        pool = paging.pool_rows * KV_BYTES_PER_TOKEN
        lanes = LANE_FIXED_BASE_32K + self.lanes * (
            LANE_FIXED_PER_LANE_32K + (self.allocated_context - 32768) * CONTEXT_BYTES_PER_TOKEN_PER_LANE
        )
        lanes += bucket_dram_bytes(self.decode_buckets, self.lanes)  # the extra decode buckets' views and traces
        if prefill_pack_enabled(lanes=self.lanes):  # the pack's persistent state and bucket traces
            lanes += pack_dram_bytes(pack_buckets_from_env(lanes=self.lanes), self.allocated_context)
        logger.info(
            f"batch pool admission: free {free / 2**30:.2f} GiB per device, pool {pool / 2**30:.2f} GiB, lane states "
            f"~{lanes / 2**30:.2f} GiB, margin {POOL_ADMISSION_MARGIN_BYTES / 2**30:.2f} GiB"
        )
        if pool + lanes + POOL_ADMISSION_MARGIN_BYTES > free:
            raise ValueError(
                f"the KV pool of {paging.num_pages} blocks x {paging.page_size} rows needs {pool / 2**30:.2f} GiB per "
                f"device beside ~{lanes / 2**30:.2f} GiB of lane states and a {POOL_ADMISSION_MARGIN_BYTES / 2**30:.2f} GiB "
                f"margin, the free DRAM is {free / 2**30:.2f} GiB: set {MAX_TOKENS_ALL_USERS_VARIABLE} lower (or "
                f"--max-model-len, --max-num-seqs)"
            )

    def _require_batch_ready(self) -> Qwen38PlainLaneDecoder:
        if self.decoder is None or self.chain is None or self.decoder.ls is None:
            raise Qwen38ChatChainError("batch mode: allocate_kv_cache opens the chain and captures the lanes first")
        if self.batch_failed:
            raise Qwen38ChatChainError("a lane call failed earlier; the lane state is not trusted")
        return self.decoder

    def _page_row(self, row: Any, *, needed: int, label: str) -> list[int]:
        """The vLLM block ids of one page-table row (zero padded): its leading nonzero entries.  Refused: block 0
        (vLLM's null block, our scratch page) among them, an id outside the pool, more than the lane table holds, or
        fewer than ``needed`` blocks (the step would write into the scratch page)."""

        ids = [int(block) for block in torch.as_tensor(row).reshape(-1).tolist()]
        count = len(ids)
        while count and ids[count - 1] == 0:
            count -= 1
        pages = ids[:count]
        if 0 in pages:
            raise ValueError(f"{label}: block 0 (vLLM's null block) inside the page-table row {pages}")
        if pages and (min(pages) < 0 or max(pages) >= self.num_blocks):
            raise ValueError(f"{label}: block ids must be in [1, {self.num_blocks}), got {pages}")
        if len(pages) > self.decoder.paging.max_blocks:
            raise ValueError(f"{label}: {len(pages)} blocks exceed the lane table's {self.decoder.paging.max_blocks}")
        if len(pages) < needed:
            raise ValueError(f"{label}: needs {needed} blocks for its tokens, the page-table row holds {len(pages)}")
        return pages

    def _release_lane_of_slot(self, slot: int) -> None:
        """``release_request`` in batch mode: the slot's lane is free (host state cleared, page row zeroed, parked at
        the next replay); vLLM frees the blocks."""

        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < self.lanes:
            raise ValueError(f"slot must be in [0, {self.lanes}), got {slot!r}")
        decoder = self._require_batch_ready()
        lane = self.lane_of_slot[slot]
        decoder.release_lane_external(lane)
        self.lane_slots[lane] = Qwen38VLLMSlot()

    def _move_lane(self, decoder: Qwen38PlainLaneDecoder, src: int, dst: int) -> None:
        """Compaction: the decoder moves live lane ``src`` into free lane ``dst``; the slot that held ``src`` now holds
        ``dst`` (the two slots swap lanes in the map, host only) and the lane's policy/RNG follows."""

        try:
            decoder.move_lane(src, dst)
        except BaseException:
            self.batch_failed = True
            raise
        slot_src, slot_dst = self.lane_of_slot.index(src), self.lane_of_slot.index(dst)
        self.lane_of_slot[slot_src], self.lane_of_slot[slot_dst] = dst, src
        self.lane_slots[dst] = self.lane_slots[src]
        self.lane_slots[src] = Qwen38VLLMSlot()
        self.lane_moves += 1
        logger.info(f"decode compaction: lane {src} -> {dst} ({decoder.last_move.get('ms', 0.0):.1f} ms)")

    def _prefill_batch(
        self,
        *,
        tokens: torch.Tensor,
        page_table: Any,
        enable_trace: bool,
        prompt_lens: Any,
        start_pos: Any,
        empty_slots: Any,
        sampling_params: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each row alone: the B=1 chain prefill, then its import into the lane that owns ``empty_slots[i]`` with the
        row's vLLM blocks as its pages.  Returns ``(out, rope_deltas)``: with ``sampling_params`` the next tokens int32
        ``[rows, 1]`` (a plain greedy row: the device greedy token; any other row: drawn on the host from the row's
        logits, penalties over the prompt), else the host logits ``[rows, 1, VOCAB_SIZE]``; ``rope_deltas`` zeros."""

        decoder = self._require_batch_ready()
        rows = int(tokens.shape[0])
        if not 1 <= rows <= self.lanes or tokens.dim() != 2:
            raise ValueError(f"batch mode: tokens must be [1..{self.lanes}, P], got {tuple(tokens.shape)}")
        if empty_slots is None or not (len(empty_slots) == len(prompt_lens) == len(start_pos) == rows):
            raise ValueError("batch mode: empty_slots, prompt_lens and start_pos must hold one entry per row")
        slots = [int(slot) for slot in empty_slots]
        if len(set(slots)) != rows or any(not 0 <= slot < self.lanes for slot in slots):
            raise ValueError(f"batch mode: empty_slots must be distinct slots in [0, {self.lanes}), got {slots}")
        if page_table is None or torch.as_tensor(page_table).shape[0] < rows:
            raise ValueError("batch mode: prefill needs the page-table row of every row")
        page_table = torch.as_tensor(page_table)
        prompts: list[list[int]] = []
        pages: list[list[int]] = []
        policies: list[Qwen38HostSamplingPolicy | None] = []
        for i in range(rows):
            if int(start_pos[i]) != 0:
                raise ValueError(
                    f"prefill must start at position 0 (no prefix caching, no chunking), got {int(start_pos[i])}"
                )
            length = int(prompt_lens[i])
            if not 1 <= length <= min(self.context_limit, tokens.shape[1]):
                raise ValueError(
                    f"prompt length {length} must be in [1, {self.context_limit}] and within tokens "
                    f"{tuple(tokens.shape)}"
                )
            ids = tokens[i, :length].tolist()
            if min(ids) < 0 or max(ids) >= TOKENIZER_SIZE:
                raise ValueError(f"prompt ids must be in [0, {TOKENIZER_SIZE}), got {min(ids)}..{max(ids)}")
            prompts.append(ids)
            pages.append(self._page_row(page_table[i], needed=-(-length // self.block_size), label=f"prefill row {i}"))
            policies.append(
                None if sampling_params is None else Qwen38HostSamplingPolicy.from_contract(sampling_params, i)
            )
        self._require_serviceable(enable_trace, recovering=True)
        if prefill_pack_enabled(lanes=self.lanes):
            return self._prefill_batch_grouped(prompts, pages, policies, slots), torch.zeros(rows, dtype=torch.long)
        outputs: list[torch.Tensor] = []
        claimed: set[int] = set()  # the lanes this call hands out (not live until their row is imported)
        for i in range(rows):
            outputs.append(self._prefill_row_single(decoder, slots[i], prompts[i], pages[i], policies[i], claimed))
        return torch.cat(outputs, dim=0), torch.zeros(rows, dtype=torch.long)

    def _take_lane(self, decoder: Qwen38PlainLaneDecoder, slot: int, claimed: set[int]) -> int:
        """The lane an admission into ``slot`` lands in.  The slot's own lane when it is the one to use (a still-live lane
        is released first, with a warning); with decode buckets on, the lowest lane that holds no request and is not in
        ``claimed`` (the lanes this prefill call already handed out), swapped into the slot -> lane map: a bucket replay
        advances only lanes 0..R-1, so live requests stay in the low lanes.  The lane is added to ``claimed``."""

        lane = self.lane_of_slot[slot]
        if decoder.positions[lane] is not None:
            logger.warning(f"prefill into slot {slot} whose lane {lane} still holds a request: released")
            decoder.release_lane_external(lane)
        if len(self.decode_buckets) == 1:
            claimed.add(lane)
            return lane
        target = next(u for u in range(self.lanes) if decoder.positions[u] is None and u not in claimed)
        if target != lane:
            other = self.lane_of_slot.index(target)  # a slot with no request (the target's position is None)
            self.lane_of_slot[slot], self.lane_of_slot[other] = target, lane
        claimed.add(target)
        return target

    def _prefill_row_single(
        self,
        decoder: Qwen38PlainLaneDecoder,
        slot: int,
        prompt: list[int],
        pages: list[int],
        policy: Qwen38HostSamplingPolicy | None,
        claimed: set[int] | None = None,
    ) -> torch.Tensor:
        """One row alone: the B=1 chain prefill, its import into the lane of ``slot`` (a still-live lane is released first)
        and the row's output (the device greedy token, a host draw, or the host logits ``[1, 1, VOCAB_SIZE]``)."""

        lane = self._take_lane(decoder, slot, set() if claimed is None else claimed)
        device_greedy = policy is not None and policy.greedy and not policy.penalized
        out = self._prefill_one(prompt, greedy_token=device_greedy)
        try:
            decoder.import_lane(self, lane, len(prompt), self.slot.ple_context, pages)
        except BaseException:
            self.batch_failed = True
            raise
        self.slot.committed = []  # the B=1 scratch state is free again
        self.slot.ple_context = None
        lane_slot = self.lane_slots[lane] = Qwen38VLLMSlot()
        if policy is not None:
            self._adopt_policy(policy, reseed=True, slot=lane_slot)
        if policy is not None and not device_greedy:
            row = out[0, 0]
            if policy.penalized:
                prompt_ids = torch.tensor(prompt, dtype=torch.long)
                apply_penalties(row, prompt_ids, torch.zeros(0, dtype=torch.long), policy)
            generator = position_generator(policy.seed, len(prompt) - 1)
            out = torch.tensor([[sample_from_row(row, policy, generator)]], dtype=torch.int32)
        return out

    def _prefill_batch_grouped(
        self,
        prompts: list[list[int]],
        pages: list[list[int]],
        policies: list[Qwen38HostSamplingPolicy | None],
        slots: list[int],
    ) -> torch.Tensor:
        """Pack on (P3d-4; the default with at least 8 lanes, ``QWEN38_PREFILL_PACK`` overrides): plan the rows into
        packs and single-path rows (:func:`qwen38_pack_plan.plan_prefill_groups`; ``QWEN38_PACK_FORCE=1`` packs every row that fits a bucket), run the
        groups in plan order and return the outputs in row order.  A packed row lands in its lane from the pack
        (``admit_pack``, the vLLM blocks as its pages) and takes its first token from the pack's fp32 logits: a plain greedy
        row is the host argmax (torch.argmax returns the lowest id among equal maxima, the device greedy's tie rule), any other row the
        single path's host draw (penalties over the prompt, ``position_generator(seed, P - 1)``).  Without sampling params
        the row's output is its masked logits ``[1, 1, VOCAB_SIZE]``, as in the single path.  A pack's tokens are in the
        tolerance class of the single path's, not bitwise (docs/PREFILL.md)."""

        decoder = self.decoder
        rows = len(prompts)
        lengths = [len(prompt) for prompt in prompts]
        groups = plan_prefill_groups(
            lengths, decoder.pack_buckets, self.lanes, force=pack_force_enabled(), context=self.allocated_context
        )
        self.last_prefill_groups = groups
        logger.info(f"prefill groups for lengths {lengths}: {groups}")
        outputs: dict[int, torch.Tensor] = {}
        claimed: set[int] = set()  # the lanes this call hands out (not live until their group is admitted)
        for group in groups:
            if group[0] == "single":
                (i,) = group[1]
                outputs[i] = self._prefill_row_single(decoder, slots[i], prompts[i], pages[i], policies[i], claimed)
                continue
            _kind, bucket, members = group
            lanes = [self._take_lane(decoder, slots[i], claimed) for i in members]
            try:
                logits = decoder.admit_pack(
                    lanes, [prompts[i] for i in members], bucket, pages=[pages[i] for i in members]
                )
            except BaseException:
                self.batch_failed = True
                raise
            logger.info(f"prefill pack S={bucket} users={len(members)} rows={members}: {decoder.last_pack}")
            for u, (i, lane) in enumerate(zip(members, lanes)):
                policy = policies[i]
                lane_slot = self.lane_slots[lane] = Qwen38VLLMSlot()
                row = logits[u].clone()
                row[TOKENIZER_SIZE:] = float("-inf")
                if policy is None:
                    outputs[i] = row.view(1, 1, VOCAB_SIZE)
                    continue
                self._adopt_policy(policy, reseed=True, slot=lane_slot)
                if policy.greedy and not policy.penalized:
                    outputs[i] = torch.tensor([[int(row.argmax())]], dtype=torch.int32)
                    continue
                if policy.penalized:
                    apply_penalties(
                        row, torch.tensor(prompts[i], dtype=torch.long), torch.zeros(0, dtype=torch.long), policy
                    )
                generator = position_generator(policy.seed, len(prompts[i]) - 1)
                outputs[i] = torch.tensor([[sample_from_row(row, policy, generator)]], dtype=torch.int32)
        return torch.cat([outputs[i] for i in range(rows)], dim=0)

    def _decode_batch(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: Any,
        page_table: Any,
        enable_trace: bool,
        slot_remap: Any,
        reset_sampling_state: bool,
        sampling_params: Any,
        prompt_tokens: Any,
        output_tokens: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """One replay of the lane trace for the live rows.  Row ``i`` is live when ``start_pos[i] >= 0``; it runs in lane
        ``lane_of_slot[i]`` after ``slot_remap`` was applied to the slot -> lane map.  Every lane that holds a request
        must have a live row (the trace advances every lane).  Lanes without a request run the warm token on the scratch
        page, parked at position 0.  Returns the next tokens int32 ``[lanes, 1]`` under ``sampling_params`` (device
        greedy ids; host draws for the other live rows; padding rows 0), else the host logits ``[lanes, 1, VOCAB_SIZE]``
        (padding rows 0).  When any entry of ``sampling_params.enable_log_probs`` is true (the whole padded list, as the
        plugin tests it) the return is ``(tokens, logprobs)``, ``logprobs`` float32 ``[lanes]``: for a live row with
        ``enable_log_probs`` the raw logprob of its final token (``log_softmax`` of its fp32 row, padding at ``-inf``,
        before the penalties, temperature, top-k and top-p), 0 for every other row.  Such a row's token comes from the
        route it would take without the request; the request only adds the row's read (with the fallback lanes' rows,
        through the same row read) and the host computation."""

        started = time.perf_counter()
        decoder = self._require_batch_ready()
        lanes = self.lanes
        if tuple(tokens.shape) != (lanes, 1):
            raise ValueError(f"batch mode: tokens must be [{lanes}, 1], got {tuple(tokens.shape)}")
        positions = [int(position) for position in torch.as_tensor(start_pos).reshape(-1).tolist()]
        if len(positions) != lanes:
            raise ValueError(f"batch mode: start_pos must hold {lanes} rows, got {len(positions)}")
        if page_table is None or torch.as_tensor(page_table).shape[0] < lanes:
            raise ValueError(f"batch mode: page_table must hold {lanes} rows")
        page_table = torch.as_tensor(page_table)
        if self.decode_compact:  # before the slot -> lane map is read: the moved lanes are what the rows resolve to
            for src, dst in decoder.compaction_moves():
                self._move_lane(decoder, src, dst)
        remap = parse_slot_remap(slot_remap, lanes)
        new_map = [self.lane_of_slot[remap[row]] for row in range(lanes)]
        live_rows = [row for row, position in enumerate(positions) if position >= 0]
        if not live_rows:
            raise ValueError("batch mode: a decode needs a live row (start_pos >= 0)")
        holders = {lane for lane in range(lanes) if decoder.positions[lane] is not None}
        row_lanes = {row: new_map[row] for row in live_rows}
        orphans = sorted(holders - set(row_lanes.values()))
        if orphans:
            raise Qwen38ChatChainError(
                f"contract error: lanes {orphans} hold live requests but no live row of this decode maps to them "
                f"(rows {live_rows}, slot -> lane {new_map}); the replay advances every lane and would corrupt them"
            )
        strangers = [row for row in live_rows if row_lanes[row] not in holders]
        if strangers:
            raise Qwen38ChatChainError(
                f"contract error: live rows {strangers} map to lanes with no prefilled request (slot -> lane {new_map})"
            )
        lane_tokens = [WARM_TOKEN_ID] * lanes
        pages: dict[int, list[int]] = {}
        policies: dict[int, Qwen38HostSamplingPolicy] = {}
        for row in live_rows:
            lane, position = row_lanes[row], positions[row]
            token_id = int(tokens[row, 0])
            if not 0 <= token_id < TOKENIZER_SIZE:
                raise ValueError(f"decode token must be in [0, {TOKENIZER_SIZE}), got {token_id} (row {row})")
            if position >= self.context_limit:
                raise ValueError(f"row {row}: position {position} is past the context limit {self.context_limit}")
            mirror = decoder.lane_position(lane)
            if position != mirror:
                raise Qwen38ChatChainError(
                    f"runner position {position} vs device position {mirror} (row {row}, lane {lane})"
                )
            pages[row] = self._page_row(
                page_table[row], needed=position // self.block_size + 1, label=f"decode row {row}"
            )
            lane_tokens[lane] = token_id
            if sampling_params is not None:
                policies[row] = Qwen38HostSamplingPolicy.from_contract(sampling_params, row)
        host_rows = [row for row, policy in policies.items() if not (policy.greedy and not policy.penalized)]
        lp_rows = [row for row in live_rows if row in policies and policies[row].logprobs]
        lp_lanes = {row_lanes[row] for row in lp_rows}
        histories = {
            row: penalty_history(prompt_tokens, output_tokens, row) for row in host_rows if policies[row].penalized
        }
        if not enable_trace and not self.trace_mode_warned:
            self.trace_mode_warned = True
            logger.warning("trace_mode is not honoured: every step replays the captured traces")
        # Accepted: the map moves, the lanes' page rows and policies follow, then the one replay.
        self._pending_note = (
            {remap[row]: row for row in range(lanes) if remap[row] != row} if remap != list(range(lanes)) else None
        )
        self.lane_of_slot = new_map
        for row in live_rows:
            decoder.set_page_row(row_lanes[row], pages[row])
            if row in policies:
                self._adopt_policy(policies[row], reseed=reset_sampling_state, slot=self.lane_slots[row_lanes[row]])
        try:
            decoder.sync_page_table()
            decoder.park_idle([lane for lane in range(lanes) if lane not in holders])
            step_started = time.perf_counter()
            lane_ids = decoder.step(lane_tokens)
            step_s = time.perf_counter() - step_started
        except BaseException:
            self.batch_failed = True
            raise
        gather_started = time.perf_counter()
        logits = None
        candidates_s = draw_s = fallback_s = normalizer_s = fallback_read_s = fallback_draw_s = 0.0
        fallback_logits: dict[int, torch.Tensor] = {}  # lane -> its logits row, when only the fallback lanes were read
        drawn: dict[int, int] = {}
        use_candidates = (
            self.batch_candidates
            and sampling_params is not None
            and bool(host_rows)
            and getattr(decoder, "has_candidates", False)
        )
        if use_candidates:
            drawn, fallback_rows, candidates_s, draw_s, normalizer_s = self._draw_candidate_rows(
                decoder, host_rows, policies, row_lanes, positions, histories
            )
            if fallback_rows:
                fallback_started = time.perf_counter()
                read_lanes = sorted({row_lanes[row] for row in fallback_rows} | lp_lanes)  # one read for both
                if self._row_read_serves(decoder, len(read_lanes)):
                    fallback_logits = dict(zip(read_lanes, self._read_lane_rows(read_lanes)))
                else:
                    logits = self._read_lane_logits()
                    self.candidate_counters["full_read_steps"] += 1
                fallback_read_s = fallback_s = time.perf_counter() - fallback_started
                self.candidate_counters["fallback_steps"] += 1
        elif sampling_params is None or host_rows:
            logits = self._read_lane_logits()
        if lp_lanes and logits is None and not lp_lanes <= set(fallback_logits):
            # no fallback read this step (or none that covers them): only the logprobs rows' lanes are read
            lanes_to_read = sorted(lp_lanes)
            if self._row_read_serves(decoder, len(lanes_to_read)):
                fallback_logits = dict(zip(lanes_to_read, self._read_lane_rows(lanes_to_read)))
            else:
                logits = self._read_lane_logits()
                self.candidate_counters["full_read_steps"] += 1
        if sampling_params is None:
            out = torch.zeros(lanes, 1, VOCAB_SIZE, dtype=torch.float32)
            for row in live_rows:
                out[row, 0] = logits[row_lanes[row]]
        else:
            out = torch.zeros(lanes, 1, dtype=torch.int32)
            raw_rows: dict[int, torch.Tensor] = {}  # a penalized logprobs row's row as read, before the penalties
            for row in live_rows:
                lane = row_lanes[row]
                if row in drawn:
                    out[row, 0] = drawn[row]
                    continue
                if row in host_rows:
                    draw_started = time.perf_counter()
                    values = fallback_logits[lane] if lane in fallback_logits else logits[lane]
                    if row in histories:
                        if policies[row].logprobs:
                            raw_rows[row] = values.clone()
                        apply_penalties(values, *histories[row], policies[row])
                    generator = position_generator(policies[row].seed, positions[row])
                    out[row, 0] = sample_from_row(
                        values,
                        policies[row],
                        generator,
                        branch_counts=self.host_branch_counts if use_candidates else None,
                    )
                    if use_candidates:
                        fallback_draw_s += time.perf_counter() - draw_started
                        fallback_s += time.perf_counter() - draw_started
                    continue
                token = int(lane_ids[lane])
                if not 0 <= token < TOKENIZER_SIZE:
                    if not self._warned_padded_greedy:
                        self._warned_padded_greedy = True
                        logger.warning(
                            f"device greedy token {token} is outside the tokenizer [0, {TOKENIZER_SIZE}): "
                            "falling back to the host argmax over the masked full logits"
                        )
                    if logits is None:
                        logits = self._read_lane_logits()
                    token = int(logits[lane].argmax())
                out[row, 0] = token
        log_probs = None
        if sampling_params is not None and logprobs_requested(sampling_params):
            log_probs = torch.zeros(lanes, dtype=torch.float32)
            for row in lp_rows:  # every token is final: the logprob of each from its row as read
                lane = row_lanes[row]
                raw = raw_rows.get(row)
                if raw is None:
                    raw = fallback_logits[lane] if lane in fallback_logits else logits[lane]
                log_probs[row] = raw_logprob(raw, int(out[row, 0]))
            self.candidate_counters["logprobs_rows"] += len(lp_rows)
        now = time.perf_counter()
        self.decode_stats.append(
            {
                "rows": len(live_rows),
                "bucket": getattr(getattr(decoder, "active", None), "lanes", self.lanes),  # R: the step's decode width
                "total_s": now - started,
                "step_s": step_s,
                "gather_s": now - gather_started,
                "overhead_s": (now - started) - step_s,
                "candidates_s": candidates_s,
                "draw_s": draw_s,
                "normalizer_s": normalizer_s,
                "fallback_read_s": fallback_read_s,
                "fallback_draw_s": fallback_draw_s,
                "fallback_s": fallback_s if use_candidates else (now - gather_started if host_rows else 0.0),
                "sampled_rows": len(host_rows),
                "logprobs_rows": len(lp_rows),
            }
        )
        self.decode_calls += 1
        if self.decode_calls % DECODE_LOG_INTERVAL == 0:
            recent = list(self.decode_stats)[-DECODE_LOG_INTERVAL:]
            mean = lambda key: sum(entry[key] for entry in recent) / len(recent)  # noqa: E731
            counters = self.candidate_counters
            norm_steps = [entry["normalizer_s"] for entry in recent if entry["normalizer_s"] > 0.0]
            fallback_steps = [entry for entry in recent if entry["fallback_read_s"] > 0.0]
            fallback_mean = lambda key: (  # noqa: E731
                sum(entry[key] for entry in fallback_steps) / len(fallback_steps) * 1e3 if fallback_steps else 0.0
            )
            branches = ", ".join(f"{name} {count}" for name, count in self.host_branch_counts.items() if count)
            logger.info(
                f"batch decode: {self.decode_calls} calls, last {len(recent)}: mean rows {mean('rows'):.2f}, "
                f"bucket R {mean('bucket'):.1f} (steps per bucket {dict(getattr(decoder, 'bucket_steps', {}))}), "
                f"total {mean('total_s') * 1e3:.2f} ms, replay {mean('step_s') * 1e3:.2f} ms, "
                f"gather+sample {mean('gather_s') * 1e3:.2f} ms, overhead {mean('overhead_s') * 1e3:.2f} ms; "
                f"split: candidates {mean('candidates_s') * 1e3:.2f} ms, host draw {mean('draw_s') * 1e3:.2f} ms, "
                f"fallback gather {mean('fallback_s') * 1e3:.2f} ms, mean sampled rows {mean('sampled_rows'):.2f}; "
                f"candidates {'on' if self.batch_candidates else 'off'}: {counters['candidate_rows']} rows drawn in "
                f"{counters['candidate_steps']} steps, fallback steps {counters['fallback_steps']} (rows: top_k "
                f"{counters['fallback_top_k']}, penalty {counters['fallback_penalty']}, guard "
                f"{counters['fallback_guard']}, nonfinite {counters['fallback_nonfinite']}, nucleus "
                f"{counters['fallback_nucleus']}; logprobs rows {counters['logprobs_rows']}); normalizer {'on' if self.batch_normalizer else 'off'}: "
                f"{counters['normalizer_rows']} rows in {counters['normalizer_steps']} steps, device+readback "
                f"{(sum(norm_steps) / len(norm_steps) * 1e3) if norm_steps else 0.0:.2f} ms per normalizer step "
                f"({len(norm_steps)} of the last {len(recent)}); fallback steps {len(fallback_steps)} of the last "
                f"{len(recent)}: device read {fallback_mean('fallback_read_s'):.2f} ms + host draw "
                f"{fallback_mean('fallback_draw_s'):.2f} ms per fallback step, "
                f"row read {'on' if self.batch_row_read else 'off'}: "
                f"{counters['row_read_steps']} steps / {counters['row_read_rows']} lanes, full gather "
                f"{counters['full_read_steps']} steps; host branches: {branches or 'none'}"
            )
        return out if log_probs is None else (out, log_probs)

    def _draw_candidate_rows(
        self,
        decoder: Any,
        host_rows: list[int],
        policies: dict[int, Qwen38HostSamplingPolicy],
        row_lanes: dict[int, int],
        positions: list[int],
        histories: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[dict[int, int], list[int], float, float, float]:
        """The step's sampled rows from the device candidate rows: one trace replay and a ``[lanes, 256]`` readback, then
        per row the candidate law and the same generator as the full-row path (:func:`sample_from_candidates`).  A step with
        a top-k-off row (the normalizer on, :func:`needs_normalizer`) replays the candidate + normalizer trace instead and
        also reads the lanes' shard normalizers.  Returns ``(tokens by row, rows that need the full logits, candidate
        seconds, host draw seconds, normalizer-step seconds)``; a row whose policy or candidates cannot be exact
        (:class:`CandidateFallback`) is counted per reason and left to the full-logits path.
        """

        counters = self.candidate_counters
        use_normalizer = self.batch_normalizer and getattr(decoder, "has_normalizer", False)
        norm_rows = [row for row in host_rows if use_normalizer and needs_normalizer(policies[row])]
        started = time.perf_counter()
        shard_lse = None
        try:
            if norm_rows:
                inverse = [1.0] * self.lanes
                for row in norm_rows:
                    inverse[row_lanes[row]] = 1.0 / policies[row].temperature
                rows, shard_lse = decoder.read_candidates_normalized(inverse)
            else:
                rows = decoder.read_candidates()
        except BaseException:
            self.batch_failed = True
            raise
        candidates_s = time.perf_counter() - started
        drawn_started = time.perf_counter()
        values, ids, valid = parse_candidate_rows(rows)
        drawn: dict[int, int] = {}
        fallback_rows: list[int] = []
        for row in host_rows:
            lane = row_lanes[row]
            policy = policies[row]
            try:
                if not bool(valid[lane]):
                    raise CandidateFallback("nonfinite", f"lane {lane}'s candidate row is malformed")
                reason = candidate_route(
                    policy, normalizer=use_normalizer
                )  # the reason a top-k-off row is counted under
                if reason is not None:
                    raise CandidateFallback(reason)
                lse = combine_shard_normalizers(shard_lse[lane]) if row in norm_rows else None
                generator = position_generator(policy.seed, positions[row])
                drawn[row] = sample_from_candidates(
                    values[lane], ids[lane], policy, generator, histories.get(row), lse=lse
                )
                if lse is not None:
                    counters["normalizer_rows"] += 1
            except CandidateFallback as fallback:
                counters[f"fallback_{fallback.reason}"] += 1
                fallback_rows.append(row)
        counters["candidate_steps"] += 1
        counters["candidate_rows"] += len(drawn)
        counters["normalizer_steps"] += 1 if norm_rows else 0
        return (
            drawn,
            fallback_rows,
            candidates_s,
            time.perf_counter() - drawn_started,
            candidates_s if norm_rows else 0.0,
        )

    def _row_read_serves(self, decoder: Any, lanes: int) -> bool:
        """True when a fallback step with ``lanes`` lanes reads only those rows: the knob is on, the decoder captured
        the
        row-read traces and ``lanes`` fits the largest bucket."""

        return (
            self.batch_row_read
            and getattr(decoder, "has_row_read", False)
            and row_read_bucket(lanes, decoder.row_read_buckets) is not None
        )

    def _read_lane_rows(self, lanes: list[int]) -> torch.Tensor:
        """The last replay's logits of the given lanes, fp32 ``[len(lanes), VOCAB_SIZE]``, the LM head's padding
        masked."""

        try:
            logits = self.decoder.read_logits_rows(lanes)
        except BaseException:
            self.batch_failed = True
            raise
        logits[:, TOKENIZER_SIZE:] = float("-inf")
        self.candidate_counters["row_read_steps"] += 1
        self.candidate_counters["row_read_rows"] += len(lanes)
        return logits

    def _read_lane_logits(self) -> torch.Tensor:
        """The last replay's logits of every lane, fp32 ``[lanes, VOCAB_SIZE]``, the LM head's padding masked."""

        try:
            logits = self.decoder.read_full_logits()
        except BaseException:
            self.batch_failed = True
            raise
        logits[:, TOKENIZER_SIZE:] = float("-inf")
        return logits

    def _warmup_batch(self) -> None:
        """Batch mode's traced warmup, under the closed miss guard: one prompt through slot 0 (the B=1 chain prefill,
        the import), two lane steps (device greedy ids, then the host logits gather), the slot released.  The pool's
        first blocks take the prompt's rows (a request rewrites every block it reads before it reads it)."""

        length = len(WARMUP_PROMPT_TOKEN_IDS)
        needed = (length + 1) // self.block_size + 1
        if self.num_blocks < needed + 1:
            raise ValueError(f"warmup needs {needed} blocks beside the scratch block, the pool has {self.num_blocks}")
        table = torch.zeros(self.lanes, self.decoder.paging.max_blocks, dtype=torch.int32)
        table[0, :needed] = torch.arange(1, needed + 1, dtype=torch.int32)
        prompt = torch.tensor([WARMUP_PROMPT_TOKEN_IDS], dtype=torch.int32)
        out, _ = self.prefill_forward(
            tokens=prompt,
            page_table=table[:1],
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[length],
            start_pos=[0],
            empty_slots=[0],
            sampling_params=greedy_batch_sampling_params(1),
        )
        row_tokens = torch.zeros(self.lanes, 1, dtype=torch.int32)
        row_tokens[0, 0] = int(out.reshape(-1)[0])
        positions = torch.full((self.lanes,), -1, dtype=torch.int32)
        positions[0] = length
        ids = self.decode_forward(
            tokens=row_tokens,
            start_pos=positions,
            page_table=table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=False,
            sampling_params=greedy_batch_sampling_params(self.lanes),
        )
        row_tokens[0, 0] = int(ids[0, 0])
        positions[0] = length + 1
        self.decode_forward(
            tokens=row_tokens,
            start_pos=positions,
            page_table=table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=False,
        )
        if self.batch_candidates and getattr(self.decoder, "has_candidates", False) and len(self.decode_buckets) > 1:
            self.decoder.warm_bucket_reads()  # every bucket's sampling traces once under the closed miss guard
        elif self.batch_candidates and getattr(self.decoder, "has_candidates", False):
            self.decoder.read_candidates()  # one candidate replay and readback under the closed miss guard
            if self.batch_normalizer and getattr(self.decoder, "has_normalizer", False):
                self.decoder.read_candidates_normalized([1.0] * self.lanes)  # and one candidate + normalizer replay
            if self.batch_row_read and getattr(self.decoder, "has_row_read", False):
                for size in self.decoder.row_read_buckets:  # and every row-read bucket's replay and readback
                    self.decoder.read_logits_rows(list(range(size)))
        self.release_request(0)
        self.lane_of_slot = list(range(self.lanes))
        self.lane_slots = [Qwen38VLLMSlot() for _ in range(self.lanes)]
        self._pending_note = None
        self.decode_stats.clear()
        self.decode_calls = 0
        try:
            self.chain.reset_and_seed(SEED_TOKEN_ID)
        except BaseException:
            self.slot.failed = True
            raise
        self.slot = Qwen38VLLMSlot()

    # -- the per-token step and its readback -----------------------------------------------------------------------

    def _require_serviceable(self, enable_trace: bool, *, recovering: bool = False) -> None:
        """A failed chain call refuses decodes; a prefill may recover, its reset_and_seed re-verifies the device."""

        if self.slot.failed and not recovering:
            raise Qwen38ChatChainError("a chain call failed earlier; the device state is not trusted")
        if not enable_trace and not self.trace_mode_warned:
            self.trace_mode_warned = True
            logger.warning("trace_mode is not honoured: every step replays the captured traces")

    def _step(self, token_id: int) -> int:
        """Write x_t into the token row, HEAD(t), the PLE row for x_t, TAIL(t); returns the residue t mod 4."""

        residue = len(self.slot.committed) % RESIDUE_CLASSES
        write_token = (
            self.chain.feed_token_row
            if os.environ.get(TOKEN_FEEDBACK_VARIABLE, "1") == "1"
            else self.chain.write_token_row
        )
        if os.environ.get(PLE_EARLY_DECODE_VARIABLE) == "1":
            # lookup + host row before HEAD is enqueued: only the upload sits between HEAD and TAIL
            host_row, next_context = self.chain.prepare_ple_row(token_id, self.slot.ple_context)
            write_token(token_id)
            self.chain.execute_head(residue)
            self.chain.upload_ple_row(host_row)
            self.slot.ple_context = next_context
            self.chain.execute_tail(residue)
            self.slot.committed.append(token_id)
            return residue
        write_token(token_id)
        self.chain.execute_head(residue)
        self.slot.ple_context = self.chain.refresh_ple_row(token_id, self.slot.ple_context)
        self.chain.execute_tail(residue)
        self.slot.committed.append(token_id)
        return residue

    def _greedy_token(self, residue: int) -> torch.Tensor:
        """The device's greedy token (the tail's resolve into the token row) as int32 ``[1, 1]``.  The resolve does not
        mask the LM-head padding rows; an id at or above the tokenizer size (a server ``error`` finish) falls back to
        the full logits, whose padding is masked, for this step."""

        token = int(self.chain.read_token_row())
        if not 0 <= token < TOKENIZER_SIZE:
            if not self._warned_padded_greedy:
                self._warned_padded_greedy = True
                logger.warning(
                    f"device greedy token {token} is outside the tokenizer [0, {TOKENIZER_SIZE}): "
                    "falling back to the host argmax over the masked full logits"
                )
            token = int(self._read_logits(residue).argmax())
        return torch.tensor([[token]], dtype=torch.int32)

    def _read_logits(self, residue: int) -> torch.Tensor:
        """TAIL(residue)'s full-vocabulary gather (blocking), the padding rows masked, as ``[1, 1, VOCAB_SIZE]``."""

        logits = self.chain.sampling.read_full_logits(residue)
        self.chain.mark_ple_fenced()  # the gather is blocking: the TAIL (after the PLE upload) has finished
        logits[TOKENIZER_SIZE:] = float("-inf")
        return logits.view(1, 1, VOCAB_SIZE)
