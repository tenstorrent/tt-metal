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
token instead of the row.  The adapter reads the same full row and samples on the host with vLLM's sampler semantics
(``vllm/v1/sample/sampler.py``: penalties, temperature, top-k, top-p, exponential-noise draw) computed over the row's
top candidates wherever that is exact, and over the full row otherwise. vLLM's own ``Sampler`` sorts the full
248,320-wide row. The reduced route preserves its distribution; the token stream for a given seed differs because
the noise is drawn over the candidates. Historical fork sampling timings are not measurements of this port.
The plugin decides host or device per step before the call (min_p, logit_bias, bad_words, allowed_token_ids,
min_tokens, structured output and logprobs stay on its host sampler) and a step without ``sampling_params`` returns
the row as before.  Prefill sampling (mode ``"all"``) is refused at warmup: prefill returns the row.

The prefill form is the environment's (``prefill_form``): ``QWEN38_PREFILL_SLAB`` (default 2048) opens the chain
with a prefill slab of that many rows ahead of the 128-row and 32-row chunks; ``0`` or ``off`` serves without the
slab, where ``QWEN38_LONG_CHUNKS=1`` still turns the 128-row chunks on.  The slab's resident dense weights are
admitted against the free DRAM when the chain opens; a refusal ends the start-up naming the knob, never a silent
fallback.  One log line after the open states the form the chain opened in.

vLLM imports this module in its registry inspection, frontend, engine and worker processes: no device is opened
here and ``sys.argv`` is not read.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from ttnn.tools.trace_allocation_tracker import TRACE_ALLOC_TRACKING

import ttnn
from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.checkpoint import PINNED_CHECKPOINT_REVISION
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import (
    MODE,
    construct_live_decode_diagnostic,
    prepare_live_decode_diagnostic,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    CHUNK_PREFILL_MIN_ROWS,
    RESIDUE_CLASSES,
    SEED_TOKEN_ID,
    WARM_CHUNK_TOKEN_IDS,
    Qwen38ChatChainError,
    Qwen38TracedChain,
)
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
    TP_SIZE,
    is_slab_rows,
)

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

    @property
    def greedy(self) -> bool:
        return self.temperature < SAMPLING_EPS

    @property
    def penalized(self) -> bool:
        return self.presence_penalty != 0.0 or self.frequency_penalty != 0.0 or self.repetition_penalty != 1.0

    @classmethod
    def from_contract(cls, sampling_params: Any) -> "Qwen38HostSamplingPolicy":
        """The one slot's row of the plugin's per-row lists (tensors and scalars accepted); refused before the device
        is touched when it holds more than one row or asks for logprobs (never sent on one device)."""

        def one(name: str, default: Any = None) -> Any:
            value = getattr(sampling_params, name, default)
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                if len(value) != 1:
                    raise ValueError(f"one resident slot: sampling_params.{name} must hold one row, got {len(value)}")
                value = value[0]
            return value

        if one("enable_log_probs", False):
            raise ValueError(
                "logprobs are not produced under device sampling: the plugin keeps them on its host sampler"
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
        )


def penalty_history(prompt_tokens: Any, output_tokens: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """The one slot's prompt and output token ids out of the contract's ``[rows, L]`` int32 histories (``-1`` pads
    the rows and the batch), sent by the plugin whenever a penalty is active."""

    if prompt_tokens is None or output_tokens is None:
        raise ValueError("a penalty is active but the plugin sent no prompt_tokens / output_tokens history")

    def ids(history: Any) -> torch.Tensor:
        history = torch.as_tensor(history)
        if history.dim() != 2 or history.shape[0] < 1:
            raise ValueError(f"token history must be [rows, L], got {tuple(history.shape)}")
        row = history[0].to(torch.long)
        return row[(row >= 0) & (row < VOCAB_SIZE)]

    return ids(prompt_tokens), ids(output_tokens)


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
) -> int:
    """vLLM's sampler over one fp32 row (``Sampler.sample`` + ``apply_top_k_top_p_pytorch`` + ``random_sample``):
    greedy below the temperature epsilon; else temperature, the top-k set (every value equal to the k-th kept), the
    nucleus (a token is kept while the probability mass ranked above it is below top_p; the first always), then the
    exponential-noise argmax over the kept set.  Computed on the row's top candidates when that is exact: the top-k
    set fits in them, or top-k is off and the candidates hold the nucleus (their mass, taken over the full row's
    softmax, reaches top_p).  Otherwise the full row, vLLM's own way.  Same distribution as vLLM's sampler; the noise
    is drawn over the kept set, so a seed does not reproduce vLLM's token stream."""

    if policy.greedy:
        return int(row.argmax())
    vocab = row.numel()
    top_k = policy.top_k if 0 < policy.top_k < vocab else vocab
    top_p = policy.top_p
    if top_k <= candidates:
        values, indices = row.topk(top_k)
        extra = int((row >= values[-1]).sum()) - top_k
        if extra > 0:
            values, indices = row.topk(top_k + extra)
        probs = torch.softmax(values / policy.temperature, 0)
    elif top_k == vocab:
        values, indices = row.topk(candidates)
        scaled = row / policy.temperature
        probs = torch.exp(values / policy.temperature - torch.logsumexp(scaled, 0))
        if top_p >= 1.0 or float(probs.sum()) < top_p:
            return _sample_full_row(scaled, top_k, top_p, generator)
    else:
        return _sample_full_row(row / policy.temperature, top_k, top_p, generator)
    if top_p < 1.0:
        keep = torch.cumsum(probs, 0) - probs < top_p
        keep[0] = True
        probs, indices = probs[keep], indices[keep]
    noise = torch.empty_like(probs).exponential_(generator=generator)
    return int(indices[(probs / noise).argmax()])


def _sample_full_row(scaled: torch.Tensor, top_k: int, top_p: float, generator: torch.Generator | None) -> int:
    """vLLM's full-row algorithm on the temperature-scaled row: the top-k threshold, a sort for the nucleus when
    top_p < 1, softmax, exponential noise, argmax.  Reached when the candidates cannot bound the nucleus (top-k off
    with top_p at 1, or a nucleus wider than the candidates) or top-k is wider than the candidates but on."""

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


def prefill_form(environ: Mapping[str, str] | None = None) -> tuple[bool, int | None]:
    """The prefill form the chain is opened with, as ``(long_chunks, slab_rows)``, from the environment (``os.environ``
    by default).  ``QWEN38_PREFILL_SLAB`` is the slab's row count, a multiple of 128 in 256..4096 (default 2048, the
    measured form); ``0`` or ``off`` serves without the slab.  A slab implies the 128-row chunks; without one,
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
                f"{PREFILL_SLAB_VARIABLE} must be the slab's rows, a multiple of 128 in 256..4096, or 0 / off for no "
                f"slab, got {raw!r}"
            )
        slab_rows = rows
    long_chunks = slab_rows is not None or environ.get(LONG_CHUNKS_VARIABLE, "").strip() == "1"
    return long_chunks, slab_rows


def describe_prefill_form(long_chunks: bool, slab_rows: int | None) -> str:
    """The chunk kinds a prompt runs through, largest first (the chunk driver's order)."""

    parts = [f"{slab_rows}-row slabs"] if slab_rows is not None else []
    if long_chunks:
        parts.append("128-row chunks")
    return ", then ".join(parts + ["32-row chunks"])


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

    def __init__(self, chain: Any, *, allocated_context: int, **kwargs: Any) -> None:
        self.chain = chain
        self.allocated_context = allocated_context
        self.context_limit = Qwen38ResidentContext(allocated_context).context_limit
        self.slot = Qwen38VLLMSlot()
        self.trace_mode_warned = False

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
        """The KV token pool the plugin sizes its block table from: one context per sequence."""

        return int(max_model_len) * int(max_num_seqs)

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
        The checkpoint is ``MODEL_WEIGHTS_DIR``, else the ``--model`` path, else the pinned revision of the repo id
        (``QWEN38_HF_REPO``, else ``--model``, else ``Qwen/Qwen3.8-Flash-Next``) in the local Hugging Face hub cache.
        The prefill form is ``prefill_form()``'s (the slab rows and the 128-row chunks); the slab's DRAM admission
        refusal is re-raised naming ``QWEN38_PREFILL_SLAB``."""

        if not TRACE_ALLOC_TRACKING:
            raise ValueError("the trace chain needs TT_METAL_TRACE_ALLOC_TRACKING=1 exported before ttnn is imported")
        if max_batch_size != 1 or tt_data_parallel != 1:
            raise ValueError(
                f"the Qwen3.8 adapter serves one resident slot: max_num_seqs 1 and no data parallelism, "
                f"got batch {max_batch_size}, DP {tt_data_parallel}"
            )
        if optimizations not in OPTIMIZATIONS:
            raise ValueError(f"optimizations must be one of {OPTIMIZATIONS}, got {optimizations!r}")
        long_chunks, slab_rows = prefill_form()  # refused by name before any library call
        shape = tuple(int(extent) for extent in mesh_device.shape)
        if shape != MESH_SHAPE or mesh_device.get_num_devices() != TP_SIZE:
            raise ValueError(
                f"Qwen3.8 needs a {MESH_SHAPE} mesh of {TP_SIZE} devices, "
                f"got {shape} with {mesh_device.get_num_devices()}"
            )
        if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
            raise ValueError(f"Qwen3.8 needs a Blackhole mesh, got {mesh_device.arch()}")
        limits = {capacity: Qwen38ResidentContext(capacity).context_limit for capacity in RESIDENT_QSA_CACHE_CAPACITIES}
        fitting = [capacity for capacity, limit in limits.items() if limit >= max_seq_len]
        if not fitting:
            raise ValueError(
                f"max_model_len {max_seq_len} exceeds every resident context limit {list(limits.values())}"
            )
        allocated_context = fitting[0]
        physical_ids = tuple(int(device_id) for device_id in mesh_device.get_device_ids())
        weights_dir = os.environ.get("MODEL_WEIGHTS_DIR")
        name_or_path = getattr(hf_config, "_name_or_path", None)
        if weights_dir:
            if not Path(weights_dir).is_dir():
                raise ValueError(f"MODEL_WEIGHTS_DIR must name the checkpoint directory, got {weights_dir!r}")
            checkpoint_root = Path(weights_dir)
        elif name_or_path and Path(name_or_path).is_dir():
            checkpoint_root = Path(name_or_path)
        else:
            # A repo id: the pinned revision from the hub cache under HF_HOME, no network (the launcher prefetched it).
            repo_id = os.environ.get("QWEN38_HF_REPO") or name_or_path or HF_REPO_ID
            from huggingface_hub import snapshot_download

            try:
                checkpoint_root = Path(
                    snapshot_download(repo_id, revision=PINNED_CHECKPOINT_REVISION, local_files_only=True)
                )
            except Exception as error:
                raise ValueError(
                    f"{repo_id} at revision {PINNED_CHECKPOINT_REVISION} is not in the local Hugging Face hub cache: "
                    f"set MODEL_WEIGHTS_DIR to the checkpoint directory, or HF_HOME to a cache holding that snapshot "
                    f"(QWEN38_HF_REPO overrides the repo id)"
                ) from error
        cache_root = os.environ.get("QWEN38_CACHE_ROOT")
        if not cache_root:
            raise ValueError(
                "QWEN38_CACHE_ROOT must name the cache root: <root>/caches/<label>/{components,model-io} and "
                "<root>/caches/bf4-experts, the chat server launcher's layout"
            )
        caches = Path(cache_root) / "caches"
        label = os.environ.get("QWEN38_CACHE_LABEL") or f"c{allocated_context}-vllm"
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
        try:
            chain = Qwen38TracedChain.open(
                construction,
                marker=logger.info,
                chunked_prefill=True,
                sampling=True,
                long_chunks=long_chunks,
                slab_rows=slab_rows,
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
        logger.info(f"prefill form: {describe_prefill_form(long_chunks, slab_rows)}")  # the form the chain opened in
        return cls(chain, allocated_context=allocated_context)

    def allocate_kv_cache(self, kv_cache_shape: Any, dtype: Any, num_layers: int) -> None:
        """vLLM's paged KV is not used: the chain allocated the resident QSA caches at open, sized by the context."""

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
        """The prompt into the reset state; returns the last position's logits and the zero mrope delta."""

        if sampling_params is not None:
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
                logits = self._read_logits(self._step(ids[-1]))
        except BaseException:
            self.slot.failed = True
            raise
        return logits, torch.zeros(1, dtype=torch.long)

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
    ) -> torch.Tensor:
        """One teacher-forced step with the token vLLM sampled; returns the next position's logits, or, when the
        plugin sends ``sampling_params`` (its device-sampling contract), the next token as int32 ``[1, 1]``, drawn on
        the host from the same row with vLLM's sampler semantics."""

        if "reset_batch" in kwargs:
            raise TypeError("decode_input_update_contract 1: reset_batch is not accepted")
        if reload_inputs is False:
            raise ValueError("host tokens are authoritative on this adapter: reload_inputs must be True")
        if slot_remap is not None and list(slot_remap) != [0]:
            raise ValueError(f"one resident slot: slot_remap must be None or [0], got {list(slot_remap)}")
        if tokens.shape[0] != 1:
            raise ValueError(f"one resident slot: tokens must be [1, 1], got {tuple(tokens.shape)}")
        token_id = int(tokens[0, 0])
        if not 0 <= token_id < TOKENIZER_SIZE:
            raise ValueError(f"decode token must be in [0, {TOKENIZER_SIZE}), got {token_id}")
        policy = history = None
        if sampling_params is not None:
            policy = Qwen38HostSamplingPolicy.from_contract(sampling_params)
            if policy.penalized:
                history = penalty_history(prompt_tokens, output_tokens)
        self._require_serviceable(enable_trace)
        position = int(start_pos[0])
        if position != len(self.slot.committed):
            raise Qwen38ChatChainError(f"runner position {position} vs device position {len(self.slot.committed)}")
        if policy is not None:
            self._adopt_policy(policy, reseed=reset_sampling_state)
        try:
            with self.chain.loop_guard():
                logits = self._read_logits(self._step(token_id))
        except BaseException:
            self.slot.failed = True
            raise
        if policy is None:
            return logits
        row = logits[0, 0]
        if history is not None:
            apply_penalties(row, *history, policy)
        return torch.tensor([[sample_from_row(row, policy, self.slot.generator)]], dtype=torch.int32)

    def _adopt_policy(self, policy: Qwen38HostSamplingPolicy, *, reseed: bool) -> None:
        """The request's policy, logged when it changes; the draw generator is seeded from ``seed`` for a new request
        (the prefill cleared the slot), a new seed, or the plugin's ``reset_sampling_state``."""

        slot = self.slot
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

        if can_sample_on_device:
            raise ValueError("prefill returns logits: sample_on_device_mode must be decode_only, not all")
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

        if slot != 0:
            raise ValueError(f"one resident slot: slot must be 0, got {slot}")
        self.slot.committed = []
        self.slot.ple_context = None

    def release_persistent_capture(self) -> None:
        """Worker shutdown while the mesh is open: the chain's release order, skipped after a failed chain call."""

        if self.slot.failed:
            logger.warning("a chain call failed; the traces and states are left to the mesh close")
            return
        self.chain.close()

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
        self.chain.write_token_row(token_id)
        self.chain.execute_head(residue)
        self.slot.ple_context = self.chain.refresh_ple_row(token_id, self.slot.ple_context)
        self.chain.execute_tail(residue)
        self.slot.committed.append(token_id)
        return residue

    def _read_logits(self, residue: int) -> torch.Tensor:
        """TAIL(residue)'s full-vocabulary gather (blocking), the padding rows masked, as ``[1, 1, VOCAB_SIZE]``."""

        logits = self.chain.sampling.read_full_logits(residue)
        logits[TOKENIZER_SIZE:] = float("-inf")
        return logits.view(1, 1, VOCAB_SIZE)
