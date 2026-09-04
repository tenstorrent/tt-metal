# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM serving adapter for ``meta-models/Muse-Glimmer-30B``.

This file is a *translation shim*.  Every decision that decides what the model
computes -- precision policy, layer stack, paged cache geometry, trace capture,
token feedback, sampling -- already lives in :mod:`tt.model` and
:mod:`tt.generator`, and stays there.  What is here is the vLLM-facing contract
in ``models/common/readiness_check/contract_vllm.py``:

* :meth:`MuseGlimmerForConditionalGeneration.initialize_vllm_model` builds the
  same generator the readiness runners build, through the same
  ``build_generator`` factory, so the selected datatype-sweep policy
  (``doc/datatype_sweep/selected_precision_config.json``) is loaded by exactly
  the code path that produced the full-model evidence -- weight groups,
  activation/residual dtype, prefill/decode CCL dtype, KV-cache dtype, per-role
  compute fidelities, LM-head dtype/fidelity/geometry and every layer exception;
* :meth:`allocate_kv_cache` makes vLLM the owner of the attention KV cache;
* :meth:`prefill_forward` / :meth:`decode_forward` translate vLLM's per-step
  tensors into the generator's low-level calls and nothing else;
* :meth:`read_decode_output` / :meth:`process_decode_output_host` are the async
  decode split, delegated to the generator.

Sampling
--------

There is exactly one sampling path in this port and the adapter does not add a
second one.  Serving decode replays the full-model generator's canonical split:
the model decode trace produces vocab-sharded logits, and
``models.common.sampling.SamplingGenerator``'s own trace samples them with
``tt_out_tok`` pointing at the persistent decode token input, so the sampled
token never leaves the device.  There is no host argmax, no full-logits
readback, no generic top-k greedy fallback and no Python readback/writeback
token-feedback loop on the measured path.  The generator's explicit host-sampling
compatibility mode is reached only when *vLLM itself* decides a batch cannot be
sampled on device (``min_p``, ``logit_bias``, ``bad_words``,
``allowed_token_ids``, structured outputs, or logprobs on a mesh whose device
count is not 8 or 32); that path is optional, labelled, and never the measured
one.

Cache ownership
---------------

The generator keeps both modes.  Standalone readiness generation lets each layer
keep the pool it allocated at build time; serving replaces it with vLLM's, via
:meth:`MuseGlimmerModel.adopt_external_kv_cache`, and every later
``prefill_forward``/``decode_forward`` threads that same cache back through.  The
adapter holds no cache of its own and makes no standalone-cache assumption: the
block count, the block ids and the lifetime are all vLLM's.

Attention type
--------------

The text stack is hybrid -- 39 sliding-window layers and 13 full-attention
layers -- and :meth:`get_kv_cache_spec` reports it per layer through vLLM's
hybrid KV-cache infrastructure rather than letting the plugin's default single
spec guess.  Every layer is reported as ``FullAttentionSpec`` **without** a
sliding window, which is deliberate and is the same choice
``models/tt_transformers/tt/generator_vllm.py`` makes for Gemma3 and GPT-OSS:
this model's decode passes an *absolute* position to ``paged_update_cache`` and
``paged_scaled_dot_product_attention_decode``, while vLLM's ``SlidingWindowSpec``
zero-pads a sliding group's page table past ``sliding_window / block_size``
entries, so positions beyond the window would collapse onto physical block 0 and
silently corrupt the cache.  The sliding layers still attend only their window --
the SDPA op is given ``sliding_window`` on the read side by
``tt/multichip_decoder.py`` -- so the model's semantics are unchanged; what the
uniform spec costs is memory, not correctness.  Reporting the spec per layer (as
opposed to omitting the hook) also keeps the plugin from deriving a
``FullAttentionSpec(sliding_window=2048)`` from the HF config, which would put
vLLM's block manager into exactly the sliding-window mode this model cannot
consume.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
    DEFAULT_TRACE_REGION_SIZE,
    MuseGlimmerGenerator,
    build_generator,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.model import DECODE_ROWS, HF_MODEL_ID

#: This port's directory, used as the ``model_dir`` argument of the shared factory.
MODEL_DIR = Path(__file__).resolve().parent.parent

#: Decode batch ceiling.  ``nlp_create_qkv_heads_decode`` hard-caps ``num_users``
#: at 32 (``nlp_create_qkv_heads_decode_device_operation.cpp:45-51``), so this is a
#: device-op limit rather than a choice this port makes.
MAX_NUM_SEQS = 32

#: Bytes of device DRAM one paged block costs, per local KV head, for the whole
#: 52-layer stack at the selected KV-cache dtype: ``52 layers x 2 (K,V) x 1 local
#: KV head x 64 tokens x 128 head_dim x 1.0625 B`` (BFLOAT8_B is 1024 mantissa +
#: 64 exponent bytes per 32x32 tile).  P150 stores two local KV heads and multiplies
#: this value by two; P150x2/P150x4 store one.  Used only to explain the pool size
#: in the log line; the authoritative figure is measured on hardware.
BYTES_PER_BLOCK_PER_DEVICE = 905_216

#: Total KV tokens the serving pool is sized for, across all users.
#:
#: This is a *serving-capacity* choice, not a context reduction: ``max_model_len``
#: stays at the advertised 131072 and a single request may still use all of it
#: (2048 of the pool's blocks).
#:
#: **The ceiling is measured, not budgeted on paper**
#: (``doc/vllm_integration/kv_budget_probe.json``).  The probe walks a descending
#: ladder of pool sizes and counts a rung feasible only if, at that size, the model
#: still allocates all 104 cache tensors, captures the decode and sampling traces,
#: runs a full 8192-token prefill chunk -- the largest activation working set the
#: serving path ever builds -- and replays traced decode on top of it.  Measured on
#: the shipped 52-layer build: **28672 blocks (1,835,008 tokens, 25.95 GB/device)
#: is feasible**, leaving 3.00 GB free at the prefill peak, against 27.10 GB free
#: after weights.
#:
#: 16416 blocks ships anyway -- 57 % of that proven-feasible size -- and the margin is
#: the point.  The ceiling run leaves 3.0 GB at a *single-user* prefill peak; a serving
#: process also has to absorb allocator fragmentation across thousands of requests
#: and the 39 sliding layers' persistent prefill tails, and an OOM mid-request is a
#: far worse failure than a smaller pool.  Raising it is a measured, one-line change
#: and belongs to optimized-vLLM, which re-runs the serving evidence anyway; doing it
#: here would leave this stage's committed server log and benchmarks describing a
#: configuration they were not produced with.
#:
#: At 64 tokens per block, 1,048,576 tokens is eight concurrent requests at the full
#: advertised context, or all 32 request slots at 32768 tokens each.
#: ``MUSE_GLIMMER_VLLM_KV_TOKEN_BUDGET`` overrides it.
KV_CACHE_TOKEN_BUDGET = 1_048_576

#: Padded prefill row counts warmed before the decode trace is captured.  The
#: prefill graph is shaped by the padded row count, so a length never seen before
#: compiles on its first request; this list covers every shape the readiness
#: checks and both benchmark profiles use (100-, 128- and few-hundred-token
#: prompts) plus one full chunk, so the steady serving path compiles nothing.
#: Arbitrary lengths outside the list remain ordinary inputs -- they compile once
#: and then run -- which the full-model stage exercises directly at lengths 1, 37,
#: 127, 129, 2049, 4097, 8193 and 12345 interleaved with traced decode.
PREFILL_WARMUP_LENGTHS = (32, 96, 128, 160, 256, 512, 1024, 8192)

#: Padded prefill buckets captured during warmup **when tracing is enabled**, which it is
#: not by default -- see :data:`_PREFILL_TRACE_ENV`.
#:
#: A trace is keyed by the *exact* padded row count: the graph slices the last 32-row tile, so
#: a 37-token prompt cannot be served by the 96-row graph. A wider list therefore covers more
#: prompt lengths. Five bucket sets were measured against a tracing-off control, and **no
#: traced one is correct at every length it was measured at** -- the wide set decays
#: short-prompt output after ~22 generations, and ``[96]``, ``[128]`` and ``[128,1024]`` each
#: change a long eager prompt. The two with no *per-length* failure, ``[1024]`` alone and the
#: wide set, were simply not run at 8192. See
#: ``doc/optimized_vllm/prefill_trace_discriminators.json``. That, and not any bucket count, is
#: why the default is off.
#:
#: ``(128,)`` is retained as the default *set* because it is the most useful single bucket
#: (prompts of 97-128 tokens, the commonest short-chat shape and both benchmark profiles'
#: length) and because it is the configuration the 1.29x TTFT figure was measured on.
#: ``MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS`` overrides it.
#:
#: Two facts worth keeping even though capacity turned out not to be the binding constraint:
#: what a trace removes is host dispatch, so the win falls with prompt length -- 1.33x at 128
#: padded rows, 1.00x at 8192 (``doc/optimized_full_model/prefill_trace_probe.json``,
#: ``prefill_trace_probe_8192.json``) -- and the 400 MB trace region holds 28 traces, the 29th
#: failing cleanly with ``mesh_trace.cpp:81``
#: (``doc/optimized_vllm/probe_trace_capacity.json``).
PREFILL_TRACE_BUCKETS = (128,)

#: Comma-separated override for :data:`PREFILL_TRACE_BUCKETS`, so the bucket set can be
#: swept against a fixed trace region without editing source between arms.
_PREFILL_TRACE_BUCKETS_ENV = "MUSE_GLIMMER_VLLM_PREFILL_TRACE_BUCKETS"


def _prefill_trace_buckets() -> tuple[int, ...]:
    raw = os.environ.get(_PREFILL_TRACE_BUCKETS_ENV, "").strip()
    if not raw:
        return PREFILL_TRACE_BUCKETS
    buckets = tuple(sorted({int(part) for part in raw.replace(" ", "").split(",") if part}))
    logger.warning(f"MuseGlimmer vLLM: {_PREFILL_TRACE_BUCKETS_ENV}={raw} overrides the prefill trace buckets")
    return buckets


#: Traced serving prefill is **off by default**, and the default is the finding.
#:
#: Turned on it is fast and, in isolation, exactly correct. Measured: **1.29x on single-user
#: TTFT** with one bucket (81.48 -> 62.97 ms) and **1.34x** with twenty, decode untouched,
#: +12.5 % on CI serving-burst throughput, and token-identical to the eager path everywhere it
#: was compared offline -- through the adapter on every shared prompt length, in-process
#: eager-vs-traced on the real pinned prompts, and decode-identical across four cache slots in
#: the acceptance suite.
#:
#: It is off because **capturing a prefill trace changes the result of other requests**, in two
#: distinct ways, at both ends of the bucket ladder. Every probe behind this is tabulated in
#: ``doc/optimized_vllm/prefill_trace_discriminators.json``; the narrative is in
#: ``doc/optimized_vllm/README.md`` -> *The bug this stage found*.
#:
#: * **Many buckets.** Served output decays into U+FFFD replacement characters by about the
#:   22nd generation of ordinary greedy/sampled traffic, no seeds, deterministically and
#:   byte-identically across two servers. This one has a mechanism: ttnn requires that
#:   "buffers allocated when a trace is active have to have a lifetime that ends before the
#:   trace is executed" (``tt_metal/impl/allocator/allocator.cpp:113-126``), a captured trace's
#:   intermediates are freed while their addresses stay baked into the replay, and a serving
#:   process allocates continuously.
#: * **Any bucket set, on some long prompt.** Short in-bucket prompts stay clean over 84
#:   generations and 198 benchmark replays, but **long eager prefills** -- prompts the trace
#:   never serves -- diverge from their first token. This one is **unexplained**. A larger
#:   largest bucket helps and is not enough: 1024 makes a 4097-token prompt correct, alone or as
#:   ``[128,1024]`` or as the top of a 20-bucket set, and ``[128,1024]`` still fails 8192
#:   byte-identically to ``[128]``. **No traced configuration measured is correct at every
#:   length it was measured at**, and the two cells that were not measured -- ``[1024]`` alone
#:   at 8192, and any bucket size between 128 and 1024 -- are named in
#:   ``doc/optimized_vllm/bench/run_discriminators.sh`` rather than read as passes.
#:   It is not the rule above -- a run whose only request is the 4097 one, with the bucket
#:   captured at warmup and never replayed, is still wrong, so the capture alone is sufficient.
#:   It is not an unwarmed shape: 8192 is in :data:`PREFILL_WARMUP_LENGTHS`.
#:
#: Four fixes were implemented and measured insufficient: warming every sampling mode at warmup
#: (``fixcheck/``), a blocking traced replay (``soak_blocking/``), reducing the bucket set to
#: one, and widening the largest bucket to 1024 while keeping the fast one (``[128,1024]``) --
#: the last two are in the discriminator matrix.
#: ``MuseGlimmerGenerator._guard_late_sampling_capture`` remains as a partial interlock for the
#: first failure only, not as a licence to default this on.
#:
#: ``MUSE_GLIMMER_VLLM_PREFILL_TRACE=1`` turns it on for a deployment that has soaked its own
#: prompt-length distribution **including lengths outside the buckets**, and for reproducing
#: this stage's numbers.
_PREFILL_TRACE_ENV = "MUSE_GLIMMER_VLLM_PREFILL_TRACE"


def _prefill_trace_enabled() -> bool:
    raw = os.environ.get(_PREFILL_TRACE_ENV, "").strip().lower()
    if raw in {"1", "true", "on", "yes"}:
        logger.warning(
            f"MuseGlimmer vLLM: {_PREFILL_TRACE_ENV}={raw} -- serving prefill will be TRACED. Worth 1.29x "
            "on TTFT and OFF by default because a resident prefill trace was measured to change the output "
            "of other requests -- see doc/optimized_vllm/README.md. Only enable this with a soak over your "
            "own prompt-length distribution, including prompts outside the traced buckets."
        )
        return True
    return False


#: Reduced serving target for the bring-up inner loop, as a comma-separated list of
#: layer indices, e.g. ``MUSE_GLIMMER_VLLM_LAYER_INDICES=0,3`` for one sliding and
#: one full-attention layer.  Everything else is identical to the shipped target --
#: same generator, same adapter, same registration, same page-table and cache
#: shapes, same terminal norm / LM head / sampling path, same traces -- so server
#: launch, cache ownership, trace capture and replay, the async split, on-device
#: sampling and the stale-input rules can all be debugged without reloading 52
#: layers.  It is an inner-loop tool only: it is never accuracy or performance
#: evidence, and the variable is unset for every run that produces either.
_LAYER_INDICES_ENV = "MUSE_GLIMMER_VLLM_LAYER_INDICES"


def _reduced_layer_indices() -> list[int] | None:
    raw = os.environ.get(_LAYER_INDICES_ENV, "").strip()
    if not raw:
        return None
    indices = [int(part) for part in raw.replace(" ", "").split(",") if part]
    logger.warning(
        f"MuseGlimmer vLLM: {_LAYER_INDICES_ENV}={raw} builds a REDUCED {len(indices)}-layer serving "
        "target for bring-up. Its outputs and timings are not this model's."
    )
    return indices


def _text_config(hf_config: Any) -> Any:
    return getattr(hf_config, "text_config", hf_config)


def _tt_config_block_size(default: int = 64) -> int:
    """vLLM's paged block size, read from the engine config when it is reachable.

    The model bakes the block size into every layer's ``PagedAttentionConfig`` at
    construction time, which happens before ``allocate_kv_cache`` is told what
    vLLM chose, so it has to be known here.  ``get_current_vllm_config()`` is set
    for the duration of model loading; when it is not (a direct construction in a
    test), the default matches both this port's ``page_block_size`` and the
    readiness runner's ``--block-size``.  Either way :meth:`allocate_kv_cache`
    re-checks the value against the cache vLLM actually allocates and refuses a
    mismatch rather than silently paging at the wrong stride.
    """
    try:
        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config()
        block_size = getattr(getattr(config, "cache_config", None), "block_size", None)
        if block_size:
            return int(block_size)
    except Exception:  # noqa: BLE001 -- no engine context: fall back to the default
        pass
    return int(default)


class MuseGlimmerForConditionalGeneration:
    """The vLLM-registered handle for this port.

    Named for the checkpoint's own architecture (``config.json`` declares
    ``MuseGlimmerForConditionalGeneration``); the TT plugin also reaches it under
    the ``TT``-prefixed alias it derives at config time.  The checkpoint carries a
    vision tower, but this port implements the text decoder only, so the class
    deliberately does **not** declare ``SupportsMultiModal`` -- that keeps vLLM's
    request path text-only and matches what the model actually computes.
    """

    #: Read by ``TTPlatform.check_and_update_config``.
    #:
    #: ``supports_async_decode`` is claimed because the split is implemented and
    #: proved, not because it is available: ``decode_forward(read_from_device=
    #: False)`` returns device handles, ``read_decode_output(async_read=True)``
    #: enqueues the minimal deferred read and returns its event, and
    #: ``process_decode_output_host`` does host formatting only.  Under overlap the
    #: adapter never restages tokens or positions from host -- see
    #: :meth:`decode_forward`.
    #:
    #: ``supports_prefix_caching`` stays False: nothing in this port implements or
    #: tests prefix reuse, and 39 of the 52 layers are sliding-window anyway.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    #: Read by the worker's KV-block heuristic.  False because
    #: :meth:`get_kv_cache_spec` emits one uniform attention type, so vLLM builds a
    #: single group and must not add per-sliding-group block headroom on top.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = False

    def __init__(self, generator: MuseGlimmerGenerator, *, max_num_seqs: int, vllm_config: Any = None) -> None:
        # ``vllm_config`` is accepted and ignored: vLLM's ``_check_vllm_model_init``
        # tests that a generative model's ``__init__`` takes that keyword.  See the
        # protocol shim at the bottom of this class for why the check reaches here at
        # all.  The TT loader constructs this class through
        # :meth:`initialize_vllm_model`, never through vLLM's own constructor path.
        self.vllm_config = vllm_config
        self.generator = generator
        self.max_num_seqs = int(max_num_seqs)
        self.already_warmed_up_prefill = False
        self.already_warmed_up_decode = False
        #: Whether this server captures prefill traces.  Read once, at construction,
        #: so a mid-run environment change cannot desynchronise the warmup from the
        #: capability report.
        self.prefill_trace_enabled = _prefill_trace_enabled()
        #: Padded buckets whose prefill trace was captured during warmup.
        self.prefill_trace_buckets: list[int] = []
        #: Whether the *device* copies of the decode token / position / RoPE-index
        #: inputs are current, i.e. whether the previous step was a decode that
        #: sampled on device and therefore wrote the token back and advanced the
        #: position inside the traced graph.  False after a prefill, after a
        #: host-sampled decode step, and before the first step -- in each of those
        #: cases the host inputs are authoritative and must be restaged.
        self._device_inputs_current = False

    # ----------------------------------------------------------------- loading

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: Any,
        mesh_device: Any,
        max_batch_size: int,
        max_seq_len: int | None = None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **kwargs: Any,
    ) -> "MuseGlimmerForConditionalGeneration":
        if tt_data_parallel != 1:
            raise ValueError(
                f"tt_data_parallel={tt_data_parallel} is not supported for this port: the model is "
                "tensor-parallel over the selected P150/P150x2/P150x4 mesh, so data-parallel "
                "replication is not implemented."
            )
        if optimizations is not None:
            raise ValueError(
                f"optimizations={optimizations!r} is not supported: the precision policy is a build "
                "input read from doc/datatype_sweep/selected_precision_config.json, not a preset."
            )
        num_devices = int(mesh_device.get_num_devices())
        if num_devices not in (1, 2, 4):
            raise ValueError(
                f"this port supports P150/P150x2/P150x4 (1, 2, or 4 devices); " f"vLLM opened {num_devices} device(s)."
            )
        max_num_seqs = int(max_batch_size)
        if max_num_seqs < 1 or max_num_seqs > MAX_NUM_SEQS:
            raise ValueError(
                f"max_num_seqs={max_num_seqs} is outside 1..{MAX_NUM_SEQS}; "
                "nlp_create_qkv_heads_decode hard-caps num_users at 32."
            )

        served = str(getattr(hf_config, "_name_or_path", "") or "")
        if served and Path(served).name.lower() not in HF_MODEL_ID.lower():
            raise ValueError(
                f"vLLM was launched with --model {served!r}, but this port loads {HF_MODEL_ID!r} weights. "
                "Serving one checkpoint's tokenizer against another's weights is silently wrong."
            )

        advertised = int(_text_config(hf_config).max_position_embeddings)
        if max_seq_len is None:
            max_seq_len = advertised
        max_seq_len = int(max_seq_len)
        if max_seq_len > advertised:
            raise ValueError(f"--max_model_len={max_seq_len} exceeds the checkpoint's advertised context {advertised}")
        if max_seq_len < advertised:
            logger.warning(
                f"serving --max_model_len={max_seq_len} below the advertised context {advertised}; this port "
                "supports the full advertised context (doc/context_contract.json records no capability "
                "reduction), so this is the caller's choice, not a device limit."
            )

        block_size = _tt_config_block_size()
        blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        logger.info(
            f"MuseGlimmer vLLM: building the generator (max_model_len={max_seq_len}, "
            f"max_num_seqs={max_num_seqs}, block_size={block_size})"
        )
        layer_indices = _reduced_layer_indices()
        generator = build_generator(
            MODEL_DIR,
            mesh_device,
            model_id=HF_MODEL_ID,
            layer_indices=layer_indices,
            # The build-time pool is the smallest legal one -- one sequence at the
            # advertised context -- because vLLM owns the serving pool and
            # ``allocate_kv_cache`` replaces this one and frees it.  Building at the
            # serving batch instead would allocate ``max_num_seqs x blocks_per_seq``
            # blocks up front, which at 32 users and 131072 tokens is 59 GB/device.
            max_batch_size=1,
            max_num_blocks=blocks_per_seq,
            max_seq_len=max_seq_len,
            page_block_size=block_size,
        )
        return cls(generator, max_num_seqs=max_num_seqs)

    @property
    def mesh_device(self) -> Any:
        return self.generator.mesh_device

    @property
    def model(self) -> Any:
        return self.generator.model

    # ------------------------------------------------------- serving capacity

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs: Any,
    ) -> int:
        """Total KV tokens the serving pool holds, across all users.

        vLLM turns this into a block count and hands the block ids out on demand,
        so it is a capacity statement about the *pool*, not about any one request:
        ``max_model_len`` is unaffected and a single request may still take the
        full advertised context out of it.
        """
        if max_num_seqs is not None and int(max_num_seqs) > MAX_NUM_SEQS:
            raise ValueError(
                f"max_num_seqs={max_num_seqs} exceeds this model's decode batch ceiling of {MAX_NUM_SEQS} "
                "(nlp_create_qkv_heads_decode hard-caps num_users at 32)."
            )
        seqs = int(max_num_seqs or 1)
        context = int(max_model_len or 0)
        want = context * seqs if context else KV_CACHE_TOKEN_BUDGET
        budget = int(os.environ.get("MUSE_GLIMMER_VLLM_KV_TOKEN_BUDGET", KV_CACHE_TOKEN_BUDGET))
        tokens = min(want, budget) if want else budget
        if context and tokens < context:
            raise ValueError(
                f"the KV pool budget of {budget} tokens cannot hold one request at max_model_len={context}; "
                "raise MUSE_GLIMMER_VLLM_KV_TOKEN_BUDGET or lower --max_model_len."
            )
        local_kv_heads = 2 if int(num_devices) == 1 else 1
        bytes_per_block = BYTES_PER_BLOCK_PER_DEVICE * local_kv_heads
        logger.info(
            f"MuseGlimmer vLLM: KV pool sized for {tokens} tokens across all users "
            f"(~{tokens // 64 * bytes_per_block / 1e9:.2f} GB/device), "
            f"max_model_len={context}, max_num_seqs={seqs}"
        )
        return tokens

    @classmethod
    def get_kv_cache_spec(cls, vllm_config: Any) -> dict:
        """One ``FullAttentionSpec`` per decoder layer; see the module docstring."""
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        text_config = _text_config(model_config.hf_config)
        layer_types = getattr(text_config, "layer_types", None)
        if not layer_types:
            raise ValueError(
                "MuseGlimmer's text config must declare layer_types "
                "('sliding_attention' / 'full_attention' per layer); none found"
            )
        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        common = dict(
            block_size=cache_config.block_size,
            num_kv_heads=model_config.get_num_kv_heads(vllm_config.parallel_config),
            head_size=model_config.get_head_size(),
            dtype=dtype,
        )
        spec = {}
        for index, kind in enumerate(layer_types):
            if kind not in ("sliding_attention", "full_attention"):
                raise ValueError(f"unsupported layer_type {kind!r} at layer {index}")
            spec[f"model.layers.{index}.self_attn"] = FullAttentionSpec(**common)
        return spec

    # ---------------------------------------------------------- cache ownership

    def allocate_kv_cache(self, kv_cache_shape: Any, dtype: Any, num_layers: int) -> list:
        """Allocate the vLLM-owned paged attention cache and bind the model to it.

        ``dtype`` is vLLM's *spec* dtype, which describes the logical cache
        element, not the on-device encoding.  The device tensors are allocated at
        the selected precision policy's ``kv_cache_dtype`` (BFLOAT8_B) -- the same
        dtype the datatype sweep measured and the full model shipped -- because
        that is what the paged ops in this port read and write.  The declared
        ``dtype`` is only used to size vLLM's own accounting.
        """
        model = self.generator.model
        num_blocks, kv_heads, block_size, head_dim = (int(v) for v in kv_cache_shape)
        if int(num_layers) != model.config.num_layers:
            if _reduced_layer_indices() is None:
                raise ValueError(f"vLLM asked for {num_layers} cache layers; the model has {model.config.num_layers}")
            logger.warning(
                f"MuseGlimmer vLLM: allocating {model.config.num_layers} cache layer(s) for the reduced "
                f"bring-up target instead of vLLM's {num_layers}; block accounting is unchanged."
            )
            num_layers = model.config.num_layers
        if block_size != model.config.page_block_size:
            raise ValueError(
                f"vLLM's paged block size is {block_size} but the model was built with "
                f"{model.config.page_block_size}. Launch the server with "
                f"--block_size {model.config.page_block_size}."
            )
        if kv_heads != model.plan.local_kv_heads or head_dim != model.plan.head_dim:
            raise ValueError(
                f"vLLM asked for {kv_heads} KV head(s) x {head_dim} head dim; this mesh carries "
                f"{model.plan.local_kv_heads} x {model.plan.head_dim} per device"
            )

        cache_dtype = model.precision.kv_cache_dtype
        shape = (num_blocks, kv_heads, block_size, head_dim)
        logger.info(
            f"MuseGlimmer vLLM: allocating the serving KV cache -- {num_layers} layers x 2 x {shape} "
            f"{cache_dtype} ({num_blocks * BYTES_PER_BLOCK_PER_DEVICE * model.plan.local_kv_heads / 1e9:.2f} GB/device, "
            f"{num_blocks * block_size} tokens across all users, declared dtype {dtype})"
        )
        kv_cache = [
            [
                ttnn.zeros(
                    shape,
                    dtype=cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            ]
            for _ in range(int(num_layers))
        ]
        model.adopt_external_kv_cache(kv_cache, cache_slots=self.max_num_seqs)
        return kv_cache

    # ------------------------------------------------------------------ warmup

    def warmup_model_prefill(
        self,
        *,
        kv_cache: Any,
        can_sample_on_device: bool,
        enable_trace: bool,
        **kwargs: Any,
    ) -> None:
        """Compile the serving prefill path at the padded row counts serving uses.

        ``enable_trace`` is accepted and does not decide anything here, and the
        reason is an ordering constraint rather than indifference.  This port's
        prefill trace is keyed by padded prompt length, so it is a *set* of traces,
        one per bucket, and they compete with the decode and sampling traces for one
        fixed ``trace_region_size``.  The plugin's warmup calls prefill before decode
        (``model_runner.py::warmup_model``), so capturing here would let TTFT work
        starve the per-token path -- the wrong trade in every workload.  The buckets
        are therefore captured at the end of :meth:`warmup_model_decode`, once the
        decode and sampling traces are safely resident, and this method stays what it
        always was: the compile pass that puts every serving prefill shape in the
        program cache.
        """
        if self.already_warmed_up_prefill:
            return
        generator = self.generator
        max_len = generator.model.config.max_seq_len
        lengths = sorted({length for length in PREFILL_WARMUP_LENGTHS if length <= max_len})
        logger.info(f"MuseGlimmer vLLM: prefill warmup over padded lengths {lengths}")
        for length in lengths:
            tokens = torch.zeros(1, length, dtype=torch.int32)
            generator.prefill_forward(
                tokens,
                page_table=None,
                kv_cache=kv_cache,
                prompt_lens=[length],
                sample_on_device=bool(can_sample_on_device),
            )
        self.already_warmed_up_prefill = True

    def warmup_model_decode(
        self,
        *,
        kv_cache: Any,
        max_batch_size: int,
        num_blocks: int,
        can_sample_on_device: bool,
        enable_trace: bool,
        **kwargs: Any,
    ) -> None:
        """Compile, and on the second pass capture, the traced serving decode step.

        The model decode trace and **every** sampling-mode trace are captured here,
        over the persistent inputs the steady-state step replays, so the first real
        request pays no capture -- and, more important than latency, so that *nothing
        is captured after this point*.

        The sampler keys a trace per ``(penalties, log_probs, force_argmax)`` mode and
        captures it lazily, on the first request that needs it.  A capture allocates
        from the same device memory whose addresses the already-captured traces baked
        in for their intermediates, which ttnn warns about explicitly ("Allocating
        device buffers is unsafe due to the existence of an active trace").  Leaving
        the penalised mode to be captured mid-serving is what corrupted this port's
        first optimized-vLLM arm: a penalised request arrived during the sampling
        suite, its trace was captured on top of 20 resident prefill traces, and every
        completion after it came back as replacement characters
        (``doc/optimized_vllm/README.md``).  Warming the modes here is the fix;
        ``MuseGlimmerGenerator._guard_late_sampling_capture`` is the backstop for a
        mode this list does not foresee.
        """
        generator = self.generator
        rows = DECODE_ROWS
        tokens = torch.zeros(rows, 1, dtype=torch.int32)
        positions = torch.zeros(rows, dtype=torch.int64)
        # An explicit warmup page table, one distinct block per row, rather than the
        # generator's ``None`` default.  The default gives each *cache slot* a private
        # run of ``blocks_per_seq`` blocks, which a shared serving pool cannot afford
        # for 32 slots, so slots past what the pool holds alias the last one -- and a
        # warmup that drives all 32 rows at position 0 would then have ~30 rows writing
        # the same physical block at once.  Position 0 only ever touches column 0, so a
        # constant row is a valid table and one distinct block per row is the cheapest
        # way to keep the warmup's writes disjoint.
        page_table = (
            torch.arange(rows, dtype=torch.int32).reshape(rows, 1).repeat(1, generator.model.config.blocks_per_seq)
        )
        for params in self._warmup_sampling_modes(rows) if can_sample_on_device else [None]:
            generator.decode_forward(
                tokens,
                positions,
                page_table=page_table,
                kv_cache=kv_cache,
                sample_on_device=bool(can_sample_on_device),
                sampling_params=params,
                enable_trace=bool(enable_trace),
                read_from_device=True,
                refresh_inputs=True,
            )
        if can_sample_on_device:
            # Leave the sampler in the greedy state the first real request expects; the
            # penalised pass above is a warmup artifact, not a policy.
            generator.apply_decode_sampling_state(
                self._warmup_sampling_modes(rows)[-1], start_pos=positions, reset_batch=True
            )
        # Warmup drove the device inputs with synthetic values and, when tracing,
        # advanced them; the first real decode step carries ``reset_batch`` and
        # restages them, but say so explicitly rather than relying on that.
        self._device_inputs_current = False
        self.already_warmed_up_decode = bool(enable_trace)
        logger.info(
            f"MuseGlimmer vLLM: decode warmup done (trace={'captured' if enable_trace else 'compiled'}, "
            f"sample_on_device={bool(can_sample_on_device)}, "
            f"sampling modes {len(self._warmup_sampling_modes(rows)) if can_sample_on_device else 0}, "
            f"batch capacity {max_batch_size}, {num_blocks} blocks per request)"
        )
        if enable_trace and self.prefill_trace_enabled:
            self._capture_prefill_traces(kv_cache=kv_cache, can_sample_on_device=bool(can_sample_on_device))

    @staticmethod
    def _warmup_sampling_modes(rows: int) -> list[Any]:
        """One ``SamplingParams`` per sampling-trace key serving can reach on this mesh.

        The sampler keys its traces on ``(penalties, log_probs, force_argmax)``.
        ``force_argmax`` is fixed ``False`` by this port's policy, and ``log_probs`` is
        unreachable on a device-sampled batch here -- the TT plugin routes any logprobs
        request to host sampling on a mesh whose device count is not 8 or 32, and this
        is a 4-die mesh -- which the committed server logs confirm: every
        ``Pre-compiling sampling path`` line in them reads ``log_probs_on=False``.  That
        leaves two modes, and both are warmed:

        * greedy, no penalties -- the benchmarked steady-state path;
        * penalties active -- reached by any request with a non-default presence,
          frequency or repetition penalty.  The values only have to be non-default;
          what is being warmed is the graph, not a policy.

        If a future plugin or mesh makes a third mode reachable, the generator's
        ``_guard_late_sampling_capture`` releases the prefill traces rather than let it
        capture on top of them, and says so in the log.
        """
        from models.common.sampling.generator import SamplingParams as _SamplingParams

        greedy = _SamplingParams(temperature=[0.0] * rows, top_k=[1] * rows, top_p=[1.0] * rows)
        penalised = _SamplingParams(
            temperature=[0.0] * rows,
            top_k=[1] * rows,
            top_p=[1.0] * rows,
            presence_penalty=[0.1] * rows,
            frequency_penalty=[0.1] * rows,
            repetition_penalty=[1.1] * rows,
        )
        # Greedy last so the sampler is left in the state the first real request
        # expects; the explicit restage after the loop makes that independent of order,
        # but a warmup that ends in the steady-state mode is easier to reason about.
        return [penalised, greedy]

    def _capture_prefill_traces(self, *, kv_cache: Any, can_sample_on_device: bool) -> None:
        """Capture one prefill trace per short padded bucket, after the decode trace.

        Ordering, ascending bucket width, and a failure that stops rather than
        cascades are all deliberate:

        * **after the decode trace** so a prefill capture can never take the region
          the per-token path needs (see :meth:`warmup_model_prefill`);
        * **ascending** so that if the region does run out it is the widest, least
          valuable bucket that is lost -- the trace's value falls with prompt length
          (1.33x at 128 padded rows, 1.00x at 8192);
        * **stop on the first failure**, because the generator disables capture for
          its own lifetime after one failure and says why; continuing would only
          produce more of the same warning.

        Every capture drives the ordinary ``prefill_forward`` entry point with the
        serving cache, so the traced graph is the serving graph -- the same call the
        first real request would have made, just paid here instead of by that request.
        """
        generator = self.generator
        max_len = generator.model.config.max_seq_len
        buckets = [length for length in _prefill_trace_buckets() if length <= max_len]
        if not buckets:
            return
        generator.enable_prefill_trace(max_entries=len(buckets), max_padded_len=max(buckets))
        logger.info(f"MuseGlimmer vLLM: capturing prefill traces for padded buckets {buckets}")
        for length in buckets:
            tokens = torch.zeros(1, length, dtype=torch.int32)
            generator.prefill_forward(
                tokens,
                page_table=None,
                kv_cache=kv_cache,
                prompt_lens=[length],
                sample_on_device=bool(can_sample_on_device),
            )
            if length not in generator.prefill_trace_buckets:
                logger.warning(
                    f"MuseGlimmer vLLM: bucket {length} did not capture a prefill trace; stopping here and "
                    "serving the remaining buckets eagerly."
                )
                break
        self.prefill_trace_buckets = list(generator.prefill_trace_buckets)
        logger.info(f"MuseGlimmer vLLM: prefill traces resident for padded buckets {self.prefill_trace_buckets}")
        # The captures drove the persistent decode inputs through a prefill's sampling
        # state; the first real decode step carries ``reset_batch`` and restages them,
        # but say so rather than rely on it.
        self._device_inputs_current = False

    # ----------------------------------------------------------------- prefill

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any,
        enable_trace: bool,
        prompt_lens: Any,
        start_pos: Any,
        page_tables_per_layer: Any = None,
        sampling_params: Any = None,
        empty_slots: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """One serving prefill step: ``[batch]`` sampled ids, or ``[batch, 1, vocab]``.

        The prompt length that reaches the generator is the caller's *logical*
        length.  Nothing here rounds it to the tile, the 64-token page or the 8192
        prefill chunk: the generator pads the ids with the zero-embedding pad id,
        the layer stack masks the padded tail, and the logits are sliced back to
        the logical last position -- so a request whose prompt length divides none
        of those is an ordinary input.
        """
        self._reject_per_layer_page_tables(page_tables_per_layer)
        lens = [int(length) for length in prompt_lens]
        if start_pos is not None:
            offsets = {int(p) for p in torch.as_tensor(start_pos).reshape(-1).tolist()}
            if offsets - {0}:
                raise NotImplementedError(
                    f"serving prefill starts every request at position 0; vLLM asked for {sorted(offsets)}. "
                    "Chunked prefill is disabled by the TT platform, and this port does not expose the "
                    "layer stack's continuation prefill through the serving path."
                )
        out = self.generator.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=lens,
            sample_on_device=sampling_params is not None,
            sampling_params=sampling_params,
            user_ids=list(empty_slots) if empty_slots is not None else None,
        )
        # A prefill leaves the persistent decode inputs untouched, so whatever the
        # device held is now stale with respect to the requests vLLM is running.
        self._device_inputs_current = False
        return out

    # ------------------------------------------------------------------ decode

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any,
        start_pos: Any,
        enable_trace: bool,
        read_from_device: bool,
        page_tables_per_layer: Any = None,
        sampling_params: Any = None,
        prompt_tokens: Any = None,
        output_tokens: Any = None,
        reset_batch: bool | None = None,
        slot_remap: Any = None,
        rope_deltas_all_users: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Submit one serving decode step.

        The only judgement call in this method is whether the host copies of
        ``tokens`` and ``start_pos`` may be trusted, and it is the one that decides
        whether async scheduling is safe.  When the step samples on device and the
        padded batch layout has not changed, vLLM may have built these inputs
        before the previous step's token was applied to its own state -- they are
        stale by construction.  The device copies are not: the sampler wrote the
        token straight into the persistent decode token input and the decode trace
        advanced ``current_pos`` and the RoPE index with ``plus_one``, each exactly
        once per emitted token.  So that case reads nothing from host.  Every other
        case -- the first decode after a prefill or a warmup, a layout change
        (``reset_batch``), a slot remap, or a step whose predecessor sampled on
        host and therefore never wrote the token back -- restages both from host,
        and vLLM guarantees those steps are drained of pending async work.

        The page table is refreshed on its own terms in either case: it changes
        when a sequence crosses a block boundary, which has nothing to do with the
        sampled token, and the generator copies it only when its contents actually
        differ from the last one staged.
        """
        self._reject_per_layer_page_tables(page_tables_per_layer)
        if rope_deltas_all_users is not None:
            raise NotImplementedError("this model does not use request-specific mRoPE deltas")
        sample_on_device = sampling_params is not None
        layout_changed = True if reset_batch is None else bool(reset_batch)
        if slot_remap is not None and not self._is_identity_remap(slot_remap):
            layout_changed = True

        if sample_on_device:
            self.generator.apply_decode_sampling_state(
                sampling_params,
                start_pos=start_pos,
                reset_batch=layout_changed,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
            )
        refresh_inputs = not (sample_on_device and self._device_inputs_current and not layout_changed)
        out = self.generator.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sample_on_device=sample_on_device,
            sampling_params=None,
            enable_trace=bool(enable_trace),
            read_from_device=bool(read_from_device),
            refresh_inputs=refresh_inputs,
            advance_seeds=False,
        )
        self._device_inputs_current = bool(sample_on_device and enable_trace)
        return out

    def read_decode_output(self, tt_out: Any, async_read: bool = False) -> Any:
        return self.generator.read_decode_output(tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out: Any, is_tokens: bool = False) -> torch.Tensor:
        return self.generator.process_decode_output_host(tt_out, is_tokens=is_tokens)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _reject_per_layer_page_tables(page_tables_per_layer: Any) -> None:
        """Refuse a hybrid multi-group submission rather than silently ignore it.

        :meth:`get_kv_cache_spec` reports one uniform attention type, so vLLM
        builds a single KV-cache group and never populates this argument.  If a
        future plugin change does populate it, every layer's table would have to be
        routed to its own attention layer; dropping it would page against the wrong
        pool with no error.
        """
        if page_tables_per_layer is None:
            return
        raise NotImplementedError(
            "this adapter reports one uniform KV-cache group, so it has no per-layer page-table routing; "
            f"vLLM supplied {len(page_tables_per_layer)} per-layer table(s)."
        )

    @staticmethod
    def _is_identity_remap(slot_remap: Any) -> bool:
        remap = torch.as_tensor(slot_remap).reshape(-1)
        return bool(torch.equal(remap, torch.arange(remap.numel(), dtype=remap.dtype)))

    # ---------------- vLLM ``VllmModelForTextGeneration`` protocol shim ----------
    #
    # ``ModelConfig.__post_init__`` decides whether the model can run
    # ``--runner generate`` by inspecting the class its registry resolves for the
    # checkpoint's architecture, and that happens *before*
    # ``TTPlatform.check_and_update_config`` prepends ``TT`` to the architecture
    # list.  Most TT models never notice: vLLM finds an upstream torch
    # implementation for ``LlamaForCausalLM`` and friends and inspects *that*,
    # while the plugin's prefix logic routes execution to the TT class.  This
    # checkpoint has no upstream vLLM implementation, so the inspection lands
    # here, and without these three methods it fails with
    #
    #     Value error, This model does not support `--runner generate`.
    #
    # before any TT code runs.  ``models/demos/gemma4/tt/generator_vllm.py`` carries
    # the same shim for the same reason.
    #
    # They are never called: execution goes through ``prefill_forward`` /
    # ``decode_forward``, which the TT model runner invokes directly.  They raise
    # rather than return something plausible, so a future caller that does reach
    # them fails loudly instead of silently computing nothing.

    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "MuseGlimmerForConditionalGeneration is a TT bridge; embedding happens on "
            "device inside prefill_forward / decode_forward, not through this method."
        )

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "MuseGlimmerForConditionalGeneration is a TT bridge; the TT model runner "
            "calls prefill_forward / decode_forward, not forward()."
        )

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "MuseGlimmerForConditionalGeneration is a TT bridge; logits are produced on "
            "device and surfaced through prefill_forward / decode_forward."
        )

    # --------------------------------------------------------------- reporting

    def capability_report(self) -> dict:
        """The generator's own report, plus what serving added on top of it."""
        report = dict(self.generator.capability_report())
        model_config = self.generator.model.config
        report["vllm"] = {
            "adapter": f"{type(self).__module__}:{type(self).__name__}",
            "model_capabilities": dict(self.model_capabilities),
            "max_num_seqs": self.max_num_seqs,
            "served_max_model_len": model_config.max_seq_len,
            "kv_cache_owner": "vllm (allocate_kv_cache -> MuseGlimmerModel.adopt_external_kv_cache)",
            "kv_cache_blocks": model_config.max_num_blocks,
            "kv_cache_tokens_all_users": model_config.max_num_blocks * model_config.page_block_size,
            "page_block_size": model_config.page_block_size,
            "blocks_per_seq": model_config.blocks_per_seq,
            "trace_region_size_default": DEFAULT_TRACE_REGION_SIZE,
            "prefill_warmup_lengths": list(PREFILL_WARMUP_LENGTHS),
            "prefill_trace_enabled": bool(self.prefill_trace_enabled),
            "prefill_trace_buckets_requested": list(_prefill_trace_buckets()),
            "prefill_trace_buckets_captured": list(self.prefill_trace_buckets),
            "prefill_trace_buckets_resident": list(self.generator.prefill_trace_buckets),
            "serving_counters": dict(getattr(self.generator, "serving_counters", {})),
            "device_inputs_current": self._device_inputs_current,
        }
        return report


#: The plugin derives a ``TT``-prefixed architecture name at config time; both
#: names resolve to the same class, so the registry can be given either.
TTMuseGlimmerForConditionalGeneration = MuseGlimmerForConditionalGeneration


__all__ = [
    "KV_CACHE_TOKEN_BUDGET",
    "MAX_NUM_SEQS",
    "MuseGlimmerForConditionalGeneration",
    "TTMuseGlimmerForConditionalGeneration",
]
