# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 prefill-graph bring-up on a Blackhole Galaxy (1x32 mesh, TP=32).

Three tests climbing the prefill stack, each parametrized over prefill chunk size
and over eager/traced execution:

  ``test_prefill_layer``   one decoder layer (sliding *and* global), PCC vs HuggingFace
  ``test_prefill_layers``  all 60 decoder layers + final norm, lm_head skipped
  ``test_prefill_full``    the whole graph: embed -> 60 layers -> norm -> lm_head -> softcap

Chunk sizes bracket Gemma4's 1024-token sliding window: 512 (below it, window
inactive), 1024 (exactly on it), 2048 and 4096 (above it, window masks). All four
are trace-eligible buckets — see ``GEMMA4_TRACE_PREFILL_SEQ_LENS`` and
``GEMMA4_MAX_TRACE_PREFILL_SEQ_LEN`` (4096) in ``tt/generator_trace.py``.

Weights come from the on-disk tensor cache under ``TT_CACHE_PATH``; only
``layer_scalar`` and ``embed_tokens.weight`` are read from the checkpoint (see
``utils/partial_weights.py`` for why). Set ``GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1``
to load the full host state dict instead — that is what populates a cold cache.

Usage:
    export HF_HUB_OFFLINE=1 \\
           HF_HOME=/localdev/svuckovic/huggingface \\
           HF_MODEL=google/gemma-4-31B-it \\
           TT_CACHE_PATH=/localdev/svuckovic/huggingface/tt_cache/google--gemma-4-31B-it

    # single layer, both attention types, every chunk size, eager + traced
    pytest models/demos/gemma4/demo/text_demo_prefill.py::test_prefill_layer -s

    # 60-layer prefill body at 2k, eager vs traced side by side
    pytest models/demos/gemma4/demo/text_demo_prefill.py::test_prefill_layers -k chunk_2048 -s

    # full graph, traced, 4k
    pytest models/demos/gemma4/demo/text_demo_prefill.py::test_prefill_full -k "traced and chunk_4096" -s

    # device-op breakdown of the traced replay (signposted region only)
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \\
      python -m tracy -p -r -v -m pytest \\
      models/demos/gemma4/demo/text_demo_prefill.py::test_prefill_layers -k "traced and chunk_4096"

The ``mesh_device`` fixture is function-scoped, so each parametrized case reloads
the model from cache. Filter with ``-k`` while iterating.
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import (
    TestFactory,
    build_hf_prefill_mask,
    compare_tensors,
    find_layer_idx,
    get_pcc_threshold,
    parametrize_mesh_with_fabric,
)
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.layer import Gemma4DecoderLayer
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.tt.precision import Gemma4Precision
from models.demos.gemma4.utils.partial_weights import load_cache_completion_state, load_layer_state
from models.tt_transformers.tt.common import PagedAttentionConfig

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*_args, **_kwargs):
        pass


# ── Configuration ─────────────────────────────────────────────────────────────

# TP is mesh axis 1 (MeshConfig.tp_axis) and CP is axis 0, so a (4, 8) mesh gives
# TP=8 x CP=4 and a (1, 32) mesh gives TP=32 with no context parallelism.
#
# (4, 8) is the default because TP=32 currently hangs: hidden_size 5376 / 32 = 168
# makes the embedding all-gather's row-major page 336 B, which is not 64 B aligned,
# so ttnn.all_gather falls back to composite_all_gather and deadlocks at 32 devices
# (see models/demos/gemma4/GALAXY_1x32_HANG.md). TP=8 gives 672 -> 1344 B = 21x64,
# which stays on the native path.
#
# Note the tensor cache is tagged by TP (``_tp8_`` vs ``_tp32_``), so switching mesh
# needs a matching cache; _require_cache reports it when missing.
_MESH_SHAPES = {"4x8": (4, 8), "1x32": (1, 32), "1x8": (1, 8), "1x4": (1, 4)}
GALAXY_MESH = _MESH_SHAPES[os.environ.get("GEMMA4_PREFILL_MESH", "4x8")]

# Prefill chunk sizes, chosen to bracket the 1024-token sliding window.
PREFILL_CHUNK_SIZES = [512, 1024, 2048, 4096]

MODEL_DTYPE = ttnn.bfloat16
PAGE_BLOCK_SIZE = 64

# 31B's decoder depth, asserted so a silently-truncated stack can't pass as a
# full-model run. Other variants are checked against their own config only.
_EXPECTED_31B_LAYERS = 60

# Trace command buffer. Existing Gemma4 trace tests use 200-256 MB, but all of
# them are 1x4 meshes with much shorter graphs than a 60-layer prefill body at
# 4096 tokens, so this default is a starting point rather than a measured value.
# If capture fails with an allocation error, raise it — the failure is loud at
# capture time, not silent corruption at replay.
TRACE_REGION_SIZE = int(os.environ.get("GEMMA4_PREFILL_TRACE_REGION_SIZE", 256_000_000))

_MESH_PARAMS = dict(device_params_extra={"trace_region_size": TRACE_REGION_SIZE})

_parametrize_chunk = pytest.mark.parametrize(
    "chunk", PREFILL_CHUNK_SIZES, ids=[f"chunk_{c}" for c in PREFILL_CHUNK_SIZES]
)
_parametrize_traced = pytest.mark.parametrize("traced", [False, True], ids=["eager", "traced"])


def _model_path():
    return os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", "google/gemma-4-31B-it")


def _load_full_weights():
    """True when the caller wants the full host state dict (cold-cache path)."""
    return os.environ.get("GEMMA4_PREFILL_LOAD_FULL_WEIGHTS", "0").lower() in ("1", "true", "yes")


def _guard_chunk(request, chunk):
    max_prefill = request.config.getoption("--max-prefill")
    if chunk > max_prefill:
        pytest.skip(f"chunk={chunk} > --max-prefill={max_prefill}")


def _guard_cp_rope_alignment(mesh_device, chunk):
    """Skip when CP is on but the model's RoPE cache would not line up.

    Under CP the 4D prefill RoPE cache is sharded along positions, so rank ``r``
    holds ``[r*max_seq_len/cp, (r+1)*max_seq_len/cp)`` while the model slices
    ``[0:chunk/cp]`` of that local shard. Those agree only when ``max_seq_len ==
    chunk``, which is the default. A larger GEMMA4_MAX_SEQ_LEN would silently give
    every rank the wrong positions, so refuse instead.
    """
    from models.demos.gemma4.tt.ccl import cp_degree

    cp = cp_degree(_mesh_config(mesh_device))
    if cp <= 1:
        return
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", chunk))
    if max_seq_len != chunk:
        pytest.skip(
            f"CP={cp} needs GEMMA4_MAX_SEQ_LEN == chunk so the sharded RoPE cache lines up with the "
            f"positions each rank owns, but max_seq_len={max_seq_len} and chunk={chunk}. Unset "
            f"GEMMA4_MAX_SEQ_LEN, or run with GEMMA4_PREFILL_MESH=1x32 (CP=1)."
        )
    if chunk % cp != 0:
        pytest.skip(f"chunk={chunk} is not divisible by CP={cp}")


# ── Weight loading from the tensor cache ──────────────────────────────────────


def _cache_root(model_path):
    """Absolute path of the tensor cache directory for this model + dtype."""
    args = Gemma4ModelArgs()
    args.model_cache_path = Gemma4ModelArgs.resolve_model_cache_path(model_path)
    return str(args.weight_cache_path(MODEL_DTYPE))


def _require_cache(cache_root, tp, num_layers):
    """Skip with actionable instructions unless the tensor cache looks usable.

    Without this, a cold or wrong-TP cache surfaces as ``ttnn.as_tensor`` calling
    ``from_torch(None)`` deep inside weight loading, which is an opaque crash.
    """
    if _load_full_weights():
        return

    missing = []
    if not os.path.isdir(cache_root):
        missing.append(cache_root)
    else:
        if not os.path.isdir(os.path.join(cache_root, f"layer_{num_layers - 1}")):
            missing.append(f"layer_{num_layers - 1}/")
        if not os.path.isdir(os.path.join(cache_root, "final_norm")):
            missing.append("final_norm/")
        entries = os.listdir(cache_root)
        if not any(e.startswith(f"embed_tokens.weight_tp{tp}_") for e in entries):
            missing.append(f"embed_tokens.weight_tp{tp}_*")
        if not any(e.startswith(f"lm_head.weight_tp{tp}_") for e in entries):
            missing.append(f"lm_head.weight_tp{tp}_*")

    if missing:
        pytest.skip(
            f"Tensor cache at {cache_root} is incomplete for TP={tp} (missing: {', '.join(missing)}). "
            f"Populate it by running any full-weight Gemma4 entry point on this mesh, or rerun "
            f"with GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1 to load weights from the checkpoint "
            f"(and write the cache)."
        )


def _cache_completion_state(model_path):
    """State dict handed to the model: cache-completion keys, or None for a full load."""
    if _load_full_weights():
        logger.info("GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1 — loading the full host state dict")
        return None
    return load_cache_completion_state(model_path)


def _mesh_config(mesh_device):
    tp = mesh_device.shape[1]
    return MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))


# ── Prefill inputs ────────────────────────────────────────────────────────────


def _prompt_tokens(model_path, chunk):
    """Tokenize a real prompt sized for ``chunk``, truncating/padding to exactly it.

    Reuses ``text_demo.load_demo_prompt``, whose long-context sources are already
    in ``models/tt_transformers/demo/context_cache``, so this needs no network.
    Imported lazily: ``text_demo`` applies its own pytest marks at import, which
    would otherwise surface as unknown-mark warnings when collecting this file.
    Returns ``(tokens [1, chunk] int32, tokenizer, prompt_len)``.
    """
    from transformers import AutoTokenizer

    from models.demos.gemma4.demo.text_demo import load_demo_prompt

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = load_demo_prompt(chunk, instruct=True)

    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
    else:
        encoded = tokenizer.encode(prompt, return_tensors="pt").squeeze(0)

    if encoded.shape[0] > chunk:
        encoded = encoded[:chunk]
    prompt_len = int(encoded.shape[0])
    tokens = torch.zeros(chunk, dtype=torch.int32)
    tokens[:prompt_len] = encoded.to(torch.int32)
    logger.info(f"Prompt: {prompt_len} tokens, padded to chunk {chunk}")
    return tokens.reshape(1, chunk), tokenizer, prompt_len


def _host_tensor(mesh_device, torch_tensor, dtype, layout, mesh_config=None, seq_dim=-2):
    """Host-resident ttnn tensor, replicated across the mesh.

    Kept on host (``device=None``) so it can be pushed into the same device buffer
    before every trace replay, matching ``Generator._capture_trace_prefill``.

    With a context-parallel ``mesh_config``, the sequence dimension is sharded
    across the CP axis instead of replicated, so each rank receives only the tokens
    it owns. The scatter is free here — it is just a different mesh mapper at
    staging time, with no collective involved.

    ``seq_dim`` is which axis of ``torch_tensor`` holds the sequence: -2 for 4D
    hidden states ``[1, 1, S, H]``, but **-1** for a 2D token-id tensor ``[1, S]``,
    where -2 is the size-1 batch dim and sharding it would be wrong.
    """
    return ttnn.from_torch(
        torch_tensor,
        device=None,
        dtype=dtype,
        layout=layout,
        mesh_mapper=_cp_or_replicate_mapper(mesh_device, mesh_config, seq_dim=seq_dim),
    )


def _cp_or_replicate_mapper(mesh_device, mesh_config, seq_dim=-2):
    """Shard ``seq_dim`` across the CP axis, or replicate when CP is off."""
    from models.demos.gemma4.tt.ccl import cp_degree

    if mesh_config is not None and cp_degree(mesh_config) > 1:
        shard_dims = (seq_dim, None) if mesh_config.sp_axis == 0 else (None, seq_dim)
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=shard_dims)
    return ttnn.ReplicateTensorToMesh(mesh_device)


def _cp_gather_torch(tensor, mesh_device, mesh_config):
    """Read a CP-sharded mesh tensor back to one torch tensor, in position order.

    The output of a CP prefill is sharded along the sequence axis and replicated
    across TP (the TP all-reduce leaves every column identical), so take one device
    per CP row and concatenate along the sequence. Device tensors come back in the
    mesh's row-major order, so CP row r at column 0 is index ``r * num_cols``.

    Falls back to device 0 alone when CP is off, matching ``_first_device_torch``.
    """
    from models.demos.gemma4.tt.ccl import cp_degree

    shards = ttnn.get_device_tensors(tensor)
    cp = cp_degree(mesh_config) if mesh_config is not None else 1
    if cp <= 1:
        return ttnn.to_torch(shards[0]).float()

    num_cols = mesh_device.shape[1]
    rows = [ttnn.to_torch(shards[r * num_cols]).float() for r in range(cp)]
    return torch.cat(rows, dim=-2)


def _identity_page_table(mesh_device, paged_config, mesh_config=None):
    """Single-user page table mapping virtual block i to physical block i.

    Under context parallelism the block pool is sharded along the CP axis (see
    ``Gemma4Model._cp_block_pool_override``), so each rank owns ``max_num_blocks/cp``
    blocks and addresses them locally, starting at 0. The table therefore just gets
    narrower — it stays a replicated identity, because the per-rank difference is
    carried by *which* tokens a rank holds, not by where it writes them.

    The width also bounds the fill: paged_fill_cache requires the input length to be
    <= ``page_table.shape[1] * block_size``, which here is exactly the local
    sequence length.
    """
    from models.demos.gemma4.tt.ccl import cp_degree

    cp = cp_degree(mesh_config) if mesh_config is not None else 1
    num_blocks = paged_config.max_num_blocks // cp
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    return ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


# ── Eager / traced execution ──────────────────────────────────────────────────


@dataclass
class _RunResult:
    output: object  # ttnn.Tensor produced by the measured pass (or trace replay)
    compile_s: float
    capture_s: float
    measured_s: float


def _run_graph(mesh_device, forward, *, traced, host_input, input_consumed=False):
    """Run ``forward(device_input)`` eagerly or through a captured device trace.

    Both modes go through this one call site so their timings are comparable.

    Eager: one warmup pass (kernel compile), then one measured pass, with the
    input rebuilt for each because several graphs deallocate their own input.

    Traced: a warmup pass first (metal traces cannot compile during capture),
    then capture, a warm replay, and finally a signposted measured replay. The
    input is re-pushed into the persistent device buffer before each replay.

    ``input_consumed=True`` says the graph deallocates its own input (as
    ``Gemma4DecoderLayer`` does with its residual). Eager passes the input
    straight in and lets it be consumed. The traced region instead operates on a
    ``ttnn.clone``: a trace bakes in buffer addresses, so a captured graph that
    freed the persistent input would replay against memory since handed to
    something else. The clone is the throwaway; the persistent buffer survives
    every replay.
    """

    def _eager_pass():
        """One untraced forward; returns ``(output, forward_seconds)``.

        The input allocation sits outside the timer so the eager number is
        comparable to the traced one, whose measured window is replay-only.

        When the graph consumes its input, this mirrors the traced region's
        ``ttnn.clone`` so that clone's program lands in the cache during warmup.
        A trace cannot compile new binaries during capture, and the clone only
        exists on the traced path, so without this the capture aborts with
        "Cannot load new binaries during trace capture".
        """
        device_input = ttnn.to_device(host_input, device=mesh_device)
        graph_input = ttnn.clone(device_input) if input_consumed else device_input
        t0 = time.time()
        output = forward(graph_input)
        ttnn.synchronize_device(mesh_device)
        elapsed = time.time() - t0
        # forward() consumed graph_input; device_input is only aliased when there
        # was no clone, so it always needs releasing here.
        device_input.deallocate(True)
        return output, elapsed

    # ── Warmup: compile every kernel in the graph ──────────────────────────
    t0 = time.time()
    warm_output, _ = _eager_pass()
    if warm_output is not None:
        warm_output.deallocate(True)
    compile_s = time.time() - t0

    if not traced:
        output, measured_s = _eager_pass()
        return _RunResult(output, compile_s, 0.0, measured_s)

    # ── Capture ───────────────────────────────────────────────────────────
    device_input = ttnn.to_device(host_input, device=mesh_device)
    t0 = time.time()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = forward(ttnn.clone(device_input) if input_consumed else device_input)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    capture_s = time.time() - t0

    try:
        # Warm replay — first replay can still pay one-off dispatch setup.
        ttnn.copy_host_to_device_tensor(host_input, device_input)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        # Measured replay, bracketed for tracy's --signpost filter.
        ttnn.copy_host_to_device_tensor(host_input, device_input)
        signpost("start")
        t0 = time.time()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        measured_s = time.time() - t0
        signpost("stop")
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    return _RunResult(trace_output, compile_s, capture_s, measured_s)


@contextmanager
def _lm_head_deferred(model):
    """Make the model's prefill graph stop at the post-norm hidden states.

    ``_prefill_trace_mode`` is really a "skip lm_head" switch, so this is used for
    the eager body-only run too, not just for tracing. See ``Gemma4Model.__call__``
    (models/demos/gemma4/tt/model.py) for why traced prefill defers the head: the
    lm_head over a full padded sequence at 262k vocab is ~40x the model body at
    4k tokens, and the last-token slice index varies per prompt so it cannot be
    baked into a trace.
    """
    previous = getattr(model, "_prefill_trace_mode", False)
    model._prefill_trace_mode = True
    try:
        yield
    finally:
        model._prefill_trace_mode = previous


def _log_run(label, chunk, traced, result, extra=""):
    mode = "traced" if traced else "eager"
    tok_s = chunk / result.measured_s if result.measured_s > 0 else 0.0
    capture = f" capture={result.capture_s:.2f}s" if traced else ""
    logger.info(
        f"[{label}] chunk={chunk} mode={mode} compile={result.compile_s:.2f}s{capture} "
        f"measured={result.measured_s * 1000:.1f}ms ({tok_s:.0f} tok/s){extra}"
    )


def _first_device_torch(tensor):
    """Read device 0's shard of a mesh tensor to torch float32.

    Prefill all-gathers logits inside the model, so device 0 already holds the
    full-width result (same assumption as ``Gemma4Model.process_output_prefill``).
    """
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()


# ── Test 1: a single decoder layer, PCC vs HuggingFace ────────────────────────


def _hf_text_config(model_path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    text_config._attn_implementation = "eager"
    return text_config


@parametrize_mesh_with_fabric([GALAXY_MESH], **_MESH_PARAMS)
@_parametrize_traced
@_parametrize_chunk
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "global"])
def test_prefill_layer(mesh_device, layer_type, chunk, traced, reset_seeds, request):
    """One decoder layer in prefill mode, TT (from cache) vs HuggingFace (from checkpoint).

    The two layer types exercise genuinely different attention paths:
      sliding: head_dim 256, 16 KV heads, full RoPE, 1024-token window mask
      global:  head_dim 512, 4 KV heads (replicated at TP=32), K=V tying, partial RoPE 0.25

    TT weights load from the tensor cache — attention and MLP at bfp8 per
    ``precision_overrides.json`` — while the HF reference gets the same weights at
    bf16 in an fp32 module, so the PCC gap is the deployed quantization error.

    PCC thresholds come from ``pcc_thresholds.json``; there is no ``4x8`` or
    ``1x32`` entry yet, so this falls back to the table's documented 0.99
    "unmeasured" default.

    Measured on a BH Galaxy at ``4x8`` (TP=8 x CP=4), 31B, eager and traced
    identical in every case:

        sliding: 0.99940 (512), 0.99933 (1024), 0.99932 (2048), 0.99931 (4096)
        global:  0.99986 (512), 0.99986 (1024), 0.99986 (2048), 0.99985 (4096)

    Left at the 0.99 default deliberately — every case clears it by ~4e-3, and
    pinning thresholds to measured values invites flakiness on run-to-run drift.
    """
    _guard_chunk(request, chunk)

    model_path = _model_path()
    tp = mesh_device.shape[1]
    text_config = _hf_text_config(model_path)
    layer_idx = find_layer_idx(text_config, layer_type)
    model_args = Gemma4ModelArgs.from_hf_config(text_config)
    model_args._hf_text_config = text_config
    attn_config = Gemma4AttentionConfig(model_args, layer_idx)

    cache_root = _cache_root(model_path)
    # No _require_cache here: this test builds one layer from the checkpoint state it
    # already loads (see layer_state_prefixed below), so it neither needs a warm cache
    # nor the whole-model entries (embed_tokens / lm_head / final_norm) that guard
    # checks for. The multi-layer tests below still require a warm cache.
    precision = Gemma4Precision.load(model_path, tuple(mesh_device.shape))

    # One layer's real weights: the HF reference needs all of them, the TT layer
    # needs only layer_scalar (everything else resolves from the tensor cache).
    layer_state = load_layer_state(model_path, layer_idx)
    logger.info(f"Layer {layer_idx} ({layer_type}): loaded {len(layer_state)} tensors from the checkpoint")

    # ── HuggingFace reference ─────────────────────────────────────────────
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    hf_layer = Gemma4TextDecoderLayer(text_config, layer_idx=layer_idx)
    hf_layer.load_state_dict(layer_state)
    hf_layer.eval()

    x_torch = torch.randn(1, chunk, model_args.hidden_size, dtype=torch.float32)
    attn_mask = build_hf_prefill_mask(chunk, sliding_window=attn_config.sliding_window)
    with torch.no_grad():
        hf_output = hf_layer(
            x_torch,
            per_layer_input=None,
            position_embeddings=TestFactory.create_hf_rope(text_config, chunk, layer_idx),
            attention_mask=attn_mask,
        )
    del hf_layer

    # ── TT layer from the tensor cache ────────────────────────────────────
    assert (
        "layer_scalar" in layer_state
    ), f"layer {layer_idx} has no layer_scalar in the checkpoint; the TT layer would silently fall back to 1.0"
    # Hand the layer its whole checkpoint state, not just layer_scalar. ttnn.as_tensor
    # prefers the cache file when it exists and otherwise builds from these tensors and
    # writes the cache, so this makes the test self-sufficient at any TP: a cache built
    # for a different TP (files are tagged _tp32_ / _tp8_) no longer means a hard skip.
    # We already load layer_state for the HF reference, so this costs no extra I/O.
    layer_state_prefixed = {
        f"model.language_model.layers.{layer_idx}.{key}": value for key, value in layer_state.items()
    }
    mesh_config = _mesh_config(mesh_device)
    tt_layer = Gemma4DecoderLayer(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=layer_state_prefixed,
        layer_idx=layer_idx,
        ccl_manager=CCLManager(mesh_device),
        dtype=MODEL_DTYPE,
        shared_mlp_dtype=precision.get("shared_mlp", MODEL_DTYPE),
        attention_dtype=precision.get("attention", MODEL_DTYPE),
        tensor_cache_path=cache_root,
        mesh_config=mesh_config,
        max_seq_len=chunk,
        max_local_batch_size=1,
    )
    logger.info(f"TT layer {layer_idx} layer_scalar={tt_layer.layer_scalar}")

    cos_tt, sin_tt = TestFactory.create_tt_rope_cache(
        mesh_device, text_config, chunk, layer_idx, mesh_config=mesh_config
    )
    host_input = _host_tensor(
        mesh_device,
        x_torch.unsqueeze(0).to(torch.bfloat16),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mesh_config=mesh_config,
    )

    def forward(hidden_states):
        return tt_layer(
            hidden_states,
            rope_mats=(cos_tt, sin_tt),
            position_idx=None,
            page_table=None,
            kv_cache=None,
            is_decode=False,
        )

    # The layer deallocates its own input (the residual), so the traced region
    # must operate on a clone of the persistent input buffer.
    result = _run_graph(mesh_device, forward, traced=traced, host_input=host_input, input_consumed=True)
    _log_run(f"prefill_layer:{layer_type}", chunk, traced, result)

    tt_output = _cp_gather_torch(result.output, mesh_device, mesh_config).squeeze(0)

    passing, pcc = compare_tensors(tt_output, hf_output, pcc_threshold=get_pcc_threshold(request))
    assert (
        passing
    ), f"Layer {layer_idx} ({layer_type}) prefill PCC too low at chunk={chunk}, tp={tp}, traced={traced}: {pcc}"


# ── Tests 2 and 3: the full 60-layer stack ────────────────────────────────────


def _build_prefill_model(mesh_device, model_path, chunk):
    """Create the full model from cache, sized for a single ``chunk``-token prefill.

    Returns ``(model_args, model, kv_cache, page_table_tt)``.
    """
    tp = mesh_device.shape[1]
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", chunk))
    paged_config = PagedAttentionConfig(
        block_size=PAGE_BLOCK_SIZE,
        max_num_blocks=max(1, max_seq_len // PAGE_BLOCK_SIZE),
    )

    cache_root = _cache_root(model_path)
    hf_config = Gemma4ModelArgs.load_hf_config(model_path)
    num_layers = Gemma4ModelArgs.from_hf_config(hf_config).num_hidden_layers
    _require_cache(cache_root, tp, num_layers)

    logger.info(f"Creating Gemma4 ({num_layers} layers, TP={tp}, max_seq_len={max_seq_len})...")
    t0 = time.time()
    model_args, model, kv_cache, _state_dict = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        dtype=MODEL_DTYPE,
        state_dict=_cache_completion_state(model_path),
        model_path=model_path,
        create_kv_cache=True,
        paged_attention_config=paged_config,
    )
    logger.info(f"Model ready in {time.time() - t0:.1f}s")

    return model_args, model, kv_cache, _identity_page_table(mesh_device, paged_config, _mesh_config(mesh_device))


def _prefill_body_forward(model, page_table_tt, kv_cache, get_last_token):
    """Build the prefill forward callable: tokens -> embed -> layers -> (norm, head).

    Tracing starts from *token ids*, not embeddings, because the model deallocates
    intermediate hidden states as it walks layers and would free an embeddings
    input buffer mid-capture. ``ttnn.embedding`` consumes the token tensor without
    freeing it, so the persistent input survives every replay — this is also the
    deployed traced-prefill path (``Generator._capture_trace_prefill`` calls
    ``transform_and_embed_prefill_inputs_device`` inside the capture).
    """

    def forward(tokens_device):
        embeds, page_table, chunk_page_table, chunk_start_idx = model.transform_and_embed_prefill_inputs_device(
            tokens_device, page_table_tt, None, None
        )
        return model.ttnn_prefill_forward(
            x=embeds,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
            get_last_token=get_last_token,
            user_id=0,
        )

    return forward


@parametrize_mesh_with_fabric([GALAXY_MESH], **_MESH_PARAMS)
@_parametrize_traced
@_parametrize_chunk
def test_prefill_layers(mesh_device, chunk, traced, reset_seeds, request):
    """All 60 decoder layers + final norm in prefill mode, lm_head skipped.

    This is the graph that dominates TTFT and the one traced prefill captures:
    embed -> 60x (attention with paged KV fill + SDPA, TP all-reduce, dense MLP,
    TP all-reduce) -> final norm. Excluding lm_head keeps the output at
    ``[1, 1, chunk, hidden]`` instead of ``chunk x 262144`` logits, which is what
    makes 4k prefill fit at all.

    Smoke-level checks only — there is no host-side 60-layer reference that fits
    in RAM. Asserts shape, finiteness, and a non-degenerate spread, and logs
    timings so eager and traced are directly comparable at the same chunk size.
    """
    _guard_chunk(request, chunk)
    _guard_cp_rope_alignment(mesh_device, chunk)

    model_path = _model_path()
    mesh_config = _mesh_config(mesh_device)
    model_args, model, kv_cache, page_table_tt = _build_prefill_model(mesh_device, model_path, chunk)

    tokens, _tokenizer, prompt_len = _prompt_tokens(model_path, chunk)
    # Token ids are [1, chunk], so the sequence axis is -1, not -2.
    host_input = _host_tensor(
        mesh_device, tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_config=mesh_config, seq_dim=-1
    )

    # get_last_token=-1: no host-side last-token slice, which is required inside a
    # trace and keeps eager identical to the captured graph.
    forward = _prefill_body_forward(model, page_table_tt, kv_cache, get_last_token=-1)

    with _lm_head_deferred(model):
        result = _run_graph(mesh_device, forward, traced=traced, host_input=host_input)
    _log_run("prefill_layers", chunk, traced, result, extra=f" layers={len(model.layers)}")

    hidden = _cp_gather_torch(result.output, mesh_device, mesh_config)

    assert len(model.layers) == model_args.num_hidden_layers, (
        f"built {len(model.layers)} decoder layers but the config declares "
        f"{model_args.num_hidden_layers} — the stack is truncated"
    )
    if model_args.hidden_size == 5376:  # 31B
        assert model_args.num_hidden_layers == _EXPECTED_31B_LAYERS, (
            f"31B should have {_EXPECTED_31B_LAYERS} decoder layers, config says " f"{model_args.num_hidden_layers}"
        )
    assert tuple(hidden.shape) == (
        1,
        1,
        chunk,
        model_args.hidden_size,
    ), f"prefill body returned {tuple(hidden.shape)}, expected (1, 1, {chunk}, {model_args.hidden_size})"
    assert torch.isfinite(hidden).all(), "prefill body produced non-finite hidden states"

    real = hidden[0, 0, :prompt_len, :]
    assert real.std() > 0, "prefill body hidden states are constant over the real prompt"
    logger.info(
        f"[prefill_layers] hidden states over {prompt_len} real tokens: "
        f"std={real.std():.4f} absmax={real.abs().max():.4f}"
    )


@parametrize_mesh_with_fabric([GALAXY_MESH], **_MESH_PARAMS)
@_parametrize_traced
@_parametrize_chunk
def test_prefill_full(mesh_device, chunk, traced, reset_seeds, request):
    """The whole prefill graph, through lm_head and logit softcapping.

    Adds the last-token slice, the 262k-vocab lm_head, softcapping and the TP
    all-gather on top of ``test_prefill_layers``.

    Eager runs the body and the 32-row head in one measured window. Traced runs
    the body from the captured trace and the head eagerly afterwards, via
    ``process_logits_after_prefill_trace`` — that split is the deployed path, not
    a shortcut (see ``_lm_head_deferred``). Both slice the same 32-row tile, so
    the reported numbers stay comparable.
    """
    _guard_chunk(request, chunk)
    _guard_cp_rope_alignment(mesh_device, chunk)

    model_path = _model_path()
    mesh_config = _mesh_config(mesh_device)
    model_args, model, kv_cache, page_table_tt = _build_prefill_model(mesh_device, model_path, chunk)

    tokens, tokenizer, prompt_len = _prompt_tokens(model_path, chunk)
    # Token ids are [1, chunk], so the sequence axis is -1, not -2.
    host_input = _host_tensor(
        mesh_device, tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_config=mesh_config, seq_dim=-1
    )

    last_token_idx = prompt_len - 1
    tile_start = (last_token_idx // 32) * 32
    row_in_tile = last_token_idx - tile_start

    if traced:
        forward = _prefill_body_forward(model, page_table_tt, kv_cache, get_last_token=-1)
        with _lm_head_deferred(model):
            result = _run_graph(mesh_device, forward, traced=True, host_input=host_input)
        t0 = time.time()
        logits_tt = model.process_logits_after_prefill_trace(result.output, last_token_idx)
        ttnn.synchronize_device(mesh_device)
        head_s = time.time() - t0
        _log_run("prefill_full", chunk, traced, result, extra=f" lm_head={head_s * 1000:.1f}ms (out of trace)")
    else:
        forward = _prefill_body_forward(model, page_table_tt, kv_cache, get_last_token=tile_start)
        result = _run_graph(mesh_device, forward, traced=False, host_input=host_input)
        logits_tt = result.output
        _log_run("prefill_full", chunk, traced, result, extra=" lm_head=in-window")

    logits = _first_device_torch(logits_tt)
    assert torch.isfinite(logits).all(), "prefill produced non-finite logits"
    assert logits.shape[-1] >= model_args.vocab_size, (
        f"logits width {logits.shape[-1]} < vocab_size {model_args.vocab_size} — "
        f"the TP all-gather did not reconstruct the full vocab"
    )

    next_token_logits = logits.reshape(-1, logits.shape[-1])[row_in_tile, : model_args.vocab_size]

    cap = model_args.final_logit_softcapping
    if cap and cap > 0:
        assert next_token_logits.abs().max() <= cap + 1e-2, (
            f"logits exceed the {cap} softcap (max |logit| = "
            f"{next_token_logits.abs().max():.4f}) — softcapping was not applied"
        )

    top = torch.topk(next_token_logits, k=5)
    logger.info(
        f"[prefill_full] chunk={chunk} top-5 next tokens: "
        + ", ".join(f"{tokenizer.decode([int(i)])!r}({v:.3f})" for v, i in zip(top.values, top.indices))
    )
    next_token = int(top.indices[0])
    assert tokenizer.decode([next_token]) != "", f"argmax token {next_token} decodes to an empty string"


# ── Test 4: the 60-layer graph against a host reference, by PCC ────────────────


@parametrize_mesh_with_fabric([GALAXY_MESH], **_MESH_PARAMS)
@_parametrize_traced
def test_prefill_layers_vs_cpu_reference(mesh_device, traced, reset_seeds, request):
    """The whole 60-layer prefill body vs a CPU reference, by PCC.

    ``test_prefill_layers`` can only assert shape / finiteness / spread because
    there is no host reference cheap enough to build inline — which makes it a
    smoke test. A context-parallel bug that shifted token positions would still
    produce finite, plausible output and pass it. This closes that gap: it compares
    against a full HuggingFace fp32 run cached on disk, so the 60-layer stack is
    checked the same way ``test_prefill_layer`` checks one layer.

    4096 tokens only — the largest trace-eligible chunk and the CP target. Skips
    with the generate command when the dump is absent, since building it needs a few
    minutes and ~130 GB of RAM.

    On the expected PCC (~0.94, not ~0.999): that is compounding, not a defect, and
    it is *not* caused by context parallelism. Measured on a BH Galaxy, 31B, 4k:

        CP=4, bfp8 weights (product default) ... 0.9398
        CP=4, bf16 weights ..................... 0.9401   (weight dtype is not it)
        CP=1, bfp8 weights ..................... 0.9389   (CP is not it either)
        CPU bf16 vs CPU fp32, no device ........ 0.9889   (dtype alone costs ~0.011)

    CP costs ~0.001. The number follows from the single-layer figure: one layer is
    0.9993 against the same reference setup, i.e. a per-layer relative error of
    ~0.037, and 0.037*sqrt(60) ~ 0.29 predicts ~0.958 — close to the 0.939 measured,
    with the remainder from residual paths correlating the per-layer errors. The
    per-layer error is dominated by bf16 activations and kernel differences rather
    than by weight quantization, which is why bfp8 -> bf16 changes nothing.

    So the threshold is a regression guard set just under the measured value, not an
    accuracy target. If it drops materially, suspect a real change; if it rises,
    tighten it. ``GEMMA4_DISABLE_CP=1`` reproduces the CP=1 row.

    The next-token logits agree far better than the hidden states — PCC 0.9892, with
    the top-3 tokens identical and in order — because lm_head projects onto the
    directions that carry the prediction while the hidden-state error sits mostly in
    low-magnitude channels. That is why this test also asserts argmax agreement:
    it is the stricter statement about behaviour, and the one that would actually
    break if positions were wrong.
    """
    from models.demos.gemma4.tests import cpu_prefill_reference as cpu_ref

    chunk = cpu_ref.REFERENCE_CHUNK
    _guard_chunk(request, chunk)
    _guard_cp_rope_alignment(mesh_device, chunk)

    model_path = _model_path()
    reference = cpu_ref.load(model_path, chunk)
    if reference is None:
        pytest.skip(
            f"No CPU reference at {cpu_ref.reference_path(model_path, chunk)}. Generate it once with:\n"
            f"  python -m models.demos.gemma4.tests.cpu_prefill_reference\n"
            f"(tens of minutes, ~130 GB RAM; only needed once per model)"
        )

    mesh_config = _mesh_config(mesh_device)
    model_args, model, kv_cache, page_table_tt = _build_prefill_model(mesh_device, model_path, chunk)

    # Reuse the reference's own tokens rather than re-deriving them, and verify the
    # fingerprint: comparing against a dump built from different input would be
    # worse than not comparing at all.
    tokens = reference["tokens"]
    expected_sha = reference["fingerprint"]["token_sha"]
    actual_sha = cpu_ref.hash_tokens(tokens)
    assert actual_sha == expected_sha, f"reference dump is corrupt: token sha {actual_sha} != {expected_sha}"

    fresh_tokens, _tokenizer, _prompt_len = _prompt_tokens(model_path, chunk)
    if cpu_ref.hash_tokens(fresh_tokens) != expected_sha:
        pytest.skip(
            f"CPU reference is stale: the harness now tokenizes to "
            f"{cpu_ref.hash_tokens(fresh_tokens)} but the dump was built from {expected_sha}. "
            f"Regenerate with: python -m models.demos.gemma4.tests.cpu_prefill_reference"
        )

    host_input = _host_tensor(
        mesh_device, tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_config=mesh_config, seq_dim=-1
    )
    forward = _prefill_body_forward(model, page_table_tt, kv_cache, get_last_token=-1)
    with _lm_head_deferred(model):
        result = _run_graph(mesh_device, forward, traced=traced, host_input=host_input)
    _log_run("prefill_layers_vs_cpu", chunk, traced, result, extra=f" layers={len(model.layers)}")

    tt_hidden = _cp_gather_torch(result.output, mesh_device, mesh_config)
    ref_hidden = reference["hidden"].reshape(1, 1, chunk, model_args.hidden_size)
    assert tuple(tt_hidden.shape) == tuple(
        ref_hidden.shape
    ), f"device gave {tuple(tt_hidden.shape)}, reference is {tuple(ref_hidden.shape)}"

    # Compare only the real prompt rows: positions past prompt_len are padding, and
    # padded rows carry no signal to agree on.
    real = reference["prompt_len"]
    passing, pcc = compare_tensors(
        tt_hidden[:, :, :real, :], ref_hidden[:, :, :real, :], pcc_threshold=get_pcc_threshold(request, default=0.93)
    )
    logger.info(
        f"[prefill_layers_vs_cpu] chunk={chunk} mode={'traced' if traced else 'eager'} "
        f"60-layer hidden PCC over {real} real tokens = {pcc}"
    )
    assert passing, f"60-layer prefill body PCC {pcc} below threshold over {real} real tokens"

    # Then the next-token distribution, which is what actually matters downstream and
    # is far more discriminating than hidden-state PCC: accumulated numerical error
    # spreads across 5376 channels, but the argmax either survives it or does not.
    logits_tt = model.process_logits_after_prefill_trace(result.output, reference["last_token_idx"])
    logits = _first_device_torch(logits_tt)[..., : model_args.vocab_size]
    ref_logits = reference["logits_tile"][..., : model_args.vocab_size]
    row = reference["last_token_idx"] - reference["tile_start"]

    tt_row = logits.reshape(-1, logits.shape[-1])[row]
    ref_row = ref_logits.reshape(-1, ref_logits.shape[-1])[row]
    _, logits_pcc = compare_tensors(tt_row.unsqueeze(0), ref_row.unsqueeze(0), pcc_threshold=0.0)

    tt_top = torch.topk(tt_row, k=5)
    ref_top = torch.topk(ref_row, k=5)
    logger.info(
        f"[prefill_layers_vs_cpu] next-token logits PCC = {logits_pcc}\n"
        f"    device top-5 idx={tt_top.indices.tolist()} val={[round(v, 3) for v in tt_top.values.tolist()]}\n"
        f"    cpu-ref top-5 idx={ref_top.indices.tolist()} val={[round(v, 3) for v in ref_top.values.tolist()]}"
    )
    assert int(tt_top.indices[0]) == int(ref_top.indices[0]), (
        f"argmax next token disagrees with the CPU reference: device {int(tt_top.indices[0])} "
        f"vs reference {int(ref_top.indices[0])}"
    )


# ── Test 5: the KV cache actually holds the whole sequence ─────────────────────


@parametrize_mesh_with_fabric([GALAXY_MESH], **_MESH_PARAMS)
def test_prefill_kv_cache_covers_sequence(mesh_device, reset_seeds, request):
    """The paged KV cache holds every prefilled token, in the right place.

    Nothing else reads the cache during CP prefill — the CP attention path uses the
    in-memory K/V — so a broken fill is invisible to every other test here. It was
    broken: the write destination is ``page_table[local_block]``, and the page table
    was uploaded replicated, so all four CP ranks resolved virtual block 0 to the
    same physical block and each wrote its own 1024 tokens at global positions
    0..1023. Three quarters of the cache stayed zero and the four shards were
    identical.

    The fix shards the block pool along the CP axis, so a rank's pool *is* its shard
    and local addressing is correct by construction. This test is the reader that
    proves it, via ``export_paged_kv_cache_natural_order`` — which is also the
    function a disaggregated decode target would use to ingest the cache.

    Checks, on a full-attention layer (unbounded pool, so cache position == token
    position) at 4k:
      1. every real prompt token has a non-zero cache row — catches the zero tail
      2. the CP shards differ from one another — catches the replicated-write bug,
         which produced cp identical copies
    """
    from models.demos.gemma4.tt.attention.kv_cache import export_paged_kv_cache_natural_order
    from models.demos.gemma4.tt.ccl import cp_degree

    chunk = 4096
    _guard_chunk(request, chunk)
    _guard_cp_rope_alignment(mesh_device, chunk)

    model_path = _model_path()
    mesh_config = _mesh_config(mesh_device)
    cp = cp_degree(mesh_config)
    model_args, model, kv_cache, page_table_tt = _build_prefill_model(mesh_device, model_path, chunk)

    tokens, _tokenizer, prompt_len = _prompt_tokens(model_path, chunk)
    host_input = _host_tensor(
        mesh_device, tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_config=mesh_config, seq_dim=-1
    )
    forward = _prefill_body_forward(model, page_table_tt, kv_cache, get_last_token=-1)
    with _lm_head_deferred(model):
        _run_graph(mesh_device, forward, traced=False, host_input=host_input)

    # A full-attention layer: its pool is unbounded, so cache row == token position.
    # Sliding layers under a bounded pool wrap, which is a different contract.
    layer_idx = next(i for i, lyr in enumerate(model.layers) if not lyr.self_attn.config.is_sliding)
    k_cache, _v_cache = model.layers[layer_idx].self_attn.kv_cache
    k_nat = export_paged_kv_cache_natural_order(k_cache, mesh_device, mesh_config, PAGE_BLOCK_SIZE)
    logger.info(f"[kv_cache] layer {layer_idx} exported {tuple(k_nat.shape)} (tp, tokens, kv_heads, head_dim), CP={cp}")

    assert k_nat.shape[1] >= prompt_len, (
        f"exported cache covers only {k_nat.shape[1]} tokens but the prompt is {prompt_len}; "
        f"the pool is too small or the CP shards were not concatenated"
    )

    # 1. Coverage: no real token may have an all-zero cache row. Before the fix,
    # everything past tokens_per_rank was untouched zeros.
    col0 = k_nat[0, :prompt_len]  # [tokens, kv_local, head_dim]
    per_token_absmax = col0.abs().amax(dim=(-1, -2))
    zero_tokens = int((per_token_absmax == 0).sum())
    first_zero = int((per_token_absmax == 0).nonzero()[0]) if zero_tokens else -1
    logger.info(f"[kv_cache] zero-valued token rows within the prompt: {zero_tokens} (first at {first_zero})")
    assert zero_tokens == 0, (
        f"{zero_tokens} of {prompt_len} prompt tokens have an all-zero KV row (first at index {first_zero}) — "
        f"the fill did not cover the whole sequence"
    )

    # 2. Distinctness: with a replicated page table every rank wrote the same tokens,
    # so the shards were byte-identical. They must differ now.
    if cp > 1:
        tokens_per_rank = k_nat.shape[1] // cp
        shard0 = k_nat[0, :tokens_per_rank]
        for r in range(1, cp):
            other = k_nat[0, r * tokens_per_rank : (r + 1) * tokens_per_rank]
            assert not torch.equal(shard0, other), (
                f"CP shard {r} is identical to shard 0 — every rank wrote the same tokens, "
                f"which is the replicated-page-table bug"
            )
        logger.info(f"[kv_cache] all {cp} CP shards differ, as expected")
