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


def cpu_ref_contexts():
    """Context lengths the CPU-reference PCC test is parametrized over.

    Every entry is a device target with a reachable reference. Cases whose reference
    has not been generated yet skip with the command to generate it, so the list can
    lead the files on disk.
    """
    from models.demos.gemma4.tests.cpu_prefill_reference import LONG_REFERENCE_CONTEXTS

    return LONG_REFERENCE_CONTEXTS


def _per_token_pcc(actual, expected):
    """PCC per token row, contracting over hidden only.

    Whole-tensor PCC pools every token together, so a component shared across the
    sequence carries the score and token order barely registers. Correlating each row
    against the reference row at the same position removes that: a row that ends up in
    the wrong place is scored against a different token's activations.
    """
    a = actual.reshape(-1, actual.shape[-1]).double()
    b = expected.reshape(-1, expected.shape[-1]).double()
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    denom = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
    return (a * b).sum(dim=-1) / denom


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


def _build_prefill_model(mesh_device, model_path, chunk, context_len=None):
    """Create the full model from cache, sized for a ``context_len``-token prefill.

    ``context_len`` defaults to a single ``chunk``. When larger, prefill runs as
    ``context_len / chunk`` chunks and the model is told the chunk size so it can lay
    the RoPE cache out chunk-major per CP rank and size the ring KV cache slabs.

    Returns ``(model_args, model, kv_cache, page_table_tt)``.
    """
    tp = mesh_device.shape[1]
    context_len = context_len or chunk
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", context_len))
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
        prefill_chunk_size=chunk if context_len > chunk else None,
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


# ── Test 6: long-context prefill as a chunk sequence under CP ─────────────────

# Context lengths to walk, each prefilled as context/4096 chunks of 4096. 4096 is
# the chunk the CP ring path supports: the 1024-token sliding window rounds to a
# 32-tile halo, which exactly fills the 1024-token per-rank slab at CP=4. Smaller
# chunks would need a multi-hop halo, which the op rejects.
LONG_CONTEXT_LENGTHS = [32768, 65536, 131072, 262144]
LONG_CONTEXT_CHUNK = 4096

# Chunks that pay a one-time program compile rather than steady-state cost: chunk 0 is
# the first run of the mask CP path, chunk 1 the first run of the ring path. Both are
# tens of seconds against a sub-second steady state, so every average excludes them.
_N_WARMUP_CHUNKS = 2


@parametrize_mesh_with_fabric([GALAXY_MESH], **_MESH_PARAMS)
@pytest.mark.parametrize("context_len", LONG_CONTEXT_LENGTHS, ids=[f"ctx_{c // 1024}k" for c in LONG_CONTEXT_LENGTHS])
def test_prefill_long_context_chunked(mesh_device, context_len, reset_seeds, request):
    """Prefill ``context_len`` tokens as a sequence of 4096-token chunks under CP.

    This is the disaggregated-prefill target: the KV cache stays CP-sharded (no
    gather for the fill), and chunks after the first read history back through
    ring_joint SDPA, which gathers the prefix around the CP ring internally.

    Chunk 0 goes through the mask-based CP path — the chunked ring mode needs a
    complete predecessor Q group and chunk 0 has none — but still writes the ring
    cache so chunk 1 has history to read.

    Smoke-level assertions: there is no host reference at these lengths (the CPU
    reference tops out around 4k). Each chunk's output must be finite and
    non-degenerate, and the last chunk's statistics must be in the same range as the
    first, which is what breaks if the ring read drifts as the prefix grows.
    """
    from models.demos.gemma4.tt.attention import ring_prefill
    from models.demos.gemma4.tt.ccl import cp_degree

    chunk = LONG_CONTEXT_CHUNK
    _guard_chunk(request, chunk)
    mesh_config = _mesh_config(mesh_device)
    cp = cp_degree(mesh_config)
    if cp <= 1:
        pytest.skip(f"long-context chunked prefill targets CP>1; mesh {tuple(mesh_device.shape)} gives CP={cp}")
    assert context_len % chunk == 0, f"context {context_len} must be a multiple of chunk {chunk}"

    model_path = _model_path()
    n_chunks = context_len // chunk
    model_args, model, kv_cache, page_table_tt = _build_prefill_model(
        mesh_device, model_path, chunk, context_len=context_len
    )
    logger.info(f"[long_ctx] {context_len} tokens = {n_chunks} x {chunk}, CP={cp}, layers={len(model.layers)}")

    # Real prompt text for the first chunk; the remainder is deterministic filler.
    # Content does not matter for these assertions, position handling does.
    tokens_first, _tokenizer, _prompt_len = _prompt_tokens(model_path, chunk)
    torch.manual_seed(1234)

    ring_prefill.reset_ring_attention_calls()
    stats = []
    # Device time per chunk, excluding the host readback below. The readback exists
    # only so the test can assert on the output; counting it would understate the
    # model by a wide margin at long contexts, where gathering 4096x5376 back to host
    # each chunk is real time that a production prefill never pays.
    chunk_device_s = []
    t_start = time.time()
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk
        tokens = (
            tokens_first if chunk_idx == 0 else torch.randint(0, model_args.vocab_size, (1, chunk), dtype=torch.int32)
        )
        host_input = _host_tensor(
            mesh_device, tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_config=mesh_config, seq_dim=-1
        )
        device_input = ttnn.to_device(host_input, device=mesh_device)

        t_chunk = time.time()
        with _lm_head_deferred(model):
            embeds, page_table, chunk_page_table, _ = model.transform_and_embed_prefill_inputs_device(
                device_input, page_table_tt, None, None
            )
            out = model.ttnn_prefill_forward(
                x=embeds,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start,
                kv_cache=kv_cache,
                get_last_token=-1,
                user_id=0,
            )
        # ttnn dispatch is async, so the elapsed time is meaningless until the device
        # has actually finished; this sync is what makes the number a measurement.
        ttnn.synchronize_device(mesh_device)
        device_s = time.time() - t_chunk
        chunk_device_s.append(device_s)

        # Per-chunk perf. Cost per chunk RISES with ring depth: chunk k attends over k
        # preceding chunks, so the full-attention layers do more work each time while
        # the sliding layers stay flat. A flat curve at large depth would suggest the
        # history read is not happening.
        #
        # The running figure deliberately excludes the warmup chunks. A cumulative
        # average that includes them climbs monotonically for the whole run — 128 to
        # 2028 tok/s at 256k — as the fixed ~67 s compile amortizes over more tokens.
        # That reads as the model getting faster with context, which is the opposite of
        # the real trend in the per-chunk column right next to it.
        done = chunk_start + chunk
        if chunk_idx < _N_WARMUP_CHUNKS:
            running = "COMPILE (mask CP path), excluded from averages"
            if chunk_idx == 1:
                running = "COMPILE (ring path), excluded from averages"
        else:
            steady_so_far = chunk_device_s[_N_WARMUP_CHUNKS:]
            running = (
                f"steady avg {len(steady_so_far) * chunk / sum(steady_so_far):.0f} tok/s "
                f"over {len(steady_so_far)} chunk(s)"
            )
        logger.info(
            f"[long_ctx_perf] chunk {chunk_idx + 1}/{n_chunks} [{chunk_start}, {done}) "
            f"device={device_s * 1000:.0f}ms ({chunk / device_s:.0f} tok/s) | {running} | "
            f"ring_depth={chunk_idx}"
        )

        hidden = _cp_gather_torch(out, mesh_device, mesh_config)
        out.deallocate(True)
        finite = bool(torch.isfinite(hidden).all())
        std = float(hidden.std())
        absmax = float(hidden.abs().max())
        stats.append((chunk_idx, finite, std, absmax))
        logger.info(
            f"[long_ctx] chunk {chunk_idx + 1}/{n_chunks} [{chunk_start}, {chunk_start + chunk}) "
            f"finite={finite} std={std:.4f} absmax={absmax:.2f}"
        )
        assert finite, f"chunk {chunk_idx} at [{chunk_start}, {chunk_start + chunk}) produced non-finite output"
        assert std > 0.01, f"chunk {chunk_idx} output is degenerate (std={std})"

    elapsed = time.time() - t_start
    device_total = sum(chunk_device_s)

    # The first two chunks each pay a one-time program compile: chunk 0 is the first
    # run of the mask CP path, chunk 1 the first run of the ring path. They are tens of
    # seconds against ~1.4 s steady state, so averaging them in would misreport
    # throughput by more than 2x and would make a last/first ratio look like a 30x
    # speedup rather than the cost curve it is meant to show.
    n_warmup = min(_N_WARMUP_CHUNKS, len(chunk_device_s))
    steady = chunk_device_s[n_warmup:] or chunk_device_s
    steady_tokens = len(steady) * chunk
    steady_s = sum(steady)
    logger.info(
        f"[long_ctx_perf] TOTAL {context_len} tokens in {device_total:.1f}s device "
        f"({context_len / device_total:.0f} tok/s) | wall {elapsed:.1f}s incl. host readback "
        f"({context_len / elapsed:.0f} tok/s)"
    )
    logger.info(
        f"[long_ctx_perf] warmup (compile): "
        f"{', '.join(f'chunk{i}={chunk_device_s[i]:.1f}s' for i in range(n_warmup))} — excluded below"
    )
    logger.info(
        f"[long_ctx_perf] STEADY STATE over {len(steady)} chunks: {steady_tokens} tokens in "
        f"{steady_s:.1f}s ({steady_tokens / steady_s:.0f} tok/s) | per-chunk "
        f"mean={steady_s / len(steady) * 1000:.0f}ms min={min(steady) * 1000:.0f}ms "
        f"max={max(steady) * 1000:.0f}ms"
    )
    # Cost per chunk should grow with ring depth, but only through the 10
    # full-attention layers — the 50 sliding layers are flat in context length — so the
    # growth is expected to be mild rather than linear in depth.
    if len(steady) > 1:
        logger.info(
            f"[long_ctx_perf] ring-depth cost: first steady chunk (depth {n_warmup}) "
            f"{steady[0] * 1000:.0f}ms -> last (depth {n_chunks - 1}) {steady[-1] * 1000:.0f}ms "
            f"= {steady[-1] / steady[0]:.2f}x over {n_chunks - 1 - n_warmup} extra chunks of history"
        )

    # The ring read must not drift as the prefix grows: a halo pointing at the wrong
    # predecessor slab, or a cache offset walking off, shows up as the last chunk's
    # scale departing from the first's rather than as an outright failure.
    # Every chunk after the first must have read history through the ring, on every
    # layer. A silent fallback to the mask path would leave the assertions above
    # intact while each chunk attended only within itself.
    expected_ring_calls = (n_chunks - 1) * len(model.layers)
    actual_ring_calls = ring_prefill.ring_attention_calls()
    logger.info(f"[long_ctx] ring history reads: {actual_ring_calls} (expected {expected_ring_calls})")
    assert actual_ring_calls == expected_ring_calls, (
        f"ring attention ran {actual_ring_calls} times, expected {expected_ring_calls} "
        f"(({n_chunks} - 1) chunks x {len(model.layers)} layers) — history was not read through the ring"
    )

    first_std, last_std = stats[0][2], stats[-1][2]
    assert last_std / first_std < 5.0 and first_std / last_std < 5.0, (
        f"output scale drifted across chunks: first std={first_std:.4f}, last std={last_std:.4f} — "
        f"suspect the ring history read"
    )


# ── Test 7: long-context chunked prefill against a CPU reference, by PCC ──────


@parametrize_mesh_with_fabric([GALAXY_MESH], **_MESH_PARAMS)
@pytest.mark.parametrize(
    "context_len",
    cpu_ref_contexts(),
    ids=[f"ctx_{c // 1024}k" for c in cpu_ref_contexts()],
)
def test_prefill_long_context_vs_cpu_reference(mesh_device, context_len, reset_seeds, request):
    """Chunked CP prefill against a whole-sequence CPU reference, chunk by chunk.

    ``test_prefill_long_context_chunked`` can only assert finiteness, spread and a
    ring-read count. This is the numerical check: each chunk's output is compared to
    the corresponding slice of a flat CPU forward over the same tokens, so a chunk
    that read the wrong history — or none — shows up as a PCC collapse on that chunk
    rather than as plausible-looking output.

    Chunk 0 has no history, so its PCC is the single-chunk baseline. Chunks after it
    exercise the ring read, and their PCC is the evidence that it works.

    Capped at 32768: the reference forward grows quadratically in the full-attention
    layers, so 256k extrapolates to ~51 h with an ~8.8 TB eager attention matrix per
    layer. The larger device targets rest on the ring-read count plus the op-level
    PCC of 0.99975 from the ring_joint probe.
    """
    from models.demos.gemma4.tests import cpu_prefill_reference as cpu_ref
    from models.demos.gemma4.tt.attention import ring_prefill
    from models.demos.gemma4.tt.ccl import cp_degree

    chunk = LONG_CONTEXT_CHUNK
    _guard_chunk(request, chunk)
    mesh_config = _mesh_config(mesh_device)
    cp = cp_degree(mesh_config)
    if cp <= 1:
        pytest.skip(f"targets CP>1; mesh {tuple(mesh_device.shape)} gives CP={cp}")

    model_path = _model_path()
    reference = cpu_ref.load_long(model_path, context_len)
    if reference is None:
        pytest.skip(
            f"No CPU reference at {cpu_ref.long_reference_path(model_path, context_len)}. Generate with:\n"
            f"  GEMMA4_CPU_REF_CONTEXT={context_len} python -m models.demos.gemma4.tests.cpu_prefill_reference"
        )

    tokens_all = reference["tokens"]
    ref_hidden = reference["hidden"]  # [1, context_len, hidden]
    n_chunks = context_len // chunk
    model_args, model, kv_cache, page_table_tt = _build_prefill_model(
        mesh_device, model_path, chunk, context_len=context_len
    )

    # Same tokens the reference used, so a drift in tokenization cannot be mistaken
    # for a numerical problem.
    fresh, _tok, _plen = cpu_ref.build_token_sequence(model_path, chunk, context_len, model_args.vocab_size)
    if cpu_ref.hash_tokens(fresh) != cpu_ref.hash_tokens(tokens_all):
        pytest.skip("CPU reference is stale — token sequence changed; regenerate it")

    ring_prefill.reset_ring_attention_calls()
    threshold = get_pcc_threshold(request, default=0.93)
    pccs = []
    tok_mins = []
    bad_fracs = []
    decoys = []
    decoy_bads = []
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk
        tokens = tokens_all[:, chunk_start : chunk_start + chunk].contiguous()
        host_input = _host_tensor(
            mesh_device, tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_config=mesh_config, seq_dim=-1
        )
        device_input = ttnn.to_device(host_input, device=mesh_device)
        with _lm_head_deferred(model):
            embeds, page_table, chunk_page_table, _ = model.transform_and_embed_prefill_inputs_device(
                device_input, page_table_tt, None, None
            )
            out = model.ttnn_prefill_forward(
                x=embeds,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start,
                kv_cache=kv_cache,
                get_last_token=-1,
                user_id=0,
            )
        ttnn.synchronize_device(mesh_device)

        tt_hidden = _cp_gather_torch(out, mesh_device, mesh_config)
        out.deallocate(True)
        ref_slice = ref_hidden[:, chunk_start : chunk_start + chunk, :].reshape(1, 1, chunk, model_args.hidden_size)
        _passing, pcc = compare_tensors(tt_hidden, ref_slice, pcc_threshold=0.0)
        pccs.append(float(pcc))

        # Whole-tensor PCC is dominated by structure shared across every token, so it
        # is blunt about token ORDER: on the 8k reference, a chunk against its own
        # reversal still scores 0.907. That is the failure this layout could plausibly
        # produce (chunk-major CP maps local row chunk*L+j to global chunk*C+r*L+j, so
        # an off-by-one in the permutation misorders rows without losing content).
        # Per-token PCC contracts over hidden only, so a misplaced row scores as the
        # wrong token: adjacent tokens sit at 0.77 and distant ones at 0.06.
        tok_pcc = _per_token_pcc(tt_hidden, ref_slice)
        tok_mins.append(float(tok_pcc.min()))
        if chunk_idx == n_chunks - 1:
            last_token_pcc = float(tok_pcc[-1])

        # Where the bad rows are, not just how bad the worst one is. A systematic
        # misordering drags most rows down; a handful of stragglers means something
        # local. Reported against prompt_len because chunk 0 carries the real prompt
        # only up to that point and the remainder is filler, which the device and the
        # reference could plausibly treat differently.
        bad = (tok_pcc < 0.80).nonzero().flatten()
        bad_fracs.append(bad.numel() / float(chunk))
        if bad.numel():
            plen = reference.get("prompt_len")
            positions = (bad + chunk_start).tolist()
            tail = f", prompt_len={plen}, bad beyond prompt_len={sum(p >= plen for p in positions)}" if plen else ""
            logger.info(
                f"[long_ctx_pcc] ctx={context_len} chunk {chunk_idx} has {bad.numel()}/{chunk} rows "
                f"below 0.80 ({100.0 * bad.numel() / chunk:.2f}%){tail}; "
                f"first={positions[:8]} last={positions[-8:]}"
            )

        # In-run discrimination control: the same output against a DIFFERENT chunk's
        # reference. Without it a high PCC is unreadable — it could mean the model is
        # right, or that any two slices of this reference look alike. They do not
        # (chunk0 vs chunk1 is 0.112), and this asserts that in-run rather than
        # relying on the offline spot check.
        if chunk_idx > 0:
            decoy_start = (chunk_idx - 1) * chunk
            decoy = ref_hidden[:, decoy_start : decoy_start + chunk, :].reshape(1, 1, chunk, model_args.hidden_size)
            _p, decoy_pcc = compare_tensors(tt_hidden, decoy, pcc_threshold=0.0)
            decoys.append(float(decoy_pcc))
            decoy_bads.append(float((_per_token_pcc(tt_hidden, decoy) < 0.80).float().mean()))

        logger.info(
            f"[long_ctx_pcc] ctx={context_len} chunk {chunk_idx + 1}/{n_chunks} "
            f"[{chunk_start}, {chunk_start + chunk}) PCC={pcc} "
            f"per_token_min={tok_pcc.min():.5f} per_token_mean={tok_pcc.mean():.5f}"
            + (f" decoy={decoys[-1]:.5f} decoy_rows_below_0.8={100 * decoy_bads[-1]:.2f}%" if chunk_idx > 0 else "")
        )

    ring_calls = ring_prefill.ring_attention_calls()

    # End of context, called out separately. This is the state a disaggregated setup
    # actually hands to the decode host, and it is the strictest point in the run: it
    # sits at the maximum ring depth, having read every preceding chunk. The per-chunk
    # list below buries it, so report it on its own line — including the final token,
    # whose hidden state seeds the first decode step.
    logger.info(
        f"[long_ctx_pcc] ctx={context_len} END-OF-CONTEXT [{context_len - chunk}, {context_len}): "
        f"PCC={pccs[-1]:.5f} rows_below_0.8={100 * bad_fracs[-1]:.2f}% "
        f"last_token_PCC={last_token_pcc:.5f} (ring depth {n_chunks - 1} chunks)"
    )
    logger.info(
        f"[long_ctx_pcc] ctx={context_len} PCC min={min(pccs):.5f} max={max(pccs):.5f} "
        f"chunk0={pccs[0]:.5f} ring_reads={ring_calls} "
        f"rows_below_0.8 per chunk: {', '.join(f'{100 * f:.2f}%' for f in bad_fracs)} "
        f"(per-token mins: {', '.join(f'{m:.3f}' for m in tok_mins)})"
    )
    assert ring_calls == (n_chunks - 1) * len(model.layers), (
        f"ring attention ran {ring_calls} times, expected {(n_chunks - 1) * len(model.layers)} — "
        f"the history read did not happen, so these PCCs would not be testing it"
    )
    worst = min(pccs)
    assert worst >= threshold, (
        f"worst chunk PCC {worst:.5f} below {threshold} (per-chunk: " f"{', '.join(f'{p:.5f}' for p in pccs)})"
    )

    # Row-misordering check, applied to the RING chunks only.
    #
    # The statistic is the fraction of token rows below 0.80, not the worst row.
    # Calibrated against the reference: every misordering tried (roll by 1 token, by a
    # tile, by a CP slab, full reversal) puts 9.0-11.3% of rows under that line, and a
    # wrong chunk puts 100% there. A correct run measured 0.15%.
    #
    # Chunk 0 is excluded because it is the control, not a subject: it runs the mask CP
    # path with no history, and it carries a pre-existing ~6.25% at this precision —
    # the same baseline behind the long-standing 0.94 aggregate, which earlier
    # ablations pinned on neither weight dtype nor CP. Including it here would assert
    # against model accuracy rather than against the ring. The 3% line sits well above
    # what a correct ring produces and well below what any misordering produces.
    ring_bad = bad_fracs[1:]
    if ring_bad and max(ring_bad) >= 0.01:
        raise AssertionError(
            f"{100 * max(ring_bad):.2f}% of token rows in a ring chunk fall below per-token "
            f"PCC 0.80 (chunk 0 control: {100 * bad_fracs[0]:.2f}%). Misordering signatures "
            f"measured on this reference are 9-11% and a wrong chunk is 4-100%; a correct run "
            f"is under 0.2%. Per-chunk: {', '.join(f'{100 * f:.2f}%' for f in bad_fracs)}"
        )

    # The comparison has to be able to tell chunks apart, or the numbers above mean
    # nothing — so score each ring chunk against the PREVIOUS chunk's reference too and
    # require that to look clearly wrong.
    #
    # This is asserted on the per-token fraction rather than the aggregate because the
    # aggregate turned out not to discriminate here: past chunk 1 the sequence is
    # filler, and two filler chunks correlate 0.917-0.942 against each other while a
    # correct chunk scores 0.990 — a margin of ~0.05, which is not evidence of
    # anything. The same decoys separate cleanly per-token (4.25-9.33% of rows bad
    # versus under 0.2%), so that is what the assertion uses.
    if decoy_bads:
        assert min(decoy_bads) >= 0.02, (
            f"a decoy chunk scored only {100 * min(decoy_bads):.2f}% bad rows — the comparison "
            f"cannot tell adjacent chunks apart, so the real numbers "
            f"({', '.join(f'{100 * f:.2f}%' for f in ring_bad)}) are not evidence the ring read "
            f"the right history"
        )

    # Gate the end-of-context state explicitly. It is already covered by the per-chunk
    # assertions above, but this is the output the prefill actually exists to produce —
    # the hidden states handed to the decode host — and it deserves to fail on its own
    # terms rather than as one entry in a list. The final token additionally seeds the
    # first decode step, so it is checked individually; a single token is noisier than
    # a chunk, hence the looser bound.
    assert pccs[-1] >= threshold, (
        f"end-of-context PCC {pccs[-1]:.5f} below {threshold} for tokens "
        f"[{context_len - chunk}, {context_len}) at ring depth {n_chunks - 1}"
    )
    assert last_token_pcc >= 0.90, (
        f"final token (index {context_len - 1}) PCC {last_token_pcc:.5f} below 0.90 — this is the "
        f"hidden state a disaggregated decode host would start from"
    )


@torch.no_grad()
@parametrize_mesh_with_fabric([GALAXY_MESH], **_MESH_PARAMS)
@pytest.mark.parametrize("context_len", [65536, 131072, 262144], ids=["ctx_64k", "ctx_128k", "ctx_256k"])
def test_prefill_long_context_prefix_pcc(mesh_device, context_len, request, reset_seeds):
    """Ground-truth PCC for the leading chunks of a run that is too long to reference.

    A whole-sequence CPU reference stops being possible well below these lengths, so
    the 64k/128k/256k targets otherwise rest only on the ring-read count. This
    recovers real ground truth for part of them.

    It works because ``build_token_sequence`` draws its filler from a single seeded
    stream in fixed-size chunks, and chunk 0 is the prompt regardless of length. The
    first 32768 tokens of a 256k sequence are therefore byte-identical to the whole
    32k sequence (verified here by hash, not assumed), and attention is causal — token
    i depends only on tokens <= i. So the first 8 chunks of a 256k prefill must
    reproduce the 32k reference exactly, and any corruption from operating at 256k
    scale (cache capacity, gather buffers, RoPE at large offsets, the ring at 63
    chunks) shows up as a PCC drop on chunks the reference does cover.

    What this does NOT establish: tokens past 32768 have no reference here. Their
    correctness still rests on the ring-read count and the op-level probe. The check
    is a floor on the long runs, not a full verification of them.
    """
    from models.demos.gemma4.tests import cpu_prefill_reference as cpu_ref
    from models.demos.gemma4.tt.attention import ring_prefill
    from models.demos.gemma4.tt.ccl import cp_degree

    chunk = LONG_CONTEXT_CHUNK
    _guard_chunk(request, chunk)
    mesh_config = _mesh_config(mesh_device)
    cp = cp_degree(mesh_config)
    if cp <= 1:
        pytest.skip(f"targets CP>1; mesh {tuple(mesh_device.shape)} gives CP={cp}")

    ref_len = max(c for c in cpu_ref.LONG_REFERENCE_CONTEXTS if c < context_len)
    model_path = _model_path()
    reference = cpu_ref.load_long(model_path, ref_len)
    if reference is None:
        pytest.skip(
            f"No CPU reference at {cpu_ref.long_reference_path(model_path, ref_len)}. Generate with:\n"
            f"  GEMMA4_CPU_REF_CONTEXT={ref_len} python -m models.demos.gemma4.tests.cpu_prefill_reference"
        )

    ref_hidden = reference["hidden"]
    ref_tokens = reference["tokens"]
    n_chunks = context_len // chunk
    ref_chunks = ref_len // chunk

    model_args, model, kv_cache, page_table_tt = _build_prefill_model(
        mesh_device, model_path, chunk, context_len=context_len
    )
    tokens_all, _tok, _plen = cpu_ref.build_token_sequence(model_path, chunk, context_len, model_args.vocab_size)

    # The whole premise. If the long sequence does not actually start with the
    # reference's tokens, every PCC below would compare unrelated text and the drop
    # would be blamed on the ring.
    prefix_sha = cpu_ref.hash_tokens(tokens_all[:, :ref_len])
    if prefix_sha != cpu_ref.hash_tokens(ref_tokens):
        pytest.skip(
            f"ctx={context_len} does not start with the ctx={ref_len} reference tokens "
            f"({prefix_sha} vs {cpu_ref.hash_tokens(ref_tokens)}); regenerate the reference"
        )

    ring_prefill.reset_ring_attention_calls()
    threshold = get_pcc_threshold(request, default=0.93)
    pccs, tok_mins, bad_fracs = [], [], []
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk
        tokens = tokens_all[:, chunk_start : chunk_start + chunk].contiguous()
        host_input = _host_tensor(
            mesh_device, tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_config=mesh_config, seq_dim=-1
        )
        device_input = ttnn.to_device(host_input, device=mesh_device)
        with _lm_head_deferred(model):
            embeds, page_table, chunk_page_table, _ = model.transform_and_embed_prefill_inputs_device(
                device_input, page_table_tt, None, None
            )
            out = model.ttnn_prefill_forward(
                x=embeds,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start,
                kv_cache=kv_cache,
                get_last_token=-1,
                user_id=0,
            )
        ttnn.synchronize_device(mesh_device)
        tt_hidden = _cp_gather_torch(out, mesh_device, mesh_config)
        out.deallocate(True)

        if chunk_idx < ref_chunks:
            ref_slice = ref_hidden[:, chunk_start : chunk_start + chunk, :].reshape(1, 1, chunk, model_args.hidden_size)
            _passing, pcc = compare_tensors(tt_hidden, ref_slice, pcc_threshold=0.0)
            tok_pcc = _per_token_pcc(tt_hidden, ref_slice)
            pccs.append(float(pcc))
            tok_mins.append(float(tok_pcc.min()))
            bad_fracs.append(float((tok_pcc < 0.80).float().mean()))
            logger.info(
                f"[prefix_pcc] ctx={context_len} chunk {chunk_idx + 1}/{n_chunks} "
                f"[{chunk_start}, {chunk_start + chunk}) PCC={pcc} per_token_min={tok_pcc.min():.5f} "
                f"rows_below_0.8={100 * bad_fracs[-1]:.2f}%"
            )
        else:
            # Past the reference. Still assert the run stays numerically sane, so a
            # late blow-up is not silently carried into the chunks that follow.
            assert torch.isfinite(tt_hidden).all(), f"chunk {chunk_idx} produced non-finite values"

    ring_calls = ring_prefill.ring_attention_calls()
    logger.info(
        f"[prefix_pcc] ctx={context_len} referenced {ref_chunks}/{n_chunks} chunks (ctx={ref_len} reference) "
        f"PCC min={min(pccs):.5f} max={max(pccs):.5f} per_token_min={min(tok_mins):.5f} ring_reads={ring_calls} "
        f"rows_below_0.8: {', '.join(f'{100 * f:.2f}%' for f in bad_fracs)}"
    )
    assert ring_calls == (n_chunks - 1) * len(model.layers), (
        f"ring attention ran {ring_calls} times, expected {(n_chunks - 1) * len(model.layers)} — "
        f"the history read did not happen"
    )
    worst = min(pccs)
    assert worst >= threshold, (
        f"worst referenced-chunk PCC {worst:.5f} below {threshold} at ctx={context_len} "
        f"(per-chunk: {', '.join(f'{p:.5f}' for p in pccs)})"
    )
    # Misordering check on the ring chunks, same calibration as
    # test_prefill_long_context_vs_cpu_reference: 9-11% of rows below per-token 0.80 is
    # what any row permutation produces, ~0.15% is what a correct run produces. Chunk 0
    # is the no-history control and carries the pre-existing baseline, so it is
    # reported but not asserted on.
    ring_bad = bad_fracs[1:]
    if ring_bad and max(ring_bad) >= 0.01:
        raise AssertionError(
            f"{100 * max(ring_bad):.2f}% of token rows in a ring chunk fall below per-token PCC "
            f"0.80 at ctx={context_len} (chunk 0 control: {100 * bad_fracs[0]:.2f}%) — misordering "
            f"signatures are 9-11%, correct is ~0.15%. Per-chunk: "
            f"{', '.join(f'{100 * f:.2f}%' for f in bad_fracs)}"
        )


# ── Test 9: traced long-context chunked prefill (production shape) ────────────


@torch.no_grad()
@parametrize_mesh_with_fabric([GALAXY_MESH], **_MESH_PARAMS)
@pytest.mark.parametrize("context_len", LONG_CONTEXT_LENGTHS, ids=[f"ctx_{c // 1024}k" for c in LONG_CONTEXT_LENGTHS])
@pytest.mark.parametrize("readback_all", [True, False], ids=["readback_all", "readback_final"])
def test_prefill_long_context_traced(mesh_device, context_len, readback_all, reset_seeds, request):
    """Chunked prefill driven entirely by two captured traces.

    This is the deployment shape: warm the programs once, capture, then serve every
    request by replaying traces. Eager dispatch costs ~74% of prefill wall time at this
    size (1055.6 ms vs 272.6 ms on the 60-layer body), so the untraced numbers say very
    little about production throughput.

    Two traces, because there are exactly two distinct graphs. Chunk 0 has no history
    and runs the mask CP path; chunks 1..N-1 all run the ring path over a fixed-size Q
    slab against a fixed-capacity cache, so they share one graph. What differs between
    them is only the per-chunk scalars, and those now live in metadata tensors the
    kernels read on-device (see CCLManager.get_ring_metadata), which is what lets one
    capture serve 63 chunks.

    Everything that varies per chunk is refreshed on the host BETWEEN replays: the token
    input, the ring metadata, and the pinned RoPE slice. A trace records addresses, not
    values, so each of those had to be given a fixed address first.

    Two ttnn fixes were needed to make one capture valid for every chunk, both found by
    this test's PCC check rather than by its timings, which were happy throughout:
    compute_gather_valid_Ht capped the gather at the creating chunk's prefix, and the
    compact sliding halo's source group (linear in chunk index) was baked into the
    all-gather descriptor. Both are now derived on-device from kv_actual_isl.

    Traced output matches eager to five decimals at 32k:
      eager  0.94585 0.98890 0.98901 0.99042 0.99024 0.98987 0.98965 0.98899
      traced 0.94585 0.98890 0.98901 0.99042 0.99024 0.98987 0.98965 0.98899
    The perf half stands regardless: ~206 ms per replayed ring chunk vs ~1016 ms eager.
    """
    from models.demos.gemma4.tests import cpu_prefill_reference as cpu_ref
    from models.demos.gemma4.tt.attention import ring_prefill
    from models.demos.gemma4.tt.ccl import cp_degree

    chunk = LONG_CONTEXT_CHUNK
    _guard_chunk(request, chunk)
    mesh_config = _mesh_config(mesh_device)
    cp = cp_degree(mesh_config)
    if cp <= 1:
        pytest.skip(f"targets CP>1; mesh {tuple(mesh_device.shape)} gives CP={cp}")
    assert context_len % chunk == 0

    model_path = _model_path()
    n_chunks = context_len // chunk
    model_args, model, kv_cache, page_table_tt = _build_prefill_model(
        mesh_device, model_path, chunk, context_len=context_len
    )
    tokens_all, _tok, _plen = cpu_ref.build_token_sequence(model_path, chunk, context_len, model_args.vocab_size)

    rope_local_seq = chunk // cp
    host_input = _host_tensor(
        mesh_device,
        tokens_all[:, :chunk].contiguous(),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh_config=mesh_config,
        seq_dim=-1,
    )
    device_input = ttnn.to_device(host_input, device=mesh_device)
    device_positions = ttnn.to_device(
        _host_tensor(
            mesh_device,
            torch.arange(0, chunk, dtype=torch.int32).unsqueeze(0),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_config=mesh_config,
            seq_dim=-1,
        ),
        device=mesh_device,
    )
    model.set_prefill_rope_positions(device_positions)
    model._ring_metadata_external = True
    # logical_n stays per-chunk: the sliding halo layout is built from it at program-create
    # time and needs the capturing chunk's true geometry. The gather extent no longer
    # depends on it on the metadata path (compute_gather_valid_Ht bounds to full capacity
    # there and the all-gather reader narrows per dispatch from kv_actual_isl).

    stage_breakdown = {"tokens": 0.0, "metadata": 0.0, "rope": 0.0}

    def _stage(chunk_idx):
        """Host-side refresh of everything that varies per chunk. Never inside a trace."""
        chunk_start = chunk_idx * chunk
        _t = time.time()
        staged = _host_tensor(
            mesh_device,
            tokens_all[:, chunk_start : chunk_start + chunk].contiguous(),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_config=mesh_config,
            seq_dim=-1,
        )
        ttnn.copy_host_to_device_tensor(staged, device_input)
        stage_breakdown["tokens"] += time.time() - _t
        _t = time.time()
        model.ccl_manager.set_ring_metadata(slot_idx=0, kv_actual_global=chunk_start)
        stage_breakdown["metadata"] += time.time() - _t
        _t = time.time()
        # REQUIRED for liveness, not an optimization. Ring attention's global semaphores
        # persist across replays, and back-to-back replays deadlock without this: 256k
        # runs hung at chunk 54 and 59 of 64 (deep ring depth, no error, all threads in
        # futex_wait). readback_all hides it because _cp_gather_torch issues an eager
        # ttnn.all_gather between every replay, which drains the state this restores.
        # Costs ~4ms/chunk. Belongs in the op — either resetting its own semaphores or
        # having the reset captured inside the trace — rather than in every caller.
        for _sem in model.ccl_manager.ring_attention_ccl_semaphore_handles:
            ttnn.reset_global_semaphore_value(_sem, 0)
        # Under CP the prefill RoPE cache is chunk-major per rank, so the local slice
        # advances by the per-rank slab, matching _get_rope_mats' start_pos // cp.
        # Absolute global positions for this chunk, CP-sharded the same way tokens are.
        # Contiguous sharding of [chunk_start, chunk_start+chunk) hands rank r exactly the
        # rows chunk-major CP assigns it, so the gather inside the trace lands correctly.
        pos_host = _host_tensor(
            mesh_device,
            torch.arange(chunk_start, chunk_start + chunk, dtype=torch.int32).unsqueeze(0),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_config=mesh_config,
            seq_dim=-1,
        )
        ttnn.copy_host_to_device_tensor(pos_host, device_positions)
        stage_breakdown["rope"] += time.time() - _t
        return chunk_start

    def _forward(chunk_start):
        with _lm_head_deferred(model):
            embeds, page_table, chunk_page_table, _ = model.transform_and_embed_prefill_inputs_device(
                device_input, page_table_tt, None, None
            )
            return model.ttnn_prefill_forward(
                x=embeds,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start,
                kv_cache=kv_cache,
                get_last_token=-1,
                user_id=0,
            )

    # ── Warm up: compile both graphs eagerly ──────────────────────────────────
    t0 = time.time()
    for warm_idx in (0, 1):
        out = _forward(_stage(warm_idx))
        ttnn.synchronize_device(mesh_device)
        out.deallocate(True)
    warmup_s = time.time() - t0

    # ── Capture: one trace per distinct graph ─────────────────────────────────
    t0 = time.time()
    ring_prefill.reset_ring_attention_calls()
    start0 = _stage(0)
    tid_first = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_first = _forward(start0)
    ttnn.end_trace_capture(mesh_device, tid_first, cq_id=0)

    # Which chunk the ring trace is captured at. A probe: replays must be independent of
    # this, so if correctness tracks the capture point then per-chunk state is still baked
    # into the recorded commands somewhere.
    one_trace = os.environ.get("GEMMA4_ONE_TRACE") == "1"
    cap_idx = int(os.environ.get("GEMMA4_TRACE_CAPTURE_CHUNK", "1"))
    start1 = _stage(cap_idx)
    tid_ring = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_ring = _forward(start1)
    ttnn.end_trace_capture(mesh_device, tid_ring, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    capture_s = time.time() - t0
    logger.info(f"[traced] warmup(compile)={warmup_s:.1f}s capture={capture_s:.1f}s for 2 traces")

    # Ring reads are counted in Python, and a replay runs no Python — so the counter can
    # only be read at capture, where it confirms the ring graph really is inside the
    # recorded trace. During replay it stays 0 by construction; correctness of the
    # replays is established by PCC below, not by this.
    captured_ring_calls = ring_prefill.ring_attention_calls()
    assert captured_ring_calls >= len(model.layers), (
        f"only {captured_ring_calls} ring calls recorded while capturing, expected >= {len(model.layers)} "
        f"(one per layer) — the ring path is not in the captured trace"
    )

    # Warm replay of each, so the measured pass excludes one-off dispatch setup.
    for tid, idx in ((tid_first, 0), (tid_ring, cap_idx)):
        _stage(idx)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    try:
        reference = cpu_ref.load_long(model_path, context_len)
        ref_hidden = reference["hidden"] if reference is not None else None
        if reference is not None:
            assert cpu_ref.hash_tokens(reference["tokens"]) == cpu_ref.hash_tokens(
                tokens_all
            ), "CPU reference tokens differ from the traced run's tokens"
        per_chunk, pccs, bad_fracs = [], [], []
        stage_s, readback_s = 0.0, 0.0
        t_run = time.time()
        for chunk_idx in range(n_chunks):
            t_stage = time.time()
            chunk_start = _stage(chunk_idx)
            stage_s += time.time() - t_stage
            # GEMMA4_ONE_TRACE: replay the ring trace for chunk 0 too. Chunk 0 differs only in
            # kv_actual_isl == 0, which the kernels now derive on-device, so one capture can serve
            # the whole prefill — halving trace memory and dropping the mask path's ~80ms premium.
            tid = tid_ring if (one_trace or chunk_idx > 0) else tid_first
            t_c = time.time()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            per_chunk.append(time.time() - t_c)
            out = out_first if chunk_idx == 0 else out_ring
            # Reading every chunk's hidden states to host is a test artifact — a prefill
            # server leaves the KV cache on device and reads back only the last chunk,
            # whose final row seeds the first decode step. readback="final" measures that
            # shape; "all" keeps the per-chunk PCC that verifies the ring read.
            if readback_all or chunk_idx == n_chunks - 1:
                t_rb = time.time()
                hidden = _cp_gather_torch(out, mesh_device, mesh_config)
                assert torch.isfinite(hidden).all(), f"chunk {chunk_idx} produced non-finite output"
                readback_s += time.time() - t_rb
                if ref_hidden is not None:
                    ref_slice = ref_hidden[:, chunk_start : chunk_start + chunk, :].reshape(
                        1, 1, chunk, model_args.hidden_size
                    )
                    _p, pcc = compare_tensors(hidden, ref_slice, pcc_threshold=0.0)
                    pccs.append(float(pcc))
                    bad_fracs.append(float((_per_token_pcc(hidden, ref_slice) < 0.80).float().mean()))
            logger.info(
                f"[traced_perf] chunk {chunk_idx + 1}/{n_chunks} [{chunk_start}, {chunk_start + chunk}) "
                f"device={per_chunk[-1] * 1000:.1f}ms ({chunk / per_chunk[-1]:.0f} tok/s) | "
                f"{'mask-path trace' if chunk_idx == 0 else 'ring trace'} | ring_depth={chunk_idx}"
            )
        total_s = time.time() - t_run
    finally:
        ttnn.release_trace(mesh_device, tid_first)
        ttnn.release_trace(mesh_device, tid_ring)

    ring_chunks = per_chunk[1:]
    device_s = sum(per_chunk)
    # Three different numbers, because conflating them understates the model by ~2x.
    #   device   — execute_trace + synchronize. What the hardware spends on prefill.
    #   staging  — token upload, ring metadata, pinned RoPE refresh. Real work a
    #              deployment also pays, though it should overlap rather than serialize.
    #   readback — gathering every chunk's hidden states to host so this test can assert
    #              on them. Test-only: a prefill server keeps the KV cache on device and
    #              reads back at most the final chunk.
    logger.info(
        f"[traced_perf] DEVICE {context_len} tokens in {device_s:.1f}s "
        f"({context_len / device_s:.0f} tok/s) | staging {stage_s:.1f}s | "
        f"readback {readback_s:.1f}s (test-only) | wall {total_s:.1f}s"
    )
    logger.info(
        f"[traced_perf] staging breakdown: "
        + ", ".join(f"{k}={v:.1f}s ({1000 * v / n_chunks:.0f}ms/chunk)" for k, v in stage_breakdown.items())
    )
    logger.info(
        f"[traced_perf] TOTAL {context_len} tokens in {total_s:.1f}s ({context_len / total_s:.0f} tok/s) "
        f"| chunk0(mask)={per_chunk[0] * 1000:.1f}ms | ring chunks mean={sum(ring_chunks) / len(ring_chunks) * 1000:.1f}ms "
        f"min={min(ring_chunks) * 1000:.1f}ms max={max(ring_chunks) * 1000:.1f}ms"
    )
    logger.info(
        f"[traced_perf] ring-depth cost: first={ring_chunks[0] * 1000:.1f}ms -> last={ring_chunks[-1] * 1000:.1f}ms "
        f"= {ring_chunks[-1] / ring_chunks[0]:.2f}x over {len(ring_chunks) - 1} extra chunks of history"
    )

    # The numerical check, and the whole reason this is trustworthy. A trace records the
    # values live at capture, so if the per-chunk scalars were still baked, every replay
    # would attend over chunk 1's prefix. That failure is invisible to timings and to the
    # finiteness check above; it shows up here as a PCC collapse on chunks after the
    # captured one. Runs whenever a reference exists for this length.
    if pccs:
        worst = min(pccs)
        logger.info(
            f"[traced_pcc] ctx={context_len} vs CPU reference: min={worst:.5f} max={max(pccs):.5f} "
            f"per-chunk: {', '.join(f'{p:.5f}' for p in pccs)}"
        )
        assert worst >= get_pcc_threshold(request, default=0.93), (
            f"traced replay PCC {worst:.5f} below threshold — replays are not reproducing the "
            f"eager result, which points at a per-chunk scalar frozen at capture "
            f"(per-chunk: {', '.join(f'{p:.5f}' for p in pccs)})"
        )
        # With readback="final" there is a single entry and it is a ring chunk, so no
        # chunk-0 control to drop.
        ring_bad_fracs = bad_fracs[1:] if readback_all else bad_fracs
        assert not ring_bad_fracs or max(ring_bad_fracs) < 0.01, (
            f"{100 * max(ring_bad_fracs):.2f}% of token rows in a traced ring chunk fall below "
            f"per-token PCC 0.80 — misordering signatures are 9-11%, correct is under 0.2%"
        )
    else:
        logger.warning(
            f"[traced_pcc] no CPU reference for ctx={context_len}; perf above is measured but "
            f"replay correctness is NOT verified at this length"
        )
