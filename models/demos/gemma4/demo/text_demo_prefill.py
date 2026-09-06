# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 long-context prefill on a Blackhole Galaxy."""

import functools
import hashlib
import os
import pathlib
import time
from contextlib import contextmanager

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import find_layer_idx, parametrize_mesh_with_fabric
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.utils.partial_weights import load_cache_completion_state
from models.tt_transformers.tt.common import PagedAttentionConfig

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*_args, **_kwargs):
        pass


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_DTYPE = ttnn.bfloat16
PAGE_BLOCK_SIZE = 64
TRACE_REGION_SIZE = int(os.environ.get("GEMMA4_PREFILL_TRACE_REGION_SIZE", 256_000_000))


def _model_path():
    return os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", "google/gemma-4-31B-it")


def _load_full_weights():
    """True when the caller wants the full host state dict (cold-cache path)."""
    return os.environ.get("GEMMA4_PREFILL_LOAD_FULL_WEIGHTS", "0").lower() in ("1", "true", "yes")


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


# Public-domain long text, tokenized to stand in for a real prompt. Cached on first use
# under the same context_cache the tt_transformers demos populate, keyed by URL digest, so
# a run needs the network once per machine and never again.
_TOKEN_TEXT_URL = "https://www.gutenberg.org/cache/epub/135/pg135.txt"
_TOKEN_TEXT_CACHE = pathlib.Path("models/tt_transformers/demo/context_cache")


@functools.lru_cache(maxsize=None)
def _text_token_stream(model_path):
    """The source text tokenized once per process, as ``[1, n]`` int32 ids.

    No chat template: it would append a question after the context, so a 32k and a 128k
    sequence would diverge at their tails and stop being prefixes of one another.
    """
    from transformers import AutoTokenizer

    cache_file = _TOKEN_TEXT_CACHE / hashlib.md5(_TOKEN_TEXT_URL.encode()).hexdigest()
    if cache_file.exists():
        text = cache_file.read_text()
    else:
        import requests

        resp = requests.get(_TOKEN_TEXT_URL, timeout=60)
        resp.raise_for_status()
        text = resp.text
        _TOKEN_TEXT_CACHE.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return torch.tensor(tokenizer.encode(text), dtype=torch.int32).unsqueeze(0)


def _prefill_tokens(model_path, context_len, vocab_size, source="text"):
    """Deterministic token ids for a ``context_len``-token prefill.

    ``text`` (the default) tokenizes real prose, so attention sees real token statistics
    rather than uniform ids -- which is what the timings are meant to represent, and what
    any future value check would need. The stream is tiled when the text runs short of the
    requested length; repetition is still language.

    ``random`` draws uniform ids from a fixed seed instead. Cheaper, and it needs neither
    the network nor a tokenizer, so it is the fallback when the corpus cannot be fetched --
    but token statistics are not a real prompt's.

    Either way the result is prefix-consistent by construction, since every length is a
    slice of one stream: the first N tokens of a 256k sequence are the whole of an N-token
    sequence.
    """
    if source == "random":
        gen = torch.Generator().manual_seed(0)
        return torch.randint(0, vocab_size, (1, context_len), dtype=torch.int32, generator=gen)
    assert source == "text", f"unknown token source {source!r}, expected 'text' or 'random'"

    ids = _text_token_stream(model_path)
    if ids.shape[-1] < context_len:
        ids = ids.repeat(1, -(-context_len // ids.shape[-1]))
    ids = ids[:, :context_len].clone()
    assert int(ids.max()) < vocab_size, f"tokenizer emitted id {int(ids.max())} outside vocab {vocab_size}"
    return ids


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


def _hf_text_config(model_path):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    text_config._attn_implementation = "eager"
    return text_config


# ── The prefill model under test ────────────────────────────────────────────


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


# Gemma4's sliding window, a model fact rather than a knob: ring_joint requires
# ``halo_tokens <= N_local_q`` (ring_joint_sdpa_device_operation.cpp) and the window rounds
# up to a 32-tile = 1024-token halo, so the per-rank Q slab ``chunk / cp`` must be at least
# this. Every test that picks a chunk size is checked against it.
_SLIDING_WINDOW_TOKENS = 1024


# ── Traced long-context chunked prefill (production shape) ────────────────────


@torch.no_grad()
# Mesh is a test arg: -k 8x4 for TP=4 x CP=8, -k 4x8 for TP=8 x CP=4. TP=32 is absent
# because hidden 5376/32 = 168 makes the embedding all-gather's page 336 B, not 64 B
# aligned, so it falls back to composite_all_gather and deadlocks (GALAXY_1x32_HANG.md).
# The tensor cache is tagged by TP, so each mesh needs its own (_tp4_ / _tp8_).
@parametrize_mesh_with_fabric([(8, 4), (4, 8)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
# Legality depends on the mesh -- the halo rule needs chunk >= window * cp -- so illegal
# combinations skip on that arithmetic rather than reaching ring_joint's TT_FATAL after a
# 90 s model load. Bigger is faster with sharply diminishing returns; the sweep shows where
# that flattens. Every context length stays divisible by every chunk size.
# Real prose by default; swap in "random" here to compare against uniform ids, or when the
# corpus cannot be fetched. See _prefill_tokens.
@pytest.mark.parametrize("token_source", ["text"], ids=lambda t: t)
@pytest.mark.parametrize("chunk_size", [4096, 8192, 16384, 32768], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("context_len", [32768, 65536, 131072, 262144], ids=lambda c: f"ctx_{c // 1024}k")
@pytest.mark.parametrize("readback_all", [True, False], ids=["readback_all", "readback_final"])
def test_prefill_long_context_traced(
    mesh_device, context_len, chunk_size, readback_all, token_source, reset_seeds, request
):
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

    Two ttnn fixes were needed to make one capture valid for every chunk. Both were found
    by a per-chunk PCC check against a host reference -- since retired with that reference,
    and worth restoring with its replacement -- rather than by timings, which were happy
    throughout:
    compute_gather_valid_Ht capped the gather at the creating chunk's prefix, and the
    compact sliding halo's source group (linear in chunk index) was baked into the
    all-gather descriptor. Both are now derived on-device from kv_actual_isl.

    Traced output matches eager to five decimals at 32k:
      eager  0.94585 0.98890 0.98901 0.99042 0.99024 0.98987 0.98965 0.98899
      traced 0.94585 0.98890 0.98901 0.99042 0.99024 0.98987 0.98965 0.98899
    The perf half stands regardless: ~206 ms per replayed ring chunk vs ~1016 ms eager.

    Why 8x4 is the mesh worth running, given the chunk size is tied to it. A 256k prefill,
    device time only, measured back to back:

        4x8  TP=8 CP=4  chunk 4096  slab 1024  64 chunks  22.4 s  11.7k tok/s  200 -> 488 ms
        8x4  TP=4 CP=8  chunk 8192  slab 1024  32 chunks  18.7 s  14.0k tok/s  304 -> 885 ms
        8x4  TP=4 CP=8  chunk 4096  slab  512  -- TT_FATAL, halo 1024 > slab 512

    8x4 is ~17% faster end to end despite each chunk costing more, because it runs half
    as many of them and the per-device Q slab is the same 1024 tokens either way. Trading
    tensor parallelism for context parallelism is the win at long context.

    The third row is why the chunk has to scale with CP rather than staying at 4096:
    keeping the chunk while doubling CP halves the Q slab below the sliding window, and
    ring_joint refuses it instead of attending over a truncated history. That combination
    is a param here, and it skips on the halo arithmetic before the model loads.
    """
    from models.demos.gemma4.tt.attention import ring_prefill
    from models.demos.gemma4.tt.ccl import cp_degree

    chunk = chunk_size
    mesh_config = _mesh_config(mesh_device)
    cp = cp_degree(mesh_config)
    if cp <= 1:
        pytest.skip(f"targets CP>1; mesh {tuple(mesh_device.shape)} gives CP={cp}")
    # The halo rule, as a skip rather than a TT_FATAL 90 s into a model load: under CP the
    # sliding layers fetch their window from a single predecessor rank, so the per-rank Q slab
    # has to cover the window. The 10 global layers would be fine; the 50 sliding ones are not.
    if chunk < _SLIDING_WINDOW_TOKENS * cp:
        pytest.skip(
            f"chunk={chunk} gives a {chunk // cp}-token Q slab at CP={cp}, under the "
            f"{_SLIDING_WINDOW_TOKENS}-token sliding window; ring_joint needs "
            f"chunk >= window*cp = {_SLIDING_WINDOW_TOKENS * cp} (its halo is single-hop)"
        )
    if context_len % chunk != 0:
        pytest.skip(f"context_len={context_len} is not a whole number of {chunk}-token chunks")

    model_path = _model_path()
    n_chunks = context_len // chunk
    model_args, model, kv_cache, page_table_tt = _build_prefill_model(
        mesh_device, model_path, chunk, context_len=context_len
    )
    tokens_all = _prefill_tokens(model_path, context_len, model_args.vocab_size, token_source)

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

    # One trace for the whole prefill: every chunk takes the ring path, and chunk 0 differs
    # from the rest only in kv_actual_isl == 0, which the kernels derive on-device. Capture at
    # chunk 0, which used to be rejected host-side because the halo layout demanded a complete
    # predecessor group; it now builds from one group, so the first chunk is capturable like
    # any other and the warmup that compiles the graph is also the chunk being captured.

    # ── Warm up: compile the graph that will be captured ──────────────────────
    t0 = time.time()
    out = _forward(_stage(0))
    ttnn.synchronize_device(mesh_device)
    out.deallocate(True)
    warmup_s = time.time() - t0

    # ── Capture ───────────────────────────────────────────────────────────────
    t0 = time.time()
    ring_prefill.reset_ring_attention_calls()
    cap_start = _stage(0)
    tid_ring = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_ring = _forward(cap_start)
    ttnn.end_trace_capture(mesh_device, tid_ring, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    capture_s = time.time() - t0
    logger.info(f"[traced] warmup(compile)={warmup_s:.1f}s capture={capture_s:.1f}s for 1 trace")

    # Ring reads are counted in Python, and a replay runs no Python — so the counter can
    # only be read at capture, where it confirms the ring graph really is inside the
    # recorded trace. During replay it stays 0 by construction, so this says nothing about
    # whether the replays are numerically right -- nothing here does, until the replacement
    # reference lands.
    captured_ring_calls = ring_prefill.ring_attention_calls()
    assert captured_ring_calls >= len(model.layers), (
        f"only {captured_ring_calls} ring calls recorded while capturing, expected >= {len(model.layers)} "
        f"(one per layer) — the ring path is not in the captured trace"
    )

    # Warm replay, so the measured pass excludes one-off dispatch setup.
    _stage(0)
    ttnn.execute_trace(mesh_device, tid_ring, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    try:
        per_chunk = []
        stage_s, readback_s = 0.0, 0.0
        t_run = time.time()
        for chunk_idx in range(n_chunks):
            t_stage = time.time()
            chunk_start = _stage(chunk_idx)
            stage_s += time.time() - t_stage
            # One capture serves every chunk, chunk 0 included: it differs only in
            # kv_actual_isl == 0, which the kernels derive on-device.
            t_c = time.time()
            ttnn.execute_trace(mesh_device, tid_ring, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            per_chunk.append(time.time() - t_c)
            out = out_ring
            # Reading every chunk's hidden states to host is a test artifact — a prefill
            # server leaves the KV cache on device and reads back only the last chunk,
            # whose final row seeds the first decode step. readback="final" measures that
            # shape; "all" gathers every chunk, which costs wall time but checks each one
            # for finiteness instead of only the last.
            if readback_all or chunk_idx == n_chunks - 1:
                t_rb = time.time()
                hidden = _cp_gather_torch(out, mesh_device, mesh_config)
                assert torch.isfinite(hidden).all(), f"chunk {chunk_idx} produced non-finite output"
                readback_s += time.time() - t_rb
            # Cumulative device and wall alongside the per-chunk cost, so the run can be
            # read without summing the column by hand. These are the same two totals the
            # DEVICE/TOTAL summary lines report at the end: device is execute_trace +
            # synchronize only, wall additionally carries staging and test-only readback.
            logger.info(
                f"[traced_perf] chunk {chunk_idx + 1}/{n_chunks} [{chunk_start}, {chunk_start + chunk}) "
                f"device={per_chunk[-1] * 1000:.1f}ms ({chunk / per_chunk[-1]:.0f} tok/s) | "
                f"total device={sum(per_chunk):.1f}s wall={time.time() - t_run:.1f}s"
            )
        total_s = time.time() - t_run
    finally:
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
        f"| chunk0(ring)={per_chunk[0] * 1000:.1f}ms | ring chunks mean={sum(ring_chunks) / len(ring_chunks) * 1000:.1f}ms "
        f"min={min(ring_chunks) * 1000:.1f}ms max={max(ring_chunks) * 1000:.1f}ms"
    )
    logger.info(
        f"[traced_perf] ring-depth cost: first={ring_chunks[0] * 1000:.1f}ms -> last={ring_chunks[-1] * 1000:.1f}ms "
        f"= {ring_chunks[-1] / ring_chunks[0]:.2f}x over {len(ring_chunks) - 1} extra chunks of history"
    )

    # No value check here. A trace records the values live at capture, so a per-chunk scalar
    # frozen at capture would make every replay attend over chunk 0's prefix -- a failure
    # invisible to both the timings and the finiteness check above. A per-chunk PCC against a
    # host reference used to catch exactly that; until the replacement reference lands, this
    # test measures perf and proves liveness only.
    logger.warning(
        f"[traced_perf] ctx={context_len}: perf above is measured, but replay correctness is "
        f"NOT verified — no reference to compare against"
    )


# ── One decoder layer at a given chunk depth, perf only ───────────────────────


def _perf_layer_tag(layer_type):
    return "sliding" if layer_type == "sliding_attention" else "global"


def _perf_signposts(layer_type, chunk_idx):
    """The signpost pair bracketing one measured replay.

    Named per (type, chunk) so a single profiler CSV holding the whole sweep can be
    sliced to exactly one cell::

        tt-perf-report --start-signpost gemma4-layer-global-chunk7-start \\
                       --end-signpost   gemma4-layer-global-chunk7-stop  REPORT.csv
    """
    base = f"gemma4-layer-{_perf_layer_tag(layer_type)}-chunk{chunk_idx}"
    return f"{base}-start", f"{base}-stop"


@torch.no_grad()
# pytest.ini caps tests at 300s, which a single cell clears easily but a 64-chunk sweep
# does not: the model load alone is minutes, and `chunkall-both` then measures 128 cells.
# Same override test_batched_prefill_perf uses for the same reason.
@pytest.mark.timeout(7200)
# Mesh is a test arg; see the traced test for why TP=32 is absent.
@parametrize_mesh_with_fabric([(8, 4), (4, 8)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
# Warm replays before the measured one. analyze_ops_perf.py skips 2 + this many
# invocations (eager compile + capture + warmups) to find the measured replay, so the two
# move together.
# Real prose by default; swap in "random" here to compare against uniform ids, or when the
# corpus cannot be fetched. See _prefill_tokens.
@pytest.mark.parametrize("token_source", ["text"], ids=lambda t: t)
@pytest.mark.parametrize("warmup_iters", [5], ids=lambda n: f"warm{n}")
# The chunk index only means something relative to a context length and a chunk size, so
# both are args here rather than borrowed constants. chunk_idx covers the deepest geometry
# on offer (256k in 8192-token chunks); a shallower one skips its out-of-range indices.
@pytest.mark.parametrize("context_len", [262144], ids=lambda c: f"ctx_{c // 1024}k")
@pytest.mark.parametrize("chunk_size", [8192], ids=lambda c: f"sz{c}")
@pytest.mark.parametrize(
    "layer_type", ["full_attention", "sliding_attention", "both"], ids=["global", "sliding", "both"]
)
# Every index is its own param, so `-k chunk37` addresses chunk 37 directly. "all" runs the
# whole depth sweep in one process, the only way to pay the model load once — see the
# docstring on cache warmth for why that ordering also makes the numbers better.
@pytest.mark.parametrize("chunk_idx", [*range(262144 // 8192), "all"], ids=lambda c: f"chunk{c}")
def test_prefill_layer_perf_chunk_n(
    mesh_device, chunk_idx, layer_type, chunk_size, context_len, warmup_iters, token_source, reset_seeds, request
):
    """Time ONE decoder layer at a chosen chunk depth, traced, with per-chunk signposts.

    Chunk 0 is the cheap case — no history, no KV cache, no ring — and is the fast
    iteration loop. This test answers the question that actually sizes a 256k prefill:
    what does a layer cost at chunk N, with N*``chunk_size`` tokens of history behind
    it. The two types diverge hard with depth, which is the whole point of measuring them
    separately:

      global:  attends the full prefix, so the ring gather grows with N
      sliding: attends a 1024-token window, so it should stay roughly flat past chunk 0

    **Geometry is the canonical traced test's, not a reconstruction of it.** The model is
    built by ``_build_prefill_model(..., context_len=context_len)`` — the same call
    ``test_prefill_long_context_traced`` makes — and the per-chunk staging is the same
    four steps its ``_stage`` does: tokens, ring metadata, ring semaphore reset, absolute
    CP-sharded RoPE positions. Only one layer out of the 60 is then driven. Rebuilding a
    standalone layer would have meant re-deriving the paged block pool, the CP block
    override, the identity page table and the RoPE offset by hand, and every one of those
    is a chance to measure a geometry the real run never uses.

    Input is the same deterministic token sequence the traced test stages
    (``_prefill_tokens``), sliced at this chunk's offset and embedded by the model's own
    ``embed_tokens`` — the same tokens at the same positions the canonical run consumes.

    **Cache warmth.** Every chunk from 0 up to the highest requested one is replayed; only
    the requested ones are measured. A replay writes its own K/V, so this is what gives a
    measured chunk the prefix a real run would have — ``chunk37`` on its own would otherwise
    attend over a cache that is zero everywhere except chunk 37. Timing would survive that
    (the ring gather extent comes from the ``kv_actual_global`` scalar, which the kernels
    turn into ``kv_actual_isl`` on-device) but the values would be meaningless, and the
    difference is invisible in a timing. Nothing here asserts on values beyond finiteness
    regardless, since the layer's true input is 7 layers deep and cannot be reproduced
    standalone.

    Warms ``warmup_iters`` replays per cell and measures EXACTLY
    ONE, so the profiled region holds one invocation of each op and the report reads
    directly as per-layer.

    Usage::

        # one cell — the chunk index is a param, so no env var is involved
        TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \\
          python -m tracy -p -r -v -m pytest \\
          'models/demos/gemma4/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk7-global-4x8]' -sv
        tt-perf-report --start-signpost gemma4-layer-global-chunk7-start \\
                       --end-signpost   gemma4-layer-global-chunk7-stop  REPORT.csv

        # the whole depth sweep, one model load, chunk-major
        ... 'test_prefill_layer_perf_chunk_n[blackhole-chunkall-both-4x8]' -sv

    ``models/demos/gemma4/tests/sweep_layer_perf.py`` drives the sweep and files the
    per-cell tt-perf-report output.
    """
    from models.demos.gemma4.tt.attention import ring_prefill
    from models.demos.gemma4.tt.ccl import cp_degree

    chunk = chunk_size
    mesh_config = _mesh_config(mesh_device)
    cp = cp_degree(mesh_config)
    if cp <= 1:
        pytest.skip(f"targets CP>1; mesh {tuple(mesh_device.shape)} gives CP={cp}")
    if chunk < _SLIDING_WINDOW_TOKENS * cp:
        pytest.skip(
            f"chunk {chunk} / CP {cp} = {chunk // cp} tokens per rank, below the "
            f"{_SLIDING_WINDOW_TOKENS}-token sliding window; ring_joint needs "
            f"chunk >= window*cp = {_SLIDING_WINDOW_TOKENS * cp}"
        )
    assert context_len % chunk == 0
    n_chunks = context_len // chunk
    if chunk_idx != "all" and int(chunk_idx) >= n_chunks:
        pytest.skip(f"chunk {chunk_idx} is past the {n_chunks} chunks of {chunk} in {context_len} tokens")

    # ``chunkall`` walks every depth in one model load; ``chunkN`` measures exactly one.
    # Selection is the pytest param, so a subset is a list of node ids -- one load each.
    chunk_idxs = list(range(n_chunks)) if chunk_idx == "all" else [int(chunk_idx)]
    layer_types = ["full_attention", "sliding_attention"] if layer_type == "both" else [layer_type]
    warm_iters = warmup_iters

    model_path = _model_path()
    text_config = _hf_text_config(model_path)
    model_args, model, kv_cache, page_table_tt = _build_prefill_model(
        mesh_device, model_path, chunk, context_len=context_len
    )
    tokens_all = _prefill_tokens(model_path, context_len, model_args.vocab_size, token_source)

    layer_idxs = {lt: find_layer_idx(text_config, lt) for lt in layer_types}
    type_desc = ", ".join(f"{_perf_layer_tag(lt)}=layer{layer_idxs[lt]}" for lt in layer_types)
    logger.info(
        f"[layer_perf_chunk] ctx={context_len} chunk={chunk} n_chunks={n_chunks} cp={cp} | "
        f"cells={len(chunk_idxs) * len(layer_types)} chunks={chunk_idxs[0]}..{chunk_idxs[-1]} "
        f"types=({type_desc}) warmup_replays={warm_iters}"
    )

    # ── Pinned per-chunk inputs. A trace records addresses, not values, so everything
    # that varies per chunk has to live at a fixed address and be refreshed on the host
    # between replays. Same three buffers the canonical traced test pins.
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
    # This test owns the ring metadata writes (they must happen outside any traced region).
    model._ring_metadata_external = True

    def _stage(idx):
        """Host-side refresh of everything that varies per chunk. Never inside a trace.

        Lifted from ``test_prefill_long_context_traced._stage`` — same four steps in the
        same order, because a divergence here is a divergence in what is being measured.
        """
        chunk_start = idx * chunk
        staged = _host_tensor(
            mesh_device,
            tokens_all[:, chunk_start : chunk_start + chunk].contiguous(),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_config=mesh_config,
            seq_dim=-1,
        )
        ttnn.copy_host_to_device_tensor(staged, device_input)
        model.ccl_manager.set_ring_metadata(slot_idx=0, kv_actual_global=chunk_start)
        # REQUIRED for liveness, not an optimization: ring attention's global semaphores
        # persist across replays and back-to-back replays deadlock without this. See the
        # long note in test_prefill_long_context_traced.
        for _sem in model.ccl_manager.ring_attention_ccl_semaphore_handles:
            ttnn.reset_global_semaphore_value(_sem, 0)
        pos_host = _host_tensor(
            mesh_device,
            torch.arange(chunk_start, chunk_start + chunk, dtype=torch.int32).unsqueeze(0),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            mesh_config=mesh_config,
            seq_dim=-1,
        )
        ttnn.copy_host_to_device_tensor(pos_host, device_positions)
        return chunk_start

    def _make_forward(lt):
        """One layer's forward, assembled the way ``Gemma4Model.__call__`` assembles it.

        Six lines mirroring the model's layer loop for a single index: embed, gather this
        layer type's RoPE on-device, call the layer. The RoPE gather has to be INSIDE the
        traced region — it reads the pinned positions tensor, which is what lets one
        capture serve every chunk depth.
        """
        idx = layer_idxs[lt]
        layer = model.layers[idx]
        cache = model.tt_kv_cache[idx]
        # A KV-sharing layer skips its own K/V projection and cache write, and needs the
        # source layer's K/V passed in. find_layer_idx returns the FIRST layer of a type,
        # which sharing maps *from* rather than *to*, so this should never fire — but
        # measuring a shared layer as if it were standalone would quietly under-report it.
        assert idx not in model.kv_shared_layer_map, (
            f"layer {idx} ({lt}) shares KV from layer {model.kv_shared_layer_map[idx]}; "
            f"timing it standalone would omit the K/V projection and cache write"
        )
        # The 2D caches are what the on-device gather indexes; they only exist when the
        # model was built with a real HF text config (create_tt_model sets it). Without
        # them there is no per-chunk RoPE and every depth would silently use chunk 0's.
        assert lt in model.rope_caches_2d, (
            f"model has no 2D RoPE cache for {lt} (built without _hf_text_config?) — "
            f"per-chunk RoPE would be wrong, refusing to measure"
        )
        cos_2d, sin_2d = model.rope_caches_2d[lt]

        def forward(chunk_start):
            embeds, page_table, chunk_page_table, _ = model.transform_and_embed_prefill_inputs_device(
                device_input, page_table_tt, None, None
            )
            cos = ttnn.unsqueeze_to_4D(ttnn.embedding(model._rope_prefill_positions, cos_2d, layout=ttnn.TILE_LAYOUT))
            sin = ttnn.unsqueeze_to_4D(ttnn.embedding(model._rope_prefill_positions, sin_2d, layout=ttnn.TILE_LAYOUT))
            return layer(
                embeds,
                rope_mats=(cos, sin),
                position_idx=None,
                page_table=page_table,
                kv_cache=cache,
                is_decode=False,
                batch_size=1,
                user_id=0,
                chunk_start_idx=chunk_start,
                chunk_page_table=chunk_page_table,
            )

        return forward

    # ── Compile and capture one trace per layer type. Captured at the first chunk index
    # being measured; the per-chunk scalars all live in metadata tensors the kernels read
    # on-device, so one capture serves every depth.
    traces, outs = {}, {}
    capture_at = chunk_idxs[0]
    for lt in layer_types:
        fwd = _make_forward(lt)
        t0 = time.time()
        compile_out = fwd(_stage(capture_at))
        ttnn.synchronize_device(mesh_device)
        compile_out.deallocate(True)
        compile_s = time.time() - t0

        t0 = time.time()
        ring_prefill.reset_ring_attention_calls()
        cap_start = _stage(capture_at)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs[lt] = fwd(cap_start)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        traces[lt] = tid
        capture_s = time.time() - t0
        # A replay runs no Python, so this counter can only be read at capture. It confirms
        # the ring graph really is inside the recorded trace rather than a mask-path
        # fallback that would make every depth cost the same.
        assert ring_prefill.ring_attention_calls() >= 1, (
            f"no ring attention call recorded while capturing the {_perf_layer_tag(lt)} layer — "
            f"the ring path is not in the captured trace, so depth would not be measured"
        )
        logger.info(
            f"[layer_perf_chunk] {_perf_layer_tag(lt)} layer_idx={layer_idxs[lt]} "
            f"compile={compile_s:.1f}s capture={capture_s:.1f}s"
        )

    # A layer at chunk N attends over everything before it, and that prefix lives in the KV
    # cache. A fresh process starts with the cache zeroed, so the prefix has to be put there
    # somehow before a measured chunk means anything: measured at chunk 32, a zeroed cache
    # reads 17.95ms against 21.01ms once the prefix is present, with non-overlapping warm
    # spreads. Ring cost depends on the cache holding real data, so this is not optional.
    measured_set = set(chunk_idxs)
    fill_upto = max(chunk_idxs)
    n_fill = (fill_upto + 1) - len(measured_set)

    # GEMMA4_PERF_KV_FILL picks how the prefix gets into the cache before a measured chunk:
    #
    #   random (default) — write random values straight into the cache tensors, once. Cost is
    #                      flat in N, which is the whole point: replay fill made chunk 63 cost
    #                      126 extra replays, and under the profiler every one is recorded.
    #   replay           — replay chunks 0..N-1 once each so they write their own K/V. Exact,
    #                      and the reference this mode is calibrated against.
    #   none             — leave the cache zeroed.
    #
    # Random is the default on the argument that ring cost depends on the cache holding
    # *something* rather than on what specifically. That argument is checkable rather than
    # assumed: summary.csv puts the replay-filled timing (timings_measured_ms) beside the
    # profiled one for the same chunk, so every profiled cell is its own A/B. At chunk 32 the
    # two reference points are 21.01ms replay-filled and 17.95ms zeroed; random landing near
    # 17.95 would mean the difference was never the data at all — back-to-back replay heating
    # being the other candidate — and that this mode measures the wrong thing.
    kv_fill = os.environ.get("GEMMA4_PERF_KV_FILL", "random").strip().lower()
    assert kv_fill in ("replay", "random", "none"), f"GEMMA4_PERF_KV_FILL must be replay|random|none, got {kv_fill!r}"

    def _randomize_kv_cache():
        """Fill each measured layer's K/V cache with random values, in place.

        In place because a captured trace holds the addresses it saw; allocating a fresh
        cache tensor here would leave the trace reading the old one.
        """
        for lt in layer_types:
            cache = model.tt_kv_cache[layer_idxs[lt]]
            for tensor in cache:
                host = ttnn.from_torch(
                    # Small values, not unit-scale: the cache holds post-projection K/V, and a
                    # wildly out-of-range prefix could push SDPA into saturation and change what
                    # is being measured for a second, unrelated reason.
                    0.1 * torch.randn(list(tensor.shape), dtype=torch.float32),
                    dtype=tensor.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                ttnn.copy_host_to_device_tensor(host, tensor)
        ttnn.synchronize_device(mesh_device)

    if kv_fill == "random":
        t0 = time.time()
        _randomize_kv_cache()
        logger.info(
            f"[layer_perf_chunk] GEMMA4_PERF_KV_FILL=random — wrote random K/V into "
            f"{len(layer_types)} layer cache(s) in {time.time() - t0:.1f}s, no fill replays"
        )
    elif kv_fill == "none":
        logger.info("[layer_perf_chunk] GEMMA4_PERF_KV_FILL=none — cache stays zeroed")
    elif n_fill:
        logger.info(
            f"[layer_perf_chunk] GEMMA4_PERF_KV_FILL=replay — replaying {n_fill} unmeasured "
            f"chunk(s) below/between the requested ones so each measured chunk sees a real prefix"
        )

    # replay mode walks every chunk from 0; the other modes visit only what is measured.
    replay_order = range(fill_upto + 1) if kv_fill == "replay" else sorted(measured_set)

    results = []
    try:
        for idx in replay_order:
            for lt in layer_types:
                if idx not in measured_set:
                    # Fill only: one unsignposted replay to write this chunk's K/V.
                    _stage(idx)
                    ttnn.execute_trace(mesh_device, traces[lt], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    continue

                tag = _perf_layer_tag(lt)
                sp_start, sp_stop = _perf_signposts(lt, idx)

                # Warm replays, deliberately outside the signposts: the first replay after a
                # capture still pays one-off dispatch setup.
                warm = []
                for _ in range(warm_iters):
                    _stage(idx)
                    t_i = time.time()
                    ttnn.execute_trace(mesh_device, traces[lt], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    warm.append(time.time() - t_i)

                # THE measured run: exactly one replay between the signposts.
                chunk_start = _stage(idx)
                signpost(sp_start)
                t_i = time.time()
                ttnn.execute_trace(mesh_device, traces[lt], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                measured_s = time.time() - t_i
                signpost(sp_stop)

                best_warm = min(warm) if warm else measured_s
                noisy = bool(warm) and not (min(warm) * 0.8 <= measured_s <= max(warm) * 1.25)
                results.append(
                    {
                        "chunk_idx": idx,
                        "layer_type": lt,
                        "tag": tag,
                        "layer_idx": layer_idxs[lt],
                        "chunk_start": chunk_start,
                        "measured_ms": measured_s * 1000,
                        "warm_best_ms": best_warm * 1000,
                        "warm_worst_ms": (max(warm) if warm else measured_s) * 1000,
                        "noisy": noisy,
                        "start_signpost": sp_start,
                        "stop_signpost": sp_stop,
                    }
                )
                # One machine-readable line per cell, so the sweep script can report timings
                # without waiting on tt-perf-report.
                logger.info(
                    f"[layer_perf_chunk] RESULT type={tag} chunk={idx} ring_depth={idx} "
                    f"kv_actual_global={chunk_start} measured_ms={measured_s * 1000:.2f} "
                    f"tok_s={chunk / measured_s:.0f} warm_best_ms={best_warm * 1000:.2f} "
                    f"warm_worst_ms={(max(warm) if warm else measured_s) * 1000:.2f} "
                    f"noisy={int(noisy)} signposts={sp_start},{sp_stop}"
                )
                if noisy:
                    logger.warning(
                        f"[layer_perf_chunk] {tag} chunk={idx} measured {measured_s * 1000:.2f}ms is outside "
                        f"the warm spread [{min(warm) * 1000:.2f}, {max(warm) * 1000:.2f}]ms — treat as noisy"
                    )

        # Finiteness on the last cell only. Reading back every cell would add an eager
        # all-gather between replays, which is real host work in the middle of a perf sweep
        # (and, per the canonical test's notes, incidentally masks the ring deadlock this
        # ordering is meant to exercise honestly).
        hidden = _cp_gather_torch(outs[layer_types[-1]], mesh_device, mesh_config)
    finally:
        for tid in traces.values():
            ttnn.release_trace(mesh_device, tid)

    assert torch.isfinite(hidden).all(), f"{layer_types[-1]} layer produced non-finite output"
    assert float(hidden.std()) > 0.001, f"{layer_types[-1]} layer output is degenerate"

    # Depth curve, and the whole-model extrapolation that makes it actionable. 31B is
    # 50 sliding + 10 global.
    n_sliding, n_global = 50, 10
    by_type = {}
    for r in results:
        by_type.setdefault(r["tag"], []).append(r)
    for tag, rows in by_type.items():
        span = (
            f"{rows[0]['measured_ms']:.2f}ms @chunk{rows[0]['chunk_idx']} -> "
            f"{rows[-1]['measured_ms']:.2f}ms @chunk{rows[-1]['chunk_idx']} "
            f"({rows[-1]['measured_ms'] / rows[0]['measured_ms']:.2f}x)"
            if len(rows) > 1
            else f"{rows[0]['measured_ms']:.2f}ms @chunk{rows[0]['chunk_idx']}"
        )
        logger.info(f"[layer_perf_chunk] {tag} depth curve: {span}")
    if len(by_type) == 2:
        for idx in chunk_idxs:
            g = next((r for r in results if r["chunk_idx"] == idx and r["tag"] == "global"), None)
            s = next((r for r in results if r["chunk_idx"] == idx and r["tag"] == "sliding"), None)
            if g and s:
                est = n_global * g["measured_ms"] + n_sliding * s["measured_ms"]
                logger.info(
                    f"[layer_perf_chunk] ESTIMATE chunk={idx} "
                    f"{n_global}x global({g['measured_ms']:.2f}ms) + {n_sliding}x sliding({s['measured_ms']:.2f}ms) "
                    f"= {est:.0f}ms of a 60-layer chunk (excludes embedding/head and the "
                    f"inter-layer CCL not in a single-layer graph)"
                )
