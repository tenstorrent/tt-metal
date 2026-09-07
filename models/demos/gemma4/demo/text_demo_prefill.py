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

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*_args, **_kwargs):
        pass


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_DTYPE = ttnn.bfloat16
GEMMA4_SLIDING_WINDOW_TOKENS = 1024
PREFILL_CHUNK_SIZES = (4096, 8192, 16384, 32768)
LAYER_PERF_CONTEXT_LENGTHS = (262144,)
TRACE_REGION_SIZE = int(os.environ.get("GEMMA4_PREFILL_TRACE_REGION_SIZE", 256_000_000))


def _model_path():
    return os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", "google/gemma-4-31B-it")


def _load_full_weights():
    """True when the caller wants the full host state dict (cold-cache path)."""
    return os.environ.get("GEMMA4_PREFILL_LOAD_FULL_WEIGHTS", "0").lower() in ("1", "true", "yes")


# ── Weight loading from the tensor cache ──────────────────────────────────────


def _cache_root(model_path, mesh_shape):
    """Absolute path of the tensor cache directory for this model, dtype, and mesh."""
    args = Gemma4ModelArgs()
    args.model_cache_path = Gemma4ModelArgs.resolve_model_cache_path(model_path)
    return str(args.weight_cache_path(MODEL_DTYPE, mesh_shape=mesh_shape))


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
    """Create a CP sharding mapper for ``seq_dim``, or a replication mapper."""
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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return torch.tensor(tokenizer.encode(text), dtype=torch.int32).unsqueeze(0)


def _get_prefill_tokens(model_path, context_len, vocab_size, source="text"):
    """Return prefix-consistent token IDs from text or a fixed-seed random stream."""
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


# ── Eager / traced execution ──────────────────────────────────────────────────


@contextmanager
def _lm_head_deferred(model):
    """Temporarily skip the LM head and return post-norm hidden states."""
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
    """Create a CP prefill model with ring caches for one or more chunks."""
    mesh_config = _mesh_config(mesh_device)
    if mesh_config.prefill.sp <= 1:
        raise ValueError("This demo requires context parallel prefill")
    tp = mesh_device.shape[1]
    context_len = context_len or chunk
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", context_len))

    cache_root = _cache_root(model_path, mesh_device.shape)
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
        mesh_config=mesh_config,
        create_kv_cache=False,
        prefill_chunk_size=chunk,
    )
    logger.info(f"Model ready in {time.time() - t0:.1f}s")

    return model_args, model, kv_cache


# ── Traced long-context chunked prefill (production shape) ────────────────────


@torch.no_grad()
@parametrize_mesh_with_fabric([(8, 4), (4, 8)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize("token_source", ["text"], ids=lambda t: t)
@pytest.mark.parametrize("chunk_size", PREFILL_CHUNK_SIZES, ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("context_len", [32768, 65536, 131072, 262144], ids=lambda c: f"ctx_{c // 1024}k")
@pytest.mark.parametrize("readback_all", [True, False], ids=["readback_all", "readback_final"])
def test_prefill_long_context_traced(
    mesh_device, context_len, chunk_size, readback_all, token_source, reset_seeds, request
):
    """Measure all prefill chunks using one replayed ring-attention trace."""
    from models.demos.gemma4.tt.ccl import cp_degree

    chunk = chunk_size
    mesh_config = _mesh_config(mesh_device)
    cp = cp_degree(mesh_config)
    if cp <= 1:
        pytest.skip(f"targets CP>1; mesh {tuple(mesh_device.shape)} gives CP={cp}")
    if chunk < GEMMA4_SLIDING_WINDOW_TOKENS * cp:
        pytest.skip(
            f"chunk={chunk} gives a {chunk // cp}-token Q slab at CP={cp}, under the "
            f"{GEMMA4_SLIDING_WINDOW_TOKENS}-token sliding window; ring_joint needs "
            f"chunk >= window*cp = {GEMMA4_SLIDING_WINDOW_TOKENS * cp} (its halo is single-hop)"
        )
    if context_len % chunk != 0:
        pytest.skip(f"context_len={context_len} is not a whole number of {chunk}-token chunks")

    model_path = _model_path()
    n_chunks = context_len // chunk
    model_args, model, kv_cache = _build_prefill_model(
        mesh_device=mesh_device,
        model_path=model_path,
        chunk=chunk,
        context_len=context_len,
    )
    tokens_all = _get_prefill_tokens(model_path, context_len, model_args.vocab_size, token_source)

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
                device_input, None, None, None
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

    # ── Warm up: compile the graph that will be captured ──────────────────────
    t0 = time.time()
    out = _forward(_stage(0))
    ttnn.synchronize_device(mesh_device)
    out.deallocate(True)
    warmup_s = time.time() - t0

    # ── Capture ───────────────────────────────────────────────────────────────
    t0 = time.time()
    cap_start = _stage(0)
    tid_ring = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_ring = _forward(cap_start)
    ttnn.end_trace_capture(mesh_device, tid_ring, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    capture_s = time.time() - t0
    logger.info(f"[traced] warmup(compile)={warmup_s:.1f}s capture={capture_s:.1f}s for 1 trace")

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
            # Report per-chunk latency and cumulative device and wall time.
            logger.info(
                f"[traced_perf] chunk {chunk_idx + 1}/{n_chunks} [{chunk_start}, {chunk_start + chunk}) "
                f"device={per_chunk[-1] * 1000:.1f}ms ({chunk / per_chunk[-1]:.0f} tok/s) | "
                f"total device={sum(per_chunk):.1f}s wall={time.time() - t_run:.1f}s"
            )
        total_s = time.time() - t_run
    finally:
        ttnn.release_trace(mesh_device, tid_ring)

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
        f"| chunks mean={device_s / len(per_chunk) * 1000:.1f}ms "
        f"min={min(per_chunk) * 1000:.1f}ms max={max(per_chunk) * 1000:.1f}ms"
    )
    logger.info(
        f"[traced_perf] ring-depth cost: first={per_chunk[0] * 1000:.1f}ms -> last={per_chunk[-1] * 1000:.1f}ms "
        f"= {per_chunk[-1] / per_chunk[0]:.2f}x over {len(per_chunk) - 1} extra chunks of history"
    )


# ── Per-layer prefill timing ────────────────────────────────────────────────


def _perf_layer_tag(layer_type):
    return "sliding" if layer_type == "sliding_attention" else "global"


def _perf_signposts(layer_type, chunk_idx):
    """Return profiler signposts for one layer and chunk."""
    base = f"gemma4-layer-{_perf_layer_tag(layer_type)}-chunk{chunk_idx}"
    return f"{base}-start", f"{base}-stop"


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4), (4, 8)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize("token_source", ["text"], ids=lambda t: t)
@pytest.mark.parametrize("context_len", LAYER_PERF_CONTEXT_LENGTHS, ids=lambda c: f"ctx_{c // 1024}k")
@pytest.mark.parametrize("chunk_size", PREFILL_CHUNK_SIZES, ids=lambda c: f"sz{c}")
@pytest.mark.parametrize(
    "layer_type", ["full_attention", "sliding_attention", "both"], ids=["global", "sliding", "both"]
)
@pytest.mark.parametrize(
    "chunk_idx",
    [*range(max(LAYER_PERF_CONTEXT_LENGTHS) // min(PREFILL_CHUNK_SIZES)), "all"],
    ids=lambda c: f"chunk{c}",
)
def test_prefill_layer_perf_chunk_n(
    mesh_device, chunk_idx, layer_type, chunk_size, context_len, token_source, reset_seeds, request
):
    """Measure selected layer/chunk pairs with one trace per layer type.

    Each layer is compiled and captured once, then each selected chunk is measured once.
    GEMMA4_PERF_KV_FILL selects random, replay-filled, or zeroed cache contents.
    Inputs are token embeddings, so this is an isolated-layer benchmark.
    """
    from models.demos.gemma4.tt.attention.global_kv_cache import pack_global_rope_device, pack_sliding_rope_device
    from models.demos.gemma4.tt.attention.ring_prefill import PackedRingKVCache
    from models.demos.gemma4.tt.ccl import cp_degree

    chunk = chunk_size
    mesh_config = _mesh_config(mesh_device)
    cp = cp_degree(mesh_config)
    if cp <= 1:
        pytest.skip(f"targets CP>1; mesh {tuple(mesh_device.shape)} gives CP={cp}")
    if chunk < GEMMA4_SLIDING_WINDOW_TOKENS * cp:
        pytest.skip(
            f"chunk {chunk} / CP {cp} = {chunk // cp} tokens per rank, below the "
            f"{GEMMA4_SLIDING_WINDOW_TOKENS}-token sliding window; ring_joint needs "
            f"chunk >= window*cp = {GEMMA4_SLIDING_WINDOW_TOKENS * cp}"
        )
    assert context_len % chunk == 0, "context_len must be a whole number of chunks"
    n_chunks = context_len // chunk
    if chunk_idx != "all" and not 0 <= int(chunk_idx) < n_chunks:
        pytest.skip(f"chunk {chunk_idx} is outside the {n_chunks} chunks of {chunk} in {context_len} tokens")

    chunk_idxs = list(range(n_chunks)) if chunk_idx == "all" else [int(chunk_idx)]
    layer_types = ["full_attention", "sliding_attention"] if layer_type == "both" else [layer_type]

    model_path = _model_path()
    text_config = _hf_text_config(model_path)
    model_args, model, _ = _build_prefill_model(
        mesh_device=mesh_device,
        model_path=model_path,
        chunk=chunk,
        context_len=context_len,
    )
    tokens_all = _get_prefill_tokens(model_path, context_len, model_args.vocab_size, token_source)

    layer_idxs = {lt: find_layer_idx(text_config, lt) for lt in layer_types}
    type_desc = ", ".join(f"{_perf_layer_tag(lt)}=layer{layer_idxs[lt]}" for lt in layer_types)
    logger.info(
        f"[layer_perf_chunk] ctx={context_len} chunk={chunk} n_chunks={n_chunks} cp={cp} | "
        f"cells={len(chunk_idxs) * len(layer_types)} chunks={chunk_idxs[0]}..{chunk_idxs[-1]} "
        f"types=({type_desc})"
    )

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

    def _stage(idx):
        """Refresh tokens, ring metadata, semaphores, and RoPE positions before replay."""
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
        # Reset persistent semaphores before each replay.
        for semaphore in model.ccl_manager.ring_attention_ccl_semaphore_handles:
            ttnn.reset_global_semaphore_value(semaphore, 0)
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
        """Build a layer forward with RoPE gathered inside the trace."""
        idx = layer_idxs[lt]
        layer = model.layers[idx]
        assert layer.self_attn.ring_kv_cache is not None, f"layer {idx} has no ring cache"
        assert idx not in model.kv_shared_layer_map, (
            f"layer {idx} ({lt}) shares KV from layer {model.kv_shared_layer_map[idx]}; "
            f"timing it standalone would omit the K/V projection and cache write"
        )
        assert lt in model.rope_caches_2d, (
            f"model has no 2D RoPE cache for {lt} (built without _hf_text_config?) — "
            f"per-chunk RoPE would be wrong, refusing to measure"
        )
        cos_2d, sin_2d = model.rope_caches_2d[lt]
        pack_rope = pack_global_rope_device if lt == "full_attention" else pack_sliding_rope_device

        def forward(chunk_start):
            embeds, _, _, _ = model.transform_and_embed_prefill_inputs_device(device_input, None, None, None)
            cos = ttnn.unsqueeze_to_4D(ttnn.embedding(model._rope_prefill_positions, cos_2d, layout=ttnn.TILE_LAYOUT))
            sin = ttnn.unsqueeze_to_4D(ttnn.embedding(model._rope_prefill_positions, sin_2d, layout=ttnn.TILE_LAYOUT))
            packed_rope = (*pack_rope(cos, sin), model._packed_global_rope_trans_mat)
            return layer(
                hidden_states=embeds,
                rope_mats=(cos, sin),
                position_idx=None,
                page_table=None,
                kv_cache=None,
                is_decode=False,
                batch_size=1,
                user_id=0,
                chunk_start_idx=chunk_start,
                packed_global_rope=packed_rope if lt == "full_attention" else None,
                packed_sliding_rope=packed_rope if lt == "sliding_attention" else None,
            )

        return forward

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
        cap_start = _stage(capture_at)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs[lt] = fwd(cap_start)
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        traces[lt] = tid
        capture_s = time.time() - t0
        logger.info(
            f"[layer_perf_chunk] {_perf_layer_tag(lt)} layer_idx={layer_idxs[lt]} "
            f"compile={compile_s:.1f}s capture={capture_s:.1f}s"
        )

    measured_set = set(chunk_idxs)
    fill_upto = max(chunk_idxs)
    n_fill = (fill_upto + 1) - len(measured_set)

    kv_fill = os.environ.get("GEMMA4_PERF_KV_FILL", "random").strip().lower()
    assert kv_fill in ("replay", "random", "none"), f"GEMMA4_PERF_KV_FILL must be replay|random|none, got {kv_fill!r}"

    def _initialize_ring_caches(randomize):
        """Initialize the ring-cache tensors in place, preserving captured addresses."""
        for lt in layer_types:
            cache = model.layers[layer_idxs[lt]].self_attn.ring_kv_cache
            tensors = (cache.kv,) if isinstance(cache, PackedRingKVCache) else cache
            for tensor in tensors:
                host = ttnn.from_torch(
                    0.1 * torch.randn(list(tensor.shape), dtype=torch.float32)
                    if randomize
                    else torch.zeros(list(tensor.shape), dtype=torch.float32),
                    dtype=tensor.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
                ttnn.copy_host_to_device_tensor(host, tensor)
        ttnn.synchronize_device(mesh_device)

    if kv_fill in ("random", "none"):
        t0 = time.time()
        _initialize_ring_caches(randomize=kv_fill == "random")
        logger.info(
            f"[layer_perf_chunk] GEMMA4_PERF_KV_FILL={kv_fill} — initialized "
            f"{len(layer_types)} ring cache(s) in {time.time() - t0:.1f}s"
        )
    elif n_fill:
        logger.info(
            f"[layer_perf_chunk] GEMMA4_PERF_KV_FILL=replay — replaying {n_fill} unmeasured "
            f"chunk(s) below/between the requested ones so each measured chunk sees a real prefix"
        )

    replay_order = range(fill_upto + 1) if kv_fill == "replay" else sorted(measured_set)

    results = []
    try:
        for idx in replay_order:
            for lt in layer_types:
                if idx not in measured_set:
                    _stage(idx)
                    ttnn.execute_trace(mesh_device, traces[lt], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    continue

                tag = _perf_layer_tag(lt)
                sp_start, sp_stop = _perf_signposts(lt, idx)

                chunk_start = _stage(idx)
                signpost(sp_start)
                t_i = time.time()
                ttnn.execute_trace(mesh_device, traces[lt], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)
                measured_s = time.time() - t_i
                signpost(sp_stop)

                results.append(
                    {
                        "chunk_idx": idx,
                        "layer_type": lt,
                        "tag": tag,
                        "layer_idx": layer_idxs[lt],
                        "chunk_start": chunk_start,
                        "measured_ms": measured_s * 1000,
                        "start_signpost": sp_start,
                        "stop_signpost": sp_stop,
                    }
                )
                logger.info(
                    f"[layer_perf_chunk] RESULT type={tag} chunk={idx} ring_depth={idx} "
                    f"kv_actual_global={chunk_start} measured_ms={measured_s * 1000:.2f} "
                    f"tok_s={chunk / measured_s:.0f} signposts={sp_start},{sp_stop}"
                )

        hidden = _cp_gather_torch(outs[layer_types[-1]], mesh_device, mesh_config)
    finally:
        for tid in traces.values():
            ttnn.release_trace(mesh_device, tid)

    assert torch.isfinite(hidden).all(), f"{layer_types[-1]} layer produced non-finite output"
    assert float(hidden.std()) > 0.001, f"{layer_types[-1]} layer output is degenerate"

    n_sliding = model_args.layer_types.count("sliding_attention")
    n_global = model_args.layer_types.count("full_attention")
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
                    f"= {est:.0f}ms of a {n_global + n_sliding}-layer chunk (excludes embedding/head and the "
                    f"inter-layer CCL not in a single-layer graph)"
                )
