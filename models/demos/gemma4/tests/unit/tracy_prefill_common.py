# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared prefill trace profiling helpers (pplx_embed-style Tracy workflow).

Tracy records the full session (model load, trace capture, replays). Only trace
replay is treated as inference for optimization; ``start``/``stop`` signposts mark
the measured replay slice in ``ops_perf_results_*.csv``.
"""

import os
from contextlib import contextmanager

from loguru import logger

import ttnn
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.generator_trace import skip_gemma4_full_prefill_warmup
from models.tt_transformers.tt.generator import SUPPORTED_PREFILL_BATCH_SIZES

from .test_prefill_trace_parity import _build_tokens, _create_page_table, _page_params

try:
    from tracy import signpost

    _HAS_SIGNPOST = True
except ModuleNotFoundError:
    _HAS_SIGNPOST = False


def _sync_mesh(mesh_device):
    ttnn.synchronize_device(mesh_device)


def _flush_device_profiler(mesh_device):
    """Drain device profiler buffers so a later traced region does not overflow."""
    ttnn.ReadDeviceProfiler(mesh_device)
    _sync_mesh(mesh_device)


def _traced_prefill(generator, kv_cache, tokens, page_table, prompt_lens):
    out = generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        enable_trace=True,
        warmup_prefill=False,
    )
    _sync_mesh(generator.model[0].mesh_device)
    return out


@contextmanager
def _signpost_start_stop(emit: bool):
    if emit and _HAS_SIGNPOST:
        signpost("start")
    try:
        yield
    finally:
        if emit and _HAS_SIGNPOST:
            signpost("stop")


def build_prefill_trace_fixtures(batch_size, prefill_len, vocab_size):
    """Build tokens, page table, and seq-len limits for one (batch, ISL) bucket."""
    tokens, prompt_lens, kernel_len = _build_tokens(batch_size, prefill_len, vocab_size)
    max_new_tokens = 32
    max_seq_len = max(prefill_len + max_new_tokens, 4096)
    paged_cfg = _page_params(batch_size, prefill_len, max_new_tokens)
    page_table = _create_page_table(batch_size, paged_cfg)
    max_batch_size = next(b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size)
    return {
        "tokens": tokens,
        "prompt_lens": prompt_lens,
        "kernel_len": kernel_len,
        "paged_cfg": paged_cfg,
        "page_table": page_table,
        "max_seq_len": max_seq_len,
        "max_batch_size": max_batch_size,
    }


def load_prefill_trace_generator(mesh_device, model_path, fixtures):
    """Load Gemma4Generator for a single trace bucket (inside Tracy session)."""
    generator, kv_caches, _tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=fixtures["max_batch_size"],
        max_seq_len=fixtures["max_seq_len"],
        paged_attention_config=fixtures["paged_cfg"],
    )
    model_args = generator.model_args[0]
    kernel_len = fixtures["kernel_len"]
    assert model_args.can_enable_trace(
        kernel_len, batch_size=fixtures["max_batch_size"]
    ), f"Trace not enabled for kernel_len={kernel_len} batch_size={fixtures['max_batch_size']}"
    skip_gemma4_full_prefill_warmup(generator)
    return generator, kv_caches


def run_prefill_trace_capture(
    generator,
    kv_caches,
    fixtures,
):
    """First traced prefill: compiles kernels and captures the device trace."""
    return _traced_prefill(
        generator,
        kv_caches,
        fixtures["tokens"],
        fixtures["page_table"],
        fixtures["prompt_lens"],
    )


def run_prefill_trace_replay(
    generator,
    kv_caches,
    fixtures,
):
    """Replay an existing prefill trace (steady-state inference path)."""
    return _traced_prefill(
        generator,
        kv_caches,
        fixtures["tokens"],
        fixtures["page_table"],
        fixtures["prompt_lens"],
    )


def run_prefill_trace_capture_and_replays(
    generator,
    kv_caches,
    fixtures,
    mesh_device,
    *,
    emit_signposts: bool = False,
    include_measured_replay: bool | None = None,
):
    """Capture trace, warm replay once, then optional signposted measured replay."""
    if include_measured_replay is None:
        include_measured_replay = emit_signposts

    run_prefill_trace_capture(generator, kv_caches, fixtures)
    run_prefill_trace_replay(generator, kv_caches, fixtures)

    if include_measured_replay:
        with _signpost_start_stop(emit_signposts):
            run_prefill_trace_replay(generator, kv_caches, fixtures)

    _flush_device_profiler(mesh_device)


def run_prefill_trace_tracy_session(
    mesh_device,
    model_path,
    batch_size,
    prefill_len,
    vocab_size,
    *,
    emit_signposts: bool = True,
):
    """Full pplx-style Tracy session: load, capture, warm replay, signposted replay."""
    fixtures = build_prefill_trace_fixtures(batch_size, prefill_len, vocab_size)
    mesh_key = "x".join(str(d) for d in mesh_device.shape)
    logger.info(
        "Prefill trace session: model={} mesh={} batch={} prompt_len={} kernel_len={} signposts={}",
        os.path.basename(model_path.rstrip("/")),
        mesh_key,
        batch_size,
        prefill_len,
        fixtures["kernel_len"],
        emit_signposts,
    )

    generator, kv_caches = load_prefill_trace_generator(mesh_device, model_path, fixtures)
    run_prefill_trace_capture_and_replays(
        generator,
        kv_caches,
        fixtures,
        mesh_device,
        emit_signposts=emit_signposts,
    )
