# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill performance harness: steady-state timings for the 8B prefill.

Unlike ``galaxy_prefill_kv_pcc.py`` this runs NO CPU golden and NO PCC — it exists purely to time
the prefill forward pass:

  * a WARMUP prefill per configuration, so JIT kernel builds and program-cache misses land OUTSIDE
    the measured region — a cold prefill and a warm one are different workloads, and only the warm
    one is the number anybody ships;
  * then ``PREFILL_PERF_ITERS`` measured prefills, bracketed by tracy signposts
    (``prefill_<chunk>_start`` / ``_end``) so ``tt-perf-report --start-signpost`` can slice the
    device ops down to one configuration's prefill.

Because ``TtPrefillRuntime`` can serve several chunk sizes from one build, the whole one-shot vs
chunked sweep runs in a SINGLE process — one checkpoint load and one device bring-up for every
configuration, which also makes the configurations directly comparable.

Plain wall-clock (no profiler — this is the honest end-to-end number)::

    pytest models/demos/llama3_1_8b_d_p/tests/prefill_perf.py -k 8x4 -s

Under the profiler (device op breakdown; adds per-op overhead, so ignore its wall-clock)::

    python -m tracy -r -p -v -m pytest models/demos/llama3_1_8b_d_p/tests/prefill_perf.py -k 8x4

NOTE ON PROFILING COST: post-processing holds the whole tracy ops-times CSV in a pandas frame, and
that file grows ~7 GB per profiled prefill. Keep the profiled configuration count x (warmup+iters)
low — a sweep that is fine unprofiled will OOM the post-processing step.

===============================  ==========================================================
``HF_MODEL``                     checkpoint dir (required; the test skips without it)
``PREFILL_PERF_SEQ_LEN``         prompt length (default 2048)
``PREFILL_PERF_CHUNK_SIZES``     comma-separated chunk sizes to sweep; ``0`` means one-shot
                                 (a single chunk of the whole prompt). Default ``0,1024,512``
``PREFILL_PERF_LAYERS``          layer count (default: the model's 32)
``PREFILL_PERF_ITERS``           measured prefills per configuration (default 2)
``PREFILL_PERF_WARMUP``          warmup prefills per configuration (default 1)
===============================  ==========================================================
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_1_8b_d_p.tt.model_config import ModelArgs
from models.demos.llama3_1_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
from tracy import signpost

from .test_factory import make_mesh_config, parametrize_mesh_with_fabric


def _checkpoint_dir():
    path = os.getenv("HF_MODEL")
    if not path or not os.path.isdir(path):
        pytest.skip("set HF_MODEL to a downloaded Llama-3.1-8B-Instruct checkpoint to run the prefill perf harness")
    if not any(f.endswith(".safetensors") for f in os.listdir(path)):
        pytest.skip(f"HF_MODEL={path} has no *.safetensors (config-only dir)")
    return path


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.timeout(0)
def test_prefill_perf(mesh_device, device_params, reset_seeds):
    """Warm, signposted prefill sweep on the target mesh — timings only, no correctness check."""
    checkpoint = _checkpoint_dir()
    seq_len = int(os.getenv("PREFILL_PERF_SEQ_LEN", "2048"))
    iters = int(os.getenv("PREFILL_PERF_ITERS", "2"))
    warmups = int(os.getenv("PREFILL_PERF_WARMUP", "1"))

    # "0" is spelled one-shot: a single chunk covering the whole prompt.
    raw = os.getenv("PREFILL_PERF_CHUNK_SIZES", "0,1024,512")
    chunk_sizes = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        c = int(tok) or seq_len
        if c not in chunk_sizes:
            chunk_sizes.append(c)

    mesh_config = make_mesh_config(mesh_device)
    sp, tp = mesh_config.sp, mesh_config.tp

    model_args = ModelArgs(mesh_device=mesh_device, max_seq_len=seq_len)
    cfg = model_args.hf_config
    num_layers = int(os.getenv("PREFILL_PERF_LAYERS", str(cfg.num_hidden_layers)))

    for c in chunk_sizes:
        assert seq_len % c == 0, f"seq_len {seq_len} must be a multiple of chunk_size {c}"
        assert c % (ttnn.TILE_SIZE * sp) == 0, f"chunk_size {c} must be a multiple of {32 * sp}"

    logger.info(
        f"prefill perf: checkpoint={checkpoint} seq_len={seq_len} layers={num_layers} "
        f"mesh={sp}x{tp} chunk_sizes={chunk_sizes} warmup={warmups} iters={iters}"
    )

    # Device-layout weights only — this harness never builds the torch reference.
    logger.info("Loading checkpoint (Meta layout, for the device)...")
    device_state = ModelArgs.load_state_dict(checkpoint, convert_to_meta_format=True)

    # One runtime serves every chunk size in the sweep: the largest is the default and the rest
    # are declared so their indexed-RoPE tables get built up front.
    ordered = sorted(chunk_sizes, reverse=True)
    runtime = TtPrefillRuntime(
        mesh_device=mesh_device,
        hf_config=cfg,
        state_dict=device_state,
        config=TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=seq_len,
            mesh_shape=(sp, tp),
            default_chunk_size=ordered[0],
            additional_chunk_sizes=tuple(ordered[1:]),
            num_users=1,
            sp_axis=mesh_config.sp_axis,
            tp_axis=mesh_config.tp_axis,
            topology=ttnn.Topology.Linear,
            attn_weight_dtype=ttnn.bfloat16,
            mlp_weight_dtype=ttnn.bfloat16,
            owns_kv_cache=True,
        ),
    )
    del device_state

    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, cfg.vocab_size, (seq_len,), generator=g).tolist()

    def one_prefill(chunk_size):
        for c in range(seq_len // chunk_size):
            start = c * chunk_size
            runtime.prefill_chunk(
                runtime.make_chunk_input(tokens[start : start + chunk_size], chunk_size),
                slot_id=0,
                actual_start=start,
                actual_end=start + chunk_size,
                chunk_size=chunk_size,
            )
        ttnn.synchronize_device(mesh_device)

    summary = []
    for chunk_size in chunk_sizes:
        num_chunks = seq_len // chunk_size
        mode = "one-shot" if num_chunks == 1 else f"chunked x{num_chunks}"
        label = f"{mode} (chunk={chunk_size})"

        # --- warmup: JIT builds + program-cache misses land here, outside the signposts ---
        cold = None
        for w in range(warmups):
            t0 = time.perf_counter()
            one_prefill(chunk_size)
            dt = time.perf_counter() - t0
            cold = dt if cold is None else cold
            logger.info(f"[{label}] warmup {w}: {dt * 1e3:.1f} ms")

        # --- measured region ---
        latencies = []
        signpost(header=f"prefill_{chunk_size}_start")
        for i in range(iters):
            t0 = time.perf_counter()
            one_prefill(chunk_size)
            dt = time.perf_counter() - t0
            latencies.append(dt)
            logger.info(f"[{label}] iter {i}: {dt * 1e3:.1f} ms  ({seq_len / dt:,.0f} tok/s)")
        signpost(header=f"prefill_{chunk_size}_end")

        best = min(latencies)
        avg = sum(latencies) / len(latencies)
        summary.append((label, num_chunks, best, avg, cold))
        # cold is None when warmups=0 — the report run does that deliberately.
        cold_txt = f"{cold * 1e3:.1f} ms" if cold is not None else "n/a"
        logger.info(
            f"[{label}] best={best * 1e3:.1f} ms ({seq_len / best:,.0f} tok/s)  "
            f"avg={avg * 1e3:.1f} ms  cold={cold_txt}"
        )

    logger.info(f"=== PREFILL PERF SUMMARY ({seq_len} tok, {num_layers} layers, {sp}x{tp}) ===")
    logger.info(f"{'config':<28} {'chunks':>7} {'best ms':>10} {'tok/s':>10} {'avg ms':>10} {'cold ms':>10}")
    for label, nchunks, best, avg, cold in summary:
        cold_col = f"{cold * 1e3:>10.1f}" if cold is not None else f"{'n/a':>10}"
        logger.info(
            f"{label:<28} {nchunks:>7} {best * 1e3:>10.1f} {seq_len / best:>10,.0f} " f"{avg * 1e3:>10.1f} {cold_col}"
        )
