# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunked-prefill throughput for Llama-3.1-8B at production parallelism (SP=4 x TP=8).

Reports the same metric the other prefill models in this suite report, so the numbers are
comparable: median wall time over N repetitions of the whole chunk loop, converted to tok/s. See
``gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py`` and ``minimax_m3/tests/perf/profile_prefill.py``.

Measured, not asserted. There is no throughput target for this model yet -- tt-blaze#4137 and its
component issues carry accuracy criteria only -- so a floor here would be a number someone invented
rather than a requirement. This prints and logs; gating comes when a target exists.

What the measurement deliberately excludes, because none of it recurs per request in serving:
  * weight load and upload (once per process)
  * JIT warm-up -- ``compile()`` runs first, so every KV-length bucket the loop will hit is already
    compiled. Without it the first iteration pays kernel build time and reads as a 10x regression.
  * the KV read-back ``kv_cache_pcc_check`` does. That walks the whole cache to host in Python,
    undoing two shardings, and costs more than the prefill it validates -- which is exactly why the
    45s window in the #4150 accuracy run is not a perf number.

Opt-in on the real checkpoint, like the other real-weight cells, so it stays out of the module
stage and cannot break on a checkpoint move.
"""

import os
import statistics
import time

import pytest
from loguru import logger

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tests.mesh_profiles import galaxy_torus_xy_device_params
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache
from models.demos.llama_3p1_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

ITERS = int(os.environ.get("LLAMA31_PREFILL_PERF_ITERS", "5"))


@pytest.mark.skipif(
    os.environ.get("LLAMA31_8B_REAL_WEIGHTS") != "1",
    reason="opt-in: reads the real 16 GB checkpoint (set LLAMA31_8B_REAL_WEIGHTS=1)",
)
@pytest.mark.parametrize("chunk_size, max_seq_len", [(1024, 2048)], ids=["c1024-s2048"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-sp4-tp8-4x8")],
    indirect=["mesh_device", "device_params"],
)
def test_prefill_throughput(mesh_device, device_params, chunk_size, max_seq_len, reset_seeds):
    """Median tok/s over the full chunk loop, 32 layers, real weights, warm."""
    import torch

    from models.demos.llama_3p1_8b_d_p.reference.model import load_hf_state_dict

    rows, cols = mesh_device.shape
    config = TtPrefillRuntimeConfig(
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
        mesh_shape=(rows, cols),
        num_layers=Llama31_8BConfig.NUM_LAYERS,
        num_users=1,
        tp_axis=1,
        vocab_size=Llama31_8BConfig.VOCAB_SIZE,
    )
    runtime = TtPrefillRuntime(mesh_device=mesh_device, config=config, state_dict=load_hf_state_dict())
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=Llama31_8BConfig.NUM_LAYERS,
        max_seq_len=max_seq_len,
        sp_axis=config.sp_axis,
        num_users=1,
        chunk_size=chunk_size,
        num_kv_heads_per_chip=Llama31_8BConfig.NUM_KEY_VALUE_HEADS // config.tp_factor,
    )

    token_ids = torch.randint(0, Llama31_8BConfig.VOCAB_SIZE, (max_seq_len,)).tolist()
    n_chunks = max_seq_len // chunk_size

    # Warm every bucket the loop will hit, so no iteration pays a first-run JIT cost.
    t0 = time.perf_counter()
    runtime.compile(kv_cache)
    logger.info(f"[perf] warm-up (compile) {(time.perf_counter() - t0):.1f}s")

    times = []
    for i in range(ITERS):
        t0 = time.perf_counter()
        # prefill_prompt ends in synchronize_device, so this wall time is device-complete.
        runtime.prefill_prompt(token_ids, kv_cache, slot_id=0)
        dt = time.perf_counter() - t0
        times.append(dt)
        logger.info(f"[perf] iter {i}: {dt * 1e3:.1f} ms  {max_seq_len / dt:.1f} tok/s")

    median = statistics.median(times)
    logger.info(
        f"[perf] PREFILL THROUGHPUT over {ITERS} iters: median {max_seq_len / median:.1f} tok/s "
        f"({median * 1e3:.1f} ms for {max_seq_len} tok in {n_chunks} chunk(s) of {chunk_size}), "
        f"best {max_seq_len / min(times):.1f} tok/s, mesh {rows}x{cols} "
        f"(sp={config.sp_factor} tp={config.tp_factor}), {Llama31_8BConfig.NUM_LAYERS} layers"
    )
    logger.info(
        f"[perf] per-chunk median {median / n_chunks * 1e3:.1f} ms ({chunk_size} tok, {chunk_size // config.sp_factor}/chip)"
    )

    assert median > 0
    ttnn.synchronize_device(mesh_device)


@pytest.mark.skipif(
    os.environ.get("LLAMA31_8B_REAL_WEIGHTS") != "1",
    reason="opt-in: reads the real 16 GB checkpoint (set LLAMA31_8B_REAL_WEIGHTS=1)",
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((4, 8), galaxy_torus_xy_device_params(), id="galaxy-sp4-tp8-4x8")],
    indirect=["mesh_device", "device_params"],
)
def test_prefill_scaling_with_context(mesh_device, device_params, reset_seeds):
    """How per-chunk cost grows as the KV history it attends over grows.

    The single-length number is the less interesting half of prefill performance. What decides
    whether long context is usable is the *shape* of the curve: each chunk attends over every token
    before it, so per-chunk cost has a term that grows with the prefix while the per-token weight
    cost stays flat. Sweeping prompt length against a fixed chunk separates the two -- flat
    per-chunk time means the fixed overhead still dominates, rising means the quadratic attention
    term has taken over, and the crossover is the number worth knowing.

    One process and one cache allocation for the whole sweep: the 16 GB load and the upload cost
    minutes each and would otherwise be paid per point for data that does not depend on them.
    """
    import torch

    from models.demos.llama_3p1_8b_d_p.reference.model import load_hf_state_dict

    chunk_size, max_seq_len = 1024, 8192
    rows, cols = mesh_device.shape
    config = TtPrefillRuntimeConfig(
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
        mesh_shape=(rows, cols),
        num_layers=Llama31_8BConfig.NUM_LAYERS,
        num_users=1,
        tp_axis=1,
        vocab_size=Llama31_8BConfig.VOCAB_SIZE,
    )
    runtime = TtPrefillRuntime(mesh_device=mesh_device, config=config, state_dict=load_hf_state_dict())
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=Llama31_8BConfig.NUM_LAYERS,
        max_seq_len=max_seq_len,
        sp_axis=config.sp_axis,
        num_users=1,
        chunk_size=chunk_size,
        num_kv_heads_per_chip=Llama31_8BConfig.NUM_KEY_VALUE_HEADS // config.tp_factor,
    )
    runtime.compile(kv_cache)

    token_ids = torch.randint(0, Llama31_8BConfig.VOCAB_SIZE, (max_seq_len,)).tolist()
    rows_out = []
    for n_tokens in [1024, 2048, 4096, 8192]:
        samples = []
        for _ in range(3):
            t0 = time.perf_counter()
            runtime.prefill_prompt(token_ids[:n_tokens], kv_cache, slot_id=0)
            samples.append(time.perf_counter() - t0)
        dt = statistics.median(samples)
        n_chunks = n_tokens // chunk_size
        rows_out.append((n_tokens, dt, n_tokens / dt, dt / n_chunks))
        logger.info(
            f"[perf] {n_tokens:>5} tok ({n_chunks} chunk(s)): {dt * 1e3:7.1f} ms  "
            f"{n_tokens / dt:7.1f} tok/s  per-chunk {dt / n_chunks * 1e3:6.1f} ms"
        )

    logger.info("[perf] SCALING (chunk=1024, 32 layers, mesh 4x8, bf8 KV)")
    for n_tokens, dt, tps, per_chunk in rows_out:
        logger.info(
            f"[perf]   {n_tokens:>5} tok  {dt * 1e3:8.1f} ms  {tps:8.1f} tok/s  per-chunk {per_chunk * 1e3:6.1f} ms"
        )
    # Per-chunk cost should not *fall materially* as the prefix grows: a later chunk attends over
    # strictly more history than an earlier one, so a real drop means the cache-read is not seeing
    # what it should. It is deliberately not an equality or a monotonicity check -- measured
    # per-chunk time is flat to ~2% out to 8k (the growing KV read is negligible beside the fixed
    # per-chunk cost at this GQA ratio), so ordering within that band is jitter, and requiring
    # monotonicity just fails on noise. This catches a gross regression only; that the later chunks
    # genuinely attend over the earlier ones is established by the KV-PCC and continuation cells in
    # tests/unit/test_model_vs_ref.py, which compare against a reference rather than a stopwatch.
    per_chunks = [r[3] for r in rows_out]
    assert min(per_chunks) >= per_chunks[0] * 0.9, f"per-chunk cost fell materially as context grew: {per_chunks}"
    ttnn.synchronize_device(mesh_device)
