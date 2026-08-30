# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE (32 chips, BH Galaxy): steady-state perf harness for the prefill attention block.

Drives ``tt/attention/prefill.py::attention_forward`` in a loop and does **nothing else** — no torch
reference, no PCC. That is deliberate: the reference is what makes the correctness tests unusable as
perf vehicles, because a 96-head float reference at 8K tokens is ~25 GB of host scores. Correctness
lives in ``tests/unit/test_attention_vs_ref.py``; this file only measures.

Run it under the light profiler (op-level, not per-line)::

    python3 -m tracy -p -r -o <out> --check-exit-code -a device_kernel_duration -t 5000 \\
        -m "pytest models/demos/mistral_medium_d_p/tests/test_attention_perf.py -k c640"

which drops ``ops_perf_results_*.csv`` under ``generated/profiler/reports/``. Signposts bracket the
measured region so warmup and setup can be sliced off.

**chunk_local is the knob.** One ``attention_forward`` call consumes one prefill chunk;
``chunk_local`` is that chunk's rows PER CHIP and ``chunk_global = sp * chunk_local`` is what the
whole mesh consumes per call. At SP=8:

    chunk_local  128 ->  1024 global   the correctness tests' shape; one q_chunk per chip, so this
                                       is the overhead floor, not a throughput number
    chunk_local  640 ->  5120 global   PRODUCTION: prefill_producer_manifest.example.yaml sets
                                       chunk_size: 5120 as the runner default
    chunk_local 1024 ->  8192 global   is cost still amortising, or flat?

**cached_len matters as much.** The ring gathers KV over ``[0, cached_len + chunk_global)``, so a
chunk late in a long prompt moves far more data than chunk 0. Measuring only chunk 0 would flatter
the ring badly, so the sweep carries a steady-state point at the production chunk size.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.mistral_medium_d_p.tt.attention import allocate_kv_cache
from models.demos.mistral_medium_d_p.tt.rope import build_indexed_rope

from .test_factory import mesh_setup, parametrize_mesh_with_fabric
from .unit.shapes import HEAD_DIM, HIDDEN, YARN, per_chip
from .unit.test_attention_vs_ref import _build_attention, _chunk_order, _place_sp, _random_attn_weights

try:
    from tracy import signpost
except ImportError:  # harness must still run outside the profiler

    def signpost(header, message=None):
        pass


WARMUP_ITERS = 2
MEASURED_ITERS = 8

# (chunk_local, prefix_chunks) — prefix_chunks sets cached_len = prefix_chunks * chunk_global.
PERF_CASES = [
    (128, 0),
    (640, 0),
    (1024, 0),
    (640, 4),  # steady state: 5th chunk of a 5120-token-per-chunk prefill, 20480 already cached
    (1024, 4),  # same prefix depth at the larger chunk, to check the per-token win survives it
]


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("chunk_local, prefix_chunks", PERF_CASES, ids=[f"c{c}_p{p}" for c, p in PERF_CASES])
def test_attention_prefill_perf(mesh_device, device_params, chunk_local, prefix_chunks, reset_seeds):
    """Loop one ``attention_forward`` at a fixed (chunk_local, cached_len) and signpost the steady state."""
    torch.manual_seed(0)
    mesh_config, ccl = mesh_setup(mesh_device)
    sp, tp = mesh_config.sp, mesh_config.tp

    chunk_global = sp * chunk_local
    cached_len = prefix_chunks * chunk_global
    # Room for the prefix, the measured chunk, and one spare so the cache-backed ring path is taken
    # (an exactly-full cache would trip the one-shot all-gather bootstrap instead).
    cache_global = (prefix_chunks + 2) * chunk_global

    w = _random_attn_weights(seed=5)
    attn = _build_attention(mesh_device, mesh_config, ccl, w, cache_global)
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=cache_global,
        sp_axis=mesh_config.sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        n_kv_local=per_chip(tp)["n_kv"],
    )
    rope_mats = build_indexed_rope(
        mesh_device,
        head_dim=HEAD_DIM,
        max_seq_len=cache_global,
        chunk_size=chunk_global,
        sp_axis=mesh_config.sp_axis,
        **YARN,
    )

    def make_input(seed_offset=0):
        torch.manual_seed(100 + seed_offset)
        x = torch.randn(1, 1, chunk_global, HIDDEN) * 0.1
        idx, _ = _chunk_order(cached_len, sp, chunk_local)
        return _place_sp(x, mesh_device, mesh_config, idx)

    # Fill the prefix so cached_len is genuinely backed by cache contents, not just an integer.
    for c in range(prefix_chunks):
        pre_cached = c * chunk_global
        torch.manual_seed(200 + c)
        xp = torch.randn(1, 1, chunk_global, HIDDEN) * 0.1
        idx_p, _ = _chunk_order(pre_cached, sp, chunk_local)
        attn(
            _place_sp(xp, mesh_device, mesh_config, idx_p),
            rope_mats=rope_mats,
            kv_cache=kv_cache,
            cached_len=pre_cached,
            indexed_rope=True,
        )
    ttnn.synchronize_device(mesh_device)

    # attention_forward deallocates its input, so each iteration needs its own tensor. Build them all up
    # front: allocating inside the timed loop would put host->device writes in the measured region.
    inputs = [make_input(i) for i in range(WARMUP_ITERS + MEASURED_ITERS)]

    for i in range(WARMUP_ITERS):
        out = attn(inputs[i], rope_mats=rope_mats, kv_cache=kv_cache, cached_len=cached_len, indexed_rope=True)
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh_device)

    signpost(header="start")
    for i in range(WARMUP_ITERS, WARMUP_ITERS + MEASURED_ITERS):
        out = attn(inputs[i], rope_mats=rope_mats, kv_cache=kv_cache, cached_len=cached_len, indexed_rope=True)
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh_device)
    signpost(header="stop")

    logger.info(
        f"PERF chunk_local={chunk_local} chunk_global={chunk_global} cached_len={cached_len} "
        f"cache_global={cache_global} SP={sp} TP={tp} iters={MEASURED_ITERS}"
    )
