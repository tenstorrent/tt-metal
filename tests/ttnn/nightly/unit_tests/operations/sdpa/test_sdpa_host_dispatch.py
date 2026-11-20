# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cache-hit host dispatch time of the prefill SDPA ops (untraced).

Each case calls the op once to build and cache its program, then times calls that hit the program cache. The device is
synchronized before every timed call, so the time is the host's work to dispatch one call (argument checks, program
cache lookup, runtime-argument patching, any host-side tensor ops) and never waits on a full command queue.

Results are logged and appended as CSV rows to $SDPA_HOST_BENCH_OUT (default: no file). The test only asserts that the
op runs; it does not compare against a target.
"""

import os
import statistics
import time

import pytest
import torch
import ttnn
from loguru import logger

ITERS = 50
# Back-to-back bursts: BURST calls enqueued without waiting (the queue never fills at this size), timed together. The
# per-call mean of a burst is less sensitive to the host's wake-up jitter after a device synchronize than one call.
BURST, BURSTS = 16, 20


def _tt(x, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _time_calls(device, fn):
    out = fn()  # program-cache miss
    ttnn.synchronize_device(device)
    _deallocate(out)
    times = []
    for _ in range(ITERS):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
        ttnn.synchronize_device(device)
        _deallocate(out)
    bursts = []
    for _ in range(BURSTS):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(BURST)]
        t1 = time.perf_counter()
        bursts.append((t1 - t0) * 1e6 / BURST)
        ttnn.synchronize_device(device)
        for out in outs:
            _deallocate(out)
    return statistics.median(times), min(times), statistics.median(bursts)


def _deallocate(out):
    for t in out if isinstance(out, (tuple, list)) else [out]:
        t.deallocate(True)


def _record(name, median_us, min_us, burst_us):
    logger.info(
        f"SDPA host dispatch {name}: median {median_us:.1f} us, min {min_us:.1f} us, burst {burst_us:.1f} us per call"
    )
    path = os.environ.get("SDPA_HOST_BENCH_OUT")
    if path:
        with open(path, "a") as f:
            f.write(f"{name},{median_us:.1f},{min_us:.1f},{burst_us:.1f}\n")


def _config(device, q_chunk, k_chunk):
    grid = device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=k_chunk)


# name, b, nh, nkv, s, d, causal, mask, q_chunk, k_chunk
DENSE_CASES = [
    ("encoder_s128_nomask", 1, 16, 16, 128, 64, False, False, 128, 128),
    ("encoder_s128_mask", 1, 16, 16, 128, 64, False, True, 128, 128),
    ("encoder_s512_mask", 2, 16, 16, 512, 64, False, True, 128, 128),
    ("tiny_s32_mask", 1, 8, 8, 32, 64, False, True, 32, 32),
    ("llm_gqa_causal_s2048", 1, 32, 8, 2048, 128, True, False, 128, 128),
]


@pytest.mark.parametrize(
    "name, b, nh, nkv, s, d, causal, mask, q_chunk, k_chunk", DENSE_CASES, ids=[c[0] for c in DENSE_CASES]
)
def test_sdpa_host_dispatch(device, name, b, nh, nkv, s, d, causal, mask, q_chunk, k_chunk):
    torch.manual_seed(0)
    q = _tt(torch.randn(b, nh, s, d), device)
    k = _tt(torch.randn(b, nkv, s, d), device)
    v = _tt(torch.randn(b, nkv, s, d), device)
    attn_mask = _tt(torch.zeros(b, 1, s, s), device) if mask else None
    config = _config(device, q_chunk, k_chunk)

    def call():
        return ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, program_config=config
        )

    _record(name, *_time_calls(device, call))


def test_joint_sdpa_host_dispatch(device):
    torch.manual_seed(0)
    b, nh, s, js, d = 1, 24, 1024, 128, 64
    q, k, v = (_tt(torch.randn(b, nh, s, d), device) for _ in range(3))
    jq, jk, jv = (_tt(torch.randn(b, nh, js, d), device) for _ in range(3))
    config = _config(device, 128, 128)

    def call():
        return ttnn.transformer.joint_scaled_dot_product_attention(
            q, k, v, jq, jk, jv, joint_strategy="rear", program_config=config
        )

    _record("joint_s1024_j128", *_time_calls(device, call))


def test_ring_distributed_sdpa_host_dispatch(device):
    torch.manual_seed(0)
    b, nh, nkv, s, d, ring_size = 1, 8, 1, 2048, 128, 4
    q = _tt(torch.randn(b, nh, s, d), device)
    k = _tt(torch.randn(b, nkv, s, d), device)
    v = _tt(torch.randn(b, nkv, s, d), device)
    config = _config(device, 128, 128)

    def call():
        return ttnn.transformer.ring_distributed_scaled_dot_product_attention(
            q, k, v, ring_size=ring_size, ring_id=1, program_config=config
        )

    _record("ring_distributed_s2048_r4", *_time_calls(device, call))
