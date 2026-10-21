# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""deepseek_v4_prefill_sdpa: sliding window plus selected compressed-cache rows, no mask (Blackhole only)."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from ttnn.operations.transformer_golden import MASKED_INDEX, deepseek_v4_prefill_sdpa_golden
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_test_utils import pcc, to_dev

pytestmark = pytest.mark.use_module_device


def make_inputs(H, S, K, W, T, TOPK, n_valid_fn, seed=0):
    """q, window_kv, cache_kv and cache_idx with each row's valid slots scattered among sentinels."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, S, K, generator=gen).to(torch.bfloat16)
    window_kv = torch.randn(1, 1, W + S, K, generator=gen).to(torch.bfloat16)
    cache_kv = torch.randn(1, 1, T, K, generator=gen).to(torch.bfloat16) if T else None
    cache_idx = None
    if TOPK:
        cache_idx = torch.full((1, 1, S, TOPK), MASKED_INDEX, dtype=torch.int64)
        for s in range(S):
            nv = min(TOPK, n_valid_fn(s))
            slots = torch.randperm(TOPK, generator=gen)[:nv]
            cache_idx[0, 0, s, slots] = torch.randperm(T, generator=gen)[:nv]
    return q, window_kv, cache_kv, cache_idx


def start_tensor(start, device):
    return to_dev(torch.tensor([[[[start]]]], dtype=torch.int32), device, ttnn.uint32)


def run(device, q, window_kv, cache_kv, cache_idx, start, W, k_chunk_size=128, sink=None, **kwargs):
    if cache_kv is not None:
        kwargs["cache_kv"] = to_dev(cache_kv, device, ttnn.bfloat16)
        kwargs["cache_idx"] = to_dev(cache_idx.to(torch.int32), device, ttnn.uint32)
    if sink is not None:
        kwargs["attention_sink"] = to_dev(sink, device, ttnn.bfloat16)
    out = ttnn.transformer.deepseek_v4_prefill_sdpa(
        to_dev(q, device, ttnn.bfloat16),
        to_dev(window_kv, device, ttnn.bfloat16),
        start_tensor(start, device),
        window_size=W,
        k_chunk_size=k_chunk_size,
        **kwargs,
    )
    return ttnn.to_torch(out)


def golden(q, window_kv, cache_kv, cache_idx, start, W, sink=None, **kwargs):
    return deepseek_v4_prefill_sdpa_golden(
        q.float(),
        window_kv.float(),
        start,
        cache_kv=None if cache_kv is None else cache_kv.float(),
        cache_idx=cache_idx,
        window_size=W,
        attention_sink=sink,
        **kwargs,
    )


@run_for_blackhole()
@pytest.mark.parametrize(
    "H, S, K, W, T, TOPK, start, k_chunk_size, n_valid",
    [
        # Window only: the prompt's first chunk (rows before position 0 skipped) and a later one.
        (32, 64, 128, 32, 0, 0, 0, 32, None),
        (32, 64, 128, 32, 0, 0, 256, 32, None),
        # First chunk with a cache whose rows are all sentinels.
        (32, 64, 128, 32, 256, 64, 0, 32, lambda s: 0),
        # Sentinels scattered through every row, partial last chunks.
        (32, 64, 128, 32, 256, 64, 512, 32, lambda s: (s * 7) % 65),
        # Start inside the window: some window rows are before position 0.
        (32, 64, 128, 64, 256, 64, 32, 64, lambda s: 1 + s % 64),
        # V4.1 geometry: head dim 512, window 128, top-512 entries, 128-key chunks.
        (32, 128, 512, 128, 4096, 512, 8192, 128, lambda s: 512 - (s * 3) % 512),
        # Head counts that are not a whole tile: V4.1's 16 local heads, and one spilling into a second tile.
        (16, 128, 512, 128, 4096, 512, 8192, 128, lambda s: 512 - (s * 3) % 512),
        (40, 64, 128, 32, 256, 64, 512, 32, lambda s: (s * 7) % 65),
    ],
    ids=[
        "window-first-chunk",
        "window-later-chunk",
        "empty-cache-rows",
        "scattered-sentinels",
        "start-inside-window",
        "v41-geometry",
        "v41-geometry-16-heads",
        "40-heads",
    ],
)
def test_deepseek_v4_prefill_sdpa(device, H, S, K, W, T, TOPK, start, k_chunk_size, n_valid):
    q, window_kv, cache_kv, cache_idx = make_inputs(H, S, K, W, T, TOPK, n_valid or (lambda s: 0), seed=start + S)
    out = run(device, q, window_kv, cache_kv, cache_idx, start, W, k_chunk_size)
    ref = golden(q, window_kv, cache_kv, cache_idx, start, W)
    assert tuple(out.shape) == (1, H, S, K)
    assert pcc(out, ref) > 0.999, f"pcc {pcc(out, ref)}"


@run_for_blackhole()
@pytest.mark.parametrize("H", [32, 16, 40])
def test_deepseek_v4_prefill_sdpa_attention_sink(device, H):
    S, K, W, T, TOPK = 64, 512, 128, 1024, 512
    q, window_kv, cache_kv, cache_idx = make_inputs(H, S, K, W, T, TOPK, lambda s: 300 + s, seed=7)
    scale = K**-0.5
    sink = (torch.linspace(-4, 8, H) / scale).reshape(1, 1, 1, H).to(torch.bfloat16)
    out = run(device, q, window_kv, cache_kv, cache_idx, 4096, W, sink=sink)
    ref = golden(q, window_kv, cache_kv, cache_idx, 4096, W, sink=sink)
    assert pcc(out, ref) > 0.999, f"pcc {pcc(out, ref)}"


@run_for_blackhole()
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("start", [0, 256])
@pytest.mark.parametrize("empty_tail", [False, True], ids=["tail", "empty-tail"])
def test_deepseek_v4_prefill_sdpa_entry_causality(device, ratio, start, empty_tail):
    """Every query lists every entry row written by the chunk's end, in random order, as an unrestricted top-k over
    them would: the op keeps only the rows each query has seen (and, with ``empty_tail``, no tail row)."""
    H, S, K, W = 16, 128, 512, 128
    T = (start + S) // ratio
    gen = torch.Generator().manual_seed(start + ratio)
    q = torch.randn(1, H, S, K, generator=gen).to(torch.bfloat16)
    window_kv = torch.randn(1, 1, W + S, K, generator=gen).to(torch.bfloat16)
    cache_kv = torch.randn(1, 1, T, K, generator=gen).to(torch.bfloat16)
    cache_idx = torch.stack([torch.randperm(T, generator=gen) for _ in range(S)]).reshape(1, 1, S, T)
    options = dict(compress_ratio=ratio, empty_tail=empty_tail)
    out = run(device, q, window_kv, cache_kv, cache_idx, start, W, **options)
    ref = golden(q, window_kv, cache_kv, cache_idx, start, W, **options)
    assert pcc(out, ref) > 0.999, f"pcc {pcc(out, ref)}"


@run_for_blackhole()
def test_deepseek_v4_prefill_sdpa_start_is_runtime(device):
    """A new start position or a longer cache reuses the program, as a trace replay needs."""
    H, S, K, W, TOPK = 32, 64, 128, 32, 64
    device.clear_program_cache()
    for start, T in ((0, 256), (16, 256), (1024, 512)):
        q, window_kv, cache_kv, cache_idx = make_inputs(H, S, K, W, T, TOPK, lambda s: 1 + s % TOPK, seed=start)
        out = run(device, q, window_kv, cache_kv, cache_idx, start, W, k_chunk_size=32)
        ref = golden(q, window_kv, cache_kv, cache_idx, start, W)
        assert pcc(out, ref) > 0.999, f"start {start}: pcc {pcc(out, ref)}"
    assert device.num_program_cache_entries() == 1
