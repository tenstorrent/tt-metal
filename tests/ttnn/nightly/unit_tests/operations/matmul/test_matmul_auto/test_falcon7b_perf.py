# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Falcon-7B model performance benchmark (bounty req #4: model usage proof)."""

import logging
import math
import os
import time

import pytest
import torch

import ttnn

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
logger = logging.getLogger(__name__)

# Representative Falcon-7B matmul shapes: (name, M, K, N)
FALCON_7B_SHAPES = [
    ("attn_qkv", 32, 4544, 4672),
    ("attn_dense", 32, 4544, 4544),
    ("mlp_up", 32, 4544, 18176),
    ("mlp_down", 32, 18176, 4544),
    ("prefill_qkv_128", 128, 4544, 4672),
    ("prefill_dense_128", 128, 4544, 4544),
    ("batch32_qkv", 1024, 4544, 4672),
    ("batch32_mlp_up", 1024, 4544, 18176),
]


def _tile_pad(x):
    return ((x + 31) // 32) * 32


def _measure(device, input_a, input_b, config=None, warmup=5, runs=10):
    """Measure average latency in microseconds."""
    for _ in range(warmup):
        out = ttnn.matmul(input_a, input_b, program_config=config) if config else ttnn.matmul(input_a, input_b)
    ttnn.synchronize_device(device)

    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(runs):
        out = ttnn.matmul(input_a, input_b, program_config=config) if config else ttnn.matmul(input_a, input_b)
    ttnn.synchronize_device(device)
    elapsed = (time.perf_counter() - start) * 1e6
    ttnn.deallocate(out)
    return elapsed / runs


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("name,m,k,n", FALCON_7B_SHAPES)
def test_single_shape_no_regression(device, name, m, k, n):
    """Each Falcon-7B shape: auto must not be >5% slower than default."""
    from ttnn._experimental.auto_config.matmul_auto import MatmulAutoConfig

    M, K, N = _tile_pad(m), _tile_pad(k), _tile_pad(n)
    input_a = ttnn.from_torch(torch.randn(1, M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch.randn(K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    t_default = _measure(device, input_a, input_b)

    selector = MatmulAutoConfig()
    result = selector.select(input_a, input_b)
    selected = result.selected_config

    if selected and selected.config and selected.backend == "matmul":
        t_auto = _measure(device, input_a, input_b, config=selected.config)
    else:
        t_auto = t_default

    speedup = t_default / t_auto if t_auto > 0 else 1.0
    logger.info(f"  {name:25s} default={t_default:8.0f}µs auto={t_auto:8.0f}µs speedup={speedup:.3f}x")

    assert (
        speedup >= 0.95
    ), f"{name}: auto {speedup:.3f}x is >5% slower (auto={t_auto:.0f}µs vs default={t_default:.0f}µs)"


def test_geomean_speedup_across_all_shapes(device):
    """Geometric mean speedup across ALL Falcon-7B shapes must be >= 1.0x."""
    from ttnn._experimental.auto_config.matmul_auto import MatmulAutoConfig

    speedups = []
    rows = []
    for name, m, k, n in FALCON_7B_SHAPES:
        M, K, N = _tile_pad(m), _tile_pad(k), _tile_pad(n)
        input_a = ttnn.from_torch(torch.randn(1, M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        input_b = ttnn.from_torch(torch.randn(K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        t_default = _measure(device, input_a, input_b)
        selector = MatmulAutoConfig()
        result = selector.select(input_a, input_b)
        selected = result.selected_config

        if selected and selected.config and selected.backend == "matmul":
            t_auto = _measure(device, input_a, input_b, config=selected.config)
            cfg_name = selected.config_family
        else:
            t_auto = t_default
            cfg_name = "fallback"

        sp = t_default / t_auto if t_auto > 0 else 1.0
        speedups.append(sp)
        rows.append((name, t_default, t_auto, sp, cfg_name))
        ttnn.deallocate(input_a)
        ttnn.deallocate(input_b)

    # Print formatted results table
    geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    print("\n" + "=" * 80)
    print("FALCON-7B MATMUL AUTO-CONFIG PERFORMANCE RESULTS")
    print("=" * 80)
    print(f"{'Shape':<25} {'Default(µs)':>12} {'Auto(µs)':>12} {'Speedup':>8} {'Config':<15}")
    print("-" * 80)
    for name, t_def, t_auto, sp, cfg in rows:
        marker = "✓" if sp >= 1.0 else "✗"
        print(f"{name:<25} {t_def:>12.0f} {t_auto:>12.0f} {sp:>7.3f}x {cfg:<15} {marker}")
    print("-" * 80)
    print(f"{'GEOMETRIC MEAN':<25} {'':>12} {'':>12} {geomean:>7.4f}x")
    print("=" * 80)
    logger.info(f"  Falcon-7B Geomean Speedup: {geomean:.4f}x ({len(speedups)} shapes)")

    assert geomean >= 1.0, f"Geomean {geomean:.4f}x below 1.0x. Speedups: {[f'{s:.3f}x' for s in speedups]}"


@pytest.mark.parametrize("name,m,k,n", FALCON_7B_SHAPES[:4])
def test_falcon_correctness(device, name, m, k, n):
    """Verify auto output matches torch.matmul within PCC threshold."""
    from ttnn._experimental.auto_config import matmul_auto

    M, K, N = _tile_pad(m), _tile_pad(k), _tile_pad(n)
    torch_a = torch.randn(1, M, K)
    torch_b = torch.randn(K, N)
    torch_out = torch.matmul(torch_a, torch_b)

    input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = matmul_auto(input_a, input_b)
    output = ttnn.to_torch(tt_out)

    from tests.ttnn.utils_for_testing import check_with_pcc

    passed, msg = check_with_pcc(torch_out, output, pcc=0.99)
    assert passed, f"Falcon7B {name}: PCC check failed: {msg}"
