# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolated device profiling tests for matmul_auto vs ttnn.matmul default.

Addresses @alnah005's feedback: "isolated device profiling of the matmul op
itself would give us much more reliable evidence. Comparison plots across
representative shapes showing matmul_auto vs ttnn.matmul with default config."

Each test runs ONLY the matmul op (no model-level overhead), measures latency
via device synchronization, and outputs a comparison CSV.
"""

import csv
import logging
import os
import time

import pytest

logger = logging.getLogger(__name__)

# Representative shapes covering decode, prefill, attention, MLP
PROFILING_SHAPES = [
    # (M, K, N, description)
    (32, 4096, 4096, "decode_qkv"),
    (32, 4096, 11008, "decode_mlp_up"),
    (32, 11008, 4096, "decode_mlp_down"),
    (128, 4096, 4096, "prefill_short"),
    (512, 4096, 4096, "prefill_medium"),
    (2048, 4096, 4096, "prefill_long"),
    (2048, 4096, 16384, "prefill_mlp_up"),
    (2048, 2048, 128, "attn_v_proj"),
    (2048, 128, 2048, "attn_qk"),
    (1024, 4096, 4096, "batch32_qkv"),
    (256, 256, 256, "small_square"),
    (4096, 4096, 4096, "large_square"),
]

NUM_WARMUP = 3
NUM_RUNS = 10


def _measure_matmul_latency(input_a, input_b, program_config=None, num_runs=NUM_RUNS):
    """Measure median matmul latency in microseconds with device sync."""
    import ttnn

    device = input_a.device()

    # Warmup
    for _ in range(NUM_WARMUP):
        if program_config is not None:
            _ = ttnn.matmul(input_a, input_b, program_config=program_config)
        else:
            _ = ttnn.matmul(input_a, input_b)
        ttnn.synchronize_device(device)

    # Timed runs
    times = []
    for _ in range(num_runs):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        if program_config is not None:
            _ = ttnn.matmul(input_a, input_b, program_config=program_config)
        else:
            _ = ttnn.matmul(input_a, input_b)
        ttnn.synchronize_device(device)
        elapsed_us = (time.perf_counter() - start) * 1e6
        times.append(elapsed_us)

    times.sort()
    return times[len(times) // 2]  # Median


@pytest.mark.parametrize("M,K,N,desc", PROFILING_SHAPES)
def test_isolated_matmul_auto_vs_default(M, K, N, desc, device):
    """
    Compare matmul_auto's selected config vs default ttnn.matmul for a single shape.

    Measures:
    - default_us: ttnn.matmul with no program_config
    - auto_us: ttnn.matmul with matmul_auto's selected program_config
    - speedup: default_us / auto_us

    Pass condition: auto config must not be >10% slower than default.
    """
    import torch
    import ttnn
    from ttnn._experimental.auto_config import matmul_auto
    from ttnn._experimental.auto_config.matmul_auto import MatmulAutoConfig

    # Create tensors
    torch_a = torch.randn(1, 1, M, K).bfloat16()
    torch_b = torch.randn(1, 1, K, N).bfloat16()

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)

    # 1. Measure default (no program_config)
    default_us = _measure_matmul_latency(tt_a, tt_b, program_config=None)

    # 2. Get auto-selected config
    selector = MatmulAutoConfig()
    features = selector.extract_features(tt_a, tt_b)
    result = selector.select(tt_a, tt_b)

    if result is not None and result.selected_config is not None:
        auto_config = result.selected_config.config
        auto_us = _measure_matmul_latency(tt_a, tt_b, program_config=auto_config)
    else:
        auto_us = default_us  # Fallback = same as default

    speedup = default_us / auto_us if auto_us > 0 else 1.0

    logger.info(
        "Shape (%s): M=%d K=%d N=%d | default=%.0f us | auto=%.0f us | speedup=%.2fx",
        desc,
        M,
        K,
        N,
        default_us,
        auto_us,
        speedup,
    )

    # Write to CSV
    csv_path = os.environ.get("MATMUL_AUTO_PERF_CSV", "matmul_auto_perf_comparison.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["shape", "M", "K", "N", "default_us", "auto_us", "speedup", "config_family"])
        family = result.selected_config.config_family if result and result.selected_config else "default"
        writer.writerow([desc, M, K, N, f"{default_us:.0f}", f"{auto_us:.0f}", f"{speedup:.3f}", family])

    # Assert: auto must not be >10% slower than default
    assert speedup >= 0.9, (
        f"matmul_auto is {(1 - speedup) * 100:.1f}% slower than default for "
        f"shape ({M}, {K}, {N}) [{desc}]. Got {auto_us:.0f}us vs {default_us:.0f}us"
    )


def test_aggregate_speedup_report(device):
    """
    Run all shapes and produce an aggregate speedup report.
    Verifies geometric mean speedup >= 1.0x.
    """
    import math

    import torch
    import ttnn
    from ttnn._experimental.auto_config.matmul_auto import MatmulAutoConfig

    results = []

    for M, K, N, desc in PROFILING_SHAPES:
        torch_a = torch.randn(1, 1, M, K).bfloat16()
        torch_b = torch.randn(1, 1, K, N).bfloat16()
        tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
        tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)

        default_us = _measure_matmul_latency(tt_a, tt_b)

        selector = MatmulAutoConfig()
        result = selector.select(tt_a, tt_b)

        if result and result.selected_config and result.selected_config.config:
            auto_us = _measure_matmul_latency(tt_a, tt_b, program_config=result.selected_config.config)
        else:
            auto_us = default_us

        speedup = default_us / auto_us if auto_us > 0 else 1.0
        results.append((desc, M, K, N, default_us, auto_us, speedup))

    # Print table
    logger.info("")
    logger.info("=" * 90)
    logger.info("ISOLATED DEVICE PROFILING: matmul_auto vs ttnn.matmul (default)")
    logger.info("=" * 90)
    logger.info(f"{'Shape':<20} {'M':>6} {'K':>6} {'N':>6} {'Default (us)':>12} {'Auto (us)':>12} {'Speedup':>8}")
    logger.info("-" * 90)
    for desc, M, K, N, default_us, auto_us, speedup in results:
        marker = "✓" if speedup >= 1.0 else "✗"
        logger.info(f"{desc:<20} {M:>6} {K:>6} {N:>6} {default_us:>12.0f} {auto_us:>12.0f} {speedup:>7.2f}x {marker}")

    # Geometric mean
    log_sum = sum(math.log(s) for _, _, _, _, _, _, s in results)
    geomean = math.exp(log_sum / len(results))
    logger.info("-" * 90)
    logger.info(f"{'GEOMETRIC MEAN':<20} {'':>6} {'':>6} {'':>6} {'':>12} {'':>12} {geomean:>7.2f}x")
    logger.info("=" * 90)

    assert geomean >= 1.0, f"Geometric mean speedup {geomean:.3f}x is below 1.0x target"
