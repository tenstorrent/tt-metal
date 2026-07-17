# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-phase compute profiling: attribute the ~92% compute-bound residual across phases.

Perf 1/2 proved the flagged 1x10x9472x128 shape is compute-bound (DM <=1.8%, hidden) and
that the compute residual is ~92% per-phase OVERHEAD (softmax MATH only 0.45%). This test
answers WHERE that overhead lives — which of the ~8 serialized helper phases per KV chunk
eats the time — with on-device per-phase DeviceZoneScopedN timing (clock-invariant cycles).

Shape: (1, 1, 2048, 128), bf16, fp32_dest_acc_en=False, HiFi2. This reproduces the flagged
shape's EXACT per-phase kernels — S=2048 -> 64 tiles -> _chunk_size=8 (chunk 8, same as the
flagged 296=8*37), Dt=4, dest_limit=8, fast_exp + fuse_rowsum both on — but with only 8 work
units (1 per core) x 8 KV chunks, so the device profiler buffer stays tiny and the per-zone
data is clean. Per-phase compute overhead is per-core-local and chunk-count-invariant, so the
small shape's per-phase cycles equal the flagged shape's.

The compute kernel wraps each phase in `SDPA_ZONE("Pxx_...")`, compiled in ONLY when the
descriptor injects -DSDPA_ZONE_PROFILE (env SDPA_ZONE_PROFILE=1). Default builds are
byte-identical (macro -> no-op).

Measure:
  scripts/run_safe_pytest.sh --profile \
    tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_phase_zones.py

Then aggregate per-zone begin/end cycle deltas on the MATH (TRISC1) thread from
generated/profiler/.logs/profile_log_device.csv.
"""

from __future__ import annotations

import math
import os

import torch
import ttnn
from loguru import logger

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

SHAPE = (1, 1, 2048, 128)  # chunk-8 / Dt-4 / dest_limit-8 twin of the flagged shape
ITERS = 3


def test_sdpa_phase_zones(device):
    torch.manual_seed(1234)
    B, H, S, D = SHAPE
    Q, K, V = (torch.randn(B, H, S, D) for _ in range(3))
    scale = 1.0 / math.sqrt(D)
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )

    # torch reference — this is a FULLY WORKING test (nothing stubbed), so the op must
    # produce the correct attention output, not garbage. Gate on PCC before trusting the
    # profile.
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)

    # Assert NO ablation env is active — this run must exercise the real op end to end.
    for k in ("SDPA_ABLATE_READER", "SDPA_ABLATE_WRITER", "SDPA_ABLATE_PV", "SDPA_ABLATE_SOFTMAX"):
        assert os.environ.get(k) in (None, "0"), f"{k} is set — this is meant to be an UNSTUBBED run"

    os.environ["SDPA_ZONE_PROFILE"] = "1"
    try:
        out = None
        for _ in range(ITERS):
            out = scaled_dot_product_attention(tq, tk, tv, scale=scale, compute_kernel_config=cfg)
        got = ttnn.to_torch(out).float()  # force completion
    finally:
        os.environ.pop("SDPA_ZONE_PROFILE", None)

    a, b = ref.flatten(), got.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    logger.info(f"phase-zones profiled on {SHAPE} ({ITERS} iters); PCC vs torch = {pcc:.5f}")
    assert pcc > 0.99, f"op is not producing correct output (PCC={pcc:.5f}) — not a working run"
