# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""All-L1 single-layer bench for prefill + SigLIP + expert blocks.

Runs each of the three pi0.5 transformer-block types for one layer with
EVERY tensor (weights, biases, activations) on L1. Real
pi05_libero_upstream weights, 100 timed iterations, single-chip.

Notes on placement:
  - Prefill (Gemma-2B): MLP weights cast to bf4_b to fit L1 at TP=1
    without hitting the matmul static-CB clash. Attention weights stay
    bf8_b. (See project_pi05_single_layer_l1_dram_perf memory.)
  - SigLIP & expert: bf8_b weights, fit L1 with default placement.

Designed for tracy/profile runs — no placement matrix, no knobs.

Run:
    PI0_BENCH_ALL_L1=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_blocks_all_l1.py

Or under tracy:
    PI0_BENCH_ALL_L1=1 python -m tracy -p -r -v --op-support-count 100000 \\
      -m pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_blocks_all_l1.py
"""

from __future__ import annotations

# Force prefill MLP to bf4_b BEFORE importing the prefill bench module so
# its module-level dtype constant picks this up. (Attn stays bf8 default.)
import os as _os

_os.environ.setdefault("PI0_PREFILL_BENCH_MLP_DTYPE", "bf4")

import os
import statistics
import time
from typing import List, Tuple

import pytest
import torch
import ttnn

from models.experimental.pi0_5.tests.perf.test_prefill_block_l1_vs_dram import (
    BATCH as PF_BATCH,
    S as PF_S,
    WIDTH as PF_WIDTH,
    _build_random_block as _build_prefill_block,
    _real_or_random_vlm_layer,
    _upload,
)
from models.experimental.pi0_5.tests.perf.test_siglip_block_l1_vs_dram import (
    BATCH as SG_BATCH,
    HIDDEN as SG_HIDDEN,
    S as SG_S,
    _make_siglip_config,
    _migrate_block_weights,
    _real_or_random_siglip_layer,
)
from models.experimental.pi0_5.tests.perf.test_expert_block_l1_vs_dram import (
    BATCH as EX_BATCH,
    S as EX_S,
    WIDTH as EX_WIDTH,
    _build_expert_block,
    _real_or_random_expert_layer,
)
from models.experimental.pi0_5.tt.ttnn_siglip import SigLIPBlockTTNN


BENCH_ENABLED = os.environ.get("PI0_BENCH_ALL_L1") == "1"
pytestmark = pytest.mark.skipif(
    not BENCH_ENABLED,
    reason="set PI0_BENCH_ALL_L1=1 to run the all-L1 3-block bench",
)

NUM_WARMUP = int(os.environ.get("PI0_BENCH_WARMUP", "10"))
NUM_ITER = int(os.environ.get("PI0_BENCH_ITER", "100"))

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG  # only for SDPA mask (kernel requirement)


def _stats(samples_ms: List[float]) -> Tuple[float, float, float, float]:
    return (
        statistics.mean(samples_ms),
        statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        min(samples_ms),
        max(samples_ms),
    )


def _time_prefill(device) -> Tuple[float, float, float, float]:
    raw = _real_or_random_vlm_layer(0)
    # weights L1, biases L1 — bf4 MLP comes from PI0_PREFILL_BENCH_MLP_DTYPE=bf4 above
    block = _build_prefill_block(device, L1, L1, raw=raw, layer_idx=0)
    hidden_host = torch.randn(PF_BATCH, PF_S, PF_WIDTH) * 0.5
    mask_tt = _upload(torch.zeros(PF_BATCH, 1, PF_S, PF_S), device, ttnn.bfloat16, DRAM)

    def fwd(h):
        out, _ = block.forward(
            h,
            cos=block._bench_cos,
            sin=block._bench_sin,
            attention_mask=mask_tt,
            position_ids=None,
            past_key_value=None,
            use_cache=False,
        )
        return out

    for _ in range(NUM_WARMUP):
        h = _upload(hidden_host, device, ttnn.bfloat16, L1)
        out = fwd(h)
        ttnn.deallocate(h)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    samples: List[float] = []
    for _ in range(NUM_ITER):
        h = _upload(hidden_host, device, ttnn.bfloat16, L1)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fwd(h)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(h)
        ttnn.deallocate(out)

    for t in (block._bench_cos, block._bench_sin):
        try:
            ttnn.deallocate(t)
        except RuntimeError:
            pass
    for t in block._bench_weights.values():
        try:
            ttnn.deallocate(t)
        except RuntimeError:
            pass
    ttnn.deallocate(mask_tt)
    return _stats(samples)


def _time_siglip(device) -> Tuple[float, float, float, float]:
    cfg = _make_siglip_config()
    raw = _real_or_random_siglip_layer(0)
    block = SigLIPBlockTTNN(cfg, raw, device)  # DRAM by default — migrate next
    _migrate_block_weights(block, weights_to_l1=True, biases_to_l1=True)
    hidden_host = torch.randn(SG_BATCH, SG_S, SG_HIDDEN) * 0.5

    def upload_hidden():
        return ttnn.from_torch(
            hidden_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=L1,
        )

    for _ in range(NUM_WARMUP):
        h = upload_hidden()
        out = block.forward(h)
        ttnn.deallocate(h)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    samples: List[float] = []
    for _ in range(NUM_ITER):
        h = upload_hidden()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = block.forward(h)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(h)
        ttnn.deallocate(out)

    for attr in (
        block.ln1_weight,
        block.ln1_bias,
        block.ln2_weight,
        block.ln2_bias,
        getattr(block.attention, "wqkv", None),
        getattr(block.attention, "bqkv", None),
        getattr(block.attention, "wo", None),
        getattr(block.attention, "bo", None),
        getattr(block.mlp, "fc1_weight", None),
        getattr(block.mlp, "fc1_bias", None),
        getattr(block.mlp, "fc2_weight", None),
        getattr(block.mlp, "fc2_bias", None),
    ):
        if attr is None:
            continue
        try:
            ttnn.deallocate(attr)
        except RuntimeError:
            pass
    return _stats(samples)


def _time_expert(device) -> Tuple[float, float, float, float]:
    raw = _real_or_random_expert_layer(0)
    block = _build_expert_block(device, L1, L1, raw=raw, layer_idx=0)
    hidden_host = torch.randn(EX_BATCH, EX_S, EX_WIDTH) * 0.5
    # adarms_cond goes on L1 (small, cheap); mask must be DRAM (SDPA req).
    adarms_tt = _upload(torch.randn(EX_BATCH, 1, EX_WIDTH) * 0.1, device, ttnn.bfloat16, L1)
    mask_tt = _upload(torch.zeros(EX_BATCH, 1, EX_S, EX_S), device, ttnn.bfloat16, DRAM)

    def fwd(h):
        out, _ = block.forward(
            h,
            block._bench_cos,
            block._bench_sin,
            adarms_tt,
            attention_mask=mask_tt,
            position_ids=None,
            past_key_value=None,
            use_cache=False,
        )
        return out

    for _ in range(NUM_WARMUP):
        h = _upload(hidden_host, device, ttnn.bfloat16, L1)
        out = fwd(h)
        ttnn.deallocate(h)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    samples: List[float] = []
    for _ in range(NUM_ITER):
        h = _upload(hidden_host, device, ttnn.bfloat16, L1)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fwd(h)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(h)
        ttnn.deallocate(out)

    for t in (block._bench_cos, block._bench_sin):
        try:
            ttnn.deallocate(t)
        except RuntimeError:
            pass
    for t in block._bench_weights.values():
        try:
            ttnn.deallocate(t)
        except RuntimeError:
            pass
    ttnn.deallocate(adarms_tt)
    ttnn.deallocate(mask_tt)
    return _stats(samples)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_blocks_all_l1(device):
    print("\n" + "=" * 80)
    print(f"  ALL-L1 1-layer bench  (warmup={NUM_WARMUP}, iter={NUM_ITER})")
    print("  Prefill: MLP=bf4, attn=bf8.  SigLIP & expert: bf8 weights.")
    print("=" * 80)

    print("\n>> PREFILL Gemma-2B  shape=(B=1, S=512, W=2048)")
    pf = _time_prefill(device)
    print(f"   mean={pf[0]:.2f} ms  stdev={pf[1]:.3f}  min={pf[2]:.2f}  max={pf[3]:.2f}")
    ttnn.synchronize_device(device)

    print("\n>> SIGLIP  shape=(B=1, S=256, H=1152)")
    sg = _time_siglip(device)
    print(f"   mean={sg[0]:.2f} ms  stdev={sg[1]:.3f}  min={sg[2]:.2f}  max={sg[3]:.2f}")
    ttnn.synchronize_device(device)

    print("\n>> EXPERT Gemma-300M  shape=(B=1, S=32, W=1024)")
    ex = _time_expert(device)
    print(f"   mean={ex[0]:.2f} ms  stdev={ex[1]:.3f}  min={ex[2]:.2f}  max={ex[3]:.2f}")

    print("\n" + "=" * 80)
    print("  SUMMARY  (all tensors L1-resident)")
    print("=" * 80)
    print(f"  {'block':<12}  {'mean ms':>9}  {'stdev':>7}  {'min':>7}  {'max':>7}")
    print(f"  {'prefill':<12}  {pf[0]:>9.2f}  {pf[1]:>7.3f}  {pf[2]:>7.2f}  {pf[3]:>7.2f}")
    print(f"  {'siglip':<12}  {sg[0]:>9.2f}  {sg[1]:>7.3f}  {sg[2]:>7.2f}  {sg[3]:>7.2f}")
    print(f"  {'expert':<12}  {ex[0]:>9.2f}  {ex[1]:>7.3f}  {ex[2]:>7.2f}  {ex[3]:>7.2f}")
    print("=" * 80)
