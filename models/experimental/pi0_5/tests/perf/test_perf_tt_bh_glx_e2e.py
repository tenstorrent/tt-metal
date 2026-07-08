# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end wall-clock perf for the BH Galaxy host-bounce pipeline.

Runs Pi0_5GLXPipeline.sample_actions in a warmup + N timed iters loop and
reports per-stage StageTimings (vision, transport, prefill, kv_migration,
denoise per-step, total). The single-chip trace e2e baseline is ~44 ms; this
host-bounce pipeline is expected at ~500-700 ms steady-state.

Run:
    source _bench_runs/pi05_production.env
    PYTHONPATH=$PWD TT_METAL_HOME=$PWD \
      python_env/bin/pytest -xvs \
      models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_e2e.py
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import List

import pytest
import torch

from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_galaxy_mesh


CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42
NUM_WARMUP = int(os.environ.get("PI05_GLX_NUM_WARMUP", "1"))
NUM_ITERS = int(os.environ.get("PI05_GLX_NUM_ITERS", "3"))


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"checkpoint not found at {CHECKPOINT_DIR}",
)
def test_pi0_5_glx_pipeline_e2e_perf():
    """End-to-end Pi0_5GLXPipeline.sample_actions wall-clock + per-stage breakdown."""
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_bh_glx.pipeline import Pi0_5GLXPipeline

    num_steps = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5"))
    num_cams = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
    cfg = Pi0_5ModelConfig(
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        num_denoising_steps=num_steps,
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))

    img_h = img_w = cfg.siglip_config.image_size
    lang_len = 256

    torch.manual_seed(SEED)
    images = [torch.randn(1, 3, img_h, img_w) for _ in range(num_cams)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cams)]
    lang_tokens = torch.randint(0, 256000, (1, lang_len), dtype=torch.int32)
    lang_masks = torch.ones(1, lang_len, dtype=torch.bool)

    print(
        f"\n📋 Pi0_5GLXPipeline e2e perf — action_horizon={cfg.action_horizon}, "
        f"denoise_steps={num_steps}, cams={num_cams}"
    )
    print(f"   warmup={NUM_WARMUP}, iters={NUM_ITERS}, ckpt={CHECKPOINT_DIR}")

    with open_galaxy_mesh(l1_small_size=24576) as h:
        pipe = Pi0_5GLXPipeline(cfg, loader.categorized_weights, h)

        # Warmup — pays JIT compile cost.
        print(f"\n🔥 Warmup ({NUM_WARMUP} calls)")
        for w in range(NUM_WARMUP):
            actions, t = pipe.sample_actions(
                images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks
            )
            print(f"   warmup {w + 1}: total={t.total_ms:.1f}ms (includes JIT on iter 1)")

        # Steady-state.
        print(f"\n⏱️  Steady-state ({NUM_ITERS} calls)")
        all_timings = []
        wall_total: List[float] = []
        for i in range(NUM_ITERS):
            t0 = time.perf_counter()
            actions, t = pipe.sample_actions(
                images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks
            )
            wall_total.append((time.perf_counter() - t0) * 1000.0)
            all_timings.append(t)
            print(
                f"   iter {i + 1:2d}: total={t.total_ms:7.1f}ms  "
                f"vision={t.vision_ms:5.1f}  v→p={t.transport_v2p_ms:4.1f}  "
                f"prefill={t.prefill_ms:6.1f}  kv_mig={t.kv_migration_ms:5.1f}  "
                f"denoise(5)={t.denoise_total_ms:6.1f}"
            )

        # Aggregate.
        def _agg(field: str) -> float:
            return statistics.mean(getattr(t, field) for t in all_timings)

        print("\n" + "=" * 78)
        print(f"  PI0.5 BH-GALAXY HOST-BOUNCE E2E (N={NUM_ITERS}, warmup={NUM_WARMUP})")
        print("=" * 78)
        print(f"   Stage avg latency:")
        print(f"     vision        : {_agg('vision_ms'):7.2f} ms")
        print(f"     transport v→p : {_agg('transport_v2p_ms'):7.2f} ms")
        print(f"     prefill       : {_agg('prefill_ms'):7.2f} ms")
        print(f"     kv_migration  : {_agg('kv_migration_ms'):7.2f} ms")
        print(f"     denoise total : {_agg('denoise_total_ms'):7.2f} ms  ({num_steps} steps)")
        denoise_per_step = sum(_agg("denoise_total_ms") for _ in [0]) / num_steps
        print(f"     denoise/step  : {denoise_per_step:7.2f} ms")
        avg_total = statistics.mean(wall_total)
        mn = min(wall_total)
        mx = max(wall_total)
        sd = statistics.stdev(wall_total) if len(wall_total) > 1 else 0.0
        print("-" * 78)
        print(f"   Per-call total avg : {avg_total:7.2f} ms")
        print(f"   Per-call total min : {mn:7.2f} ms")
        print(f"   Per-call total max : {mx:7.2f} ms")
        print(f"   Per-call total stddev: {sd:7.2f} ms")
        print(f"   Chunks/s           : {(1000.0 / avg_total):7.2f}")
        print(f"   Actions/s          : {(1000.0 / avg_total) * cfg.action_horizon:7.2f}")
        print("=" * 78)

        assert torch.isfinite(actions).all(), "actions contain NaN/Inf"
        assert actions.shape == (1, cfg.action_horizon, cfg.action_dim)
