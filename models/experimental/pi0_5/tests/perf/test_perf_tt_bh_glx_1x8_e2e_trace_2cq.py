# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf (timing) tests for the 1×8 single-mesh pipeline (Pi0_5GLX1x8Pipeline).

Correctness (PCC vs torch) lives in tests/pcc/test_pcc_tt_bh_glx_1x8.py.

What it covers:
    - SigLIP DP (3 cams batch-padded to 8, all_gather, slice)
    - On-device prefix concat (replicated)
    - Prefill TP=8 (sharded MLP + all_reduce per block)
    - Replicated 5-step Euler denoise on all 8 chips
    - Output extraction via ConcatMeshToTensor + slice [:1]

Tests:
    test_perf_1x8_traced_2cq        — 2CQ trace replay (H2D on CQ1 || compute on CQ0);
                                      the headline per-chunk ms (README perf numbers).
    test_perf_1x8_traced_staged     — per-stage traced breakdown via 3 sub-traces (diagnostic).

Run:
    # optional: pin the 1×8 mesh to a device subset, e.g. TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
    export PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
    export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
    # Inherits production env defaults — set explicit flags before running.
    python_env/bin/pytest -sq \
      models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_1x8_e2e_trace_2cq.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# 1×8-specific flags (not in pi05_production.env — they're pipeline-1x8 specific).
# Set before apply_production_env_defaults so the production file can't override.
# Use setdefault so an explicit shell export still wins.
for _k, _v in {
    "PI0_TP": "8",  # 8-chip tensor parallel for prefill
    "PI0_TP8_ATTN_HEADPAR": "1",  # head-parallel attention split
    "PI0_MLP_BS": "1",  # block-sharded MLP (TP=8 tuned)
    "PI0_MLP_FUSED_RS": "0",  # fused reduce-scatter off (TP=8 uses split RS+AG)
}.items():
    os.environ.setdefault(_k, _v)


# Production perf flags now live in the pi0_5 package (pi05_production.env), loaded
# via the shared package-relative loader. Runs AFTER the 1×8-specific setdefaults
# above so those win; setdefault semantics mean an explicit shell export still wins.
from models.experimental.pi0_5.common.prod_env import apply_production_env_defaults

apply_production_env_defaults()

# Match production fold path before any ttnn / pi0_5 import (so PatchEmbeddingTTNN
# reads the env at construction time). Most of these are in pi05_production.env
# now — kept here as a belt-and-braces fallback in case the env file is missing.
os.environ.setdefault("PI0_SIGLIP_USE_FOLD", "1")
os.environ.setdefault("QWEN_NLP_CREATE_HEADS_HEAD_SPLIT", "1")
os.environ.setdefault("QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT", "1")
os.environ.setdefault("PI0_NUM_CAMERAS", "3")

import ttnn  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42
N_CAMS = int(os.environ["PI0_NUM_CAMERAS"])
LANG_LEN = 256
PERF_ITERS = int(os.environ.get("PERF_ITERS", "20"))
# One warm-up replay by default (not timed): absorbs the small first-call cost
# (~1 ms above steady-state — first CQ1 wait / cache warm-up) so the reported
# mean over PERF_ITERS is steady-state. Override with WARMUP_ITERS.
WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", "1"))

# Production env flags worth asserting present (set by models/experimental/pi0_5/common/pi05_production.env).
# Logged at test start so the run-log shows which optimizations were active.
_PROD_ENV_KEYS = (
    "PI0_EXPERT_MM_LOFI",
    "PI0_ROPE_TABLES_L1",
    "PI0_MM_SWEEP_V2",
    "PI0_DENOISE_MM_TUNE",
    "PI0_PREFILL_MM_TUNE",
    "PI0_UPSTREAM_MASKS",
    "QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT",
    "QWEN_NLP_CREATE_HEADS_HEAD_SPLIT",
    "PI0_MQA_HEAD_SPLIT",
    "PI0_SDPA_DENOISE_K_FORCE",
    "PI0_NUM_CAMERAS",
    "PI0_VLM_CHUNK_SIZE",
    "PI0_VLM_MLP_BF8_OUT",
    "PI0_VLM_MLP_MINIMAL",
    "PI0_VLM_MINIMAL_CFG",
    "PI0_SIGLIP_USE_FOLD",
    "PI0_TP",
    "PI0_TP8_ATTN_HEADPAR",
    "PI0_MLP_BS",
    "PI0_MLP_FUSED_RS",
    "PI05_NUM_DENOISE_STEPS",
)


def _print_prod_env_status():
    present = []
    missing = []
    for k in _PROD_ENV_KEYS:
        v = os.environ.get(k)
        if v is not None:
            present.append(f"{k}={v}")
        else:
            missing.append(k)
    print(f"\n[env] {len(present)}/{len(_PROD_ENV_KEYS)} production flags set:")
    for s in present:
        print(f"      {s}")
    if missing:
        print(f"[env] MISSING ({len(missing)}): {', '.join(missing)}")
    print(f"[env] N_CAMS (test) = {N_CAMS}")


pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_test_inputs(siglip_cfg):
    """Random 3 cameras + a random lang_token batch — same convention as
    pipeline.py:sample_actions and the existing perf tests."""
    torch.manual_seed(SEED)
    H = W = siglip_cfg.image_size
    images = [torch.randn(1, 3, H, W) for _ in range(N_CAMS)]
    lang_tokens = torch.randint(0, 256000, (1, LANG_LEN), dtype=torch.int64)
    return images, lang_tokens


def _make_pipeline(mesh):
    """Construct Pi0_5GLX1x8Pipeline + return (pipe, cfg). Pulled out of the
    test bodies so we can reuse from both eager + traced tests."""
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.tt_bh_glx.pipeline_1x8 import Pi0_5GLX1x8Pipeline

    cfg = Pi0_5ModelConfig(
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        num_denoising_steps=int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5")),
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    pipe = Pi0_5GLX1x8Pipeline(cfg, loader.categorized_weights, mesh)
    return pipe, cfg


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def test_perf_1x8_traced_2cq():
    """2CQ trace replay: H2D input upload on CQ1 overlapped with compute on CQ0.

    Opens the 1×8 mesh with num_command_queues=2, captures the e2e trace on
    CQ0, then does iters=PERF_ITERS replays in a CQ0/CQ1 ping-pong pattern.
    The host-overhead of input_upload should mostly hide behind compute,
    closing toward the trace_exec floor.
    """
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp8_mesh

    _print_prod_env_status()

    with open_prefill_tp8_mesh(
        tp=8,
        l1_small_size=24576,
        trace_region_size=128 * 1024 * 1024,
        num_command_queues=2,
    ) as mesh:
        pipe, cfg = _make_pipeline(mesh)
        images, lang_tokens = _build_test_inputs(cfg.siglip_config)

        pipe.capture_trace(images, lang_tokens)

        ah = cfg.action_horizon
        ad = cfg.action_dim

        # Warm-up replay(s) — not timed (kernel caches, address stability, first CQ1 wait).
        for _ in range(WARMUP_ITERS):
            _ = pipe.sample_actions_traced(images, lang_tokens)

        last_actions, times = pipe.sample_actions_traced_2cq_loop(images, lang_tokens, PERF_ITERS)
        assert last_actions.shape == (1, ah, ad), f"shape mismatch: {tuple(last_actions.shape)}"
        assert torch.isfinite(last_actions).all(), "non-finite values in actions output"

        mean = _mean(times)
        print("\n" + "=" * 72)
        print(f"1×8 pi0.5 TRACED 2CQ replay  (N_CAMS={N_CAMS})")
        print("=" * 72)
        print(f"  mean ({len(times)} iters) : {mean:.2f} ms")
        print("=" * 72)


def test_perf_1x8_traced_staged():
    """Per-stage TRACED breakdown via 3 sub-traces on the single 1×8 mesh.

    Captures vision / prefill / denoise as three independent traces (with
    persistent vision_real and per_layer_kv intermediates living across trace
    boundaries via the deterministic trace allocator). Replays each in
    sequence, times each replay with perf_counter + blocking=True.

    These are TRUE traced per-stage numbers (no eager dispatch overhead).
    """
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp8_mesh

    _print_prod_env_status()

    # 3 sub-traces share the trace region — bump from 128 MiB to 256 MiB.
    with open_prefill_tp8_mesh(tp=8, l1_small_size=24576, trace_region_size=256 * 1024 * 1024) as mesh:
        pipe, cfg = _make_pipeline(mesh)
        images, lang_tokens = _build_test_inputs(cfg.siglip_config)

        pipe.capture_traces_staged(images, lang_tokens)

        ah = cfg.action_horizon
        ad = cfg.action_dim

        for _ in range(WARMUP_ITERS):
            _ = pipe.sample_actions_traced_staged_timed(images, lang_tokens)

        runs = []
        last_actions = None
        for _ in range(PERF_ITERS):
            last_actions, t = pipe.sample_actions_traced_staged_timed(images, lang_tokens)
            runs.append(t)
        assert last_actions is not None
        assert last_actions.shape == (1, ah, ad), f"shape mismatch: {tuple(last_actions.shape)}"
        assert torch.isfinite(last_actions).all(), "non-finite values in actions output"

        keys = [
            "input_upload_ms",
            "vision_ms",
            "prefill_ms",
            "denoise_ms",
            "output_readback_ms",
            "compute_total_ms",
            "traced_total_ms",
        ]
        means = {k: _mean([r[k] for r in runs]) for k in keys}
        mins = {k: min(r[k] for r in runs) for k in keys}

        compute = means["compute_total_ms"]
        pct = lambda x: 100.0 * x / compute if compute > 0 else 0.0

        print("\n" + "=" * 72)
        print(
            f"1×8 pi0.5 TRACED per-stage breakdown   (PERF_ITERS={PERF_ITERS}, steps={cfg.num_denoising_steps}, N_CAMS={N_CAMS})"
        )
        print("=" * 72)
        print(f"  {'stage':<20} {'mean ms':>10} {'min ms':>10} {'% compute':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        print(
            f"  {'input_upload (host)':<20} {means['input_upload_ms']:10.2f} {mins['input_upload_ms']:10.2f} {'-':>10}"
        )
        print(
            f"  {'vision (trace)':<20} {means['vision_ms']:10.2f} {mins['vision_ms']:10.2f} {pct(means['vision_ms']):9.1f}%"
        )
        print(
            f"  {'prefill (trace)':<20} {means['prefill_ms']:10.2f} {mins['prefill_ms']:10.2f} {pct(means['prefill_ms']):9.1f}%"
        )
        print(
            f"  {'denoise (trace)':<20} {means['denoise_ms']:10.2f} {mins['denoise_ms']:10.2f} {pct(means['denoise_ms']):9.1f}%"
        )
        print(
            f"  {'output_readback':<20} {means['output_readback_ms']:10.2f} {mins['output_readback_ms']:10.2f} {'-':>10}"
        )
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'compute (v+p+d)':<20} {means['compute_total_ms']:10.2f} {mins['compute_total_ms']:10.2f}")
        print(f"  {'traced_total':<20} {means['traced_total_ms']:10.2f} {mins['traced_total_ms']:10.2f}")
        print("=" * 72)
