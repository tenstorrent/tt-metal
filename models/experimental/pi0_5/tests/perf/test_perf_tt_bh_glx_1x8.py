# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the 1×8 single-mesh pipeline (Pi0_5GLX1x8Pipeline).

What it covers:
    - SigLIP DP (3 cams batch-padded to 8, all_gather, slice)
    - On-device prefix concat (replicated)
    - Prefill TP=8 (sharded MLP + all_reduce per block)
    - Replicated 5-step Euler denoise on all 8 chips
    - Output extraction via ConcatMeshToTensor + slice [:1]

Two tests:
    test_perf_1x8_eager    — shape + (optional) torch-ref PCC sanity check; one
                             eager sample_actions call. No timing claims.
    test_perf_1x8_traced   — captures the full e2e trace; runs PERF_ITERS=20
                             replays; reports mean / min ms.

Run:
    export TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
    export PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
    export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
    # Inherits production env defaults — set explicit flags before running.
    python_env/bin/pytest -sq \
      models/experimental/pi0_5/tests/perf/test_perf_tt_bh_glx_1x8.py
"""
from __future__ import annotations

import os
import statistics
import time
from pathlib import Path

import pytest
import torch

# Match production fold path before any ttnn / pi0_5 import (so PatchEmbeddingTTNN
# reads the env at construction time). The full e2e also wants the matmul/head-par
# defaults the bench env sets — leave those to the user's shell.
os.environ.setdefault("PI0_SIGLIP_USE_FOLD", "1")
os.environ.setdefault("QWEN_NLP_CREATE_HEADS_HEAD_SPLIT", "1")
os.environ.setdefault("QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT", "1")
os.environ.setdefault("PI0_NUM_CAMERAS", "3")

import ttnn  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42
N_CAMS = 3
LANG_LEN = 256
PERF_ITERS = int(os.environ.get("PERF_ITERS", "20"))
WARMUP_ITERS = 3

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


def test_perf_1x8_eager():
    """Eager sample_actions call on the 1×8 pipeline.

    Asserts the output shape is (1, action_horizon, action_dim). PCC vs torch
    reference is OFF by default — set PI05_E2E_PCC=1 to enable (matches the
    existing socket-traced run script's behavior). PCC requires CPU torch
    inference which is slow; the shape check alone is enough to catch any
    structural error in the pipeline glue.
    """
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

    with open_prefill_tp4_mesh(tp=8, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:
        pipe, cfg = _make_pipeline(mesh)
        images, lang_tokens = _build_test_inputs(cfg.siglip_config)

        actions = pipe.sample_actions(images, lang_tokens=lang_tokens)

        ah = cfg.action_horizon
        ad = cfg.action_dim
        assert actions.shape == (1, ah, ad), f"shape mismatch: {tuple(actions.shape)} vs (1, {ah}, {ad})"
        assert torch.isfinite(actions).all(), "non-finite values in actions output"

        print(
            f"\n✅ 1×8 eager: shape={tuple(actions.shape)}  "
            f"mean={actions.mean().item():.4f} std={actions.std().item():.4f}"
        )


def test_perf_1x8_traced():
    """Capture the full e2e trace and time PERF_ITERS replays.

    Reports mean / min / max ms over the replay loop after a warmup. The
    trace covers SigLIP DP + all_gather + slice + prefix concat + prefill TP=8
    + 5-step denoise + output extract — exactly what production replay does.
    """
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

    with open_prefill_tp4_mesh(tp=8, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:
        pipe, cfg = _make_pipeline(mesh)
        images, lang_tokens = _build_test_inputs(cfg.siglip_config)

        # capture_trace runs an eager warmup internally (JIT compile every kernel)
        # before begin_trace_capture, so the first replay is already cache-warm.
        pipe.capture_trace(images, lang_tokens)

        ah = cfg.action_horizon
        ad = cfg.action_dim

        # Warm a few replays — the kernel caches are populated by capture_trace's
        # eager warmup, but the *trace replay* still benefits from a couple of
        # iterations to settle any first-replay cost.
        for _ in range(WARMUP_ITERS):
            _ = pipe.sample_actions_traced(images, lang_tokens)

        # Timed replays.
        ts = []
        last_actions = None
        for _ in range(PERF_ITERS):
            t0 = time.perf_counter()
            last_actions = pipe.sample_actions_traced(images, lang_tokens)
            ts.append((time.perf_counter() - t0) * 1000.0)

        assert last_actions is not None
        assert last_actions.shape == (1, ah, ad), f"shape mismatch: {tuple(last_actions.shape)}"
        assert torch.isfinite(last_actions).all(), "non-finite values in actions output"

        mean = statistics.mean(ts)
        mn = min(ts)
        mx = max(ts)
        print(
            f"\n✅ 1×8 traced replay over {PERF_ITERS} iters: "
            f"mean={mean:.2f} ms  min={mn:.2f}  max={mx:.2f}  "
            f"(28-chip baseline ≈ 43 ms — should be substantially faster, no socket hops)"
        )


@pytest.mark.skipif(
    os.environ.get("PI05_E2E_PCC", "").lower() not in ("1", "true", "yes", "on"),
    reason="PCC check off by default; set PI05_E2E_PCC=1 to enable (slow — runs CPU torch ref)",
)
def test_pcc_1x8_vs_torch():
    """OPTIONAL PCC check: compare 1×8 eager actions vs the torch
    Pi0_5Model.sample_actions reference.

    Slow (CPU reference); off by default. Target PCC ≥ 0.95 (matches
    test_denoise_expert_chain_pcc / the production traced baseline at
    PI05_E2E_PCC=1 which reports ≈ 0.9988).
    """
    from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

    with open_prefill_tp4_mesh(tp=8, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:
        pipe, cfg = _make_pipeline(mesh)
        images, lang_tokens = _build_test_inputs(cfg.siglip_config)
        img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(N_CAMS)]
        lang_masks = torch.ones(1, LANG_LEN, dtype=torch.bool)

        # FIXED_NOISE: pin RNG so eager and torch ref use the same x_0. The
        # pipeline's _refresh_noise_buffer uses torch.randn — seed before each
        # path so both pull the same noise stream.
        torch.manual_seed(SEED)
        tt_actions = pipe.sample_actions(images, lang_tokens=lang_tokens)

        # Torch reference. Loaded from the SAME checkpoint as the pipeline so
        # PCC is a fidelity number (not a model-mismatch number).
        torch.manual_seed(SEED)
        ref_model = Pi0_5Model.from_pretrained(str(CHECKPOINT_DIR))
        with torch.no_grad():
            ref_actions = ref_model.sample_actions(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                num_denoising_steps=cfg.num_denoising_steps,
            )

        # PCC over the action_horizon slice.
        a = tt_actions.flatten().float()
        b = ref_actions.flatten().float()
        m1, m2 = a.mean(), b.mean()
        s1, s2 = a.std(), b.std()
        pcc = ((a - m1) * (b - m2)).mean() / (s1 * s2)
        pcc = float(pcc.item())

        print(f"\n✅ 1×8 PCC vs torch ref: {pcc:.6f}  (shape {tuple(tt_actions.shape)})")
        assert pcc >= 0.95, f"PCC {pcc:.6f} < 0.95"
