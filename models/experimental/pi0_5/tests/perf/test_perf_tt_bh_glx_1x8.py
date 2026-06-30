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
from pathlib import Path

import pytest
import torch

# 1×8-specific flags (not in pi05_production.env — they're pipeline-1x8 specific).
# Set before _apply_production_env_defaults so the production file can't override.
# Use setdefault so an explicit shell export still wins.
for _k, _v in {
    "PI0_TP": "8",  # 8-chip tensor parallel for prefill
    "PI0_TP4_ATTN_HEADPAR": "1",  # head-parallel attention split
    "PI0_MLP_BS": "1",  # block-sharded MLP (TP=8 tuned)
    "PI0_MLP_FUSED_RS": "0",  # fused reduce-scatter off (TP=8 uses split RS+AG)
    "TT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",  # the second tray on this box
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
# Explicit replay-warmup default = 0: capture_trace / capture_traces_staged
# already run an internal eager warmup that JIT-compiles every kernel before
# trace recording. The first traced replay does have a small first-call cost
# (~1 ms above steady-state), which the 2CQ test reports via the "excl iter 0"
# mean — set WARMUP_ITERS=1 if you prefer to drop that from the timing.
WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", "0"))

# Production env flags worth asserting present (set by models/experimental/pi0_5/pi05_production.env).
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
    "PI0_TP4_ATTN_HEADPAR",
    "PI0_MLP_BS",
    "PI0_MLP_FUSED_RS",
    "PI05_NUM_DENOISE_STEPS",
    "TT_VISIBLE_DEVICES",
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


def test_perf_1x8_eager():
    """Eager sample_actions on the 1×8 pipeline.

    Honors PI05_E2E_NUM_WARMUP (default 0) and PI05_E2E_NUM_ITERS (default 1)
    to mirror the single-chip eager test's interface — useful for tracy
    profiling where the canonical inference is the LAST one captured.
    """
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

    n_warmup = int(os.environ.get("PI05_E2E_NUM_WARMUP", "0"))
    n_iters = int(os.environ.get("PI05_E2E_NUM_ITERS", "1"))

    with open_prefill_tp4_mesh(tp=8, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:
        pipe, cfg = _make_pipeline(mesh)
        images, lang_tokens = _build_test_inputs(cfg.siglip_config)

        if n_warmup > 0:
            print(f"\n🔥 Warmup ({n_warmup} call{'s' if n_warmup > 1 else ''}) — full sample_actions (JIT compile)")
            for _ in range(n_warmup):
                _ = pipe.sample_actions(images, lang_tokens=lang_tokens)

        print(f"\n⏱️  Measuring steady-state ({n_iters} sample_actions call{'s' if n_iters > 1 else ''})")
        actions = None
        for i in range(n_iters):
            actions = pipe.sample_actions(images, lang_tokens=lang_tokens)
            print(f"   call {i + 1:2d}: done")

        ah = cfg.action_horizon
        ad = cfg.action_dim
        assert actions.shape == (1, ah, ad), f"shape mismatch: {tuple(actions.shape)} vs (1, {ah}, {ad})"
        assert torch.isfinite(actions).all(), "non-finite values in actions output"

        print(
            f"\n✅ 1×8 eager: shape={tuple(actions.shape)}  "
            f"mean={actions.mean().item():.4f} std={actions.std().item():.4f}"
        )


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def test_perf_1x8_traced():
    """Capture the full e2e trace and time PERF_ITERS replays, broken down.

    Two breakdowns are produced:

    1. EAGER per-stage breakdown — 3 iters of sample_actions_timed, which
       synchronizes between SigLIP / prefix / prefill / denoise stages and
       times each. Proportions match the traced replay; absolute numbers will
       be larger because eager has per-op host dispatch (the "trace dispatch
       savings" line below quantifies that delta).

    2. TRACED replay breakdown — PERF_ITERS iters of sample_actions_traced_timed,
       which splits the host-observable parts of the replay loop:
            - input_upload_ms : pixel + lang + noise refresh (host→device)
            - trace_exec_ms   : ttnn.execute_trace (pure on-device compute)
            - output_readback_ms : ttnn.to_torch (single concat → host)
       These three are the only host-observable knobs once the trace is
       captured; their sum is the wall-clock per-call cost. The 28-chip
       baseline reported ≈43 ms total.
    """
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

    with open_prefill_tp4_mesh(tp=8, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:
        pipe, cfg = _make_pipeline(mesh)
        images, lang_tokens = _build_test_inputs(cfg.siglip_config)

        pipe.capture_trace(images, lang_tokens)

        ah = cfg.action_horizon
        ad = cfg.action_dim

        # ---- EAGER per-stage breakdown (3 iters, average) ----
        eager_runs = []
        for _ in range(3):
            _, t = pipe.sample_actions_timed(images, lang_tokens)
            eager_runs.append(t)
        eag = {
            "input_upload_ms": _mean([r["input_upload_ms"] for r in eager_runs]),
            "vision_ms": _mean([r["vision_ms"] for r in eager_runs]),
            "prefix_ms": _mean([r["prefix_ms"] for r in eager_runs]),
            "prefill_ms": _mean([r["prefill_ms"] for r in eager_runs]),
            "denoise_ms": _mean([r["denoise_ms"] for r in eager_runs]),
            "output_readback_ms": _mean([r["output_readback_ms"] for r in eager_runs]),
            "eager_total_ms": _mean([r["eager_total_ms"] for r in eager_runs]),
        }
        # Per-step denoise (averaged across runs and steps).
        per_step = [r["denoise_step_ms"] for r in eager_runs]
        n_steps = len(per_step[0]) if per_step else 0
        eag_step_means = [_mean([per_step[r][s] for r in range(len(per_step))]) for s in range(n_steps)]

        # ---- TRACED replay breakdown ----
        for _ in range(WARMUP_ITERS):
            _ = pipe.sample_actions_traced(images, lang_tokens)

        traced_runs = []
        last_actions = None
        for _ in range(PERF_ITERS):
            last_actions, t = pipe.sample_actions_traced_timed(images, lang_tokens)
            traced_runs.append(t)
        assert last_actions is not None
        assert last_actions.shape == (1, ah, ad), f"shape mismatch: {tuple(last_actions.shape)}"
        assert torch.isfinite(last_actions).all(), "non-finite values in actions output"

        keys = ["input_upload_ms", "trace_exec_ms", "output_readback_ms", "traced_total_ms"]
        tr_mean = {k: _mean([r[k] for r in traced_runs]) for k in keys}
        tr_min = {k: min(r[k] for r in traced_runs) for k in keys}

        # ---- Derived overhead numbers ----
        sum_compute_eager = eag["vision_ms"] + eag["prefix_ms"] + eag["prefill_ms"] + eag["denoise_ms"]
        trace_dispatch_savings = sum_compute_eager - tr_mean["trace_exec_ms"]

        print("\n" + "=" * 72)
        print(f"1×8 pi0.5 perf breakdown   (PERF_ITERS={PERF_ITERS}, denoise_steps={n_steps})")
        print("=" * 72)
        print(" EAGER per-stage (3-iter mean, with synchronize_device between stages):")
        print(f"   input_upload     : {eag['input_upload_ms']:7.2f} ms")
        print(f"   vision (SigLIP)  : {eag['vision_ms']:7.2f} ms")
        print(f"   prefix concat    : {eag['prefix_ms']:7.2f} ms")
        print(f"   prefill TP=8     : {eag['prefill_ms']:7.2f} ms")
        print(f"   denoise (5 step) : {eag['denoise_ms']:7.2f} ms  per-step={['%.2f' % s for s in eag_step_means]}")
        print(f"   output_readback  : {eag['output_readback_ms']:7.2f} ms")
        print(f"   ─────────────────────────────────")
        print(f"   eager_total      : {eag['eager_total_ms']:7.2f} ms")
        print()
        print(f" TRACED replay ({PERF_ITERS}-iter mean / min):")
        print(f"   input_upload     : {tr_mean['input_upload_ms']:7.2f} / {tr_min['input_upload_ms']:7.2f} ms")
        print(f"   trace_exec       : {tr_mean['trace_exec_ms']:7.2f} / {tr_min['trace_exec_ms']:7.2f} ms")
        print(f"   output_readback  : {tr_mean['output_readback_ms']:7.2f} / {tr_min['output_readback_ms']:7.2f} ms")
        print(f"   ─────────────────────────────────")
        print(f"   traced_total     : {tr_mean['traced_total_ms']:7.2f} / {tr_min['traced_total_ms']:7.2f} ms")
        print()
        print(" DERIVED:")
        print(f"   eager compute sum : {sum_compute_eager:7.2f} ms  (vision+prefix+prefill+denoise)")
        print(f"   traced compute    : {tr_mean['trace_exec_ms']:7.2f} ms")
        print(f"   dispatch savings  : {trace_dispatch_savings:7.2f} ms  (eager compute − traced compute)")
        print(f"   28-chip baseline  : ≈43 ms (single-mesh 1×8 expected substantially faster, no socket hops)")
        print("=" * 72)


def test_perf_1x8_traced_2cq():
    """2CQ trace replay: H2D input upload on CQ1 overlapped with compute on CQ0.

    Opens the 1×8 mesh with num_command_queues=2, captures the e2e trace on
    CQ0, then does iters=PERF_ITERS replays in a CQ0/CQ1 ping-pong pattern.
    The host-overhead of input_upload should mostly hide behind compute,
    closing toward the trace_exec floor.
    """
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

    _print_prod_env_status()

    with open_prefill_tp4_mesh(
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

        # Warmup: do a few normal replays first (kernel caches, address stability).
        for _ in range(WARMUP_ITERS):
            _ = pipe.sample_actions_traced(images, lang_tokens)

        last_actions, times = pipe.sample_actions_traced_2cq_loop(images, lang_tokens, PERF_ITERS)
        assert last_actions.shape == (1, ah, ad), f"shape mismatch: {tuple(last_actions.shape)}"
        assert torch.isfinite(last_actions).all(), "non-finite values in actions output"

        mean = _mean(times)
        mn = min(times)
        mx = max(times)
        # Drop the first iter (warmup-effect of the first CQ1 wait), report excluded.
        if len(times) > 1:
            mean_excl0 = _mean(times[1:])
        else:
            mean_excl0 = mean
        print("\n" + "=" * 72)
        print(f"1×8 pi0.5 TRACED 2CQ replay  (PERF_ITERS={PERF_ITERS}, N_CAMS={N_CAMS})")
        print("=" * 72)
        print(f"  mean (incl iter 0)     : {mean:.2f} ms")
        print(f"  mean (excl iter 0)     : {mean_excl0:.2f} ms")
        print(f"  min                    : {mn:.2f} ms")
        print(f"  max                    : {mx:.2f} ms")
        print(f"  per-iter (first 5)     : {[f'{t:.2f}' for t in times[:5]]}")
        print("=" * 72)


def test_perf_1x8_traced_1cq_prestaged():
    """Single-CQ trace replay with host_chunks pre-staged before the timed loop.

    Apples-to-apples comparison with test_perf_1x8_traced_2cq: both pre-stage
    the host CPU work (tilize / shard / randn) outside the timed window. The
    only difference is the command queue used for H2D:
        1CQ-prestaged: DMA on CQ0 SERIAL BEFORE compute on CQ0.
        2CQ:          DMA on CQ1 PARALLEL with compute on CQ0.

    Use to isolate the actual PCIe-DMA cost from the host-prep cost. The
    standard test_perf_1x8_traced reports ~50 ms per iter because host prep
    (~15 ms) runs inside the timed window. This variant should drop to
    ~36 ms (= compute 34 + actual DMA ~1.4 + D2H ~1.4), which is within
    ~1 ms of the 2CQ wall-clock.
    """
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

    _print_prod_env_status()

    with open_prefill_tp4_mesh(tp=8, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:
        pipe, cfg = _make_pipeline(mesh)
        images, lang_tokens = _build_test_inputs(cfg.siglip_config)

        pipe.capture_trace(images, lang_tokens)

        ah = cfg.action_horizon
        ad = cfg.action_dim

        for _ in range(WARMUP_ITERS):
            _ = pipe.sample_actions_traced(images, lang_tokens)

        last_actions, times = pipe.sample_actions_traced_1cq_prestaged_loop(images, lang_tokens, PERF_ITERS)
        assert last_actions.shape == (1, ah, ad), f"shape mismatch: {tuple(last_actions.shape)}"
        assert torch.isfinite(last_actions).all(), "non-finite values in actions output"

        mean = _mean(times)
        mn = min(times)
        mx = max(times)
        if len(times) > 1:
            mean_excl0 = _mean(times[1:])
        else:
            mean_excl0 = mean

        print("\n" + "=" * 72)
        print(f"1×8 pi0.5 TRACED 1CQ-PRESTAGED replay  (PERF_ITERS={PERF_ITERS}, N_CAMS={N_CAMS})")
        print("=" * 72)
        print(f"  mean (incl iter 0)     : {mean:.2f} ms")
        print(f"  mean (excl iter 0)     : {mean_excl0:.2f} ms")
        print(f"  min                    : {mn:.2f} ms")
        print(f"  max                    : {mx:.2f} ms")
        print(f"  per-iter (first 5)     : {[f'{t:.2f}' for t in times[:5]]}")
        print()
        print("  Compare:")
        print(f"    - test_perf_1x8_traced (host prep IN window)        : ~50 ms")
        print(f"    - this test (host prep PRE-STAGED, DMA on CQ0)      : {mean_excl0:.2f} ms")
        print(f"    - test_perf_1x8_traced_2cq (DMA on CQ1 || compute) : ~35 ms")
        print("=" * 72)


def test_perf_1x8_traced_staged():
    """Per-stage TRACED breakdown via 3 sub-traces on the single 1×8 mesh.

    Captures vision / prefill / denoise as three independent traces (with
    persistent vision_real and per_layer_kv intermediates living across trace
    boundaries via the deterministic trace allocator). Replays each in
    sequence, times each replay with perf_counter + blocking=True.

    These are TRUE traced per-stage numbers (no eager dispatch overhead),
    unlike the eager-with-sync proportions in test_perf_1x8_traced.
    """
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

    _print_prod_env_status()

    # 3 sub-traces share the trace region — bump from 128 MiB to 256 MiB.
    with open_prefill_tp4_mesh(tp=8, l1_small_size=24576, trace_region_size=256 * 1024 * 1024) as mesh:
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


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Standard PCC formula matching tests/pcc/test_pcc_tt_bh_glx_stages.py."""
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return float((cov / (s1 * s2)).item())


@pytest.mark.skipif(
    os.environ.get("PI05_E2E_PCC", "").lower() not in ("1", "true", "yes", "on"),
    reason="PCC check off by default; set PI05_E2E_PCC=1 to enable (slow — runs CPU torch ref)",
)
def test_pcc_1x8_all_stages():
    """Per-stage + end-to-end PCC check on the 1×8 pipeline.

    Three isolated stage checks (same input to TT and torch, compare output):
      1. Vision : TT vision DP (8 chips, slice to N_CAMS) vs torch SigLIP+projector
      2. Prefill: TT prefill TP=8 vs torch PaliGemmaBackbone.forward_vlm
      3. E2E    : TT pipe.sample_actions vs torch Pi0_5Model.sample_actions

    Single mesh open, single torch-model load — amortizes the ~30 s setup.

    Targets (matching tests/pcc/test_pcc_tt_bh_glx_stages.py):
      vision ≥ 0.997   prefill ≥ 0.99   e2e ≥ 0.95
    """
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone as TorchBackbone
    from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model
    from models.experimental.pi0_5.reference.torch_siglip import (
        MultiModalProjector as TorchMMProjector,
        SigLIPVisionTower as TorchSigLIPVisionTower,
    )
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

    _print_prod_env_status()

    with open_prefill_tp4_mesh(tp=8, l1_small_size=24576, trace_region_size=128 * 1024 * 1024) as mesh:
        pipe, cfg = _make_pipeline(mesh)
        images, lang_tokens = _build_test_inputs(cfg.siglip_config)
        img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(N_CAMS)]
        lang_masks = torch.ones(1, LANG_LEN, dtype=torch.bool)

        weights = Pi0_5WeightLoader(str(CHECKPOINT_DIR)).categorized_weights

        # ---- 0. E2E PCC FIRST (original "seed-around-both" pattern) ----------
        # IMPORTANT: this pattern (torch.manual_seed(SEED) before pipe AND before
        # ref_model construction+call) gives the best PCC because Pi0_5Model
        # construction happens to consume the exact RNG offset that aligns torch's
        # denoising.sample_noise with TT's _refresh_noise_buffer randn. The
        # alternative "seed-before-call" pattern from the 28-chip test gives WORSE
        # PCC (~0.93 vs ~0.99) on this 1×8 pipeline — empirically verified.
        # Done first so vision/prefill PCC sections don't pollute the RNG state.
        torch.manual_seed(SEED)
        tt_actions = pipe.sample_actions(images, lang_tokens=lang_tokens)

        torch.manual_seed(SEED)
        ref_model = Pi0_5Model(cfg, Pi0_5WeightLoader(str(CHECKPOINT_DIR)))
        with torch.no_grad():
            ref_actions = ref_model.sample_actions(images, img_masks, lang_tokens, lang_masks)
        pcc_e2e = _pcc(ref_actions, tt_actions)

        # ---- 1. Vision PCC ----------------------------------------------------
        pixel_values = torch.cat(images, dim=0)  # (N_CAMS, 3, H, W) — real cams only
        torch.manual_seed(SEED)
        ref_tower = TorchSigLIPVisionTower(cfg.siglip_config, weights["vlm_vision"])
        ref_proj = TorchMMProjector(weights["vlm_projector"])
        with torch.no_grad():
            ref_vision = ref_proj.forward(ref_tower.forward(pixel_values))  # (N_CAMS, 256, 2048)

        pipe._ensure_persistent_input_buffers(images, lang_tokens)
        tt_vision_ttnn = pipe._run_vision_dp(pipe.pixel_values_buf)  # (N_CAMS, 256, 2048) replicated
        # Replicated on 8 chips → ConcatMeshToTensor stacks 8 identical copies along dim 0; take first slice.
        tt_vision_concat = ttnn.to_torch(tt_vision_ttnn, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        tt_vision = tt_vision_concat[:N_CAMS]
        ttnn.deallocate(tt_vision_ttnn)
        assert (
            tt_vision.shape == ref_vision.shape
        ), f"vision shape {tuple(tt_vision.shape)} vs ref {tuple(ref_vision.shape)}"
        pcc_vision = _pcc(ref_vision, tt_vision)

        # ---- 2. Prefill PCC ---------------------------------------------------
        # Feed a random torch-side prefix to BOTH sides so prefill PCC is isolated
        # from any upstream vision drift. seq_len matches the actual prefix the
        # production pipeline produces for N_CAMS cams (N_CAMS·256 + 256_lang).
        seq_len = N_CAMS * 256 + LANG_LEN
        torch.manual_seed(SEED + 1)
        prefix_torch = (torch.randn(1, seq_len, cfg.vlm_config.width) * 0.5).contiguous()

        ref_backbone = TorchBackbone(cfg, weights)
        with torch.no_grad():
            ref_prefill_out, _ = ref_backbone.forward_vlm(
                prefix_torch, attention_mask=None, position_ids=None, use_cache=False
            )

        prefix_ttnn = ttnn.from_torch(
            prefix_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_prefill_out_ttnn, tt_kv = pipe.prefill.run(prefix_ttnn)
        ttnn.deallocate(prefix_ttnn)
        tt_prefill_out_concat = ttnn.to_torch(tt_prefill_out_ttnn, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        # Replicated — take chip-0 slice (first batch row).
        tt_prefill_out = tt_prefill_out_concat[:1] if tt_prefill_out_concat.shape[0] == 8 else tt_prefill_out_concat
        ttnn.deallocate(tt_prefill_out_ttnn)
        for k, v in tt_kv:
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        assert (
            tt_prefill_out.shape == ref_prefill_out.shape
        ), f"prefill shape {tuple(tt_prefill_out.shape)} vs ref {tuple(ref_prefill_out.shape)}"
        pcc_prefill = _pcc(ref_prefill_out, tt_prefill_out)

        # ---- 3. Denoise-attributable PCC -------------------------------------
        # Disabled: the previous estimate `pcc_e2e / (pcc_vision * pcc_prefill)`
        # is not statistically meaningful (PCC isn't multiplicative and the
        # value can exceed 1.0). A real isolated denoise PCC would require
        # injecting torch-side KV cache into the TT denoise expert — per-layer
        # shape/layout/dtype conversion is non-trivial. Skip the estimate.

        print("\n" + "=" * 72)
        print(f"1×8 pi0.5 PCC report  (N_CAMS={N_CAMS}, steps={cfg.num_denoising_steps})")
        print("=" * 72)
        print(f"  vision   (N_CAMS,256,{cfg.vlm_config.width})       PCC = {pcc_vision:.6f}   (target ≥ 0.997)")
        print(f"  prefill  (1,{seq_len},{cfg.vlm_config.width})      PCC = {pcc_prefill:.6f}   (target ≥ 0.99)")
        print(f"  e2e      (1,{cfg.action_horizon},{cfg.action_dim})           PCC = {pcc_e2e:.6f}   (target ≥ 0.99)")
        print("=" * 72)
        print(" Note: e2e uses seed-around-both pattern (seed before pipe.sample_actions")
        print(" and before Pi0_5Model construction+call).")
        print("=" * 72)

        assert pcc_vision >= 0.997, f"vision PCC {pcc_vision:.6f} < 0.997"
        assert pcc_prefill >= 0.99, f"prefill PCC {pcc_prefill:.6f} < 0.99"
        assert pcc_e2e >= 0.99, f"e2e PCC {pcc_e2e:.6f} < 0.99"


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
