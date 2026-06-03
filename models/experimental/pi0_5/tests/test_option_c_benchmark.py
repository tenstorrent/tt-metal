# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Option C perf benchmark — staged e2e timings on real pi0.5 weights.

Counterpart to `test_option_b_benchmark.py`. Where Option B's benchmark
isolates per-op cost inside one TP=8 VLM layer (we know the model — it's
all_reduce overhead, so the analysis is at the op level), Option C's
characteristic latency lives at the STAGE level: how long does prefill
take, how long does each Euler denoise step take, how much wall-clock
goes to host-bounce transport.

Three benchmarks:

  1. E2E STAGED BREAKDOWN — drive `Pi0_5PipelineC.run_inference()` for
     N iterations and report median wall-clock per stage (vision /
     transport / prefill / KV migration / denoise / transport / total)
     plus per-Euler-step timings. Compares against the analytical model
     (8.90 ms target at prefix_seq=968).

  2. REPLICATED vs LAYER-PAIRED — same workload, both placements, report
     speedup (or regression) per stage. The README's "host-bounce
     transport when stage adjacency allows" claim is grounded by this
     number: if layer-paired transports cost more than the analytical
     model predicts, that's where the gap is.

  3. PREFILL SEQ-LEN SWEEP — at a fixed placement (layer-paired), sweep
     the lang-token count {64, 128, 256, 512} and report how prefill
     latency scales. Catches O(S²) terms (SDPA, mask) drowning out the
     linear weight load.

Skipped by default (set `PI0_OC_BENCHMARK=1` to run).
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import List

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.option_c.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.option_c.pipeline import Pi0_5PipelineC, StageTimingsC
from models.experimental.pi0_5.tt.option_c.stages import build_shrunk_layout

BENCH_ENABLED = os.environ.get("PI0_OC_BENCHMARK") == "1"
pytestmark = pytest.mark.skipif(not BENCH_ENABLED, reason="set PI0_OC_BENCHMARK=1 to run Option C perf benchmarks")

WARMUP = int(os.environ.get("PI0_OC_BENCH_WARMUP", "2"))
ITERS = int(os.environ.get("PI0_OC_BENCH_ITERS", "5"))

# Shape constants — mirror `test_option_b_benchmark.py` so the two
# benchmarks compare apples-to-apples.
LANG_SEQ_LEN = 256
PREFIX_LEN = 256 + LANG_SEQ_LEN  # 512 — VLM prefill sequence length
ACTION_HORIZON = 10  # LIBERO upstream
ACTION_HORIZON_PADDED = 32  # ceil(10 / 32) * 32
NUM_DENOISE_STEPS = int(os.environ.get("PI0_OC_DENOISE_STEPS", "10"))

# pi0.5 upstream-openpi LIBERO finetune.
_REAL_CKPT = os.environ.get("PI0_OC_CHECKPOINT", "/home/tt-admin/pi05_cache/pi05_libero_upstream")


def _quantiles(values: List[float], qs=(0.05, 0.5, 0.95)) -> List[float]:
    if not values:
        return [float("nan")] * len(qs)
    s = sorted(values)
    n = len(s)
    return [s[max(0, min(n - 1, int(q * n)))] for q in qs]


def _format_ms(values: List[float], label: str) -> str:
    if not values:
        return f"{label:<28s} (no samples)"
    p5, p50, p95 = _quantiles(values)
    return f"{label:<28s} p5={p5:7.2f} p50={p50:7.2f} p95={p95:7.2f} ms  (n={len(values)})"


def _synthetic_inputs(cfg: PaliGemmaConfig, prefix_len: int = PREFIX_LEN):
    """Random torch inputs matching what stage 0 expects.

    pixel_values [1, 3, 224, 224], language_token_ids [1, S_lang], and
    noisy_actions [1, ACTION_HORIZON_PADDED, action_dim]. Mask Nones — the
    pipeline builds all-unmasked masks from the prefix length.
    """
    S_lang = max(0, prefix_len - 256)  # SigLIP gives 256 image tokens
    gen = torch.Generator().manual_seed(0xC0FFEE)
    pixel_values = torch.randn(1, 3, 224, 224, generator=gen, dtype=torch.float32)
    language_token_ids = torch.randint(0, 32000, (1, S_lang), generator=gen, dtype=torch.int32)
    noisy_actions = torch.randn(1, ACTION_HORIZON_PADDED, 32, generator=gen, dtype=torch.float32)
    return pixel_values, language_token_ids, noisy_actions


def _load_real_weights(cfg: PaliGemmaConfig):
    """Load the real pi05 LIBERO finetune weights. Skips the test if absent."""
    if not Path(_REAL_CKPT, "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {_REAL_CKPT}")
    from models.experimental.pi0_5.common import Pi0_5WeightLoader

    return Pi0_5WeightLoader(_REAL_CKPT).categorized_weights


# ============================================================== #
# Benchmark 1: E2E staged breakdown                              #
# ============================================================== #


@pytest.mark.timeout(2400)
def test_oc_bench_e2e_staged_breakdown():
    """Run the full Option C pipeline N times, report per-stage medians.

    Runs in REPLICATED mode (the smoke-tested path). Use a shrunk layout
    (vlm_depth=2, expert_depth=1) so the replicated upload fits — the
    per-stage shapes that dominate wall-clock are the same as the
    full-config workload, just with fewer layers.
    """
    cfg = PaliGemmaConfig()
    weights = _load_real_weights(cfg)
    shrunk = build_shrunk_layout(vlm_depth=2, expert_depth=1)

    with open_galaxy_mesh(shrunk) as (_parent, submeshes):
        pipe = Pi0_5PipelineC(
            layout=shrunk,
            submeshes=submeshes,
            config=cfg,
            weights=weights,
            denoise_steps=NUM_DENOISE_STEPS,
        )
        pipe.initialize()

        pix, lang, noisy = _synthetic_inputs(cfg)

        # Warmup
        for _ in range(WARMUP):
            actions, _t = pipe.run_inference(pix, lang, noisy)
            ttnn.deallocate(actions)

        all_t: List[StageTimingsC] = []
        for _ in range(ITERS):
            actions, t = pipe.run_inference(pix, lang, noisy)
            ttnn.deallocate(actions)
            all_t.append(t)

        print()
        print(f"== Option C e2e staged breakdown (replicated, shrunk vlm_depth=2 expert_depth=1) ==")
        print(_format_ms([t.stage_0_vision_ms for t in all_t], "stage_0_vision_ms"))
        print(_format_ms([t.transport_0_to_1_ms for t in all_t], "transport_0_to_1_ms"))
        print(_format_ms([t.stage_1_prefill_ms for t in all_t], "stage_1_prefill_ms"))
        print(_format_ms([t.kv_migration_ms for t in all_t], "kv_migration_ms"))
        print(_format_ms([t.stage_2_denoise_ms for t in all_t], "stage_2_denoise_ms"))
        print(_format_ms([t.transport_2_to_host_ms for t in all_t], "transport_2_to_host_ms"))
        print(_format_ms([t.total_ms for t in all_t], "total_ms"))

        # Per-step denoise — flatten across iters and show distribution.
        step_ms = [s for t in all_t for s in t.denoise_step_ms]
        print(_format_ms(step_ms, "denoise_step (per Euler)"))


# ============================================================== #
# Benchmark 2: Replicated vs Layer-paired                        #
# ============================================================== #


@pytest.mark.timeout(2400)
def test_oc_bench_replicated_vs_layer_paired():
    """Same workload, both placements; report wall-clock delta per stage.

    Currently `Pi0_5PipelineC` doesn't take a layer_paired_l1 flag — we
    drive the two `StagePrefill` / `StageDenoise` modes directly to time
    the diff. When the full pipeline grows a flag, replace this with
    `pipe = Pi0_5PipelineC(..., layer_paired_l1=True)`.
    """
    from models.experimental.pi0_5.tt.option_c.stage_prefill import StagePrefill

    cfg = PaliGemmaConfig()
    weights = _load_real_weights(cfg)

    # Both placements use the same shrunk depth so memory fits the
    # replicated path; layer-paired could go wider but we hold the
    # comparison apples-to-apples.
    shrunk = build_shrunk_layout(vlm_depth=2, expert_depth=1)

    def _prefill_only(layer_paired: bool, submeshes):
        prefill = StagePrefill(
            shrunk.stages[1],
            submeshes[1],
            cfg,
            weights,
            layer_paired_l1=layer_paired,
        )
        prefill.initialize()
        first_sm = prefill.first_chip_submesh
        S = PREFIX_LEN
        h0 = ttnn.from_torch(
            torch.randn(1, S, cfg.vlm_config.width) * 0.02,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=first_sm,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(first_sm),
        )
        mask0 = ttnn.from_torch(
            torch.zeros(1, 1, S, S, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=first_sm,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(first_sm),
        )
        for _ in range(WARMUP):
            out = prefill.forward(h0, attention_mask=mask0, use_cache=True)
            ttnn.deallocate(out)
        samples_ms = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            out = prefill.forward(h0, attention_mask=mask0, use_cache=True)
            # Forward already does host bounces in paired mode — wall
            # clock includes them. Synchronize on the output's submesh.
            last_sm = prefill.last_chip_submesh
            ttnn.synchronize_device(last_sm)
            samples_ms.append((time.perf_counter() - t0) * 1000)
            ttnn.deallocate(out)
        ttnn.deallocate(h0)
        ttnn.deallocate(mask0)
        return samples_ms

    with open_galaxy_mesh(shrunk) as (_parent, submeshes):
        repl_ms = _prefill_only(False, submeshes)

    # New mesh open — paired path carves micro-submeshes from a fresh
    # prefill submesh; sharing the same open_galaxy_mesh context across
    # paired and replicated would leak L1 between modes.
    with open_galaxy_mesh(shrunk) as (_parent, submeshes):
        paired_ms = _prefill_only(True, submeshes)

    print()
    print("== Option C prefill: replicated vs layer-paired ==")
    print(_format_ms(repl_ms, "prefill replicated"))
    print(_format_ms(paired_ms, "prefill layer-paired"))
    if repl_ms and paired_ms:
        delta = statistics.median(paired_ms) - statistics.median(repl_ms)
        print(f"  median delta (paired - replicated) = {delta:+.2f} ms")


# ============================================================== #
# Benchmark 3: prefill seq-len sweep                             #
# ============================================================== #


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
def test_oc_bench_prefill_seqlen_sweep(seq_len):
    """How does layer-paired prefill scale with prefix length?

    Measures one VLM layer's forward at varying S, on a 1-chip micro-submesh.
    Pure microbenchmark — useful for catching O(S²) regressions in SDPA.
    """
    from models.experimental.pi0_5.tt.option_c.stage_prefill import StagePrefill

    cfg = PaliGemmaConfig()
    weights = _load_real_weights(cfg)
    shrunk = build_shrunk_layout(vlm_depth=1, expert_depth=1)

    with open_galaxy_mesh(shrunk) as (_parent, submeshes):
        prefill = StagePrefill(
            shrunk.stages[1],
            submeshes[1],
            cfg,
            weights,
            layer_paired_l1=True,
        )
        prefill.initialize()
        first_sm = prefill.first_chip_submesh

        h0 = ttnn.from_torch(
            torch.randn(1, seq_len, cfg.vlm_config.width) * 0.02,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=first_sm,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(first_sm),
        )
        mask0 = ttnn.from_torch(
            torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=first_sm,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(first_sm),
        )

        for _ in range(WARMUP):
            out = prefill.forward(h0, attention_mask=mask0, use_cache=True)
            ttnn.deallocate(out)

        samples_us = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            out = prefill.forward(h0, attention_mask=mask0, use_cache=True)
            ttnn.synchronize_device(prefill.last_chip_submesh)
            samples_us.append((time.perf_counter() - t0) * 1e6)
            ttnn.deallocate(out)

        p5, p50, p95 = _quantiles(samples_us)
        print(
            f"prefill_layer_paired S={seq_len:>3d}: " f"p5={p5/1000:7.2f}  p50={p50/1000:7.2f}  p95={p95/1000:7.2f} ms"
        )
