# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Option B perf benchmark — staged e2e timings on real pi0.5 weights.

Mirrors `test_option_c_benchmark.py` so the two perf reports compare
apples-to-apples. The existing `test_option_b_benchmark.py` is the
microbenchmark suite (per-op breakdown, all_reduce isolation, dispatch
floor — random weights). This file is the e2e counterpart, identical in
structure to the Option C benchmark file so we get matched per-stage
numbers across the two pipelines.

Three benchmarks (1:1 with `test_option_c_benchmark.py`):

  1. E2E STAGED BREAKDOWN — drive the 4-stage Option B pipeline N times
     and report median wall-clock per stage (vision / transport / VLM 1st
     half / transport / VLM 2nd half / KV migration / denoise / total)
     plus per-Euler-step timings.

  2. TP=8 vs SUBMESH-REPLICATED — same workload, both placements, report
     speedup (or regression) per stage. Toggles the `tp_shard` flag on
     `StageVLM` / `Stage3Expert`.

  3. PREFILL SEQ-LEN SWEEP — at a fixed placement, sweep the
     lang-token count {64, 128, 256, 512} and report how stage 1 + stage
     2 VLM prefill latency scales. Catches O(S²) terms (SDPA, mask)
     drowning out the linear weight load.

Skipped by default (set `PI0_OB_E2E_BENCHMARK=1` to run).
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Dict, List

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.tt.option_b.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.option_b.pipeline import (
    Pi0_5PipelineB,
)
from models.experimental.pi0_5.tt.option_b.stages import build_shrunk_layout
from models.experimental.pi0_5.tt.option_b.transport import send_activation_via_host

BENCH_ENABLED = os.environ.get("PI0_OB_E2E_BENCHMARK") == "1"
pytestmark = pytest.mark.skipif(
    not BENCH_ENABLED, reason="set PI0_OB_E2E_BENCHMARK=1 to run Option B e2e perf benchmarks"
)

# Knob names are intentionally identical (modulo the OB_ prefix) to the
# Option C benchmark so the two are easy to drive with the same shell
# vars in a single run.
WARMUP = int(os.environ.get("PI0_OB_E2E_WARMUP", "2"))
ITERS = int(os.environ.get("PI0_OB_E2E_ITERS", "5"))

# Shape constants — same as test_option_c_benchmark.py / test_option_b_benchmark.py.
LANG_SEQ_LEN = 256
PREFIX_LEN = 256 + LANG_SEQ_LEN  # 512 — VLM prefill sequence length
ACTION_HORIZON = 10  # LIBERO upstream
ACTION_HORIZON_PADDED = 32  # ceil(10 / 32) * 32
NUM_DENOISE_STEPS = int(os.environ.get("PI0_OB_DENOISE_STEPS", "10"))

# pi0.5 upstream-openpi LIBERO finetune.
_REAL_CKPT = os.environ.get("PI0_OB_CHECKPOINT", "/home/tt-admin/pi05_cache/pi05_libero_upstream")


def _quantiles(values: List[float], qs=(0.05, 0.5, 0.95)) -> List[float]:
    if not values:
        return [float("nan")] * len(qs)
    s = sorted(values)
    n = len(s)
    return [s[max(0, min(n - 1, int(q * n)))] for q in qs]


def _format_ms(values: List[float], label: str) -> str:
    if not values:
        return f"{label:<32s} (no samples)"
    p5, p50, p95 = _quantiles(values)
    return f"{label:<32s} p5={p5:7.2f} p50={p50:7.2f} p95={p95:7.2f} ms  (n={len(values)})"


def _synthetic_inputs(cfg: PaliGemmaConfig, prefix_len: int = PREFIX_LEN):
    """Random torch inputs matching what stage 0 expects (same as Option C)."""
    S_lang = max(0, prefix_len - 256)
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


def _replicate(t: torch.Tensor, mesh, dtype=ttnn.bfloat16) -> "ttnn.Tensor":
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )


# ============================================================== #
# Benchmark 1: E2E staged breakdown                              #
# ============================================================== #


@pytest.mark.timeout(2400)
def test_ob_bench_e2e_staged_breakdown():
    """Run the full Option B pipeline N times, report per-stage medians.

    Uses the shrunk layout (vlm_depth=2, expert_depth=1) so replicated
    uploads fit per-chip L1, matching test_option_c_benchmark.py.

    Pi0_5PipelineB's `run_one_step` only fires a single expert step (the
    full denoise loop isn't part of the orchestrator yet — see
    pipeline.py line 14). To match the Option C `run_inference` contract
    we drive the 4 stages here and call `Stage3Expert.denoise` directly
    for the full Euler loop.
    """
    cfg = PaliGemmaConfig()
    weights = _load_real_weights(cfg)
    shrunk = build_shrunk_layout(vlm_depth=2, expert_depth=1)

    with open_galaxy_mesh(shrunk) as (_parent, submeshes):
        pipe = Pi0_5PipelineB(
            layout=shrunk,
            submeshes=submeshes,
            config=cfg,
            weights=weights,
        )
        pipe.initialize()
        # Match the matched workload for the denoise loop.
        pipe.stage_3.denoise_steps = NUM_DENOISE_STEPS
        pipe.stage_3.action_horizon = ACTION_HORIZON

        pix, lang, noisy = _synthetic_inputs(cfg)
        sm1, sm2, sm3 = submeshes[1], submeshes[2], submeshes[3]

        def _one_e2e() -> Dict[str, object]:
            t: Dict[str, object] = {}
            t_total0 = time.perf_counter()

            # ----- stage 0 vision -------------------------------------------
            t0 = time.perf_counter()
            prefix_s0 = pipe.stage_0.forward(pix, lang)
            t["stage_0_vision_ms"] = (time.perf_counter() - t0) * 1000

            # ----- transport 0 → 1 ------------------------------------------
            t0 = time.perf_counter()
            h_on_1 = send_activation_via_host(prefix_s0, sm1)
            ttnn.deallocate(prefix_s0)
            t["transport_0_to_1_ms"] = (time.perf_counter() - t0) * 1000

            S_prefix = h_on_1.shape[1]
            mask_s1 = _replicate(torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32), sm1)

            # ----- stage 1 VLM 1st half -------------------------------------
            t0 = time.perf_counter()
            h_after_1 = pipe.stage_1.forward(h_on_1, attention_mask=mask_s1, use_cache=True)
            t["stage_1_vlm_first_half_ms"] = (time.perf_counter() - t0) * 1000
            ttnn.deallocate(mask_s1)

            # ----- transport 1 → 2 ------------------------------------------
            t0 = time.perf_counter()
            h_on_2 = send_activation_via_host(h_after_1, sm2)
            ttnn.deallocate(h_after_1)
            t["transport_1_to_2_ms"] = (time.perf_counter() - t0) * 1000

            mask_s2 = _replicate(torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32), sm2)

            # ----- stage 2 VLM 2nd half (captures KV) -----------------------
            t0 = time.perf_counter()
            h_after_2 = pipe.stage_2.forward(h_on_2, attention_mask=mask_s2, use_cache=True)
            t["stage_2_vlm_second_half_ms"] = (time.perf_counter() - t0) * 1000
            ttnn.deallocate(h_after_2)
            ttnn.deallocate(mask_s2)

            # ----- KV migration → submesh 3 ---------------------------------
            t0 = time.perf_counter()
            prefix_kv_on_3 = pipe._migrate_kv()
            t["kv_migration_ms"] = (time.perf_counter() - t0) * 1000

            # ----- full N-step Euler denoise --------------------------------
            noisy_on_3 = _replicate(noisy, sm3)
            joint_mask_s3 = _replicate(
                torch.zeros(
                    1,
                    1,
                    ACTION_HORIZON_PADDED,
                    S_prefix + ACTION_HORIZON_PADDED,
                    dtype=torch.float32,
                ),
                sm3,
            )
            step_ms = []
            num_steps = pipe.stage_3.denoise_steps
            dt = -1.0 / num_steps
            x_t = noisy_on_3
            x_t_owned = False
            t0 = time.perf_counter()
            for i in range(num_steps):
                ts = 1.0 - i / num_steps
                B = x_t.shape[0]
                t_step0 = time.perf_counter()
                adarms_cond = pipe.stage_3.suffix.embed_adarms_cond(ts, batch_size=B)
                suffix_h = pipe.stage_3.suffix.embed_actions(x_t)
                velocity_hidden = pipe.stage_3.slice.forward(
                    suffix_h,
                    adarms_cond,
                    prefix_kv_cache=prefix_kv_on_3,
                    attention_mask=joint_mask_s3,
                )
                ttnn.deallocate(suffix_h)
                ttnn.deallocate(adarms_cond)
                v_t = pipe.stage_3.suffix.project_output(velocity_hidden)
                ttnn.deallocate(velocity_hidden)
                dx = ttnn.multiply(v_t, dt)
                ttnn.deallocate(v_t)
                x_t_new = ttnn.add(x_t, dx)
                ttnn.deallocate(dx)
                if x_t_owned:
                    ttnn.deallocate(x_t)
                x_t = x_t_new
                x_t_owned = True
                step_ms.append((time.perf_counter() - t_step0) * 1000)
            t["stage_3_denoise_ms"] = (time.perf_counter() - t0) * 1000
            t["__step_ms"] = step_ms
            ttnn.deallocate(joint_mask_s3)
            ttnn.deallocate(x_t)

            t["total_ms"] = (time.perf_counter() - t_total0) * 1000
            return t

        # Warmup
        for _ in range(WARMUP):
            _ = _one_e2e()

        all_t: List[Dict[str, object]] = []
        all_step_ms: List[float] = []
        for _ in range(ITERS):
            t = _one_e2e()
            all_step_ms.extend(t.pop("__step_ms"))
            all_t.append(t)

        print()
        print(
            f"== Option B e2e staged breakdown "
            f"(replicated, shrunk vlm_depth=2 expert_depth=1, "
            f"prefix={PREFIX_LEN}, denoise_steps={NUM_DENOISE_STEPS}, "
            f"horizon={ACTION_HORIZON}→pad{ACTION_HORIZON_PADDED}) =="
        )
        for label in (
            "stage_0_vision_ms",
            "transport_0_to_1_ms",
            "stage_1_vlm_first_half_ms",
            "transport_1_to_2_ms",
            "stage_2_vlm_second_half_ms",
            "kv_migration_ms",
            "stage_3_denoise_ms",
            "total_ms",
        ):
            print(_format_ms([t[label] for t in all_t], label))
        print(_format_ms(all_step_ms, "denoise_step (per Euler)"))


# ============================================================== #
# Benchmark 2: replicated vs TP=8                                #
# ============================================================== #


@pytest.mark.timeout(2400)
def test_ob_bench_replicated_vs_tp():
    """Same workload, both placements (StageVLM with tp_shard=False vs True),
    report wall-clock delta on the prefill stage. Mirrors Option C's
    `test_oc_bench_replicated_vs_layer_paired`.
    """
    from models.experimental.pi0_5.tt.option_b.stage_vlm import StageVLM

    cfg = PaliGemmaConfig()
    weights = _load_real_weights(cfg)
    shrunk = build_shrunk_layout(vlm_depth=2, expert_depth=1)

    def _stage1_only(tp_shard: bool, submeshes):
        # Only the 1st-half VLM stage in this microbench — Option C's analog
        # times the equivalent layer-paired prefill on its 18-chip submesh.
        sm = submeshes[1]
        spec = shrunk.stages[1]
        stage = StageVLM(spec, sm, cfg, weights, tp_shard=tp_shard)
        stage.initialize()
        S = PREFIX_LEN
        h0 = ttnn.from_torch(
            torch.randn(1, S, cfg.vlm_config.width) * 0.02,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=sm,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(sm),
        )
        mask = ttnn.from_torch(
            torch.zeros(1, 1, S, S, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=sm,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(sm),
        )
        for _ in range(WARMUP):
            out = stage.forward(h0, attention_mask=mask, use_cache=True)
            ttnn.deallocate(out)
        samples_ms = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            out = stage.forward(h0, attention_mask=mask, use_cache=True)
            ttnn.synchronize_device(sm)
            samples_ms.append((time.perf_counter() - t0) * 1000)
            ttnn.deallocate(out)
        ttnn.deallocate(h0)
        ttnn.deallocate(mask)
        return samples_ms

    with open_galaxy_mesh(shrunk) as (_parent, submeshes):
        repl_ms = _stage1_only(False, submeshes)

    with open_galaxy_mesh(shrunk) as (_parent, submeshes):
        tp_ms = _stage1_only(True, submeshes)

    print()
    print("== Option B prefill (stage 1): replicated vs TP=8 ==")
    print(_format_ms(repl_ms, "prefill replicated"))
    print(_format_ms(tp_ms, "prefill TP=8"))
    if repl_ms and tp_ms:
        delta = statistics.median(tp_ms) - statistics.median(repl_ms)
        print(f"  median delta (TP - replicated) = {delta:+.2f} ms")


# ============================================================== #
# Benchmark 3: prefill seq-len sweep                             #
# ============================================================== #


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("seq_len", [64, 128, 256, 512])
def test_ob_bench_prefill_seqlen_sweep(seq_len):
    """How does Option B prefill scale with prefix length? Mirrors Option C's
    `test_oc_bench_prefill_seqlen_sweep`. One stage-1 forward at varying S
    on the 8-chip submesh; pure microbenchmark — useful for catching O(S²)
    regressions in SDPA.
    """
    from models.experimental.pi0_5.tt.option_b.stage_vlm import StageVLM

    cfg = PaliGemmaConfig()
    weights = _load_real_weights(cfg)
    shrunk = build_shrunk_layout(vlm_depth=2, expert_depth=1)

    with open_galaxy_mesh(shrunk) as (_parent, submeshes):
        sm = submeshes[1]
        stage = StageVLM(shrunk.stages[1], sm, cfg, weights)
        stage.initialize()

        h0 = ttnn.from_torch(
            torch.randn(1, seq_len, cfg.vlm_config.width) * 0.02,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=sm,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(sm),
        )
        mask = ttnn.from_torch(
            torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=sm,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(sm),
        )

        for _ in range(WARMUP):
            out = stage.forward(h0, attention_mask=mask, use_cache=True)
            ttnn.deallocate(out)

        samples_us = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            out = stage.forward(h0, attention_mask=mask, use_cache=True)
            ttnn.synchronize_device(sm)
            samples_us.append((time.perf_counter() - t0) * 1e6)
            ttnn.deallocate(out)

        p5, p50, p95 = _quantiles(samples_us)
        print(f"prefill stage1 S={seq_len:>3d}: " f"p5={p5/1000:7.2f}  p50={p50/1000:7.2f}  p95={p95/1000:7.2f} ms")
