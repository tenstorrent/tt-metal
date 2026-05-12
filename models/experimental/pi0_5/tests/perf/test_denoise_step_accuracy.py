# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 denoise-step accuracy sweep — self-consistency at different
`num_denoising_steps` values on real lerobot/pi05_base weights.

Idea:
  Take the 10-step output as the reference. For each smaller N, run
  sample_actions with the SAME initial noise and inputs. Compare to the
  10-step output by:
    - cosine similarity (direction)
    - L2 / RMSE (magnitude)
    - max per-element |delta| (worst-case wobble)

  This answers "can we drop the denoise steps without changing the action
  output?" — a *necessary* condition before any expensive sim/real eval.

  Per the Dense-Jump Flow Matching paper (arxiv 2509.13574), vanilla
  flow-matching policies often *peak* in accuracy at 2-4 steps; we test
  for that empirically here.

Skipped if the checkpoint isn't present locally.
"""

from pathlib import Path
from typing import Dict

import pytest
import torch
import ttnn

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_base"
SEED = 0
LANG_SEQ_LEN = 32

# Steps to test, in execution order. 10 first (reference), then descending.
STEP_SWEEP = [10, 5, 4, 3, 2, 1]

# Pass/fail threshold (informational): cosine >= this AND max delta <= max_delta
COSINE_OK = 0.995
MAX_DELTA_OK = 0.05

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def _build_inputs(device):
    torch.manual_seed(SEED)
    image = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    img_mask = torch.ones(1, dtype=torch.bool)
    lang_tokens = torch.randint(0, 256000, (1, LANG_SEQ_LEN), dtype=torch.int32)
    lang_masks = torch.ones(1, LANG_SEQ_LEN, dtype=torch.bool)

    image_ttnn = ttnn.from_torch(
        image,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    img_mask_ttnn = ttnn.from_torch(
        img_mask.float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    lang_tokens_ttnn = ttnn.from_torch(
        lang_tokens.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    lang_masks_ttnn = ttnn.from_torch(
        lang_masks.to(torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return image_ttnn, img_mask_ttnn, lang_tokens_ttnn, lang_masks_ttnn


def _set_num_steps(model, n: int) -> None:
    """Reconfigure model.num_denoising_steps without rebuilding weights.

    Rebuilds the cheap precomputed lists (timesteps + adarms_cond) so the
    fast-path indexing in sample_actions is consistent with the new N.
    Old precomputed tensors are not deallocated explicitly — they just
    become unreachable list entries. Memory usage per swap is ~60 KB.
    """
    model.denoise_config.num_steps = n
    model._precompute_bs1_timestep_tensors()
    model._precompute_bs1_adarms_cond()


def _set_initial_noise(model, device, seed: int) -> None:
    """Replace model.x_t_ttnn with a fresh (deterministic) noise tensor."""
    g = torch.Generator().manual_seed(seed)
    cfg = model.config
    noise = torch.randn(1, cfg.action_horizon, cfg.action_dim, generator=g)
    model.x_t_ttnn = ttnn.from_torch(
        noise,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _run_one(model, image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn) -> torch.Tensor:
    out = model.sample_actions(
        images=[image_ttnn],
        img_masks=[img_mask],
        lang_tokens=lang_tokens_ttnn,
        lang_masks=lang_masks_ttnn,
        state=None,
    )
    ttnn.synchronize_device(model.device)
    acts = ttnn.to_torch(out)
    return acts[:, : model.config.action_horizon, : model.config.action_dim].float()


def _compare(out: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """cosine sim, RMS error, max absolute delta — flat vector comparison."""
    a = out.flatten()
    b = ref.flatten()
    cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    rms = torch.sqrt(((a - b) ** 2).mean()).item()
    mx = (a - b).abs().max().item()
    return {"cosine": cos, "rms": rms, "max_delta": mx}


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576}],
    indirect=True,
)
def test_pi0_5_denoise_step_accuracy_sweep(device):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.ttnn_pi0_5_model import Pi0_5ModelTTNN

    print(f"\n📋 Loading PI0.5 TTNN model from {CHECKPOINT_DIR}")
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    # Build with max num_steps so the precomputed pad fits all sweep values.
    cfg = Pi0_5ModelConfig(num_denoising_steps=max(STEP_SWEEP))
    model = Pi0_5ModelTTNN(cfg, loader, device)
    print(f"✅ Model loaded")

    image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn = _build_inputs(device)

    results: Dict[int, Dict[str, float]] = {}
    ref_actions: torch.Tensor = None

    for n in STEP_SWEEP:
        _set_num_steps(model, n)
        _set_initial_noise(model, device, SEED + 1)  # same noise every call
        with torch.no_grad():
            actions = _run_one(model, image_ttnn, img_mask, lang_tokens_ttnn, lang_masks_ttnn)
        if ref_actions is None:
            ref_actions = actions
            results[n] = {"cosine": 1.0, "rms": 0.0, "max_delta": 0.0}
            print(f"  N={n:2d}: reference captured, action shape={tuple(actions.shape)}")
        else:
            results[n] = _compare(actions, ref_actions)
            r = results[n]
            print(f"  N={n:2d}: cos={r['cosine']:.6f}  rms={r['rms']:.6f}  " f"max|Δ|={r['max_delta']:.6f}")

    # Projected per-call latency at each N. Prefix prefill is ~34 ms fixed,
    # per-step expert ~10.8 ms (from our perf measurements with all opts).
    PREFIX_MS = 34.0
    PER_STEP_MS = (142.0 - PREFIX_MS) / 10.0  # 10.8 ms / step

    print("\n" + "=" * 84)
    print("  PI0.5 DENOISE-STEP ACCURACY + PERF SWEEP (real pi05_base weights, Blackhole)")
    print("=" * 84)
    print(
        f"  {'N steps':>8}  {'~latency':>10}  {'~actions/s':>11}  "
        f"{'cosine':>10}  {'rms':>10}  {'max|Δ|':>10}  {'verdict':>10}"
    )
    print("-" * 84)
    for n in STEP_SWEEP:
        r = results[n]
        lat = PREFIX_MS + n * PER_STEP_MS
        fps = 1000.0 / lat * cfg.action_horizon
        if n == max(STEP_SWEEP):
            verdict = "REF"
        elif r["cosine"] >= COSINE_OK and r["max_delta"] <= MAX_DELTA_OK:
            verdict = "OK ✅"
        else:
            verdict = "DEGRADE ⚠"
        print(
            f"  {n:>8d}  {lat:>7.1f} ms  {fps:>9.0f}  "
            f"{r['cosine']:>10.6f}  {r['rms']:>10.6f}  {r['max_delta']:>10.6f}  {verdict:>10}"
        )
    print("=" * 84)
    print(f"  Threshold: cosine ≥ {COSINE_OK}  AND  max|Δ| ≤ {MAX_DELTA_OK}")
    print("=" * 84)

    # Always pass — this is a measurement test, not a correctness gate. Output is
    # the diagnostic table itself.
    assert results[max(STEP_SWEEP)]["cosine"] == 1.0
