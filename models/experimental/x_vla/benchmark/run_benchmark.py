#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end X-VLA inference benchmark — fixed metric harness.

This is the oracle. Per PROGRAM.md it is **not** modified across iterations:
optimizations replace pieces of the model, not the harness.

Backend selection is via the env var `XVLA_BACKEND`:
  - `torch_cpu` (default) — vanilla lerobot/HF X-VLA on CPU. Reference.
  - `ttnn`               — partial/full TT-NN port from `models/experimental/x_vla/tt`.

Outputs (greppable):
    inference_speed=<frames per second, where one frame = one action step>
    accuracy=<PCC% vs cached reference action chunk; 100.0 on first run>
    peak_dram=<peak DRAM MB used during inference; 0 on torch_cpu>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Bootstrap broken lerobot imports before anything else.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent.parent.parent))  # tt-metal root
sys.path.insert(0, str(_HERE))
from lerobot_bootstrap import install as _bootstrap_install  # noqa: E402

_bootstrap_install()


# =============================================================================
# Config
# =============================================================================

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "xvla_base"
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"
REFERENCE_FILE = REFERENCE_DIR / "action_chunk_seed42.npy"
SEED = 42
WARMUP_RUNS = 1
TIMED_RUNS = 3


# =============================================================================
# Input synthesis (mirrors xvla config: 3 cameras + state(8) + bart language)
# =============================================================================


def build_synthetic_batch(policy, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Construct a deterministic synthetic input batch matching xvla schema.

    Bypasses the lerobot processor pipeline (tokenizer, normalization) by
    feeding the model exactly the tensors the policy expects post-processing.
    Determinism comes from a fixed torch generator seeded with SEED.
    """
    from lerobot.utils.constants import (
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    cfg = policy.config
    g = torch.Generator(device=device).manual_seed(SEED)

    batch = {}

    # Three camera views per config: image (3,256,256), image2 (3,256,256), image3 (3,224,224).
    # The xvla policy resizes each to `resize_imgs_with_padding` (224x224) before stacking,
    # so giving it the raw shapes from the config is correct.
    img_specs = []
    for key, feat in cfg.input_features.items():
        if hasattr(feat, "type") and str(feat.type).endswith("VISUAL"):
            img_specs.append((key, tuple(feat.shape)))
    img_specs.sort()  # deterministic order
    for key, shape in img_specs:
        batch[key] = torch.rand(1, *shape, generator=g, device=device)  # in [0,1]

    # Proprio state
    batch[OBS_STATE] = torch.randn(1, cfg.robot_state_feature.shape[0], generator=g, device=device)

    # Language tokens — bart vocab=51289. The HF config advertises
    # tokenizer_max_length=1024 but `pad_language_to=max_length` in combo
    # with that value produces a sequence (text + image tokens + 3 image
    # views' worth of aux + 30 action + 32 soft prompts) that exceeds the
    # model's own `max_len_seq=512`. Real-world xvla inference uses short
    # natural-language instructions (a few words). We fix the benchmark
    # prompt length at LANG_TOKENS = 16, which keeps the merged sequence
    # comfortably under the cap and represents a typical "pick up X" prompt.
    LANG_TOKENS = 16
    # token id 0 = <s>, 2 = </s>; 1 = <pad>. Fill: <s> + N junk + </s> + pad.
    tokens = torch.full((1, LANG_TOKENS), 1, dtype=torch.long, device=device)
    tokens[0, 0] = 0
    tokens[0, 1:8] = torch.tensor([100, 200, 300, 400, 500, 600, 700])
    tokens[0, 8] = 2
    batch[OBS_LANGUAGE_TOKENS] = tokens

    return batch


# =============================================================================
# Metrics
# =============================================================================


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def peak_dram_mb_torch_cpu() -> float:
    """torch CPU has no on-device DRAM tracking; report 0."""
    return 0.0


def peak_dram_mb_ttnn() -> float:
    """Read peak DRAM allocation across all devices, in MB.

    Best-effort: ttnn API surface differs across builds. Returns 0 on failure.
    """
    try:
        import ttnn

        # Try several possible APIs; fall through silently.
        for fn_name in ("get_memory_view", "get_memory_info"):
            fn = getattr(ttnn, fn_name, None)
            if callable(fn):
                info = fn()
                # Heuristic: dict with peak fields, else 0.
                if isinstance(info, dict):
                    for k in ("peak_dram_bytes", "peak_dram", "dram_peak"):
                        if k in info:
                            return float(info[k]) / (1024 * 1024)
        return 0.0
    except Exception:
        return 0.0


# =============================================================================
# Backends
# =============================================================================


def load_policy_torch_cpu(weights: Path):
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    torch.set_grad_enabled(False)
    torch.manual_seed(SEED)
    policy = XVLAPolicy.from_pretrained(str(weights))
    policy.eval()
    return policy


def load_policy_ttnn(weights: Path):
    """Load X-VLA with the TT-NN backend.

    On the very first iteration (no tt-nn impl yet) this falls back to
    torch CPU and prints a banner so the human can see the harness is
    waiting on the porting work — but the metric extraction stays valid.
    """
    try:
        from models.experimental.x_vla.tt.policy import load_policy_ttnn as _load  # type: ignore

        return _load(weights)
    except (ImportError, AttributeError) as e:
        print(f"[bench] no TT-NN backend yet ({e}); falling back to torch_cpu.", flush=True)
        return load_policy_torch_cpu(weights)


# =============================================================================
# Driver
# =============================================================================


def run(backend: str, weights: Path) -> dict[str, float]:
    if backend == "torch_cpu":
        policy = load_policy_torch_cpu(weights)
        peak_fn = peak_dram_mb_torch_cpu
    elif backend == "ttnn":
        policy = load_policy_ttnn(weights)
        peak_fn = peak_dram_mb_ttnn
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    batch = build_synthetic_batch(policy, device="cpu")

    # Warmup
    for _ in range(WARMUP_RUNS):
        torch.manual_seed(SEED)
        _ = policy.predict_action_chunk(batch)

    # Timed runs (deterministic — re-seed each call so output is identical too)
    times = []
    last_actions = None
    for _ in range(TIMED_RUNS):
        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        actions = policy.predict_action_chunk(batch)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        last_actions = actions

    chunk_size = policy.config.chunk_size
    mean_time = float(np.mean(times))
    inference_speed = chunk_size / mean_time  # frames per second (action steps / sec)

    actions_np = last_actions.detach().cpu().to(torch.float32).numpy()

    # Reference handling: cache on first ever run; PCC vs cached otherwise.
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    if not REFERENCE_FILE.exists():
        np.save(REFERENCE_FILE, actions_np)
        print(f"[bench] cached new reference at {REFERENCE_FILE}", flush=True)
        accuracy = 100.0
    else:
        ref = np.load(REFERENCE_FILE)
        if ref.shape != actions_np.shape:
            print(
                f"[bench] reference shape {ref.shape} != current {actions_np.shape}; "
                f"treating as 0% accuracy",
                flush=True,
            )
            accuracy = 0.0
        else:
            pcc = pearson_correlation(actions_np, ref)
            # Map PCC in [-1, 1] -> percent in [0, 100]. PCC=1 -> 100, PCC<=0 -> 0.
            accuracy = max(0.0, pcc) * 100.0

    return {
        "inference_speed": inference_speed,
        "accuracy": accuracy,
        "peak_dram": peak_fn(),
        "mean_time_s": mean_time,
        "chunk_size": chunk_size,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend",
        default=os.environ.get("XVLA_BACKEND", "torch_cpu"),
        choices=["torch_cpu", "ttnn"],
    )
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = p.parse_args()

    results = run(args.backend, args.weights)
    # Greppable lines (PROGRAM.md):
    print(f"inference_speed={results['inference_speed']:.4f}")
    print(f"accuracy={results['accuracy']:.4f}")
    print(f"peak_dram={results['peak_dram']:.2f}")
    # Extra context (not grepped):
    print(f"# backend={args.backend} mean_time_s={results['mean_time_s']:.4f} "
          f"chunk_size={results['chunk_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
