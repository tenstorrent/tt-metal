#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Relative-PCC evaluation — implementation fidelity only.

Loads X-VLA in one or more backends (torch_cpu fp32 serves as the
reference, ttnn is the port), runs the same synthetic inputs through
each, and reports PCC + max-abs-error of the predicted action chunk vs
the reference. Both backends use the SAME `num_denoising_steps` so
algorithmic differences don't leak into the number.

This is independent of `benchmark/run_benchmark.py` — the benchmark
oracle keeps its own (possibly algorithm-varying) PCC cache; this script
answers a narrower question.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

# Route ttnn imports to the xvla tt-metal tree, and patch broken lerobot
# imports — same bootstraps as the benchmark.
_HERE = Path(__file__).resolve().parent
_XV = _HERE.parent
_BENCH = _XV / "benchmark"


def _load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_ttnn_env = _load_file_module("xvla_ttnn_env", _XV / "tt" / "ttnn_env.py")
_ttnn_env.install()
_bootstrap = _load_file_module("xvla_lerobot_bootstrap", _BENCH / "lerobot_bootstrap.py")
_bootstrap.install()
_bench = _load_file_module("xvla_run_benchmark", _BENCH / "run_benchmark.py")


# ------------------------------------------------------------------
# Loaders — override num_denoising_steps post-hoc so both backends run
# the same algorithm.
# ------------------------------------------------------------------

def load_policy(backend: str, weights: Path, steps: int, tokenizer_max_length: int = 32):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    torch.set_grad_enabled(False)
    if backend == "torch_cpu":
        config = PreTrainedConfig.from_pretrained(str(weights))
        config.dtype = "float32"  # reference
        config.num_denoising_steps = steps
        config.tokenizer_max_length = tokenizer_max_length
        policy = XVLAPolicy.from_pretrained(str(weights), config=config)
        policy.eval()
        return policy
    elif backend == "ttnn":
        tt_policy = _load_file_module("xvla_tt_policy", _XV / "tt" / "policy.py")
        policy = tt_policy.load_policy_ttnn(weights)
        policy.config.num_denoising_steps = steps  # override the hard-coded 1
        policy.config.tokenizer_max_length = tokenizer_max_length
        return policy
    else:
        raise ValueError(backend)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rel_err(test: np.ndarray, ref: np.ndarray) -> float:
    """Mean |test - ref| / std(ref). Scale-invariant sanity check."""
    scale = ref.astype(np.float64).std() + 1e-12
    return float(np.mean(np.abs(test - ref)) / scale)


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------

def run_one(policy, seed: int) -> np.ndarray:
    """Deterministic inference for a single seed. Returns [1,chunk,action_dim]."""
    batch = _bench.build_synthetic_batch(policy, device="cpu")
    # Re-seed the torch RNG so flow-matching noise x1 is the same across backends.
    torch.manual_seed(seed)
    actions = policy.predict_action_chunk(batch)
    return actions.detach().cpu().to(torch.float32).numpy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path,
                   default=_XV / "weights" / "xvla_base")
    p.add_argument("--backends", type=str, default="torch_cpu,ttnn",
                   help="Comma list. torch_cpu is always included as the reference.")
    p.add_argument("--steps", type=int, default=10,
                   help="num_denoising_steps for both reference and tests.")
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of seeds to average over.")
    args = p.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if "torch_cpu" not in backends:
        backends = ["torch_cpu"] + backends

    print(f"Relative PCC vs torch_cpu fp32, num_denoising_steps={args.steps}, seeds={args.seeds}")
    print(f"Backends: {backends}")
    print()

    # Run reference FIRST and record its outputs per seed.
    print("[1/2] Loading reference (torch_cpu fp32)...")
    ref_policy = load_policy("torch_cpu", args.weights, args.steps)
    ref_outs: dict[int, np.ndarray] = {}
    for seed in range(42, 42 + args.seeds):
        ref_outs[seed] = run_one(ref_policy, seed)
    del ref_policy  # free memory before loading ttnn

    results: dict[str, dict] = {}
    results["torch_cpu"] = {"pcc": [1.0] * args.seeds,
                            "max_abs": [0.0] * args.seeds,
                            "rel_err": [0.0] * args.seeds}

    # Evaluate each non-reference backend.
    for backend in backends:
        if backend == "torch_cpu":
            continue
        print(f"[2/2] Loading {backend}...")
        pol = load_policy(backend, args.weights, args.steps)
        pccs, maes, rels = [], [], []
        for seed in range(42, 42 + args.seeds):
            out = run_one(pol, seed)
            ref = ref_outs[seed]
            if out.shape != ref.shape:
                print(f"  WARN: seed={seed} shape mismatch {out.shape} vs {ref.shape}")
                pccs.append(0.0); maes.append(float("inf")); rels.append(float("inf"))
                continue
            pccs.append(pcc(out, ref))
            maes.append(float(np.max(np.abs(out - ref))))
            rels.append(rel_err(out, ref))
        results[backend] = {"pcc": pccs, "max_abs": maes, "rel_err": rels}

    # Report.
    print()
    print(f"{'backend':<12s} {'mean PCC':>10s} {'min PCC':>10s} "
          f"{'mean |err|/std(ref)':>22s} {'max abs err':>14s}")
    print("-" * 74)
    for backend in backends:
        r = results[backend]
        print(f"{backend:<12s} "
              f"{np.mean(r['pcc']):>10.6f} "
              f"{np.min(r['pcc']):>10.6f} "
              f"{np.mean(r['rel_err']):>22.4e} "
              f"{np.max(r['max_abs']):>14.4e}")
    print()
    print("Interpretation: a clean port has mean PCC >= 0.9990 and rel_err < 0.01.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
