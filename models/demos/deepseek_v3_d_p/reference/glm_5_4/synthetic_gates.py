# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic damp-gate weights for GLM-5.4 until the official weights are published.

Used by both weight tiers: tier 1 (tests/glm54_weights.py: all weights random, reduced config) and tier 2 (the
glm_5_4 runner adapter: GLM-5.3 FP8 weights for every shared tensor). The gate formula is A1 (``ops.damp_gate_reference``):

    beta_k = sigmoid(RMSNorm(x_{k-1}; norm_weight) @ proj_weight^T + proj_bias)

Initialization, per loop pass k:

* ``proj_bias`` = logit(GATE_BETA_CENTER[k]): it sets where beta sits (pass 1 ~0.2, pass 2 ~0.9), so performance
  runs see a pass-2 input close to a realistic mix and a realistic MoE load downstream.
* ``proj_weight`` ~ N(0, (GATE_LOGIT_STD / sqrt(hidden))^2): the normed input has unit RMS, so for one token the
  logit's spread over weight draws is GATE_LOGIT_STD; beta then varies per token (pass 1 about 0.11-0.33 at
  +-2 sigma) and a gate that reads the wrong input or drops the weight is visible.
* ``norm_weight`` = 1 + 0.1 * N(0, 1): a non-trivial gain, so a dropped gain (A1e) is visible; its RMS is ~1, so
  it barely changes the spread.

On real activations the logit's mean also moves by w . mean(normed x), because hidden states share a dominant
direction across tokens, so where beta sits depends on the seed. Measured on GLM-5.3 activations (golden trace
vllm-glm53-indexer-kcache-55k, 8192 of 56320 rows: x0 = layer-25 output, h1 = layer-53 output, gate 2 fed
x1 = gate-1 output), the pass-1 median beta over seeds 0-7 and 42 ranged 0.15-0.33. GLM54_SYNTHETIC_SEED = 1 was
picked from that run: beta1 2/50/98th percentiles 0.147 / 0.215 / 0.282, beta2 0.857 / 0.910 / 0.945. Changing
the seed or the generator needs that measurement repeated (tests/glm54/test_glm54_weights.py does it when the
trace is present). Gates are bf16 (A8). All draws are fp32, then cast.
"""

import math

import torch

# Loop pass -> beta the bias centres the gate on (user, 2026-11-23: beta1 ~0.1-0.3, beta2 ~0.9).
GATE_BETA_CENTER = {1: 0.2, 2: 0.9}
# Logit standard deviation per token on a unit-RMS input: (logit(0.3) - logit(0.1)) / 4, so +-2 sigma around a
# 0.2 centre spans about 0.11..0.33.
GATE_LOGIT_STD = 0.34
GATE_NORM_GAIN_STD = 0.1
# Seed of every GLM-5.4 synthetic weight (layers in tier 1, gates in both tiers); see the module docstring.
GLM54_SYNTHETIC_SEED = 1
# Generator version: bump when the drawn values for a given seed change (cache and provenance identity).
SYNTHETIC_GATES_VERSION = 1


def gate_seed(seed: int, loop_pass: int) -> int:
    """Seed of loop pass ``loop_pass``'s gate, disjoint from the per-layer seeds (seed + 10 * layer + 0..3)."""
    return seed + 100_000 + loop_pass


def synthetic_gate_weights(hidden: int, loop_pass: int, *, seed: int, per_channel: bool = False) -> dict:
    """One loop pass's gate weights ``{"norm_weight", "proj_weight", "proj_bias"}`` in bf16.

    ``per_channel`` gives a [hidden, hidden] Linear (one beta per channel), else [1, hidden] (one beta per token);
    A1a: the width is read from the weight, synthetic runs use the scalar form.
    """
    if loop_pass not in GATE_BETA_CENTER:
        raise ValueError(f"loop pass must be one of {sorted(GATE_BETA_CENTER)}, got {loop_pass}")
    g = torch.Generator().manual_seed(gate_seed(seed, loop_pass))
    width = hidden if per_channel else 1
    center = GATE_BETA_CENTER[loop_pass]
    norm_weight = 1.0 + GATE_NORM_GAIN_STD * torch.randn(hidden, generator=g)
    proj_weight = torch.randn(width, hidden, generator=g) * (GATE_LOGIT_STD / math.sqrt(hidden))
    proj_bias = torch.full((width,), math.log(center / (1.0 - center)))
    return {
        "norm_weight": norm_weight.to(torch.bfloat16),
        "proj_weight": proj_weight.to(torch.bfloat16),
        "proj_bias": proj_bias.to(torch.bfloat16),
    }


def synthetic_gates(hidden: int, loop_count: int, *, seed: int, per_channel: bool = False) -> list[dict]:
    """Gate weights for loop passes 1..loop_count, the ``gate_weights`` list ``glm54_stack_reference`` takes."""
    return [synthetic_gate_weights(hidden, k, seed=seed, per_channel=per_channel) for k in range(1, loop_count + 1)]


__all__ = [
    "GATE_BETA_CENTER",
    "GATE_LOGIT_STD",
    "GATE_NORM_GAIN_STD",
    "GLM54_SYNTHETIC_SEED",
    "SYNTHETIC_GATES_VERSION",
    "gate_seed",
    "synthetic_gate_weights",
    "synthetic_gates",
]
