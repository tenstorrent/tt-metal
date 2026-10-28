# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Weight facts for CONTRACT §6 A / §7: per-layer decay, phase step, slow-mode memory and chunk overflow bounds.

Reads the safetensors directly (reference venv or python_env; needs only torch + safetensors), writes JSON:

    ttq run --model flowstate --kind cpu -- <venv>/bin/python .../weight_facts.py --out <goldens>/weight_facts.json

Per layer and per s in {1/12, 0.25, 1, 2, 6}, with log Lbar_p = s * lambda_p * Delta_p (complex64, as the reference):
  max_decay      max_p |log|Lbar_p||            (the per-step decay rate; D0: 2.07 at s = 1 in layer 5)
  max_phase      max_p |Im log Lbar_p|          (rad per step; D0: 131 at s = 1 in layer 3)
  slow_kept_L    min_p |Lbar_p|^L for L in (682, 2048, 4096, 16384)
  c_max          the largest chunk c with |Lbar_p|^-c * max|Bu| < 1e30 for every p (route S's unbounded scaling)
  n_overflow_c   modes that overflow at c = 32 and c = 64 (route S may treat them separately, D0 item 3)
max|Bu| defaults to 1e3 (an upper estimate; the B0 goldens report the measured value per layer).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = "ibm-granite/granite-timeseries-flowstate-r1"
REVISION = "b80e0f1b1c2d90edebe3d4f243ad046a3f935daa"
SCALES = {"1/12": 1 / 12, "0.25": 0.25, "1": 1.0, "2": 2.0, "6": 6.0}


def layer_facts(w, i, max_bu):
    p = f"encoder.layers.{i}.ssm."
    lam = -torch.exp(w[p + "log_Lambda_real"]) + 1j * w[p + "Lambda_imag"]
    delta = torch.exp(w[p + "log_Delta"])
    out = {
        "delta_min": delta.min().item(),
        "delta_max": delta.max().item(),
        "neg_re_lambda_min": (-lam.real).min().item(),
        "neg_re_lambda_max": (-lam.real).max().item(),
    }
    for name, s in SCALES.items():
        log_lbar = s * lam * delta
        rate = (-log_lbar.real).double()  # -ln|Lbar_p| > 0
        budget = 30 * math.log(10) - math.log(max_bu)
        out[f"s={name}"] = {
            "max_decay": rate.max().item(),
            "max_phase": log_lbar.imag.abs().max().item(),
            "slow_kept_L": {L: math.exp(-rate.min().item() * L) for L in (682, 2048, 4096, 16384)},
            "c_max": int(budget / rate.max().item()),
            "n_overflow_c": {c: int((rate * c > budget).sum()) for c in (32, 64)},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-bu", type=float, default=1e3)
    args = ap.parse_args()
    w = load_file(hf_hub_download(REPO, "model.safetensors", revision=REVISION))
    n = 1 + max(int(k.split(".")[2]) for k in w if k.startswith("encoder.layers."))
    facts = {"max_bu": args.max_bu, "layers": {i: layer_facts(w, i, args.max_bu) for i in range(n)}}
    args.out.write_text(json.dumps(facts, indent=1) + "\n")
    for i, f in facts["layers"].items():
        s1 = f["s=1"]
        print(
            f"layer {i}: decay {s1['max_decay']:.3f}/step, phase {s1['max_phase']:.1f} rad/step, "
            f"slow kept@2048 {s1['slow_kept_L'][2048]:.2f}, c_max {s1['c_max']} (s=1), "
            f"c_max {f['s=6']['c_max']} (s=6)"
        )


if __name__ == "__main__":
    main()
