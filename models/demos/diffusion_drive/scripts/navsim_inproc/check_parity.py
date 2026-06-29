#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Single-forward parity check for the in-process DiffusionDrive TTNN agent.

Run in the ``navsim`` conda env (torch 2.0.1) to confirm the on-device stack
still matches the PyTorch reference under that env *before* committing to a full
PDM eval.  Mirrors ``tests/pcc/test_pcc_stage4.py`` but exercises the in-process
AGENT and the real checkpoint (latent=False) — i.e. the deployed eval config.

  conda activate navsim
  PYTHONPATH="$TT_METAL_HOME:$NAVSIM_DEVKIT_ROOT" \
    python models/demos/diffusion_drive/scripts/navsim_inproc/check_parity.py \
      --checkpoint "$DD_CHECKPOINT_PATH" \
      --anchors    "$DD_ANCHOR_PATH"

--checkpoint/--anchors default to $DD_CHECKPOINT_PATH / $DD_ANCHOR_PATH when set
(see the demo README "Eval environment" block).  Exit code 0 if trajectory
PCC >= threshold (default 0.999), else 1.
"""

from __future__ import annotations

import argparse
import os

import torch

# Sibling import: running this script puts its own dir on sys.path[0], so the
# agent module (imported as a bare top-level module, like the hydra _target_) is
# found here too.
from diffusiondrive_ttnn_inproc_agent import DiffusionDriveTtnnInprocAgent


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom < 1e-12:
        return 1.0
    return (a @ b).item() / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DD_CHECKPOINT_PATH"), help="default: $DD_CHECKPOINT_PATH")
    ap.add_argument("--anchors", default=os.environ.get("DD_ANCHOR_PATH"), help="default: $DD_ANCHOR_PATH")
    ap.add_argument("--threshold", type=float, default=0.999)
    args = ap.parse_args()
    if not args.checkpoint or not args.anchors:
        ap.error("set --checkpoint/--anchors or export $DD_CHECKPOINT_PATH / $DD_ANCHOR_PATH")

    from models.demos.diffusion_drive.reference.model import DiffusionDriveConfig, load_model

    print(f"[parity] torch {torch.__version__}")

    # 1. Clean PyTorch reference (CPU), real checkpoint, latent=False (deployed config).
    ref_cfg = DiffusionDriveConfig(plan_anchor_path=args.anchors, latent=False)
    ref_model = load_model(args.checkpoint, ref_cfg, device=torch.device("cpu")).eval()

    # 2. Synthetic features at production resolution, WITH batch dim (as the harness
    #    passes them).  randn (not zeros) so every channel carries variance.
    torch.manual_seed(42)
    features = {
        "camera_feature": torch.randn(1, 3, 256, 1024),
        "lidar_feature": torch.randn(1, 1, 256, 256),
        "status_feature": torch.randn(1, 8),
    }

    # 3. Reference forward — pin DDIM noise (DD-5).
    torch.manual_seed(1234)
    with torch.no_grad():
        ref_out = ref_model(features)
    print(f"[parity] reference trajectory shape {tuple(ref_out['trajectory'].shape)}")

    # 4. In-process agent: opens device + builds the full TTNN stack, then forward
    #    on the SAME noise stream.  config=None is fine — the parity path calls
    #    forward() directly and never touches the feature builders.
    agent = DiffusionDriveTtnnInprocAgent(config=None, checkpoint_path=args.checkpoint, anchors_path=args.anchors)
    print("[parity] building TTNN stack on device (JIT compile at full res — minutes)...")
    agent.initialize()
    torch.manual_seed(1234)
    agent_out = agent.forward(features)
    print(f"[parity] agent trajectory shape     {tuple(agent_out['trajectory'].shape)}")

    # 5. PCC on the trajectory (the only output PDM scores).
    traj_pcc = _pcc(agent_out["trajectory"], ref_out["trajectory"])
    print(f"\n[parity] trajectory PCC = {traj_pcc:.6f}  (threshold {args.threshold})")

    agent._close()
    ok = traj_pcc >= args.threshold
    print("[parity] RESULT:", "OK" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
