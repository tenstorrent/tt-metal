#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""In-process pi0.5 real-robot demo: live cameras + robot → TT policy → actions.

Swaps the LIBERO simulator for a `RobotInterface`. The TT policy runs unchanged
(`ttnn_1x8` = 8-chip mesh trace+2CQ, ~31 ms/chunk; `ttnn` = single chip).

Defaults to LOG-ONLY (actions printed, not sent). Pass `--enable-motion` to
command the arm. With no real RobotInterface wired it uses a MockRobot, so the
full policy loop + latency run with no hardware.

Run (1×8 mesh, log-only smoke):
    export PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
    export PI0_TOKENIZER_PATH=/home/tt-admin/pi05_cache/tokenizer/paligemma_tokenizer.model
    export TT_METAL_HOME=$PWD PYTHONPATH=$PWD TT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
    python_env/bin/python models/experimental/pi0_5/demo/demo_realrobot.py \
      --checkpoint $PI05_CHECKPOINT_DIR --backend ttnn_1x8 \
      --task "pick up the black bowl" --steps 40

To drive a real arm: implement RobotInterface (see demo/robot_runtime.py +
demo/README.md), import it here in place of MockRobot, and add --enable-motion.
"""
from __future__ import annotations

import argparse
import os
import sys

# Resolve repo root from this file so the script runs from any tt-metal checkout.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description="pi0.5 in-process real-robot demo (TT kernels).")
    ap.add_argument("--checkpoint", default=os.environ.get("PI05_CHECKPOINT_DIR"), help="pi05_libero checkpoint dir")
    ap.add_argument("--backend", default="ttnn_1x8", choices=["ttnn_1x8", "ttnn", "pytorch"])
    ap.add_argument("--task", required=True, help="natural-language task prompt")
    ap.add_argument(
        "--replan-steps", type=int, default=5, help="actions applied per predicted chunk before re-planning"
    )
    ap.add_argument(
        "--num-denoise-steps",
        type=int,
        default=int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5")),
        help="flow-matching Euler steps (5 = perf-tuned)",
    )
    ap.add_argument("--action-horizon", type=int, default=None, help="chunk size (auto-read from config.json if unset)")
    ap.add_argument("--steps", type=int, default=200, help="max control steps for the episode")
    ap.add_argument("--state-in-prompt", default="false", choices=["true", "false"])
    ap.add_argument("--device-id", type=int, default=0, help="single-chip device id (backend=ttnn)")
    ap.add_argument(
        "--enable-motion", action="store_true", help="actually send actions to the robot (default: log only)"
    )
    args = ap.parse_args()

    if not args.checkpoint:
        ap.error("--checkpoint is required (or set PI05_CHECKPOINT_DIR)")

    from models.experimental.pi0_5.demo.policy import build_policy
    from models.experimental.pi0_5.demo.robot_runtime import MockRobot, run_realrobot

    # TODO(real robot): replace MockRobot with your RobotInterface implementation.
    robot = MockRobot()
    if args.enable_motion and isinstance(robot, MockRobot):
        print("[warn] --enable-motion set but robot is MockRobot (no hardware) — nothing will move.", flush=True)

    print(
        f"[demo] backend={args.backend} task={args.task!r} replan={args.replan_steps} N={args.num_denoise_steps}",
        flush=True,
    )
    with build_policy(
        args.checkpoint,
        backend=args.backend,
        action_horizon=args.action_horizon,
        state_in_prompt=(args.state_in_prompt == "true"),
        device_id=args.device_id,
    ) as adapter:
        result = run_realrobot(
            adapter,
            robot,
            args.task,
            max_steps=args.steps,
            replan_steps=args.replan_steps,
            num_denoising_steps=args.num_denoise_steps,
            enable_motion=args.enable_motion,
        )
    print(f"[demo] result: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
