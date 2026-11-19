# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Alpamayo 2 Edge on one Blackhole P100: camera frames and a navigation instruction -> reasoning and 64 waypoints.

Frames are `DIR/cam{id}_{t_ms}.png` files, any size, RGB. The driving profile takes 3 frames from cameras 0, 2, 3
and 5 and 5 frames from cameras 1 and 6 (22 in all); `reference/chat.py` `CAMERA_DISPLAY_NAMES` names each camera's
position on the car. Only the capture times relative to the latest frame enter the prompt, so `t_ms` may use any
epoch. The ego history defaults to driving straight at `--speed`; `--ego-history`
takes an `.npz` with `xyz` [20, 3] and `rot` [20, 3, 3], 0.1 s apart (the action space's `dt`), in the ego frame
at the latest time.

    python models/experimental/alpamayo2_edge/demo/demo.py --frames DIR --nav "Turn left at the intersection."
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

import ttnn
from models.experimental.alpamayo2_edge.reference.config import load_config, resolve_model_dir
from models.experimental.alpamayo2_edge.reference.inputs import CAMERA_FRAME_COUNTS, HISTORY_STEPS, prepare_inputs
from models.experimental.alpamayo2_edge.tt.model import Alpamayo2EdgeTT

_FRAME_NAME = re.compile(r"cam(\d+)_(-?\d+)\.png")


def load_frames(directory: Path) -> tuple[list[np.ndarray], list[int], list[int]]:
    """HxWx3 uint8 frames, camera ids and capture times in microseconds of every `cam{id}_{t_ms}.png` in `directory`.

    Other files are skipped with a warning. The per-camera frame counts are checked before any image is decoded.
    """
    paths, cameras, timestamps_us = [], [], []
    for path in sorted(directory.iterdir()):
        match = _FRAME_NAME.fullmatch(path.name)
        if match is None:
            print(f"warning: skipping {path.name}, not a cam{{id}}_{{t_ms}}.png frame", file=sys.stderr)
            continue
        paths.append(path)
        cameras.append(int(match[1]))
        timestamps_us.append(int(match[2]) * 1000)
    counts = dict(sorted(Counter(cameras).items()))
    if counts != CAMERA_FRAME_COUNTS:
        raise SystemExit(f"{directory} must hold frames per camera {CAMERA_FRAME_COUNTS}, found {counts}")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")))
    return frames, cameras, timestamps_us


def straight_history(speed_mps: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Ego poses `dt` seconds apart of a car driving straight along +x at `speed_mps`, ending at the origin."""
    xyz = torch.zeros(HISTORY_STEPS, 3)
    xyz[:, 0] = torch.arange(-(HISTORY_STEPS - 1), 1, dtype=torch.float32) * (speed_mps * dt)
    return xyz, torch.eye(3).expand(HISTORY_STEPS, 3, 3).contiguous()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--frames", type=Path, required=True, help="directory of cam{id}_{t_ms}.png frames")
    parser.add_argument("--nav", required=True, help='navigation instruction, e.g. "Continue straight."')
    parser.add_argument("--seed", type=int, default=0, help="seeds the CoT sampling and the flow noise")
    parser.add_argument("--speed", type=float, default=10.0, help="m/s of the default straight ego history")
    parser.add_argument("--ego-history", type=Path, help=".npz with xyz [20, 3] and rot [20, 3, 3]")
    args = parser.parse_args(argv)

    cfg = load_config(resolve_model_dir())
    frames, cameras, timestamps_us = load_frames(args.frames)
    if args.ego_history is None:
        xyz, rot = straight_history(args.speed, cfg.action.dt)
    else:
        history = np.load(args.ego_history)
        xyz, rot = history["xyz"], history["rot"]

    inputs = prepare_inputs(cfg, frames, cameras, timestamps_us, xyz, rot, args.nav)
    print(f"{len(frames)} frames, prompt of {inputs.input_ids.shape[1]} tokens", flush=True)

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        model = Alpamayo2EdgeTT.from_pretrained(mesh_device, cfg)
        prediction = model.infer(inputs, seed=args.seed)
    finally:
        ttnn.close_mesh_device(mesh_device)

    tokenizer = Tokenizer.from_file(str(cfg.model_dir / "tokenizer.json"))
    print("Chain of causation:")
    print(tokenizer.decode(prediction.cot_tokens[0].tolist(), skip_special_tokens=False))
    print(f"Future waypoints (x, y, z in meters, ego frame at the latest time, {cfg.action.dt} s apart):")
    for i, (x, y, z) in enumerate(prediction.xyz[0].tolist()):
        print(f"{i:2d}: {x:9.3f} {y:9.3f} {z:9.3f}")


if __name__ == "__main__":
    main()
