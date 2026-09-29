# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Full-depth CPU reference forward of the MiniMax-H3 transformer on deterministic inputs, and PSNR of a device
output against it. The device side is `test_zz_cpu_ref.py` (same inputs, current env knobs), so every speed knob can
be scored as "distance to the fp32 CPU reference" per forward, next to the tip's own distance.

    # CPU (hours at the 15 s shape, ~1 h at 5 s on 64 cores, ~15 min at 2 s); writes inputs + reference outputs
    MINIMAX_H3_MODEL_PATH=... python cpu_reference_forward.py reference --shape 5s --out ref_5s.npz
    # after the device test wrote tt_<tag>.npz
    python cpu_reference_forward.py score ref_5s.npz tt_tip.npz tt_rec3.npz
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

# (num_text, num_audio, num_video, grid): the transformer test's production 5 s point and two cheaper ones
SHAPES = {
    "2s": (512, 166, 14112, (24, 42)),  # 14 latent frames of 24x42 patches
    "5s": (512, 414, 37296, (24, 42)),  # the transformer test's prod_768p_5s
    "15s": (512, 1242, 108864, (24, 42)),  # 108 latent frames: the served 15 s 16:9 clip's shape class
}


def _test_module():
    """The transformer test module carries the packed-layout metadata and checkpoint readers; import it lazily."""
    from models.tt_dit.tests.models.minimax_h3 import test_transformer_minimax_h3 as t

    return t


def build_inputs(shape: str, seed: int, timesteps: list[float]) -> dict:
    t = _test_module()
    num_text, num_audio, num_video, grid = SHAPES[shape]
    per_modality = t._modality_metadata(num_text, num_audio, num_video, grid, ())
    segments = [per_modality["text"], per_modality["audio"], per_modality["video"]]
    position_ids = torch.cat([s["pos"] for s in segments])
    tags = torch.cat([s["tags"] for s in segments])
    ts_idx = torch.cat([s["ts"] for s in segments])
    num_timesteps = int(ts_idx.max().item()) + 1
    if len(timesteps) != num_timesteps:
        raise ValueError(f"shape {shape} has {num_timesteps} timestep slots, got {len(timesteps)} values")

    directory = Path(os.environ["MINIMAX_H3_MODEL_PATH"]) / "transformer"
    config = {k: v for k, v in json.loads((directory / "config.json").read_text()).items() if not k.startswith("_")}
    video_patch_dim = config["in_channels"] * int(np.prod(config["patch_size"]))

    g = torch.Generator().manual_seed(seed)
    video_input = torch.randn((1, num_video, video_patch_dim), generator=g, dtype=torch.float32)
    audio_input = torch.randn((1, num_audio, config["audio_in_channels"]), generator=g, dtype=torch.float32)
    prompt_input = torch.randn((1, num_text, config["text_dim"]), generator=g, dtype=torch.float32)
    return dict(
        shape=shape,
        seed=seed,
        num_text=num_text,
        num_audio=num_audio,
        num_video=num_video,
        grid=np.array(grid),
        position_ids=position_ids.numpy(),
        tags=tags.numpy(),
        ts_idx=ts_idx.numpy(),
        video_input=video_input.numpy(),
        audio_input=audio_input.numpy(),
        prompt_input=prompt_input.numpy(),
        timestep=np.array(timesteps, dtype=np.float32),
    )


def run_reference(inp: dict, threads: int) -> dict:
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    t = _test_module()
    torch.set_num_threads(threads)
    directory = Path(os.environ["MINIMAX_H3_MODEL_PATH"]) / "transformer"
    config = {k: v for k, v in json.loads((directory / "config.json").read_text()).items() if not k.startswith("_")}
    start = time.time()
    model = MiniMaxH3Transformer3DModel(**config)
    model.load_state_dict(t._load_reference_state_dict(directory), strict=True)
    model = model.to(torch.float32).eval()
    print(f"reference model loaded in {time.time() - start:.0f} s ({sum(p.numel() for p in model.parameters()) / 1e9:.1f} B params)", flush=True)

    num_text, num_audio, num_video = int(inp["num_text"]), int(inp["num_audio"]), int(inp["num_video"])
    text_indices = torch.arange(num_text)
    audio_indices = torch.arange(num_text, num_text + num_audio)
    video_indices = torch.arange(num_text + num_audio, num_text + num_audio + num_video)
    start = time.time()
    with torch.no_grad():
        out = model(
            hidden_states=torch.from_numpy(inp["video_input"]),
            audio_hidden_states=torch.from_numpy(inp["audio_input"]),
            encoder_hidden_states=torch.from_numpy(inp["prompt_input"]),
            timestep=torch.from_numpy(inp["timestep"]),
            timestep_indices=torch.from_numpy(inp["ts_idx"]),
            token_tags=torch.from_numpy(inp["tags"]),
            position_ids=torch.from_numpy(inp["position_ids"]),
            video_indices=video_indices,
            audio_indices=audio_indices,
            text_indices=text_indices,
            return_dict=True,
        )
    elapsed = time.time() - start
    print(f"reference forward: {elapsed:.0f} s", flush=True)
    return dict(ref_video=out.sample.float().numpy(), ref_audio=out.audio_sample.float().numpy(), ref_seconds=elapsed)


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """dB, peak = the reference's own range (as models/tt_dit/tests/models/minimax_h3/common.py::psnr)."""
    ref = ref.astype(np.float64).reshape(-1)
    test = test.astype(np.float64).reshape(-1)
    mse = np.mean((ref - test) ** 2)
    peak = ref.max() - ref.min()
    return float("inf") if mse == 0 else float(20 * np.log10(peak) - 10 * np.log10(mse))


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    return float(np.corrcoef(a, b)[0, 1])


def score(ref_path: str, tt_paths: list[str]) -> None:
    ref = np.load(ref_path)
    print(f"{'output':32s} {'video PSNR':>10s} {'video PCC':>10s} {'audio PSNR':>10s} {'audio PCC':>10s}")
    for p in tt_paths:
        tt = np.load(p)
        nv, na = ref["ref_video"].shape[1], ref["ref_audio"].shape[1]
        v, a = tt["tt_video"][:, :nv], tt["tt_audio"][:, :na]
        print(
            f"{Path(p).stem:32s} {psnr(ref['ref_video'], v):10.2f} {pcc(ref['ref_video'], v):10.6f} "
            f"{psnr(ref['ref_audio'], a):10.2f} {pcc(ref['ref_audio'], a):10.6f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--shape", choices=sorted(SHAPES), default="5s")
    r.add_argument("--seed", type=int, default=1234)
    r.add_argument("--timesteps", default="0.9,0.65", help="one sigma per slot: video, audio")
    r.add_argument("--threads", type=int, default=os.cpu_count())
    r.add_argument("--out", required=True)
    r.add_argument("--inputs-only", action="store_true", help="write the inputs and skip the CPU forward")
    s = sub.add_parser("score")
    s.add_argument("ref")
    s.add_argument("tt", nargs="+")
    args = ap.parse_args()
    if args.cmd == "score":
        score(args.ref, args.tt)
        return
    inp = build_inputs(args.shape, args.seed, [float(x) for x in args.timesteps.split(",")])
    if not args.inputs_only:
        inp.update(run_reference(inp, args.threads))
    np.savez(args.out, **inp)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
