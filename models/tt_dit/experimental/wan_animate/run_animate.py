# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2-Animate-14B on a TT Galaxy: reference image + driving video -> mp4.

Animation mode (the reference character performs the driving video's motion):

    python models/tt_dit/experimental/wan_animate/run_animate.py \\
        --image cat.png --video dance.mp4 --mode animate --seconds 3 \\
        --wan-repo ~/Wan2.2 --preprocess-ckpt ~/Wan2.2-Animate-14B/process_checkpoint \\
        --pre-python ~/wan_pre/bin/python --retarget --out out/cat_dance.mp4

Replacement mode (the reference character replaces the person, background kept, relit):

    python .../run_animate.py --image cat.png --video dance.mp4 --mode replace \\
        --relighting-lora ~/Wan2.2-Animate-14B/relighting_lora/adapter_model.safetensors ...

Already preprocessed (official preprocess_data.py output dir)?  Pass ``--src-root DIR`` instead of
--image/--video/--wan-repo. The device run happens in a pytest process (the repo's mesh fixtures
open the Galaxy); this script only prepares inputs and environment.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
TEST = os.path.join(REPO, "models", "tt_dit", "experimental", "tests", "test_pipeline_wan_animate.py")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("inputs")
    src.add_argument("--src-root", help="official preprocess_data.py output dir (src_pose.mp4, src_face.mp4, ...)")
    src.add_argument("--image", help="reference character image")
    src.add_argument("--video", help="driving video")
    src.add_argument("--wan-repo", help="Wan2.2 checkout (for wan/modules/animate/preprocess)")
    src.add_argument("--preprocess-ckpt", help="Wan2.2-Animate-14B/process_checkpoint (det/, pose2d/, sam2/)")
    src.add_argument("--retarget", action="store_true", help="animate mode: basic pose retargeting")
    src.add_argument(
        "--pre-python",
        default=sys.executable,
        help="python with the upstream preprocessing deps (e.g. ~/wan_pre/bin/python)",
    )
    src.add_argument("--work-dir", default="wan_animate_work", help="where preprocessing output goes")

    gen = p.add_argument_group("generation")
    gen.add_argument("--mode", choices=["animate", "replace"], default="animate")
    gen.add_argument("--seconds", type=float, default=3.0, help="cap on clip length")
    gen.add_argument("--width", type=int, help="default: preprocessed size rounded down to 16")
    gen.add_argument("--height", type=int)
    gen.add_argument(
        "--resolution-area",
        type=int,
        nargs=2,
        default=[832, 480],
        metavar=("W", "H"),
        help="preprocessing target area (keeps the video's aspect)",
    )
    gen.add_argument("--steps", type=int, default=20)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--prompt")
    gen.add_argument("--relighting-lora", help="replace mode: relighting LoRA (safetensors or .ckpt)")
    gen.add_argument("--checkpoint", default="Wan-AI/Wan2.2-Animate-14B-Diffusers")
    gen.add_argument("--out", default="wan_animate_out.mp4")

    dev = p.add_argument_group("device")
    dev.add_argument(
        "--mesh", default="bh_4x8_linear", help="pytest id: bh_4x8_linear | bh_4x8_ring | wh_4x8_ring | bh_2x4_linear"
    )
    args = p.parse_args()

    src_root = args.src_root
    if src_root is None:
        missing = [n for n in ("image", "video", "wan_repo", "preprocess_ckpt") if getattr(args, n) is None]
        if missing:
            p.error(f"without --src-root, these are required: {', '.join('--' + m.replace('_', '-') for m in missing)}")
        sys.path.insert(0, os.path.join(REPO))
        from models.tt_dit.experimental.wan_animate.video_io import run_official_preprocess

        src_root = os.path.join(args.work_dir, f"preprocess_{args.mode}")
        run_official_preprocess(
            wan_repo=args.wan_repo,
            ckpt_path=args.preprocess_ckpt,
            video=args.video,
            image=args.image,
            save_path=src_root,
            mode=args.mode,
            width=args.resolution_area[0],
            height=args.resolution_area[1],
            retarget=args.retarget,
            python=os.path.expanduser(args.pre_python),
        )

    env = dict(os.environ)
    env.update(
        WAN_ANIMATE_SRC_ROOT=os.path.abspath(src_root),
        WAN_ANIMATE_MODE=args.mode,
        WAN_ANIMATE_OUT=os.path.abspath(args.out),
        WAN_ANIMATE_SECONDS=str(args.seconds),
        WAN_ANIMATE_STEPS=str(args.steps),
        WAN_ANIMATE_SEED=str(args.seed),
        WAN_ANIMATE_CHECKPOINT=args.checkpoint,
    )
    for flag, name in (
        (args.width, "WAN_ANIMATE_WIDTH"),
        (args.height, "WAN_ANIMATE_HEIGHT"),
        (args.prompt, "WAN_ANIMATE_PROMPT"),
        (args.relighting_lora, "WAN_ANIMATE_RELIGHT_LORA"),
    ):
        if flag is not None:
            env[name] = str(flag)
    if args.mode == "replace" and not args.relighting_lora:
        print("warning: replace mode without --relighting-lora (lighting will not match the scene)", flush=True)

    cmd = [sys.executable, "-m", "pytest", TEST, "-k", args.mesh, "-s", "-x", "--timeout=0"]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
