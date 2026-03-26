# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
E2E perf: mesh uses 2 command queues and trace region 0 by default (see _open_lingbot_mesh_device in
inference_ttnn). Override with LINGBOT_VA_NUM_COMMAND_QUEUES / LINGBOT_VA_TRACE_REGION_SIZE.

Workload: same three PNGs as ``inference_ttnn.py`` / README under
``tests/demo/sample_images/robotwin/`` (override with LINGBOT_VA_E2E_IMAGES_DIR).

Each ``run_inference`` opens the mesh and tears it down. Re-invoking ``run_inference`` in the
same Python process can leave MetalContext invalid (TT_FATAL context_id). This test runs each
invocation in a **subprocess** via ``pytest ...::test_lingbot_va`` (same test as
``test_perf_ttnn_lingbot_va`` device perf) so each iteration gets a fresh process and device
lifecycle. ``LINGBOT_VA_PERF_SAVE_DIR`` / ``LINGBOT_VA_E2E_IMAGES_DIR`` / ``LINGBOT_VA_E2E_PROMPT``
are set for the child; ``LINGBOT_VA_CHECKPOINT`` is forced to the resolved checkpoint path.

Timed iterations default to 5 (``LINGBOT_VA_E2E_NUM_ITERS``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from loguru import logger

from models.experimental.lingbot_va.tests.demo.inference_ttnn import load_message_from_files, run_inference
from models.perf.perf_utils import prep_perf_report


def _repo_root() -> Path:
    # models/experimental/lingbot_va/tests/perf/this_file.py -> parents[5] == tt-metal root
    return Path(__file__).resolve().parents[5]


def _resolve_checkpoint_path() -> Path | None:
    ckpt = os.environ.get("LINGBOT_VA_CHECKPOINT", "").strip()
    if ckpt:
        p = Path(ckpt)
        return p if p.is_dir() else None
    tt_metal_home = os.environ.get("TT_METAL_HOME", "").strip()
    if tt_metal_home:
        p = Path(tt_metal_home) / "models/experimental/lingbot_va/reference/checkpoints"
        return p if p.is_dir() else None
    return None


def _resolve_robotwin_images_dir() -> Path:
    override = os.environ.get("LINGBOT_VA_E2E_IMAGES_DIR", "").strip()
    if override:
        return Path(override).resolve()
    return (Path(__file__).resolve().parent.parent / "demo" / "sample_images" / "robotwin").resolve()


def _validate_robotwin_images(images_dir: Path) -> None:
    for name in (
        "observation.images.cam_high.png",
        "observation.images.cam_left_wrist.png",
        "observation.images.cam_right_wrist.png",
    ):
        p = images_dir / name
        if not p.is_file():
            pytest.skip(f"Missing Robotwin sample image: {p}. Set LINGBOT_VA_E2E_IMAGES_DIR or add PNGs.")


def _lingbot_e2e_runner_main() -> None:
    """One isolated run_inference invocation for E2E perf subprocess timing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="Lift the cup from the table")
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "observation.images.cam_high.png",
        "observation.images.cam_left_wrist.png",
        "observation.images.cam_right_wrist.png",
    ):
        image_path = args.images_dir / name
        if not image_path.is_file():
            print(f"Missing Robotwin sample image: {image_path}", file=sys.stderr)
            sys.exit(2)

    message = load_message_from_files(
        str(args.images_dir / "observation.images.cam_high.png"),
        str(args.images_dir / "observation.images.cam_left_wrist.png"),
        str(args.images_dir / "observation.images.cam_right_wrist.png"),
        prompt=args.prompt,
    )
    out = run_inference(
        message=message,
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
    )
    if "action" not in out or out["action"] is None:
        print("run_inference did not return action", file=sys.stderr)
        sys.exit(3)


def _run_e2e_subprocess(
    *,
    checkpoint_path: Path,
    save_dir: Path,
    images_dir: Path,
    prompt: str,
    env: dict,
) -> None:
    child_env = dict(env)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--lingbot-e2e-runner",
        "--checkpoint",
        str(checkpoint_path),
        "--save-dir",
        str(save_dir),
        "--images-dir",
        str(images_dir),
        "--prompt",
        prompt,
    ]
    subprocess.run(cmd, check=True, cwd=str(_repo_root()), env=child_env)


@pytest.mark.timeout(0)
@pytest.mark.models_performance_bare_metal
def test_e2e_perf():
    checkpoint_path = _resolve_checkpoint_path()
    if checkpoint_path is None:
        pytest.skip("Lingbot checkpoint not found. Set LINGBOT_VA_CHECKPOINT or TT_METAL_HOME.")

    num_iterations = int(os.environ.get("LINGBOT_VA_E2E_NUM_ITERS", "3"))
    batch_size = 1
    save_dir = Path(__file__).resolve().parent / "out_perf_e2e"
    expected_compile_time = float(os.environ.get("LINGBOT_VA_EXPECTED_COMPILE_TIME_S", "180.0"))
    expected_throughput = float(os.environ.get("LINGBOT_VA_EXPECTED_THROUGHPUT_FPS", "0.5"))
    prompt = os.environ.get("LINGBOT_VA_E2E_PROMPT", "Lift the cup from the table")

    images_dir = _resolve_robotwin_images_dir()
    _validate_robotwin_images(images_dir)
    logger.info(f"E2E perf images dir: {images_dir}")

    previous_overrides = {
        "LINGBOT_VA_NUM_INFERENCE_STEPS": os.environ.get("LINGBOT_VA_NUM_INFERENCE_STEPS"),
        "LINGBOT_VA_ACTION_NUM_INFERENCE_STEPS": os.environ.get("LINGBOT_VA_ACTION_NUM_INFERENCE_STEPS"),
        "LINGBOT_VA_FRAME_CHUNK_SIZE": os.environ.get("LINGBOT_VA_FRAME_CHUNK_SIZE"),
        "LINGBOT_VA_NUM_COMMAND_QUEUES": os.environ.get("LINGBOT_VA_NUM_COMMAND_QUEUES"),
        "LINGBOT_VA_TRACE_REGION_SIZE": os.environ.get("LINGBOT_VA_TRACE_REGION_SIZE"),
    }
    os.environ["LINGBOT_VA_NUM_INFERENCE_STEPS"] = "1"
    os.environ["LINGBOT_VA_ACTION_NUM_INFERENCE_STEPS"] = "1"
    os.environ["LINGBOT_VA_FRAME_CHUNK_SIZE"] = "1"
    os.environ.setdefault("LINGBOT_VA_NUM_COMMAND_QUEUES", "2")
    os.environ.setdefault("LINGBOT_VA_TRACE_REGION_SIZE", "0")

    child_env = os.environ.copy()

    try:
        logger.info("E2E warmup (subprocess: one full run_inference)...")
        compile_start = time.time()
        _run_e2e_subprocess(
            checkpoint_path=checkpoint_path,
            save_dir=save_dir,
            images_dir=images_dir,
            prompt=prompt,
            env=child_env,
        )
        compile_time = time.time() - compile_start

        logger.info(f"Running {num_iterations} timed e2e iteration(s) in subprocess(es)...")
        iter_start = time.time()
        for _ in range(num_iterations):
            _run_e2e_subprocess(
                checkpoint_path=checkpoint_path,
                save_dir=save_dir,
                images_dir=images_dir,
                prompt=prompt,
                env=child_env,
            )
        total_infer_time = time.time() - iter_start
    finally:
        for key, value in previous_overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    inference_time = total_infer_time / num_iterations
    throughput_fps = batch_size / inference_time

    logger.info(f"Average subprocess run time (timed loop): {1000.0 * inference_time:.2f} ms")
    logger.info(f"Average model performance (from timed loop): {throughput_fps:.4f} fps")

    prep_perf_report(
        model_name="lingbot-va-e2e",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput,
        comments=f"iters_{num_iterations}_subprocess_robotwin_2cq",
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--lingbot-e2e-runner":
        sys.argv.pop(1)
        _lingbot_e2e_runner_main()
