# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""E2E perf: timed ``run_inference`` in an isolated spawn process per iteration.

Metal context is not reliably reusable across repeated ``run_inference`` calls in one process;
each iteration uses ``multiprocessing`` ``spawn`` so the device lifecycle matches a fresh run.

Env (optional): ``LINGBOT_VA_CHECKPOINT``, ``TT_METAL_HOME``, ``LINGBOT_VA_E2E_IMAGES_DIR``,
``LINGBOT_VA_E2E_PROMPT``, ``LINGBOT_VA_E2E_NUM_ITERS`` (default 3), ``LINGBOT_VA_NUM_COMMAND_QUEUES``,
``LINGBOT_VA_TRACE_REGION_SIZE``, compile/throughput expectations for ``prep_perf_report``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest
from loguru import logger

from models.experimental.lingbot_va.tests.demo.inference_ttnn import load_message_from_files, run_inference
from models.perf.perf_utils import prep_perf_report


def _repo_root() -> Path:
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


def _e2e_worker(
    repo_root: str,
    checkpoint_path: str,
    save_dir: str,
    images_dir: str,
    prompt: str,
) -> None:
    """Executed only inside a spawn child (fresh interpreter + Metal context)."""
    os.chdir(repo_root)
    idir = Path(images_dir)
    for name in (
        "observation.images.cam_high.png",
        "observation.images.cam_left_wrist.png",
        "observation.images.cam_right_wrist.png",
    ):
        if not (idir / name).is_file():
            raise FileNotFoundError(idir / name)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    message = load_message_from_files(
        str(idir / "observation.images.cam_high.png"),
        str(idir / "observation.images.cam_left_wrist.png"),
        str(idir / "observation.images.cam_right_wrist.png"),
        prompt=prompt,
    )
    out = run_inference(
        message=message,
        checkpoint_path=Path(checkpoint_path),
        save_dir=Path(save_dir),
    )
    if "action" not in out or out["action"] is None:
        raise RuntimeError("run_inference did not return action")


def _run_e2e_spawned(
    *,
    checkpoint_path: Path,
    save_dir: Path,
    images_dir: Path,
    prompt: str,
) -> None:
    repo = str(_repo_root())
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_e2e_worker,
        args=(repo, str(checkpoint_path), str(save_dir), str(images_dir), prompt),
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        pytest.fail(f"E2E worker process exited with code {proc.exitcode}")


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
    logger.info("E2E perf images dir: {}", images_dir)

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

    try:
        logger.info("E2E warmup (spawn: one full run_inference)...")
        compile_start = time.time()
        _run_e2e_spawned(
            checkpoint_path=checkpoint_path,
            save_dir=save_dir,
            images_dir=images_dir,
            prompt=prompt,
        )
        compile_time = time.time() - compile_start

        logger.info("Running {} timed e2e iteration(s) (spawn each)...", num_iterations)
        iter_start = time.time()
        for _ in range(num_iterations):
            _run_e2e_spawned(
                checkpoint_path=checkpoint_path,
                save_dir=save_dir,
                images_dir=images_dir,
                prompt=prompt,
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

    logger.info("Average spawn run time (timed loop): {:.2f} ms", 1000.0 * inference_time)
    logger.info("Average model performance (timed loop): {:.4f} fps", throughput_fps)

    prep_perf_report(
        model_name="lingbot-va-e2e",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=batch_size / expected_throughput,
        comments=f"iters_{num_iterations}_spawn_robotwin_2cq",
    )
