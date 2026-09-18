# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU VBench scoring with original appearance inputs and bounded temporal resolution.

All frames and seeds remain in the gate. Only RAFT/AMT receive a spatially reduced
copy; this is a distinct temporal evaluation policy, not full-resolution VBench.
The copy is losslessly encoded so compression does not introduce a second change.
"""

import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from models.tt_dit.utils.vbench import score_vbench

APPEARANCE_METRICS = ["subject_consistency", "background_consistency", "imaging_quality"]


def motion_decision(scores, *, pairs, threshold, required):
    """Return VBench's binary decision as soon as the remaining pairs cannot alter it."""
    moving = 0
    consumed = 0
    for consumed, score in enumerate(scores, 1):
        moving += score > threshold
        if moving >= required:
            return True
        if moving + pairs - consumed < required:
            return False
    if consumed != pairs:
        raise ValueError("Incomplete optical-flow results")
    return False


def dynamic_degree(video_path):
    import torch
    from easydict import EasyDict
    from vbench.dynamic_degree import DynamicDegree
    from vbench.third_party.RAFT.core.utils_core.utils import InputPadder
    from vbench.utils import init_submodules

    model = init_submodules(["dynamic_degree"])["dynamic_degree"]["model"]
    dynamic = DynamicDegree(EasyDict(model=model, small=False, mixed_precision=False, alternate_corr=False), "cpu")
    frames = dynamic.get_frames(str(video_path))
    dynamic.set_params(frame=frames[0], count=len(frames))

    def scores():
        for first, second in zip(frames[:-1], frames[1:]):
            first, second = InputPadder(first.shape).pad(first, second)
            _, flow = dynamic.model(first, second, iters=20, test_mode=True)
            yield dynamic.get_score(first, flow)

    with torch.no_grad():
        return float(
            motion_decision(
                scores(), pairs=len(frames) - 1, threshold=dynamic.params["thres"], required=dynamic.params["count_num"]
            )
        )


def video_info(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot read VBench video: {path}")
        return tuple(
            cap.get(prop)
            for prop in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FRAME_COUNT,
                cv2.CAP_PROP_FPS,
            )
        )
    finally:
        cap.release()


@contextmanager
def temporal_video(path, max_width):
    import imageio_ffmpeg

    width, height, count, fps = video_info(path)
    if max_width <= 0:
        raise ValueError("Temporal width must be positive")
    if width <= max_width:
        yield path
        return
    target_height = 2 * round(height * max_width / width / 2)
    with tempfile.TemporaryDirectory(prefix="vbench-temporal-") as directory:
        reduced = Path(directory) / "temporal.mp4"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-i",
                str(path),
                "-vf",
                f"scale={max_width}:{target_height}:flags=area",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "ultrafast",
                "-threads",
                "1",
                "-an",
                str(reduced),
            ],
            check=True,
        )
        if video_info(reduced) != (max_width, target_height, count, fps):
            raise ValueError("Temporal resize changed the frame count or frame rate")
        print(f"VBench temporal input: {max_width}x{target_height}, {int(count)} frames at {fps} fps", flush=True)
        yield reduced


def score_clip_ci(path, *, prompt, temporal_width=960):
    scores = score_vbench(str(path), prompt=prompt, dimensions=APPEARANCE_METRICS)
    with temporal_video(path, temporal_width) as reduced:
        scores.update(score_vbench(str(reduced), prompt=prompt, dimensions=["motion_smoothness"]))
        scores["dynamic_degree"] = dynamic_degree(reduced)
    return scores
