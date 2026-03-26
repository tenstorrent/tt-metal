# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Video compositor for the robotics demo suite.

Composes individual environment frames into quad-view (2x2) or
side-by-side layouts, overlays live performance metrics, model
labels, and Tenstorrent branding. Supports both live display
(returns numpy frames for Streamlit) and MP4 file recording.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _put_text(frame: np.ndarray, text: str, pos: Tuple[int, int],
              scale: float = 0.6, color: Tuple = (255, 255, 255),
              thickness: int = 1, bg: bool = True):
    """Draw text with optional dark background for readability."""
    if not _HAS_CV2:
        return frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bg:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame, (pos[0] - 2, pos[1] - th - 4),
                      (pos[0] + tw + 2, pos[1] + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)
    return frame


def compose_quad_view(
    frames: List[np.ndarray],
    labels: Optional[List[str]] = None,
    metrics: Optional[List[Dict]] = None,
    output_width: int = 1280,
    output_height: int = 720,
) -> np.ndarray:
    """
    Compose up to 4 frames into a 2x2 grid with labels and metrics.

    Args:
        frames: List of 1-4 RGB numpy arrays (any size, will be resized).
        labels: Optional per-frame labels (e.g. model name + task).
        metrics: Optional per-frame metrics dicts with keys like
                 'inference_ms', 'freq_hz', 'distance'.
        output_width: Target composite width.
        output_height: Target composite height.

    Returns:
        Composite RGB frame of shape (output_height, output_width, 3).
    """
    while len(frames) < 4:
        frames.append(np.zeros_like(frames[0]) if frames else
                       np.zeros((output_height // 2, output_width // 2, 3), dtype=np.uint8))

    cell_w = output_width // 2
    cell_h = output_height // 2
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames[:4]):
        row, col = divmod(idx, 2)
        if _HAS_CV2:
            resized = cv2.resize(frame, (cell_w, cell_h))
        else:
            resized = _simple_resize(frame, cell_w, cell_h)

        y0, x0 = row * cell_h, col * cell_w
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = resized

        if labels and idx < len(labels):
            _put_text(canvas, labels[idx], (x0 + 8, y0 + 58),
                      scale=0.55, color=(0, 255, 200))

        if metrics and idx < len(metrics):
            m = metrics[idx]
            y_off = y0 + cell_h - 18
            if "inference_ms" in m:
                _put_text(canvas, f"Inf: {m['inference_ms']:.0f}ms",
                          (x0 + 8, y_off), scale=0.45, color=(200, 200, 200))
                y_off -= 22
            if "freq_hz" in m:
                _put_text(canvas, f"Freq: {m['freq_hz']:.1f} Hz",
                          (x0 + 8, y_off), scale=0.45, color=(200, 200, 200))
                y_off -= 22
            if "distance" in m:
                _put_text(canvas, f"Dist: {m['distance']:.3f}m",
                          (x0 + 8, y_off), scale=0.45, color=(200, 200, 200))

    # Grid lines
    if _HAS_CV2:
        cv2.line(canvas, (cell_w, 0), (cell_w, output_height), (60, 60, 60), 1)
        cv2.line(canvas, (0, cell_h), (output_width, cell_h), (60, 60, 60), 1)

    return canvas


def compose_side_by_side(
    left_frames: List[np.ndarray],
    right_frames: List[np.ndarray],
    left_label: str = "PI0",
    right_label: str = "SmolVLA",
    left_metrics: Optional[Dict] = None,
    right_metrics: Optional[Dict] = None,
    output_width: int = 1280,
    output_height: int = 720,
) -> np.ndarray:
    """
    Compose two sets of frames side-by-side for model comparison.

    Takes the first frame from each list, puts left model on the left
    half and right model on the right half, with labels and metrics.
    """
    half_w = output_width // 2
    canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)

    for side, frames, label, mets in [
        (0, left_frames, left_label, left_metrics),
        (1, right_frames, right_label, right_metrics),
    ]:
        x0 = side * half_w
        frame = frames[0] if frames else np.zeros((output_height, half_w, 3), dtype=np.uint8)
        if _HAS_CV2:
            resized = cv2.resize(frame, (half_w, output_height))
        else:
            resized = _simple_resize(frame, half_w, output_height)
        canvas[:, x0:x0 + half_w] = resized

        _put_text(canvas, label, (x0 + 10, 30), scale=0.8, color=(0, 255, 200))

        if mets:
            y_off = output_height - 15
            for key in ["inference_ms", "freq_hz", "distance"]:
                if key in mets:
                    txt = f"{key}: {mets[key]:.1f}" if isinstance(mets[key], float) else f"{key}: {mets[key]}"
                    _put_text(canvas, txt, (x0 + 10, y_off), scale=0.45, color=(200, 200, 200))
                    y_off -= 20

    if _HAS_CV2:
        cv2.line(canvas, (half_w, 0), (half_w, output_height), (100, 100, 100), 2)

    return canvas


def _simple_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest-neighbor resize fallback when cv2 is not available."""
    src_h, src_w = img.shape[:2]
    ys = np.linspace(0, src_h - 1, h).astype(int)
    xs = np.linspace(0, src_w - 1, w).astype(int)
    return img[np.ix_(ys, xs)]


class VideoRecorder:
    """Records composed frames to video or image sequence."""

    def __init__(self, path: Optional[str] = None, fps: int = 20):
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"robotics_demo_{ts}.mp4"
        self.path = path
        self.fps = fps
        self._writer = None
        self._frames: list = []
        self._frame_count = 0

        try:
            import imageio
            self._writer = imageio.get_writer(path, fps=fps, format="FFMPEG")
        except Exception:
            # Fallback: buffer frames in memory, save on close
            self._writer = None

    def write_frame(self, frame: np.ndarray):
        self._frame_count += 1
        if self._writer is not None:
            try:
                self._writer.append_data(frame)
                return
            except Exception:
                self._writer = None
        self._frames.append(frame)

    def close(self) -> str:
        if self._writer is not None:
            try:
                self._writer.close()
                return self.path
            except Exception:
                pass

        # Fallback: save buffered frames
        if self._frames:
            try:
                import imageio
                imageio.mimsave(self.path, self._frames, fps=self.fps)
            except Exception:
                # Last resort: save as numbered PNGs
                out_dir = self.path.rsplit(".", 1)[0] + "_frames"
                os.makedirs(out_dir, exist_ok=True)
                for i, f in enumerate(self._frames):
                    try:
                        from PIL import Image
                        Image.fromarray(f).save(os.path.join(out_dir, f"frame_{i:05d}.png"))
                    except Exception:
                        np.save(os.path.join(out_dir, f"frame_{i:05d}.npy"), f)
                self.path = out_dir
        return self.path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration_seconds(self) -> float:
        return self._frame_count / self.fps if self.fps > 0 else 0.0
