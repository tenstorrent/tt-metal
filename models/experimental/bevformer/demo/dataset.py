# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Source-neutral sequence manifests for the BEVFormer demo.

A manifest is intentionally smaller than any training framework's dataset configuration. Every
frame contains exactly what inference needs, so datasets with the same multi-camera metadata schema
can be converted without adding provider-specific branches here.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def _matrix4(value, context):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{context}: expected a 4x4 matrix")
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"{context}: expected a 4x4 matrix")
        rows.append(tuple(float(element) for element in row))
    return tuple(rows)


@dataclass(frozen=True)
class SequenceFrame:
    token: str
    scene_token: str
    timestamp: int
    image_paths: tuple[Path, ...]
    lidar2img: tuple[tuple[tuple[float, ...], ...], ...]
    can_bus: tuple[float, ...]


@dataclass(frozen=True)
class SequenceManifest:
    path: Path
    camera_order: tuple[str, ...]
    frames: tuple[SequenceFrame, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        dataset_root: str | Path | None = None,
        *,
        camera_order: Sequence[str],
        can_bus_dims: int,
    ):
        path = Path(path).resolve()
        root = Path(dataset_root).resolve() if dataset_root is not None else path.parent
        with path.open(encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict) or not isinstance(raw.get("frames"), list):
            raise ValueError(f"{path}: expected an object with a 'frames' list")
        camera_order = tuple(camera_order)
        if raw.get("camera_order") != list(camera_order):
            raise ValueError(f"{path}: camera_order must be {list(camera_order)}")

        frames = []
        for index, item in enumerate(raw["frames"]):
            context = f"{path}: frame {index}"
            if not isinstance(item, dict):
                raise ValueError(f"{context}: expected an object")
            cameras = item.get("cameras")
            can_bus = item.get("can_bus")
            if not isinstance(cameras, dict) or set(cameras) != set(camera_order):
                raise ValueError(f"{context}: cameras must contain exactly {list(camera_order)}")
            if not isinstance(can_bus, list) or len(can_bus) != can_bus_dims:
                raise ValueError(f"{context}: can_bus must contain {can_bus_dims} values")
            ordered_cameras = [cameras[name] for name in camera_order]
            if not all(isinstance(camera, dict) for camera in ordered_cameras):
                raise ValueError(f"{context}: each camera must be an object")
            images = [camera.get("image") for camera in ordered_cameras]
            if not all(isinstance(image, str) and image for image in images):
                raise ValueError(f"{context}: every camera must contain a non-empty image path")
            image_paths = tuple(Path(image) if Path(image).is_absolute() else root / image for image in images)
            missing = [str(image) for image in image_paths if not image.is_file()]
            if missing:
                raise FileNotFoundError(f"{context}: missing camera images: {missing}")
            frames.append(
                SequenceFrame(
                    token=str(item.get("token", index)),
                    scene_token=str(item.get("scene_token", "default")),
                    timestamp=int(item.get("timestamp", index)),
                    image_paths=image_paths,
                    lidar2img=tuple(
                        _matrix4(camera.get("lidar2img"), f"{context} {name} lidar2img")
                        for name, camera in zip(camera_order, ordered_cameras, strict=True)
                    ),
                    can_bus=tuple(float(value) for value in can_bus),
                )
            )
        if not frames:
            raise ValueError(f"{path}: manifest contains no frames")
        return cls(path=path, camera_order=camera_order, frames=tuple(frames))

    @property
    def scenes(self):
        """Scene tokens in first-appearance order."""
        return tuple(dict.fromkeys(frame.scene_token for frame in self.frames))

    def sequence(
        self,
        *,
        num_output_frames: int,
        start_frame: int = 0,
        scene_token: str | None = None,
        scene_index: int = 0,
    ) -> Sequence[SequenceFrame]:
        """One preparation frame followed by ``num_output_frames`` chronological frames."""
        if num_output_frames < 1:
            raise ValueError("num_output_frames must be positive")
        if scene_token is None:
            try:
                scene_token = self.scenes[scene_index]
            except IndexError as error:
                raise ValueError(f"scene_index {scene_index} is outside 0..{len(self.scenes) - 1}") from error
        selected = sorted(
            (frame for frame in self.frames if frame.scene_token == scene_token), key=lambda frame: frame.timestamp
        )
        if not selected:
            raise ValueError(f"scene {scene_token!r} is not present in {self.path}")
        required = num_output_frames + 1
        result = selected[start_frame : start_frame + required]
        if len(result) != required:
            raise ValueError(f"scene {scene_token!r} has {len(selected)} frames; start {start_frame} needs {required}")
        return result
