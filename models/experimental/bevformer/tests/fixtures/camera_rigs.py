# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic approximate camera rigs for PCC and performance tests.

These are geometry fixtures, not production calibration. Real inference reads per-camera
``lidar2img`` matrices from the dataset manifest.
"""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from models.experimental.bevformer.model_config import CAMERA_IMAGE_SIZE


NUSCENES_NARROW_FOCAL_PX = 1266.4
NUSCENES_NARROW_PRINCIPAL_PX = (816.3, 491.5)
NUSCENES_WIDE_FOCAL_PX = 809.2
NUSCENES_WIDE_PRINCIPAL_PX = (829.2, 481.8)


@dataclass(frozen=True)
class CameraSpec:
    """One synthetic camera mounting and its pixel intrinsics."""

    name: str
    yaw_deg: float
    focal_px: Tuple[float, float]
    principal_px: Tuple[float, float]
    reference_size: Tuple[int, int]
    pitch_deg: float = 0.0
    translation_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def _narrow(name: str, yaw_deg: float) -> CameraSpec:
    return CameraSpec(
        name=name,
        yaw_deg=yaw_deg,
        focal_px=(NUSCENES_NARROW_FOCAL_PX, NUSCENES_NARROW_FOCAL_PX),
        principal_px=NUSCENES_NARROW_PRINCIPAL_PX,
        reference_size=CAMERA_IMAGE_SIZE,
    )


_NUSCENES_CAMERAS = {
    "CAM_FRONT": _narrow("CAM_FRONT", 0.0),
    "CAM_FRONT_RIGHT": _narrow("CAM_FRONT_RIGHT", -55.0),
    "CAM_FRONT_LEFT": _narrow("CAM_FRONT_LEFT", 55.0),
    "CAM_BACK": CameraSpec(
        name="CAM_BACK",
        yaw_deg=180.0,
        focal_px=(NUSCENES_WIDE_FOCAL_PX, NUSCENES_WIDE_FOCAL_PX),
        principal_px=NUSCENES_WIDE_PRINCIPAL_PX,
        reference_size=CAMERA_IMAGE_SIZE,
    ),
    "CAM_BACK_LEFT": _narrow("CAM_BACK_LEFT", 110.0),
    "CAM_BACK_RIGHT": _narrow("CAM_BACK_RIGHT", -110.0),
}
NUSCENES_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
NUSCENES_CAMERA_RIG = tuple(_NUSCENES_CAMERAS[name] for name in NUSCENES_CAMERA_ORDER)


def ring_camera_rig(
    num_cams: int,
    input_size: Tuple[int, int],
    overlap: float = 1.25,
) -> Tuple[CameraSpec, ...]:
    """Synthesize an evenly spaced 360-degree ring with overlapping views."""
    if num_cams < 1:
        raise ValueError(f"num_cams must be positive, got {num_cams}")
    width, height = input_size
    fov = math.radians(360.0 / num_cams) * overlap
    if fov >= math.pi:
        raise ValueError(f"{num_cams} cameras at overlap {overlap} need a >=180 degree field of view")
    focal = (width / 2.0) / math.tan(fov / 2.0)
    return tuple(
        CameraSpec(
            name=f"CAM_{index}",
            yaw_deg=index * 360.0 / num_cams,
            focal_px=(focal, focal),
            principal_px=(width / 2.0, height / 2.0),
            reference_size=(width, height),
        )
        for index in range(num_cams)
    )


def _intrinsic_matrix(spec: CameraSpec, input_size: Tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    width, height = input_size
    reference_width, reference_height = spec.reference_size
    matrix = torch.eye(4, dtype=dtype)
    matrix[0, 0] = spec.focal_px[0] * width / reference_width
    matrix[1, 1] = spec.focal_px[1] * height / reference_height
    matrix[0, 2] = spec.principal_px[0] * width / reference_width
    matrix[1, 2] = spec.principal_px[1] * height / reference_height
    return matrix


def _extrinsic_matrix(spec: CameraSpec, dtype: torch.dtype) -> torch.Tensor:
    yaw = math.radians(spec.yaw_deg)
    pitch = math.radians(spec.pitch_deg)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    forward = torch.tensor([cos_yaw * cos_pitch, sin_yaw * cos_pitch, -sin_pitch], dtype=dtype)
    right = torch.tensor([sin_yaw, -cos_yaw, 0.0], dtype=dtype)
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack((right, down, forward))
    translation = torch.tensor(spec.translation_m, dtype=dtype)
    matrix = torch.eye(4, dtype=dtype)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -rotation @ translation
    return matrix


def build_lidar2img(
    specs: Sequence[CameraSpec],
    input_size: Tuple[int, int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compose ``K @ [R | -R t]`` for every synthetic camera."""
    return torch.stack([_intrinsic_matrix(spec, input_size, dtype) @ _extrinsic_matrix(spec, dtype) for spec in specs])
