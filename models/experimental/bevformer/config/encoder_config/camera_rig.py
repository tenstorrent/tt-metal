# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic surround-view camera rigs and their lidar-to-image matrices.

``lidar2img`` decides which BEV queries project into which camera, so it decides
``bev_mask``, the spatial-cross-attention rebatch length ``max_len``, and therefore
every spatial-path tensor shape. Random matrices make those shapes an artifact of
the RNG draw order rather than a property of the model, so both correctness and
performance runs build their matrices here instead.

Frames follow the reference implementation: lidar x forward, y left, z up; camera
x right, y down, z forward. ``point_sampling_3d_to_2d`` treats row 2 of the
composed matrix as depth and normalizes rows 0 and 1 by image width and height,
so the matrix must be ``K @ [R | -R t]`` with ``K`` carrying pixel intrinsics.
"""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch


NUSCENES_IMAGE_SIZE = (1600, 900)

# The five narrow nuScenes cameras share one nominal intrinsic. Real per-camera
# calibration differs by well under 1% in focal length and by tens of pixels in
# the principal point, which leaves max_len unchanged and per-camera coverage
# within 1%, so the spread is not modelled.
NUSCENES_NARROW_FOCAL_PX = 1266.4
NUSCENES_NARROW_PRINCIPAL_PX = (816.3, 491.5)

# CAM_BACK is a different, wider-FOV unit; its ~89 degree horizontal field is what
# closes the 360 degree ring and it sets max_len.
NUSCENES_WIDE_FOCAL_PX = 809.2
NUSCENES_WIDE_PRINCIPAL_PX = (829.2, 481.8)


@dataclass(frozen=True)
class CameraSpec:
    """One camera's mounting and pixel intrinsics.

    Intrinsics are stored together with the ``reference_size`` they were measured
    at so a rig can be reused at a resized input resolution.
    """

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
        reference_size=NUSCENES_IMAGE_SIZE,
    )


# Nominal nuScenes mounting yaws. Pitch and translation stay zero: the cameras sit
# within about a metre of LIDAR_TOP and are near-horizontal, which is under a
# degree of angular error over the +-51.2 m BEV range.
NUSCENES_CAMERA_RIG: Tuple[CameraSpec, ...] = (
    _narrow("CAM_FRONT", 0.0),
    _narrow("CAM_FRONT_LEFT", 55.0),
    _narrow("CAM_FRONT_RIGHT", -55.0),
    _narrow("CAM_BACK_LEFT", 110.0),
    _narrow("CAM_BACK_RIGHT", -110.0),
    CameraSpec(
        name="CAM_BACK",
        yaw_deg=180.0,
        focal_px=(NUSCENES_WIDE_FOCAL_PX, NUSCENES_WIDE_FOCAL_PX),
        principal_px=NUSCENES_WIDE_PRINCIPAL_PX,
        reference_size=NUSCENES_IMAGE_SIZE,
    ),
)


# KITTI-360 rectified perspective intrinsics (P_rect_00), shared by both cameras
# of the stereo pair. The 0.6 m baseline is the image_00 to image_01 separation.
KITTI_FOCAL_PX = 552.554261
KITTI_PRINCIPAL_PX = (682.049453, 238.769549)
KITTI_IMAGE_SIZE = (1408, 376)
KITTI_BASELINE_M = 0.6


def _kitti(name: str, lateral_offset_m: float) -> CameraSpec:
    return CameraSpec(
        name=name,
        yaw_deg=0.0,
        focal_px=(KITTI_FOCAL_PX, KITTI_FOCAL_PX),
        principal_px=KITTI_PRINCIPAL_PX,
        reference_size=KITTI_IMAGE_SIZE,
        translation_m=(0.0, lateral_offset_m, 0.0),
    )


# A forward-facing stereo pair, not a ring: both cameras share a yaw and are
# separated only laterally, so the rig covers one frustum rather than 360 degrees.
# Roughly half the BEV grid projects into no camera at all, which is the real
# geometry of a stereo dataset and what bev_mask should reflect.
KITTI_CAMERA_RIG: Tuple[CameraSpec, ...] = (
    _kitti("CAM_LEFT", KITTI_BASELINE_M / 2.0),
    _kitti("CAM_RIGHT", -KITTI_BASELINE_M / 2.0),
)


def ring_camera_rig(
    num_cams: int,
    input_size: Tuple[int, int],
    overlap: float = 1.25,
) -> Tuple[CameraSpec, ...]:
    """Synthesize an evenly spaced ring rig covering 360 degrees.

    Used for datasets with no calibration recorded here. Focal length is solved
    from the horizontal field of view each camera must cover, widened by
    ``overlap`` so adjacent frusta intersect the way a real rig's do. This is a
    plausible geometry, not any vehicle's calibration.
    """
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
    scale_x = width / reference_width
    scale_y = height / reference_height
    matrix = torch.eye(4, dtype=dtype)
    matrix[0, 0] = spec.focal_px[0] * scale_x
    matrix[1, 1] = spec.focal_px[1] * scale_y
    matrix[0, 2] = spec.principal_px[0] * scale_x
    matrix[1, 2] = spec.principal_px[1] * scale_y
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
    """Compose ``K @ [R | -R t]`` for each camera.

    Args:
        specs: Camera mountings and intrinsics.
        input_size: Actual ``(width, height)`` of the images being projected into.
        dtype: Output dtype.

    Returns:
        Tensor of shape ``[len(specs), 4, 4]``.
    """
    return torch.stack([_intrinsic_matrix(spec, input_size, dtype) @ _extrinsic_matrix(spec, dtype) for spec in specs])


def camera_rig_for_dataset(dataset_config) -> Tuple[CameraSpec, ...]:
    """Pick the rig recorded for a dataset, falling back to a synthetic ring."""
    if dataset_config.name.startswith("nuscenes") and dataset_config.num_cams == len(NUSCENES_CAMERA_RIG):
        return NUSCENES_CAMERA_RIG
    if dataset_config.name.startswith("kitti") and dataset_config.num_cams == len(KITTI_CAMERA_RIG):
        return KITTI_CAMERA_RIG
    return ring_camera_rig(dataset_config.num_cams, dataset_config.input_size)


def lidar2img_for_dataset(dataset_config, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Lidar-to-image matrices for a dataset config, at its own input resolution."""
    return build_lidar2img(camera_rig_for_dataset(dataset_config), dataset_config.input_size, dtype)


def img_metas_for_dataset(dataset_config, batch_size: int, dtype: torch.dtype = torch.float32) -> List[dict]:
    """Build the ``img_metas`` list the encoder expects, with a deterministic rig.

    Every batch entry shares the rig, matching a single vehicle's calibration.
    """
    width, height = dataset_config.input_size
    lidar2img = lidar2img_for_dataset(dataset_config, dtype).tolist()
    return [
        {
            "img_shape": [(height, width, 3)] * dataset_config.num_cams,
            "lidar2img": lidar2img,
        }
        for _ in range(batch_size)
    ]
