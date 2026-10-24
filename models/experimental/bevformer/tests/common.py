# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the BEVFormer tests: camera rigs and dataset presets, dummy weights with
realistic statistics, random inputs shaped like the model's, and PCC checks.

BEVFormer-base's own configuration is ``model_config.py``'s; this module holds what only the tests
use.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.bevformer.model_config import (
    BEV_H,
    BEV_W,
    BOX_CENTER,
    BOX_SIZE,
    BOX_VELOCITY,
    BOX_YAW,
    CAN_BUS_DIMS,
    CENTERS_FIRST,
    CODE_CENTERS,
    CODE_SIN,
    CODE_SIZE,
    CODE_VELOCITY,
    CODE_XY,
    CODE_Z,
    DECODER_NUM_LAYERS,
    DECODER_NUM_POINTS,
    EMBED_DIMS,
    ENCODER_NUM_LAYERS,
    FEEDFORWARD_CHANNELS,
    FPN_KWARGS,
    IMAGE_HEIGHT,
    IMAGE_MEAN_BGR,
    IMAGE_WIDTH,
    NUM_CAMS,
    NUM_CLASSES,
    NUM_HEADS,
    NUM_LEVELS,
    NUM_POINTS_IN_PILLAR,
    NUM_QUERY,
    PC_RANGE,
    RAW_IMAGE_SIZE,
    RESNET_KWARGS,
    SCA_NUM_POINTS,
    SPATIAL_SHAPES,
    TSA_NUM_POINTS,
    fpn_spatial_shapes,
    padded_image_size,
)
from models.experimental.bevformer.reference.bevformer import BEVFormer, relative_can_bus
from models.experimental.bevformer.reference.decoder import DetectionTransformerDecoder, inverse_sigmoid, reg_branch
from models.experimental.bevformer.reference.encoder import BEVFormerEncoder
from models.experimental.bevformer.reference.fpn import FPN
from models.experimental.bevformer.reference.head import BEVFormerHead
from models.experimental.bevformer.reference.ms_deformable_attention import MSDeformableAttention
from models.experimental.bevformer.reference.nms_free_coder import NMSFreeCoder, denormalize_bbox
from models.experimental.bevformer.reference.nms_free_coder import bottom_center as reference_bottom_center
from models.experimental.bevformer.reference.perception_transformer import PerceptionTransformer
from models.experimental.bevformer.reference.resnet import ModulatedDeformConv2dPack, ResNet
from models.experimental.bevformer.reference.spatial_cross_attention import MSDeformableAttention3D
from models.experimental.bevformer.reference.temporal_self_attention import TemporalSelfAttention

# --- Camera rigs ------------------------------------------------------------------------------

# ``lidar2img`` decides which BEV queries project into which camera, so it decides ``bev_mask``, the
# spatial cross-attention's rebatch length and with it every spatial-path tensor shape. Random
# matrices would make those shapes an artifact of the RNG draw order, so the tests build fixed rigs.
# Frames follow the reference: lidar x forward, y left, z up; camera x right, y down, z forward.
# ``point_sampling_3d_2d`` treats row 2 of the composed matrix as depth and normalizes rows 0 and 1
# by the image width and height, so the matrix is ``K @ [R | -R t]`` with ``K`` in pixels.

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
        reference_size=RAW_IMAGE_SIZE,
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
        reference_size=RAW_IMAGE_SIZE,
    ),
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
    """nuScenes' rig for a nuScenes dataset, else a synthetic ring."""
    if dataset_config.name.startswith("nuscenes") and dataset_config.num_cams == len(NUSCENES_CAMERA_RIG):
        return NUSCENES_CAMERA_RIG
    return ring_camera_rig(dataset_config.num_cams, dataset_config.input_size)


def lidar2img_for_dataset(dataset_config, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Lidar-to-image matrices for a dataset config, at its own input resolution."""
    return build_lidar2img(camera_rig_for_dataset(dataset_config), dataset_config.input_size, dtype)


def img_metas_for_dataset(dataset_config, batch_size: int, dtype: torch.dtype = torch.float32) -> List[dict]:
    """Build the ``img_metas`` list the encoder expects, with a deterministic rig.

    Every batch entry shares the rig, matching a single vehicle's calibration. ``img_shape`` is the
    padded shape, as upstream's ``PadMultiViewImage`` sets it (``padded_image_size``), and the
    encoder normalizes the projected points by that shape. ``lidar2img`` stays in the unpadded
    images' pixels, which the padding does not move.
    """
    width, height = padded_image_size(*dataset_config.input_size)
    lidar2img = lidar2img_for_dataset(dataset_config, dtype).tolist()
    return [
        {
            "img_shape": [(height, width, 3)] * dataset_config.num_cams,
            "lidar2img": lidar2img,
        }
        for _ in range(batch_size)
    ]


# --- Dataset presets --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetPreset:
    """
    Attributes:
        name: Dataset identifier; :func:`camera_rig_for_dataset` picks nuScenes' rig for names
            starting "nuscenes", a synthetic ring otherwise.
        pc_range: (x_min, y_min, z_min, x_max, y_max, z_max) in metres.
        num_cams: Cameras.
        input_size: Image size, (width, height).
        embed_dims, num_heads, num_levels, num_points: The spatial cross-attention's deformable
            attention sizes. Every BEV pillar samples ``NUM_POINTS_IN_PILLAR`` heights.
        tsa_num_points: The temporal self-attention's points per head.
    """

    name: str
    pc_range: tuple
    num_cams: int
    input_size: tuple
    embed_dims: int
    num_heads: int
    num_levels: int
    num_points: int
    tsa_num_points: int

    @property
    def spatial_shapes(self):
        """The preset's feature levels, each as (h, w), as ``SPATIAL_SHAPES``."""
        width, height = self.input_size
        return list(fpn_spatial_shapes(height, width, self.num_levels))

    @property
    def z_cfg(self):
        """The pillars' height sampling: ``NUM_POINTS_IN_PILLAR`` heights from z_min to z_max."""
        return {"num_points": NUM_POINTS_IN_PILLAR, "start": self.pc_range[2], "end": self.pc_range[5]}


_CARLA_PC_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)
_TINY = dict(embed_dims=128, num_heads=4, num_levels=1, num_points=4, tsa_num_points=4)
# The base presets: BEVFormer-base's encoder attention sizes over the four camera levels.
_BASE = dict(
    embed_dims=EMBED_DIMS,
    num_heads=NUM_HEADS,
    num_levels=NUM_LEVELS,
    num_points=SCA_NUM_POINTS,
    tsa_num_points=TSA_NUM_POINTS,
)

PRESETS = {
    "nuscenes_tiny": DatasetPreset("nuscenes_v1.0_full_640x360", PC_RANGE, NUM_CAMS, (640, 360), **_TINY),
    "nuscenes_base": DatasetPreset("nuscenes_v1.0_full_1600x900", PC_RANGE, NUM_CAMS, RAW_IMAGE_SIZE, **_BASE),
    "carla_tiny": DatasetPreset("carla_v0.9.10_640x480", _CARLA_PC_RANGE, NUM_CAMS, (640, 480), **_TINY),
    "carla_base": DatasetPreset("carla_v0.9.10_1280x960", _CARLA_PC_RANGE, NUM_CAMS, (1280, 960), **_BASE),
}


# --- Backbone and FPN dummy weights -----------------------------------------------------------

# Tuned so the random backbone has a realistic output std at C3-C5 (order 1), DCN offsets (under a pixel on average, so samples fall between pixels)
# and DCN masks (near 0.9).
DCN_OFFSET_STD = 0.04
DCN_MASK_BIAS = 2.6
RESIDUAL_BN_GAMMA = (0.1, 0.3)


def init_dummy_backbone_weights(torch_model, seed=0, input_std=1.0):
    """Fill every parameter and BatchNorm buffer with seeded random values.

    The stem conv is divided by ``input_std``, the std of the images the model will see, so
    everything after the stem sees the unit scale the values above were tuned for.

    The reference model's own initialization cannot be used. ``ModulatedDeformConv2dPack``
    allocates its weight uninitialized, and the ResNet stores ``init_cfg`` without applying
    it, so its BatchNorms keep identity statistics no real network has.
    """
    generator = torch.Generator().manual_seed(seed)

    def normal_(tensor, std, mean=0.0):
        tensor.copy_(torch.randn(tensor.shape, generator=generator) * std + mean)

    def uniform_(tensor, low, high):
        tensor.copy_(torch.rand(tensor.shape, generator=generator) * (high - low) + low)

    offset_convs = {
        id(module.conv_offset) for module in torch_model.modules() if isinstance(module, ModulatedDeformConv2dPack)
    }

    with torch.no_grad():
        for module in torch_model.modules():
            if isinstance(module, ModulatedDeformConv2dPack):
                fan_out = module.weight.shape[0] * module.weight.shape[2] * module.weight.shape[3]
                normal_(module.weight, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.zero_()
                normal_(module.conv_offset.weight, DCN_OFFSET_STD)
                # conv_offset emits 3*K*K channels: two thirds of (y, x)-interleaved offsets, then the
                # mask logits, which go through sigmoid.
                module.conv_offset.bias.zero_()
                module.conv_offset.bias[2 * module.conv_offset.bias.numel() // 3 :] = DCN_MASK_BIAS
            elif isinstance(module, nn.Conv2d):
                if id(module) in offset_convs:
                    continue
                fan_out = module.weight.shape[0] * module.weight.shape[2] * module.weight.shape[3]
                normal_(module.weight, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                uniform_(module.weight, 0.5, 1.5)
                normal_(module.bias, 0.1)
                normal_(module.running_mean, 0.1)
                uniform_(module.running_var, 0.5, 1.5)

        # With the BatchNorm ranges above, the last BatchNorm of each branch at gamma ~1
        # makes the branch outgrow its identity path block after block, and the output
        # grows by orders of magnitude over 33 blocks. This range keeps each branch's std
        # at about its identity's (ratio 0.6-1.1).
        for module in torch_model.modules():
            if hasattr(module, "bn3") and isinstance(module.bn3, nn.modules.batchnorm._BatchNorm):
                uniform_(module.bn3.weight, *RESIDUAL_BN_GAMMA)

        if input_std != 1.0:
            torch_model.conv1.weight /= input_std

    torch_model.eval()
    return torch_model


def init_dummy_fpn_weights(torch_model, seed=0):
    """Fill the FPN's convs with seeded random weights and biases."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in torch_model.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.weight.shape[0] * module.weight.shape[2] * module.weight.shape[3]
                module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    module.bias.copy_(torch.randn(module.bias.shape, generator=generator) * 0.1)
    torch_model.eval()
    return torch_model


# --- Backbone and FPN -------------------------------------------------------------------------

# Dtypes TTResNet emits for C3-C5.
BACKBONE_OUTPUT_DTYPES = [ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16]

# Random structure at these cell sizes and amplitudes, in pixel units, plus per-pixel noise.
IMAGE_NOISE_SCALES = ((8, 80.0), (32, 40.0), (128, 20.0))
IMAGE_PIXEL_NOISE = 8.0
# The std of random_image_batch, measured over full batches; its mean is near zero. Re-measure
# it when the noise constants change: the dummy backbone's stem is scaled by it.
IMAGE_STD = 19.5


# The references only run forward; without autograd they keep no activations for backward.
def build_reference_backbone():
    return init_dummy_backbone_weights(ResNet(**RESNET_KWARGS), input_std=IMAGE_STD).requires_grad_(False)


def build_reference_fpn():
    return init_dummy_fpn_weights(FPN(**FPN_KWARGS)).requires_grad_(False)


def to_conv_layout(nchw, device, dtype, layout=ttnn.TILE_LAYOUT):
    """NCHW torch tensor -> (1, 1, N*H*W, C) device tensor, the layout conv2d reads and writes."""
    n, c, h, w = nchw.shape
    nhwc = nchw.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, c)
    return ttnn.from_torch(nhwc, device=device, dtype=dtype, layout=layout)


def from_conv_layout(tt_tensor, nchw_shape):
    """(1, 1, N*H*W, C) device tensor -> NCHW torch tensor of ``nchw_shape``."""
    n, c, h, w = nchw_shape
    return ttnn.to_torch(tt_tensor).reshape(n, h, w, c).permute(0, 3, 1, 2)


def assert_pcc(expected, actual, pcc):
    passed, message = comp_pcc(expected, actual, pcc)
    logger.info(f"PCC {message} (threshold {pcc})")
    assert passed, f"PCC {message} is below {pcc}"
    return passed, message


def assert_finite(name, tensor):
    """comp_pcc zeroes NaN and Inf before correlating, so a PCC check needs this one first."""
    assert torch.isfinite(tensor).all(), f"non-finite values in {name}"


def to_device(tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device)


def random_image_batch():
    """Random camera images in the range BEVFormer's normalized images take: smooth random
    structure at several scales plus pixel noise around the dataset mean, clamped to [0, 255]
    as pixels are, minus ``IMAGE_MEAN_BGR``. Backbone error depends on the input scale, so the
    inputs keep the real one."""
    mean = torch.tensor(IMAGE_MEAN_BGR).view(1, 3, 1, 1)
    images = mean.expand(NUM_CAMS, 3, IMAGE_HEIGHT, IMAGE_WIDTH).clone()
    for cell, amplitude in IMAGE_NOISE_SCALES:
        coarse = torch.rand(NUM_CAMS, 3, IMAGE_HEIGHT // cell + 1, IMAGE_WIDTH // cell + 1) - 0.5
        images += amplitude * F.interpolate(
            coarse, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False
        )
    images += IMAGE_PIXEL_NOISE * torch.randn(images.shape)
    return images.clamp(0, 255) - mean


# --- Encoder ----------------------------------------------------------------------------------

BEV_SHAPES = {"tiny": (50, 50), "base": (BEV_H, BEV_W)}

# Spread of the random weights on top of upstream's init, so the offsets and attention logits vary
# per query. Upstream's init alone has zero offset and
# attention weights: every query samples the same fixed pattern with uniform weights.
# The TSA offsets are drawn narrower than the SCA's. Realistic offsets barely move with the input,
# so six layers carry a bfloat16-sized perturbation through unchanged, frame to frame too;
# random offset weights at a wider spread follow every perturbation of the BEV maps they then sample,
# and the layers amplify it on their own, so the test would measure the dummy model's sensitivity,
# not the port's error. The offsets' bias (upstream's init ring, up to 4 px) still exposes a
# misordered channel.
TSA_OFFSET_STD_PX = 0.5
TSA_LOGIT_STD = 2.5
SCA_OFFSET_STD_PX = 1.7
SCA_LOGIT_STD = 1.8

# Correlation length of the random features, in cells. The FPN's and the encoder's features are
# spatially smooth; white noise instead makes every sample position error an O(1) change in the
# sampled value. The encoder's queries carry that error from layer to layer and frame to frame, and
# the decoder's refinement feeds it back into its next layer's positions. With white BEV features
# the decoder's reference on bfloat16-rounded inputs and weights (fp32 compute) loses its
# correlation with itself by the last layer on the 200x200 grid, so no bound on the port would mean
# anything.
FEATURE_CELLS = 4

# Ego translation between the two frames, in BEV fractions (x, y): a few cells on the base grid.
EGO_SHIFT_BEV_FRACTION = (0.013, -0.021)


def _spread(linear, std, generator):
    """Random weights giving outputs of std ``std`` on unit-variance inputs, bias kept."""
    with torch.no_grad():
        scale = std / math.sqrt(linear.in_features)
        linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator) * scale)


def build_reference_encoder(num_layers=ENCODER_NUM_LAYERS, seed=0):
    """BEVFormer's encoder with upstream's init plus random offset and attention spreads,
    drawn from ``seed``. The projections, FFNs and norms keep upstream's and PyTorch's init."""
    torch.manual_seed(seed)
    model = BEVFormerEncoder(num_layers=num_layers)
    generator = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if isinstance(module, TemporalSelfAttention):
            _spread(module.sampling_offsets, TSA_OFFSET_STD_PX, generator)
            _spread(module.attention_weights, TSA_LOGIT_STD, generator)
        elif isinstance(module, MSDeformableAttention3D):
            _spread(module.sampling_offsets, SCA_OFFSET_STD_PX, generator)
            _spread(module.attention_weights, SCA_LOGIT_STD, generator)
    return model.eval().requires_grad_(False)


def _smooth(batch, channels, h, w, generator):
    """``(batch, channels, h, w)`` noise, bilinear over a grid of ``FEATURE_CELLS``-cell steps."""
    coarse = torch.randn(
        batch, channels, math.ceil(h / FEATURE_CELLS), math.ceil(w / FEATURE_CELLS), generator=generator
    )
    return F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)


def random_camera_features(batch_size, generator=None, spatial_shapes=SPATIAL_SHAPES):
    """Unit-variance FPN-like features ``(num_cams, num_keys, bs, C)``, levels flattened in
    ``spatial_shapes`` order, each smooth over ``FEATURE_CELLS`` cells."""
    levels = [_smooth(NUM_CAMS * batch_size, EMBED_DIMS, h, w, generator).flatten(2) for h, w in spatial_shapes]
    features = torch.cat(levels, -1)
    features = features / features.std()
    return features.view(NUM_CAMS, batch_size, EMBED_DIMS, -1).permute(0, 3, 1, 2).contiguous()


def camera_rows(value):
    """The reference's camera features ``(num_cams, num_keys, bs, C)`` batch- and camera-major,
    ``(bs * num_cams, num_keys, C)``, as the spatial cross-attention reads them."""
    num_cams, num_keys, bs, channels = value.shape
    return value.permute(2, 0, 1, 3).reshape(bs * num_cams, num_keys, channels).contiguous()


def random_bev(bev_shape, batch_size, generator=None):
    """A unit-variance BEV map ``(bev_h * bev_w, bs, C)``, smooth over ``FEATURE_CELLS`` cells, as
    the encoder's output is: the previous BEV, the decoder's and head's BEV features, the spatial
    cross-attention's queries."""
    bev = _smooth(batch_size, EMBED_DIMS, *bev_shape, generator)
    return (bev / bev.std()).flatten(2).permute(2, 0, 1).contiguous()


def random_encoder_inputs(bev_shape, batch_size=1, seed=0, yaw_step_deg=0.0):
    """Sequence-first BEV queries and positions ``(num_query, bs, C)``, camera features and the
    cameras' ``img_metas`` (see :func:`camera_img_metas` for ``yaw_step_deg``). The queries and positions are
    white, as the learned embeddings are."""
    generator = torch.Generator().manual_seed(seed)
    num_query = bev_shape[0] * bev_shape[1]
    return dict(
        bev_query=torch.randn(num_query, batch_size, EMBED_DIMS, generator=generator),
        bev_pos=torch.randn(num_query, batch_size, EMBED_DIMS, generator=generator),
        value=random_camera_features(batch_size, generator),
        img_metas=camera_img_metas(batch_size, yaw_step_deg=yaw_step_deg),
    )


def camera_img_metas(batch_size, preset="nuscenes_base", yaw_step_deg=0.0):
    """``lidar2img`` and ``img_shape`` of a preset's six-camera rig (:func:`img_metas_for_dataset`).
    ``yaw_step_deg`` turns sample ``b``'s rig by ``b * yaw_step_deg`` about the vertical axis, so
    the samples' cameras see different BEV cells."""
    metas = img_metas_for_dataset(PRESETS[preset], batch_size)
    for b, meta in enumerate(metas):
        yaw = math.radians(b * yaw_step_deg)
        turn = torch.eye(4)
        turn[:2, :2] = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        meta["lidar2img"] = (torch.tensor(meta["lidar2img"]) @ turn).tolist()
    return metas


def preset_pc_range(preset="nuscenes_base"):
    """A preset's point-cloud range, the box its camera geometry projects pillars from."""
    return tuple(PRESETS[preset].pc_range)


def sample_ego_shifts(batch_size):
    """``(bs, 2)`` ego translations in BEV fractions: ``EGO_SHIFT_BEV_FRACTION`` times ``b + 1`` for
    sample ``b``, so a shift taken from the wrong sample shows."""
    return torch.tensor(EGO_SHIFT_BEV_FRACTION) * torch.arange(1, batch_size + 1, dtype=torch.float32)[:, None]


# --- Decoder ----------------------------------------------------------------------------------


# Spread of the random weights on top of the upstream init: sampling offsets about 1.3 px off the
# init pattern, cross-attention logits of std 0.7 and self-attention logits of std 3, so the test
# sees varied sampling and peaked softmaxes.
SAMPLING_OFFSET_STD_PX = 1.3
ATTENTION_LOGIT_STD = 0.7
SELF_ATTENTION_LOGIT_STD = 3.0

# Scale of the reg branches' last Linear over PyTorch's default init, whose outputs have std
# ~0.08: refinements of the reference points of ~0.01 in logit space per layer, and the other box
# channels with std ~0.6. Larger refinements move every later
# layer's sampling points further and amplify the decoder's error through them; near-constant
# box channels make their PCC measure noise.
REG_REFINEMENT_SCALE = 0.125
REG_BOX_SCALE = 7.5


def _init_cross_attention(msda, generator):
    """BEVFormer ``CustomMSDeformableAttention.init_weights`` (mmcv's
    ``MultiScaleDeformableAttention`` init) plus random weights."""
    heads, levels, points = msda.num_heads, msda.num_levels, msda.num_points
    in_features = msda.sampling_offsets.in_features

    thetas = torch.arange(heads, dtype=torch.float32) * (2.0 * math.pi / heads)
    grid = torch.stack([thetas.cos(), thetas.sin()], -1)
    grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(heads, 1, 1, 2).repeat(1, levels, points, 1)
    for i in range(points):
        grid[:, :, i, :] *= i + 1
    msda.sampling_offsets.bias.copy_(grid.flatten())

    # The Linears read query + query_pos, of variance 2 (see _init_self_attention), so a
    # weight of std s gives outputs of std s * sqrt(2 * in_features).
    fan_in_scale = 1.0 / math.sqrt(2 * in_features)
    msda.sampling_offsets.weight.copy_(
        torch.randn(msda.sampling_offsets.weight.shape, generator=generator) * SAMPLING_OFFSET_STD_PX * fan_in_scale
    )
    msda.attention_weights.weight.copy_(
        torch.randn(msda.attention_weights.weight.shape, generator=generator) * ATTENTION_LOGIT_STD * fan_in_scale
    )
    msda.attention_weights.bias.zero_()
    for proj in (msda.value_proj, msda.output_proj):
        bound = math.sqrt(6.0 / (proj.in_features + proj.out_features))
        proj.weight.copy_(torch.rand(proj.weight.shape, generator=generator) * 2 * bound - bound)
        proj.bias.zero_()


def _init_self_attention(mha, generator):
    """Q and K spread so the ``q . k / sqrt(head_dim)`` logits have std SELF_ATTENTION_LOGIT_STD.

    Q and K project ``query + query_pos``, of variance 2 (a LayerNorm-ed or random query plus
    a random position). A weight of std s then gives q, k of std s * sqrt(2 * embed_dims),
    and the scaled logit's std is std(q) * std(k).
    """
    embed_dims = mha.embed_dim
    std = math.sqrt(SELF_ATTENTION_LOGIT_STD / (2 * embed_dims))
    qk_rows = mha.in_proj_weight[: 2 * embed_dims]
    qk_rows.copy_(torch.randn(qk_rows.shape, generator=generator) * std)


def build_reference_decoder(seed=0):
    """BEVFormer's decoder with dummy weights: ``_init_cross_attention`` and ``_init_self_attention``
    on top of PyTorch's default init, which the rest keeps."""
    torch.manual_seed(seed)
    model = DetectionTransformerDecoder(
        num_layers=DECODER_NUM_LAYERS,
        embed_dims=EMBED_DIMS,
        num_heads=NUM_HEADS,
        feedforward_channels=FEEDFORWARD_CHANNELS,
        num_points=DECODER_NUM_POINTS,
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, MSDeformableAttention):
                _init_cross_attention(module, generator)
            elif isinstance(module, nn.MultiheadAttention):
                _init_self_attention(module, generator)
    return model.eval().requires_grad_(False)


def init_reg_branches(branches):
    """Scale each branch's last Linear's outputs: REG_REFINEMENT_SCALE on the
    rows that refine the reference points (cx, cy, cz), REG_BOX_SCALE on the others."""
    scale = torch.full((CODE_SIZE, 1), REG_BOX_SCALE)
    scale[CODE_XY] = scale[CODE_Z] = REG_REFINEMENT_SCALE
    with torch.no_grad():
        for branch in branches:
            branch[-1].weight.mul_(scale)
            branch[-1].bias.mul_(scale[:, 0])
    return branches


def build_reg_branches(seed=1):
    """BEVFormer's per-layer box regression head, ``Linear-ReLU-Linear-ReLU-Linear(code_size)``,
    with init_reg_branches' scale."""
    torch.manual_seed(seed)
    branches = init_reg_branches(nn.ModuleList(reg_branch(EMBED_DIMS, CODE_SIZE) for _ in range(DECODER_NUM_LAYERS)))
    return branches.eval().requires_grad_(False)


def reference_box_codes(outputs, reg_branches):
    """Each layer's reg branch on the reference decoder's ``(L, nq, bs, C)`` outputs: the box codes
    the TT decoder also returns, ``(L, bs, nq, code_size)`` in its ``CENTERS_FIRST`` order."""
    return torch.stack([branch(out.permute(1, 0, 2)) for branch, out in zip(reg_branches, outputs)])[..., CENTERS_FIRST]


def random_reference_points(batch_size, generator=None):
    """Uniform in [0, 1], with a slice of points on and just inside the grid edges.

    Edge points exercise out-of-bounds zero padding in the sampling and the eps clamp of
    inverse_sigmoid, which queries near the BEV border reach.
    """
    points = torch.rand(batch_size, NUM_QUERY, 3, generator=generator)
    edges = torch.tensor([0.0, 1e-3, 1.0 - 1e-3, 1.0])
    num_edge = NUM_QUERY // 10
    points[:, :num_edge] = edges[torch.randint(len(edges), (batch_size, num_edge, 3), generator=generator)]
    return points


def random_decoder_inputs(bev_shape, batch_size=1, seed=None):
    """Sequence-first query, query_pos and BEV value, and reference points in [0, 1]."""
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return dict(
        query=torch.randn(NUM_QUERY, batch_size, EMBED_DIMS, generator=generator),
        value=random_bev(bev_shape, batch_size, generator),
        query_pos=torch.randn(NUM_QUERY, batch_size, EMBED_DIMS, generator=generator),
        reference_points=random_reference_points(batch_size, generator),
    )


def layer_metrics(expected, actual, input_reference_points, bev_shape):
    """Per-layer accuracy of ``actual`` (output, reference points) against ``expected``, for logging.

    Per layer: the output PCC, the PCC of the refinement step in logit space for xy and z
    apart, and the mean xy error of the refined points in BEV pixels. The steps show the
    refinement's accuracy, which the absolute points barely reflect: they move little per layer.
    """
    (expected_output, expected_points), (actual_output, actual_points) = expected, actual
    bev_h, bev_w = bev_shape
    px_scale = torch.tensor([bev_w, bev_h], dtype=torch.float32)

    def steps(points):
        logits = inverse_sigmoid(torch.cat([input_reference_points[None], points]))
        return logits[1:] - logits[:-1]

    expected_steps, actual_steps = steps(expected_points), steps(actual_points)
    return [
        dict(
            output=comp_pcc(expected_output[layer], actual_output[layer])[1],
            refine_xy=comp_pcc(expected_steps[layer][..., :2], actual_steps[layer][..., :2])[1],
            refine_z=comp_pcc(expected_steps[layer][..., 2:], actual_steps[layer][..., 2:])[1],
            px=((actual_points[layer][..., :2] - expected_points[layer][..., :2]).abs() * px_scale).mean().item(),
        )
        for layer in range(expected_output.shape[0])
    ]


def assert_channels_close(expected, actual, pcc=0.99):
    """PCC of every last-dim channel apart, so a channel of small range cannot hide behind
    the others."""
    for channel in range(expected.shape[-1]):
        assert_pcc(expected[..., channel], actual[..., channel], pcc)


def check_decoder_outputs(expected, actual, input_reference_points, bev_shape):
    """The TT decoder's (layer outputs, refined points, box codes) as torch tensors, the outputs
    sequence-first, against the reference's: all finite, the outputs and points layer by layer, as the first layers'
    accuracy would hide a failing last layer in the stack, and the box codes channel by channel.
    Logs ``layer_metrics``."""
    for name, tensor in zip(("output", "reference points", "box codes"), actual):
        assert_finite(f"the decoder {name}", tensor)
    for layer, metrics in enumerate(layer_metrics(expected[:2], actual[:2], input_reference_points, bev_shape)):
        logger.info(f"layer {layer}: " + ", ".join(f"{key} {value:.5f}" for key, value in metrics.items()))
    for expected_layers, actual_layers in zip(expected[:2], actual[:2], strict=True):
        for expected_layer, actual_layer in zip(expected_layers, actual_layers, strict=True):
            assert_pcc(expected_layer, actual_layer, 0.99)
    assert_channels_close(expected[2], actual[2])


# --- Head -------------------------------------------------------------------------------------


def build_reference_head(bev_shape, seed=0):
    """BEVFormer's head with dummy weights over a ``bev_shape`` BEV map, all drawn from ``seed``.

    The decoder is ``build_reference_decoder``'s, the reg branches get ``init_reg_branches``'
    scale, and ``reference_points`` gets BEVFormer's xavier init, which spreads the initial
    points over the BEV grid. The rest keeps PyTorch's default init.
    """
    torch.manual_seed(seed)
    model = BEVFormerHead(*bev_shape, num_query=NUM_QUERY, embed_dims=EMBED_DIMS)
    nn.init.xavier_uniform_(model.reference_points.weight)
    nn.init.zeros_(model.reference_points.bias)
    init_reg_branches(model.reg_branches)
    model.decoder = build_reference_decoder(seed)
    return model.eval().requires_grad_(False)


# Box center error bounds in metres: PCC alone allows metres of error on x and y, which span
# 102.4 m, where nuScenes matches boxes from 0.5 m. The worst of the 900 queries carries a
# bfloat16 tail no single op's precision moves; the bounds leave headroom over it.
CENTER_MEAN_ATOL = 0.015
CENTER_MAX_ATOL = 0.35


def log_center_error(expected, actual):
    """Logs and returns the box predictions' center channels' absolute error, in metres."""
    error = (expected[..., CODE_CENTERS] - actual[..., CODE_CENTERS]).abs()
    logger.info(f"box center error: mean {error.mean():.4f} m, max {error.max():.4f} m")
    return error


def assert_centers_close(expected, actual):
    """The box predictions' center channels within ``CENTER_MEAN_ATOL`` and ``CENTER_MAX_ATOL``."""
    error = log_center_error(expected, actual)
    assert error.mean() <= CENTER_MEAN_ATOL, f"mean box center error {error.mean():.4f} m"
    assert error.max() <= CENTER_MAX_ATOL, f"max box center error {error.max():.4f} m"


def check_head_outputs(expected, actual):
    """The TT head's (class logits, box predictions) device tensors against the reference's:
    finite, the centers within ``assert_centers_close``'s bounds, and the logits and the boxes over
    all layers, then the last alone, which the coder reads and where the bfloat16 error has built up
    most. The boxes go channel by channel: a joint PCC would be the centers' alone, as x and y span
    102.4 m against z's 8 m and the other channels' O(1)."""
    actual = tuple(ttnn.to_torch(t).float() for t in actual)
    for name, tensor in zip(("class logits", "box predictions"), actual):
        assert_finite(f"the head {name}", tensor)
    assert_centers_close(expected[1], actual[1])
    for layers in (slice(None), -1):
        assert_pcc(expected[0][layers], actual[0][layers], 0.99)
        assert_channels_close(expected[1][layers], actual[1][layers])


def tt_coder_detections(coder, all_cls_scores, all_bbox_preds):
    """``coder``'s per-sample host detections of the head's ``(L, bs, num_query, *)`` device
    outputs, from the last decoder layer, as ``reference.nms_free_coder.NMSFreeCoder.decode``
    returns them: the device top-k and validity mask, then the host compaction."""
    scores, labels, _, boxes = coder.topk(all_cls_scores[-1], all_bbox_preds[-1])
    return coder.compact(scores, labels, boxes, coder.valid(scores, boxes))


def assert_boxes_close(expected, actual, pcc=0.99):
    """PCC of every decoded box channel apart, yaw through its sine and cosine, as atan2
    may land on either side of +-pi."""
    for channels in (BOX_CENTER, BOX_SIZE, BOX_VELOCITY):
        assert_channels_close(expected[..., channels], actual[..., channels], pcc)
    yaw_expected, yaw_actual = expected[..., BOX_YAW], actual[..., BOX_YAW]
    assert_channels_close(
        torch.cat([yaw_expected.sin(), yaw_expected.cos()], dim=-1),
        torch.cat([yaw_actual.sin(), yaw_actual.cos()], dim=-1),
        pcc,
    )


# The device's selected scores against the reference's at the same pairs, and the share of the
# reference's top-k the device selects. The score error is gated at its 99th percentile, with a
# looser bound on the worst pair: the second frame's BEV is off most at the grid's edges, where the previous BEV, rotated and shifted by the ego motion,
# turns to zeros and the self-attention samples across that step, so the few queries there
# score off by several times the rest. The bounds leave headroom, not so much that a regression
# in the logits passes.
SCORE_ERROR_QUANTILE = 0.99
MAX_SCORE_ERROR = 0.04
MAX_WORST_SCORE_ERROR = 0.2
MIN_TOPK_OVERLAP = 0.9


def assert_topk_close(cls_scores, bbox_preds, tt_cls_scores, tt_bbox_preds, tt_topk, pcc=0.99, bottom_center=False):
    """The coder's device top-k against the reference head's outputs, per sample: the pairs it
    selects lie within twice the score error of the reference's top-k and cover at least
    ``MIN_TOPK_OVERLAP`` of it, its scores are the reference's at the same (query, class) pairs
    within ``MAX_SCORE_ERROR`` at the ``SCORE_ERROR_QUANTILE`` and ``MAX_WORST_SCORE_ERROR`` at
    worst, and its decoded boxes are its own box predictions' at the selected
    queries, decoded on the host, at PCC ``pcc``. The box predictions themselves are the caller's
    to check against the reference's, over all queries: the selected ones span narrower ranges,
    where the error carried from the BEV lowers the PCC further.

    ``cls_scores`` and ``bbox_preds`` are the reference's last-layer ``(bs, num_query, *)``
    outputs, ``tt_cls_scores`` and ``tt_bbox_preds`` the device's as torch tensors, ``tt_topk``
    ``TTNMSFreeCoder.topk``'s outputs as torch tensors, or ``TTBEVFormer.detect``'s first four with
    ``bottom_center``, whose boxes have z at the bottom face. The device's logits differ from the
    reference's slightly, so near-equal scores may rank in a different order and the two top-k
    sets differ at their tail. The scores are compared by their error, not PCC: the top-k's span
    a narrow range, where a small error already lowers the PCC.
    """
    reference = NMSFreeCoder()
    tt_scores, tt_labels, tt_query_index, tt_boxes = tt_topk
    # The largest score error over all num_query * num_classes pairs. topk ranks the device's
    # logits exactly, which order as their exact sigmoid does.
    score_error = (cls_scores.sigmoid() - tt_cls_scores.float().sigmoid()).abs().amax((1, 2))
    for i in range(cls_scores.shape[0]):
        labels, query_index = tt_labels[i].long(), tt_query_index[i].long()
        ref_scores, ref_labels, ref_query_index, _ = reference.topk(cls_scores[i], bbox_preds[i])
        assert (tt_scores[i][1:] <= tt_scores[i][:-1]).all(), f"sample {i}: top-k scores are not sorted"
        # With e the score error and s_k the reference's k-th score: every reference top-k pair
        # scores at least s_k - e on device, so the device's k-th score does too, and every pair
        # the device selects scores at least s_k - 2e in the reference.
        selected = cls_scores[i, query_index, labels].sigmoid()
        bound = ref_scores[-1] - 2 * score_error[i]
        assert (selected >= bound).all(), f"sample {i}: a selected pair scores below the top-k bound {bound:.5f}"
        pairs = set((query_index * NUM_CLASSES + labels).tolist())
        ref_pairs = set((ref_query_index * NUM_CLASSES + ref_labels).tolist())
        overlap = len(pairs & ref_pairs) / len(ref_pairs)
        errors = (selected - tt_scores[i].float()).abs()
        selected_error = errors.quantile(SCORE_ERROR_QUANTILE).item()
        worst_error = errors.max().item()
        logger.info(
            f"sample {i}: top-{reference.max_num} overlap {overlap:.3f}, selected score error "
            f"p{SCORE_ERROR_QUANTILE * 100:.0f} {selected_error:.5f}, max {worst_error:.5f}"
        )
        assert overlap >= MIN_TOPK_OVERLAP, f"sample {i}: top-{reference.max_num} overlap {overlap:.3f}"
        assert selected_error <= MAX_SCORE_ERROR, f"sample {i}: selected score error {selected_error:.5f}"
        assert worst_error <= MAX_WORST_SCORE_ERROR, f"sample {i}: worst selected score error {worst_error:.5f}"
        expected_boxes = denormalize_bbox(tt_bbox_preds[i, query_index].float())
        if bottom_center:
            expected_boxes = reference_bottom_center(expected_boxes)
        assert_boxes_close(expected_boxes, tt_boxes[i].float(), pcc)


# --- Perception transformer and detector ------------------------------------------------------

# Ego motion between consecutive frames, sample ``b`` scaled by ``b + 1``: a few BEV cells of
# translation and a heading change large enough that the previous BEV's rotation moves most cells.
EGO_STEP_M = (2.0, -1.0)
EGO_TURN_DEG = 5.0


def build_reference_transformer(num_layers, seed=0):
    """``PerceptionTransformer`` over ``build_reference_encoder``'s encoder; the embeddings and the
    CAN-bus MLP keep upstream's and PyTorch's init."""
    torch.manual_seed(seed)
    return PerceptionTransformer(build_reference_encoder(num_layers, seed)).eval().requires_grad_(False)


def build_reference_bevformer(bev_shape):
    """The detector over a ``(bev_h, bev_w)`` grid from the dummy-weight parts above."""
    bev_h, bev_w = bev_shape
    model = BEVFormer(
        build_reference_backbone(),
        build_reference_fpn(),
        build_reference_transformer(ENCODER_NUM_LAYERS),
        build_reference_head(bev_shape),
        bev_h,
        bev_w,
    )
    return model.eval().requires_grad_(False)


# The box code channels the detector reads off small differences, the yaw's sine and the
# velocities, carry the backbone's and the encoder's bfloat16 error the most; they gate at this
# floor, every other output at the test's PCC.
LOW_PRECISION_BOX_CODES = (CODE_SIN.start, *range(CODE_VELOCITY.start, CODE_VELOCITY.stop))
LOW_PRECISION_BOX_CODE_PCC = 0.94


def check_detector_outputs(expected, actual, pcc):
    """The TT detector's class logits, box predictions, BEV and ``TTBEVFormer.detect``'s first four
    outputs, as torch tensors, against the reference's ``(cls_scores, bbox_preds, bev)``: finite, the
    BEV, the last layer's logits and boxes at ``pcc`` (``LOW_PRECISION_BOX_CODES`` at
    ``LOW_PRECISION_BOX_CODE_PCC``), and the detections by ``assert_topk_close``.
    Logs the box center error."""
    cls_scores, bbox_preds, bev = expected
    tt_cls_scores, tt_bbox_preds, tt_bev, tt_topk = actual
    for name, tensor in (("BEV", tt_bev), ("class logits", tt_cls_scores), ("boxes", tt_bbox_preds)):
        assert_finite(f"the detector's {name}", tensor)
    log_center_error(bbox_preds[-1], tt_bbox_preds[-1])
    assert_pcc(bev, tt_bev.reshape(bev.shape), pcc)
    # The last decoder layer's, which the coder reads.
    assert_pcc(cls_scores[-1], tt_cls_scores[-1], pcc)
    for channel in range(bbox_preds.shape[-1]):
        channel_pcc = min(pcc, LOW_PRECISION_BOX_CODE_PCC) if channel in LOW_PRECISION_BOX_CODES else pcc
        assert_pcc(bbox_preds[-1][..., channel], tt_bbox_preds[-1][..., channel], channel_pcc)
    assert_topk_close(
        cls_scores[-1], bbox_preds[-1], tt_cls_scores[-1], tt_bbox_preds[-1], tt_topk, pcc, bottom_center=True
    )


def random_fpn_levels(batch_size, generator, spatial_shapes=SPATIAL_SHAPES):
    """Unit-variance FPN-like levels ``(bs, num_cams, C, h, w)``, smooth as the encoder tests'."""
    levels = []
    for h, w in spatial_shapes:
        level = _smooth(batch_size * NUM_CAMS, EMBED_DIMS, h, w, generator)
        levels.append((level / level.std()).view(batch_size, NUM_CAMS, EMBED_DIMS, h, w))
    return levels


def fpn_rows(level):
    """A ``(bs, num_cams, C, h, w)`` level as the TTNN FPN emits it, ``(1, 1, bs * num_cams * h * w, C)``."""
    return level.permute(0, 1, 3, 4, 2).reshape(1, 1, -1, level.shape[2])


def frame_metas(batch_size, num_frames, generator, yaw_step_deg=0.0, motion=None):
    """Per frame, ``img_metas`` with the relative CAN bus of an ego that moves by ``EGO_STEP_M`` and
    turns by ``EGO_TURN_DEG`` per frame, both times ``b + 1`` for sample ``b``. ``motion`` scales
    each of the ``num_frames - 1`` steps, 1 by default; -1 drives one step backwards and turns the
    other way. The CAN bus's other readings are random."""
    motion = [1.0] * (num_frames - 1) if motion is None else list(motion)
    assert len(motion) == num_frames - 1, f"{len(motion)} step scales for {num_frames} frames"
    frames, previous = [], [None] * batch_size
    for i in range(num_frames):
        travelled = sum(motion[:i])
        metas = camera_img_metas(batch_size, yaw_step_deg=yaw_step_deg)
        for b, meta in enumerate(metas):
            can_bus = torch.randn(CAN_BUS_DIMS, generator=generator, dtype=torch.float64)
            heading = 0.3 + 0.2 * b + travelled * math.radians(EGO_TURN_DEG) * (b + 1)
            can_bus[0] = travelled * EGO_STEP_M[0] * (b + 1)
            can_bus[1] = travelled * EGO_STEP_M[1] * (b + 1)
            can_bus[2] = 0.0
            can_bus[-2] = heading
            can_bus[-1] = math.degrees(heading)
            meta["can_bus"] = relative_can_bus(can_bus, previous[b]).numpy()
            previous[b] = can_bus
        frames.append(metas)
    return frames


def signposted_forward(device, forward):
    """Runs ``forward()`` once between Tracy's ``start`` and ``stop`` signposts and returns what it
    returns. The call is the measured region, so host dispatch and the stage signposts inside
    ``forward`` land in it. Syncs before ``stop``: otherwise the marker can fire while the device
    is still running the last programs."""
    # Imported here: only the perf harnesses run under Tracy.
    from tracy import signpost

    ttnn.synchronize_device(device)
    # Drains and resets the device profiler buffers so the signposted region starts from
    # empty; the warmup's markers would otherwise eat into the same budget.
    ttnn.ReadDeviceProfiler(device)
    signpost("start")
    outputs = forward()
    ttnn.synchronize_device(device)
    signpost("stop")
    return outputs


@contextmanager
def signposted_trace(device, forward):
    """Captures ``forward()`` as a trace and replays it once between Tracy's ``start`` and ``stop``
    signposts; yields the trace id and the outputs the replays write into. Run ``forward`` once
    before, so the capture records already-compiled programs. The trace is released on exit.
    Stage signposts fire during the capture, outside this window: the replay runs the recorded
    device commands and does not re-enter ``forward``."""
    # Imported here: only the perf harnesses run under Tracy.
    from tracy import signpost

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = forward()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    try:
        ttnn.synchronize_device(device)
        # Drains and resets the device profiler buffers so the signposted region starts from
        # empty; the warmup's markers would otherwise eat into the same budget.
        ttnn.ReadDeviceProfiler(device)
        signpost("start")
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        signpost("stop")
        yield trace_id, outputs
    finally:
        ttnn.release_trace(device, trace_id)
