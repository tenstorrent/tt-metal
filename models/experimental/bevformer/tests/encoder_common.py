# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Configuration and helpers for the encoder tests."""

import math

import torch
import torch.nn.functional as F

from models.experimental.bevformer.config.encoder_config import get_preset_config
from models.experimental.bevformer.reference.encoder import BEVFormerEncoder
from models.experimental.bevformer.reference.spatial_cross_attention import MSDeformableAttention3D
from models.experimental.bevformer.reference.temporal_self_attention import TemporalSelfAttention
from models.experimental.bevformer.tests.camera_rig import img_metas_for_dataset

EMBED_DIMS = 256
NUM_CAMS = 6
NUM_LAYERS = 6
BEV_SHAPES = {"tiny": (50, 50), "base": (200, 200)}
# The FPN's four levels for BEVFormer-base's 928x1600 (padded 900x1600) images, as (h, w).
SPATIAL_SHAPES = ((116, 200), (58, 100), (29, 50), (15, 25))

# Spread of the random weights on top of upstream's init, after the offset and attention-logit
# spread the BEVFormer-base checkpoint's encoder shows. Upstream's init alone has zero offset and
# attention weights: every query samples the same fixed pattern with uniform weights.
# The TSA offsets are drawn narrower than the checkpoint's. Its offsets barely move with the input,
# so its six layers carry a bfloat16-sized perturbation through unchanged, frame to frame too;
# random offset weights at its spread follow every perturbation of the BEV maps they then sample,
# and the layers amplify it on their own, so the test would measure the dummy model's sensitivity,
# not the port's error. The offsets' bias (upstream's init ring, up to 4 px) still exposes a
# misordered channel.
TSA_OFFSET_STD_PX = 0.5
TSA_LOGIT_STD = 2.5
SCA_OFFSET_STD_PX = 1.7
SCA_LOGIT_STD = 1.8

# Correlation length of the random features, in cells. The FPN's and the encoder's features are
# spatially smooth; white noise instead makes every sample position error an O(1) change in the
# sampled value, which the queries carry from layer to layer and the previous BEV from frame to
# frame.
FEATURE_CELLS = 4

# Ego translation between the two frames, in BEV fractions (x, y): a few cells on the base grid.
EGO_SHIFT = (0.013, -0.021)


def _spread(linear, std, generator):
    """Random weights giving outputs of std ``std`` on unit-variance inputs, bias kept."""
    with torch.no_grad():
        scale = std / math.sqrt(linear.in_features)
        linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator) * scale)


def build_reference_encoder(num_layers=NUM_LAYERS, seed=0):
    """BEVFormer's encoder with upstream's init plus trained-like offset and attention spreads,
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
    """The reference's camera features ``(num_cams, num_keys, bs, C)`` as the TTNN encoder takes
    them, ``(bs * num_cams, num_keys, C)``."""
    num_cams, num_keys, bs, channels = value.shape
    return value.permute(2, 0, 1, 3).reshape(bs * num_cams, num_keys, channels).contiguous()


def random_bev(bev_shape, batch_size, generator=None):
    """A unit-variance BEV map ``(bev_h * bev_w, bs, C)``, smooth over ``FEATURE_CELLS`` cells,
    as the encoder's output is; the previous frame's BEV for the self-attention tests."""
    bev = _smooth(batch_size, EMBED_DIMS, *bev_shape, generator)
    return (bev / bev.std()).flatten(2).permute(2, 0, 1).contiguous()


def random_encoder_inputs(bev_shape, batch_size=1, seed=0, yaw_step_deg=0.0):
    """Sequence-first BEV queries and positions ``(num_query, bs, C)``, camera features and the
    cameras' ``img_metas`` (see :func:`img_metas` for ``yaw_step_deg``). The queries and positions are
    white, as the learned embeddings are."""
    generator = torch.Generator().manual_seed(seed)
    num_query = bev_shape[0] * bev_shape[1]
    return dict(
        bev_query=torch.randn(num_query, batch_size, EMBED_DIMS, generator=generator),
        bev_pos=torch.randn(num_query, batch_size, EMBED_DIMS, generator=generator),
        value=random_camera_features(batch_size, generator),
        img_metas=img_metas(batch_size, yaw_step_deg=yaw_step_deg),
    )


def img_metas(batch_size, preset="nuscenes_base", yaw_step_deg=0.0):
    """``lidar2img`` and ``img_shape`` of a preset's six-camera rig (``tests/camera_rig.py``).
    ``yaw_step_deg`` turns sample ``b``'s rig by ``b * yaw_step_deg`` about the vertical axis, so
    the samples' cameras see different BEV cells."""
    metas = img_metas_for_dataset(get_preset_config(preset).dataset_config, batch_size)
    for b, meta in enumerate(metas):
        yaw = math.radians(b * yaw_step_deg)
        turn = torch.eye(4)
        turn[:2, :2] = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        meta["lidar2img"] = (torch.tensor(meta["lidar2img"]) @ turn).tolist()
    return metas


def pc_range(preset="nuscenes_base"):
    """A preset's point-cloud range, the box its camera geometry projects pillars from."""
    return tuple(get_preset_config(preset).dataset_config.pc_range)


def ego_shift(batch_size):
    """``(bs, 2)`` ego translations in BEV fractions: ``EGO_SHIFT`` times ``b + 1`` for sample ``b``,
    so a shift taken from the wrong sample shows."""
    return torch.tensor(EGO_SHIFT) * torch.arange(1, batch_size + 1, dtype=torch.float32)[:, None]
