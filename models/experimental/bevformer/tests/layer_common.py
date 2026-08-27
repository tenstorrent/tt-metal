# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the single-``BEVFormerLayer`` tests.

A layer is not self-contained: the encoder derives the reference points, the
camera projections and the BEV mask once and hands them to every layer. Testing
one layer means reproducing that preamble exactly, on both paths, so the PCC and
the profile harness measure the layer and not a different projection.

Camera geometry comes from the dataset's fixed rig, not from random matrices.
``lidar2img`` decides ``bev_mask`` and therefore the spatial-cross-attention
rebatch length, which sizes every spatial-path tensor; drawing it from the RNG
would make the workload depend on the seed and on allocation order.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

import ttnn
from models.experimental.bevformer.config.encoder_config import img_metas_for_dataset
from models.experimental.bevformer.reference.encoder import BEVFormerLayer
from models.experimental.bevformer.reference.point_sampling_3d_2d import (
    generate_reference_points,
    point_sampling_3d_to_2d,
)
from models.experimental.bevformer.tt.model_preprocessing import preprocess_bevformer_layer_parameters
from models.experimental.bevformer.tt.tt_encoder import TTBEVFormerLayer
from models.experimental.bevformer.tt.tt_point_sampling_3d_2d import point_sampling_3d_to_2d_ttnn

# The encoder consumes every other key of ``get_encoder_kwargs`` itself; only
# these reach a layer.
LAYER_KWARG_KEYS = (
    "embed_dims",
    "num_heads",
    "num_levels",
    "num_points",
    "num_cams",
    "feedforward_channels",
    "batch_first",
)


def layer_kwargs_from_config(config) -> Dict[str, Any]:
    encoder_kwargs = config.get_encoder_kwargs()
    kwargs = {key: encoder_kwargs[key] for key in LAYER_KWARG_KEYS if key in encoder_kwargs}
    kwargs["batch_first"] = True
    return kwargs


@dataclass
class LayerFixture:
    """Both layer implementations plus the inputs each one expects."""

    ref_model: BEVFormerLayer
    tt_model: TTBEVFormerLayer
    ref_inputs: Dict[str, Any]
    tt_inputs: Dict[str, Any]
    img_metas: List[Dict[str, Any]]


def build_layer_fixture(device, config, bev_size, batch_size: int, dtype=ttnn.bfloat16, lidar2img=None) -> LayerFixture:
    """Build a reference and a TT ``BEVFormerLayer`` sharing one set of inputs.

    ``lidar2img`` overrides the dataset rig with a ``[batch_size, num_cams, 4, 4]``
    stack, so a diagnostic can hold everything but the camera geometry fixed.
    """
    dataset_config = config.dataset_config
    model_config = config.model_config

    bev_h, bev_w = bev_size
    num_queries = bev_h * bev_w
    embed_dims = model_config.embed_dims
    num_cams = dataset_config.num_cams
    num_levels = model_config.num_levels

    spatial_shapes = torch.tensor(dataset_config.spatial_shapes[:num_levels], dtype=torch.long)
    level_start_index = config.get_level_start_index()[:num_levels]
    bev_shape = torch.tensor([[bev_h, bev_w]])

    bev_query = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)
    bev_pos = torch.randn(batch_size, num_queries, embed_dims, dtype=torch.float32)

    key_length = sum(h * w for h, w in spatial_shapes.tolist())
    camera_features = torch.randn(num_cams, key_length, batch_size, embed_dims, dtype=torch.float32)

    img_metas = img_metas_for_dataset(dataset_config, batch_size)
    if lidar2img is None:
        lidar2img = torch.stack([torch.tensor(meta["lidar2img"], dtype=torch.float32) for meta in img_metas])
    else:
        lidar2img = lidar2img.to(torch.float32)
        for index, meta in enumerate(img_metas):
            meta["lidar2img"] = lidar2img[index].tolist()

    encoder_kwargs = config.get_encoder_kwargs()
    pc_range = encoder_kwargs["pc_range"]
    z_cfg = encoder_kwargs["z_cfg"]

    reference_points_3d = generate_reference_points(
        bev_h=bev_h,
        bev_w=bev_w,
        z_cfg=z_cfg,
        batch_size=batch_size,
        dtype=torch.float32,
    )

    ref_points_cam, ref_bev_mask = point_sampling_3d_to_2d(
        reference_points=reference_points_3d,
        pc_range=pc_range,
        lidar2img=lidar2img,
        img_metas=img_metas,
    )
    tt_points_cam, tt_bev_mask = point_sampling_3d_to_2d_ttnn(
        reference_points=reference_points_3d,
        pc_range=pc_range,
        lidar2img=lidar2img,
        img_metas=img_metas,
        device=device,
    )

    layer_kwargs = layer_kwargs_from_config(config)

    ref_model = BEVFormerLayer(**layer_kwargs)
    ref_model.eval()

    tt_model = TTBEVFormerLayer(
        device=device,
        params=preprocess_bevformer_layer_parameters(ref_model, device=device, dtype=dtype),
        **layer_kwargs,
    )

    ref_inputs = {
        "bev_query": bev_query,
        "key": camera_features,
        "value": camera_features,
        "bev_pos": bev_pos,
        "spatial_shapes": spatial_shapes,
        "bev_shape": bev_shape,
        "level_start_index": level_start_index,
        "prev_bev": None,
        "reference_points_3d": reference_points_3d,
        "reference_points_cam": ref_points_cam,
        "bev_mask": ref_bev_mask,
    }

    tt_inputs = {
        "bev_query": ttnn.from_torch(bev_query, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT),
        "key": ttnn.from_torch(camera_features, device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
        "bev_pos": ttnn.from_torch(bev_pos, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT),
        "spatial_shapes": spatial_shapes,
        "bev_shape": bev_shape,
        "level_start_index": ttnn.from_torch(level_start_index, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT),
        "prev_bev": None,
        "reference_points_3d": reference_points_3d,
        "reference_points_cam": tt_points_cam,
        "bev_mask": tt_bev_mask,
    }
    tt_inputs["value"] = tt_inputs["key"]

    return LayerFixture(
        ref_model=ref_model,
        tt_model=tt_model,
        ref_inputs=ref_inputs,
        tt_inputs=tt_inputs,
        img_metas=img_metas,
    )
