# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``reference.bevformer.BEVFormer`` as TtBEVFormer takes it."""

from types import SimpleNamespace

import torch

import ttnn
from models.experimental.bevformer.tt.model_preprocessing import (
    DEFAULT_DTYPE,
    create_perception_transformer_parameters,
)
from models.experimental.bevformer.tt.model_preprocessing_backbone import (
    create_fpn_parameters,
    create_resnet_parameters,
)
from models.experimental.bevformer.tt.model_preprocessing_head import create_head_parameters


@torch.no_grad()
def create_bevformer_parameters(model, img, device, dtype=DEFAULT_DTYPE):
    """The detector's parameters for images shaped like ``img`` ``(bs, num_cams, 3, H, W)``: the
    backbone and FPN pin every conv to that shape, so the TT detector only takes it. One reference
    forward of the backbone records the shapes, and the FPN's on its features the levels' ``(h, w)``.
    The BEV queries and their positional encoding are constants of the weights, so they are
    uploaded here, ``(1, bev_h * bev_w, C)``. ``config`` carries the reference's settings the TT
    detector is built with, so the two cannot disagree."""
    images = img.flatten(0, 1)
    # create_resnet_parameters runs the backbone to record its conv shapes; its features come from
    # that same forward, as a second ResNet101 forward on the CPU takes minutes. The forward runs
    # under ttnn's tracer, whose tensor subclass is unwrapped here.
    recorded = []
    hook = model.img_backbone.register_forward_hook(lambda _module, _inputs, outputs: recorded.append(outputs))
    try:
        backbone = create_resnet_parameters(model.img_backbone, images)
    finally:
        hook.remove()
    features = [feature.as_subclass(torch.Tensor) for feature in recorded[-1]]
    levels = model.img_neck(list(features))

    def upload(tensor):
        return ttnn.from_torch(tensor[None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return SimpleNamespace(
        config=SimpleNamespace(
            bev_h=model.bev_h,
            bev_w=model.bev_w,
            batch_size=img.shape[0],
            num_cams=img.shape[1],
            out_indices=tuple(model.img_backbone.out_indices),
            spatial_shapes=tuple(tuple(level.shape[-2:]) for level in levels),
        ),
        backbone=backbone,
        neck=create_fpn_parameters(model.img_neck, features),
        transformer=create_perception_transformer_parameters(model.transformer, device, dtype),
        head=create_head_parameters(model.head, device),
        bev_queries=upload(model.bev_embedding.weight),
        bev_pos=upload(model.positional_encoding(model.bev_h, model.bev_w)),
    )
