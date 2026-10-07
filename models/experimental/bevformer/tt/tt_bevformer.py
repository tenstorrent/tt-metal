# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the BEVFormer detector (``reference/bevformer.py``): the ResNet101-DCN backbone,
the FPN, the perception transformer and the detection head, on device end to end.

A frame's host-derived inputs (CAN bus, ego shift, previous-BEV rotation, the cameras' rebatch
plan) are prepared by :meth:`TtBEVFormer.prepare_frame` into buffers refilled in place, so the
forward runs on device only and a trace captured with a frame replays on the next one. The caller
carries the previous BEV between frames, as upstream's ``forward_test`` does.
"""

import ttnn

from models.experimental.bevformer import model_config
from models.experimental.bevformer.tt.tt_fpn import TtFPN
from models.experimental.bevformer.tt.tt_head import TtBEVFormerHead
from models.experimental.bevformer.tt.tt_perception_transformer import TtPerceptionTransformer
from models.experimental.bevformer.tt.tt_resnet import TtResNet


class TtBEVFormer:
    """The detector for the image shape its parameters were built for.

    The backbone's and FPN's convs are pinned to the image shape, and building the encoder consumes
    its cross-attentions' sampling-offset weights (see ``TTBEVFormerEncoder``), so each instance
    needs its own ``create_bevformer_parameters``.
    """

    def __init__(self, params, device):
        """``params`` from ``model_preprocessing.create_bevformer_parameters``."""
        config = params.config
        assert (
            config.num_cams == params.transformer.config.num_cams
        ), f"{config.num_cams} cameras in the images, {params.transformer.config.num_cams} in the transformer"
        assert (config.bev_h, config.bev_w) == tuple(
            params.head.bev_shape
        ), f"the detector's BEV is {config.bev_h}x{config.bev_w}, the head's {tuple(params.head.bev_shape)}"
        self.params = params
        self.batch_size = config.batch_size
        self.backbone = TtResNet(
            params.backbone.conv_args,
            params.backbone["res_model"],
            device,
            out_indices=config.out_indices,
            **model_config.tt_resnet_kwargs(),
        )
        self.neck = TtFPN(
            conv_args=params.neck.conv_args,
            conv_pth=params.neck,
            device=device,
            input_dtypes=self.backbone.output_dtypes,
            **model_config.tt_fpn_kwargs(),
        )
        self.transformer = TtPerceptionTransformer(
            params.transformer, device, bev_h=config.bev_h, bev_w=config.bev_w, spatial_shapes=config.spatial_shapes
        )
        self.head = TtBEVFormerHead(params.head, device)
        bev_pos = params.bev_pos
        if self.batch_size > 1:
            bev_pos = ttnn.repeat(bev_pos, ttnn.Shape((self.batch_size, 1, 1)))
        self.bev_pos = bev_pos

    def prepare_frame(self, img_metas, frame=None, capacity=None):
        """See ``TtPerceptionTransformer.prepare_frame``."""
        assert (
            len(img_metas) == self.batch_size
        ), f"{len(img_metas)} samples, the detector is built for {self.batch_size}"
        return self.transformer.prepare_frame(img_metas, frame, capacity)

    def __call__(self, img, frame, prev_bev=None):
        """``img`` the ``bs * num_cams`` normalized images, ``(1, 1, bs * num_cams * H * W, 3)``
        bfloat16 ROW_MAJOR, sample-major; ``frame`` from :meth:`prepare_frame`; ``prev_bev`` the
        previous frame's ``(bs, bev_h * bev_w, C)`` BEV or None. Returns the per-layer class logits
        ``(L, bs, num_query, num_classes)``, boxes ``(L, bs, num_query, code_size)`` and this
        frame's BEV ``(bs, bev_h * bev_w, C)``; the FPN frees the backbone's features, not
        ``img``."""
        levels = self.neck(list(self.backbone(img)))
        bev_embed = self.transformer(levels, self.params.bev_queries, self.bev_pos, frame, prev_bev=prev_bev)
        cls_scores, bbox_preds = self.head(bev_embed)
        return cls_scores, bbox_preds, bev_embed
