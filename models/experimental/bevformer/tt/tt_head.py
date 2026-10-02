# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's detection head, from UniAD's ``TtBEVFormerTrackHead``
(``models/experimental/uniad/tt/ttnn_head.py``).

Runs the decoder batch-first over the encoder's ``(bs, bev_h * bev_w, embed_dims)`` BEV
features, then each layer's classification branch. Parameters come from
``model_preprocessing_head.create_head_parameters``. Forward runs on device only.

The reg branches run once per layer, inside the decoder, which returns their box codes.
The reference sets a box's centre to ``sigmoid(delta + inverse_sigmoid(points))``, with
``delta`` the code's centre channels and ``points`` the layer's input reference points:
that is the decoder's refinement of its points, so its refined points (float32) replace
the centre channels, scaled to metres.
"""

import torch

import ttnn
from models.experimental.bevformer.config.decoder_config import CODE_H, CODE_WL, CODE_XY, CODE_Z
from models.experimental.bevformer.config.head_config import PC_RANGE
from models.experimental.bevformer.tt.tt_common import SCORE_DTYPE, layer_norm
from models.experimental.bevformer.tt.tt_decoder import GRID_DTYPE, TtDetectionTransformerDecoder

# TtBEVFormerHead rebuilds the box code by concatenating its parts in channel order.
assert (CODE_XY.start, CODE_XY.stop, CODE_WL.stop, CODE_Z.stop) == (0, CODE_WL.start, CODE_Z.start, CODE_H.start)


class TtBEVFormerHead:
    """BEVFormer's detection head over a ``bev_shape`` BEV map, batch-first on device."""

    def __init__(self, params, device, bev_shape, pc_range=PC_RANGE):
        """``params`` comes from ``create_head_parameters``; ``bev_shape`` is ``(bev_h, bev_w)``.
        ``pc_range`` maps the [0, 1] reference points to box centres in metres."""
        self.params = params
        self.decoder = TtDetectionTransformerDecoder(params.decoder, device, bev_shape, batch_first=True)
        pc_min = torch.tensor(pc_range[:3])
        pc_size = torch.tensor(pc_range[3:]) - pc_min

        def upload(values):
            return ttnn.from_torch(values.view(1, 1, 1, 3), dtype=GRID_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)

        self.pc_min = upload(pc_min)
        self.pc_size = upload(pc_size)

    @staticmethod
    def _cls_branch(x, branch):
        for linear, norm in branch.hidden:
            x = ttnn.relu(layer_norm(ttnn.linear(x, linear.weight, bias=linear.bias), norm))
        return ttnn.linear(x, branch.out.weight, bias=branch.out.bias, dtype=SCORE_DTYPE)

    def __call__(self, bev_embed):
        """``bev_embed`` is ``(bs, bev_h * bev_w, embed_dims)``.

        Returns every decoder layer's class logits ``(L, bs, num_query, num_classes)``
        (``SCORE_DTYPE``) and box predictions ``(L, bs, num_query, code_size)`` (``GRID_DTYPE``):
        box codes with cx, cy and cz in metres.
        """
        p = self.params
        bs = bev_embed.shape[0]
        query, query_pos, reference_points = p.query, p.query_pos, p.reference_points
        if bs > 1:
            query, query_pos, reference_points = (
                ttnn.repeat(t, (bs, 1, 1)) for t in (query, query_pos, reference_points)
            )

        hs, refined_points, box_codes = self.decoder(query, bev_embed, query_pos, reference_points, p.reg_branches)
        all_cls_scores = ttnn.stack(
            [self._cls_branch(hs[lvl], branch) for lvl, branch in enumerate(p.cls_branches)], dim=0
        )

        centres = ttnn.add(ttnn.multiply(refined_points, self.pc_size), self.pc_min)
        all_bbox_preds = ttnn.concat(
            [
                centres[..., 0:2],
                box_codes[..., CODE_WL],
                centres[..., 2:3],
                box_codes[..., CODE_H.start :],
            ],
            dim=-1,
        )
        return all_cls_scores, all_bbox_preds
