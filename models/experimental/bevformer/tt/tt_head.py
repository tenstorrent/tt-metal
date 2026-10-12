# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's detection head, from UniAD's ``TtBEVFormerTrackHead``
(``models/experimental/uniad/tt/ttnn_head.py``).

Runs the decoder batch-first over the encoder's ``(bs, bev_h * bev_w, embed_dims)`` BEV
features, then each layer's classification branch. Parameters come from
``model_preprocessing.create_head_parameters``. Forward runs on device only.

The reg branches run once per layer, inside the decoder, which returns their box codes.
The reference sets a box's center to ``sigmoid(delta + inverse_sigmoid(points))``, with
``delta`` the code's center channels and ``points`` the layer's input reference points:
that is the decoder's refinement of its points, so its refined points (float32) replace
the center channels, scaled to metres.
"""

import torch

import ttnn
from models.experimental.bevformer.model_config import (
    CODE_COS,
    CODE_H,
    CODE_SIZE,
    CODE_WL,
    CODE_XY,
    CODE_Z,
    GRID_DTYPE,
    SCORE_DTYPE,
)
from models.experimental.bevformer.tt.tt_common import layer_norm, signpost
from models.experimental.bevformer.tt.tt_decoder import TtDetectionTransformerDecoder

# The dtypes of the head's outputs. Class logits: in bfloat16 many of the num_query * num_classes
# scores tie, and the coder's top-k order departs from the reference's. Box predictions: the
# decoder's, as their centers are the refined reference points.
SCORE_DTYPE = ttnn.float32
BOX_DTYPE = GRID_DTYPE

# __call__ turns the decoder's centers-first box codes into the box code layout by one concat:
# cx cy | w l | cz | h onward, which needs this order on both sides.
_CHANNELS = [*range(CODE_SIZE)]
if not (
    _CHANNELS[CODE_XY] == [0, 1]
    and _CHANNELS[CODE_WL] == [2, 3]
    and _CHANNELS[CODE_Z] == [4]
    and _CHANNELS[CODE_H.start :] == CENTERS_FIRST[5:]
    and CENTERS_FIRST[3:5] == _CHANNELS[CODE_WL]
):
    raise ValueError("the head's box code concat expects xy | wl | z | h onward, and centers-first codes")


class TtBEVFormerHead:
    """BEVFormer's detection head, batch-first on device."""

    def __init__(self, params, device):
        """``params`` comes from ``create_head_parameters``, with the reference head's BEV shape
        and the ``pc_range`` its [0, 1] reference points map to in metres."""
        self.params = params
        self.decoder = TtDetectionTransformerDecoder(params.decoder, device, params.bev_shape, batch_first=True)
        pc_min = torch.tensor(params.pc_range[:3])
        pc_size = torch.tensor(params.pc_range[3:]) - pc_min

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
        ``model_config.py``'s box code layout with cx, cy and cz in metres.
        """
        signpost(header="TTNN Head Forward Start")
        p = self.params
        bs = bev_embed.shape[0]
        query, query_pos, reference_points = p.query, p.query_pos, p.reference_points
        if bs > 1:
            query, query_pos, reference_points = (
                ttnn.repeat(t, (bs, 1, 1)) for t in (query, query_pos, reference_points)
            )

        signpost(header="Head Decoder Start")
        hs, refined_points, box_codes = self.decoder(query, bev_embed, query_pos, reference_points, p.reg_branches)
        signpost(header="Head Decoder Complete")
        signpost(header="Head ClsBranches Start")
        all_cls_scores = ttnn.stack(
            [self._cls_branch(hs[lvl], branch) for lvl, branch in enumerate(p.cls_branches)], dim=0
        )
        signpost(header="Head ClsBranches Complete")
        signpost(header="Head BoxPreds Start")

        centers = ttnn.add(ttnn.multiply(refined_points, self.pc_size), self.pc_min)
        # The refined points and the codes' first three channels are (x, y, z).
        all_bbox_preds = ttnn.concat(
            [centers[..., 0:2], box_codes[..., 3:5], centers[..., 2:3], box_codes[..., 5:]], dim=-1
        )
        signpost(header="Head BoxPreds Complete")
        signpost(header="TTNN Head Forward End")
        return all_cls_scores, all_bbox_preds
