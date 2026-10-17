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

import ttnn
from models.experimental.bevformer.model_config import BOX_PRED_PIECES, SCORE_DTYPE
from models.experimental.bevformer.tt.tt_common import layer_norm, signpost
from models.experimental.bevformer.tt.tt_decoder import TTDetectionTransformerDecoder


class TTBEVFormerHead:
    """BEVFormer's detection head, batch-first on device."""

    def __init__(self, params, device):
        """``params`` comes from ``create_head_parameters``, with the reference head's BEV shape
        and the ``pc_range`` corner and size its [0, 1] reference points map to in metres."""
        self.params = params
        self.decoder = TTDetectionTransformerDecoder(params.decoder, device, params.bev_shape, batch_first=True)

    @staticmethod
    def _cls_branch(x, branch):
        for linear, norm in branch.hidden:
            x = ttnn.relu(layer_norm(ttnn.linear(x, linear.weight, bias=linear.bias), norm))
        return ttnn.linear(x, branch.out.weight, bias=branch.out.bias, dtype=SCORE_DTYPE)

    def __call__(self, bev_embed):
        """``bev_embed`` is ``(bs, bev_h * bev_w, embed_dims)``.

        Returns every decoder layer's class logits ``(L, bs, num_query, num_classes)``
        (``SCORE_DTYPE``) and box predictions ``(L, bs, num_query, code_size)`` (``BOX_DTYPE``):
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

        sources = {"centers": ttnn.add(ttnn.multiply(refined_points, p.pc_size), p.pc_min), "codes": box_codes}
        all_bbox_preds = ttnn.concat([sources[source][..., channels] for source, channels in BOX_PRED_PIECES], dim=-1)
        signpost(header="Head BoxPreds Complete")
        signpost(header="TTNN Head Forward End")
        return all_cls_scores, all_bbox_preds
