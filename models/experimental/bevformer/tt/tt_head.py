# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's detection head, from UniAD's ``TtBEVFormerTrackHead``
(``models/experimental/uniad/tt/ttnn_head.py``).

Runs the decoder batch-first over the encoder's ``(bs, bev_h * bev_w, embed_dims)`` BEV
features, then the classification branch of every layer, or of the last alone. Parameters come from
``model_preprocessing.create_head_parameters``. Forward runs on device only.

The reg branches run once per layer, inside the decoder, which returns their box codes.
The reference sets a box's center to ``sigmoid(delta + inverse_sigmoid(points))``, with
``delta`` the code's center channels and ``points`` the layer's input reference points:
that is the decoder's refinement of its points, so its refined points (float32) replace
the center channels, scaled to metres. The boxes keep the decoder's ``CENTERS_FIRST`` channel
order, whose center channels come first, where the refined centers fill them without slicing.
"""

import ttnn
from models.experimental.bevformer.model_config import (
    CENTERS_FIRST_CENTERS,
    CODE_ORDER,
    CODE_SIZE,
    GRID_DTYPE,
    SCORE_DTYPE,
)
from models.experimental.bevformer.tt.tt_common import device_constant, l1_linear, layer_norm, signpost, stack_layers
from models.experimental.bevformer.tt.tt_decoder import TTDetectionTransformerDecoder


def box_preds_to_torch(bbox_preds):
    """``TTBEVFormerHead``'s box predictions, a device tensor, on the host in the reference's box
    code order."""
    return ttnn.to_torch(bbox_preds)[..., CODE_ORDER]


class TTBEVFormerHead:
    """BEVFormer's detection head, batch-first on device."""

    def __init__(self, params, device, all_layers=False):
        """``params`` comes from ``create_head_parameters``, with the reference head's BEV shape
        and the ``pc_range`` corner and size its [0, 1] reference points map to in metres.

        ``all_layers`` returns every decoder layer's logits and boxes, as the reference does for
        its per-layer training losses; without it, the default, only the last layer's, the ones
        the coder reads. The decoder runs every layer either way, as each refines the previous
        one's points."""
        self.params = params
        self.all_layers = all_layers
        # The box predictions' center channels, which the refined centers fill.
        self._center_mask = device_constant(
            [float(c in range(CENTERS_FIRST_CENTERS.start, CENTERS_FIRST_CENTERS.stop)) for c in range(CODE_SIZE)],
            [1, 1, 1, CODE_SIZE],
            GRID_DTYPE,
            device,
        )
        self.decoder = TTDetectionTransformerDecoder(
            params.decoder, device, params.bev_shape, batch_first=True, all_layers=all_layers
        )

    @staticmethod
    def _cls_branch(x, branch):
        for linear, norm in branch.hidden:
            x = ttnn.relu(layer_norm(l1_linear(x, linear), norm, memory_config=ttnn.L1_MEMORY_CONFIG))
        return l1_linear(x, branch.out, dtype=SCORE_DTYPE)

    def __call__(self, bev_embed):
        """``bev_embed`` is ``(bs, bev_h * bev_w, embed_dims)``.

        Returns the class logits ``(L, bs, num_query, num_classes)`` (``SCORE_DTYPE``) and box
        predictions ``(L, bs, num_query, code_size)`` (``BOX_DTYPE``) in L1, L the number of decoder
        layers with ``all_layers``, else 1, the last layer. The boxes are the box code in
        ``model_config.py``'s ``CENTERS_FIRST`` order with cx, cy and cz in metres;
        ``box_preds_to_torch`` brings them to the host in the reference's order.
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
        signpost(header="Head Decoder End")
        # The decoder returns the last layer alone without all_layers.
        cls_branches = p.cls_branches if self.all_layers else p.cls_branches[-1:]
        signpost(header="Head ClsBranches Start")
        all_cls_scores = stack_layers([self._cls_branch(hs[i], branch) for i, branch in enumerate(cls_branches)])
        signpost(header="Head ClsBranches End")
        signpost(header="Head BoxPreds Start")

        centers = ttnn.add(ttnn.multiply(refined_points, p.pc_size), p.pc_min)
        centers = ttnn.pad(centers, [(0, 0)] * 3 + [(0, CODE_SIZE - centers.shape[-1])], 0.0)
        all_bbox_preds = ttnn.where(self._center_mask, centers, box_codes, memory_config=ttnn.L1_MEMORY_CONFIG)
        signpost(header="Head BoxPreds End")
        signpost(header="TTNN Head Forward End")
        return all_cls_scores, all_bbox_preds
