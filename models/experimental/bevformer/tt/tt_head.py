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
the center channels, scaled to metres. The boxes keep the decoder's ``CENTERS_FIRST`` channel
order, whose center channels come first, where the refined centers fill them without slicing.
"""

import ttnn
from models.experimental.bevformer.model_config import CENTERS_FIRST_CENTERS, CODE_SIZE, GRID_DTYPE, SCORE_DTYPE
from models.experimental.bevformer.tt.tt_common import l1_linear, layer_norm, signpost
from models.experimental.bevformer.tt.tt_decoder import TTDetectionTransformerDecoder


class TTBEVFormerHead:
    """BEVFormer's detection head, batch-first on device."""

    def __init__(self, params, device, all_layers=True):
        """``params`` comes from ``create_head_parameters``, with the reference head's BEV shape
        and the ``pc_range`` corner and size its [0, 1] reference points map to in metres.

        ``all_layers`` returns every decoder layer's logits and boxes, as the reference does for
        its per-layer training losses; without it only the last layer's, the ones the coder reads.
        The decoder runs every layer either way, as each refines the previous one's points."""
        self.params = params
        self.all_layers = all_layers
        # The box predictions' center channels, which the refined centers fill.
        self._center_mask = ttnn.to_device(
            ttnn.Tensor(
                [float(c in range(CENTERS_FIRST_CENTERS.stop)) for c in range(CODE_SIZE)],
                [1, 1, 1, CODE_SIZE],
                GRID_DTYPE,
                ttnn.TILE_LAYOUT,
            ),
            device,
        )
        self.decoder = TTDetectionTransformerDecoder(params.decoder, device, params.bev_shape, batch_first=True)

    @staticmethod
    def _cls_branch(x, branch):
        for linear, norm in branch.hidden:
            x = ttnn.relu(layer_norm(l1_linear(x, linear), norm, memory_config=ttnn.L1_MEMORY_CONFIG))
        return l1_linear(x, branch.out, dtype=SCORE_DTYPE)

    def __call__(self, bev_embed):
        """``bev_embed`` is ``(bs, bev_h * bev_w, embed_dims)``.

        Returns every decoder layer's class logits ``(L, bs, num_query, num_classes)``
        (``SCORE_DTYPE``) and box predictions ``(L, bs, num_query, code_size)`` (``BOX_DTYPE``):
        the box code in ``model_config.py``'s ``CENTERS_FIRST`` order with cx, cy and cz in metres,
        which ``CODE_ORDER`` takes to the reference's. Without ``all_layers``, L is 1, the last layer.
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
        layers = range(len(p.cls_branches)) if self.all_layers else [len(p.cls_branches) - 1]
        if not self.all_layers:
            refined_points, box_codes = refined_points[layers[0] :], box_codes[layers[0] :]
        signpost(header="Head ClsBranches Start")
        all_cls_scores = ttnn.stack([self._cls_branch(hs[lvl], p.cls_branches[lvl]) for lvl in layers], dim=0)
        signpost(header="Head ClsBranches End")
        signpost(header="Head BoxPreds Start")

        centers = ttnn.add(ttnn.multiply(refined_points, p.pc_size), p.pc_min)
        centers = ttnn.pad(centers, [(0, 0)] * 3 + [(0, CODE_SIZE - centers.shape[-1])], 0.0)
        all_bbox_preds = ttnn.where(self._center_mask, centers, box_codes)
        signpost(header="Head BoxPreds End")
        signpost(header="TTNN Head Forward End")
        # The caller may hold the outputs into the next frame, whose backbone needs the L1.
        return tuple(ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (all_cls_scores, all_bbox_preds))
