# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Detection head in PyTorch.

This module implements the part of BEVFormer's ``BEVFormerHead`` that runs on top of the
encoder's BEV features: the decoder half of ``PerceptionTransformer.forward`` and the
classification and regression branches. It turns ``(bs, bev_h * bev_w, embed_dims)`` BEV
features into per-layer class logits and box codes, and is the reference the TTNN head in
``tt/tt_head.py`` is checked against.

The head performs:
1. Split ``query_embedding`` into the object queries and their positional embeddings
2. Initial reference points ``sigmoid(reference_points(query_pos))``
3. The detection decoder over the BEV features, refining the points after every layer
4. Per decoder layer, the classification branch on the layer output (class logits) and the
   regression branch (box code), whose cx, cy and cz are refined from the layer's input
   reference points and scaled from [0, 1] to ``pc_range`` metres

Only inference with BEVFormer's settings is kept: box refinement (``with_box_refine``, one
branch per layer), no two-stage proposals, no losses or assigners. The BEV encoder side of
``PerceptionTransformer`` (BEV queries, positional encoding, can bus, previous BEV) is not
part of this module.

Parameter names follow the BEVFormer checkpoint's ``pts_bbox_head``: ``query_embedding``,
``cls_branches`` and ``reg_branches`` load unchanged, and ``transformer.reference_points``
and ``transformer.decoder`` load into ``reference_points`` and ``decoder``. The rest of
``pts_bbox_head`` (``bev_embedding``, ``positional_encoding``, and ``transformer``'s
``encoder``, ``level_embeds``, ``cams_embeds`` and ``can_bus_mlp``) is the encoder side's
and is left out.

Adapted from UniAD's ``BEVFormerTrackHead`` in ``models/experimental/uniad/reference/head.py``
and BEVFormer's head and transformer:
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/bevformer/modules/transformer.py

BEVFormer head configurations:
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_base.py
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py
"""

import torch
import torch.nn as nn

from models.experimental.bevformer.config.decoder_config import CODE_SIZE, REG_XY, REG_Z
from models.experimental.bevformer.config.head_config import NUM_CLASSES, PC_RANGE
from models.experimental.bevformer.reference.decoder import DetectionTransformerDecoder, inverse_sigmoid


def cls_branch(embed_dims, num_classes, num_reg_fcs=2):
    """``(Linear-LayerNorm-ReLU) x num_reg_fcs`` then ``Linear(num_classes)``, as BEVFormerHead builds it."""
    layers = []
    for _ in range(num_reg_fcs):
        layers += [nn.Linear(embed_dims, embed_dims), nn.LayerNorm(embed_dims), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.Linear(embed_dims, num_classes))


def reg_branch(embed_dims, code_size, num_reg_fcs=2):
    """``(Linear-ReLU) x num_reg_fcs`` then ``Linear(code_size)``, as BEVFormerHead builds it."""
    layers = []
    for _ in range(num_reg_fcs):
        layers += [nn.Linear(embed_dims, embed_dims), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(embed_dims, code_size))


class BEVFormerHead(nn.Module):
    """
    BEVFormer's detection head over a ``bev_h x bev_w`` BEV map.

    Args:
        bev_h, bev_w (int): BEV grid the features cover.
        num_query (int): Object queries.
        num_classes (int): Class logits per query.
        embed_dims (int): Channels of the queries and of the BEV features.
        code_size (int): Box code channels, see ``config/decoder_config.py``.
        pc_range (tuple[float]): Box centre range in metres the [0, 1] reference points map to.
        decoder (dict, optional): ``DetectionTransformerDecoder`` arguments; BEVFormer's by default.

    Each decoder layer has its own classification and regression branch; the decoder refines
    its reference points with the same regression branches.
    """

    def __init__(
        self,
        bev_h,
        bev_w,
        num_query=900,
        num_classes=NUM_CLASSES,
        embed_dims=256,
        code_size=CODE_SIZE,
        pc_range=PC_RANGE,
        decoder=None,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.query_embedding = nn.Embedding(num_query, 2 * embed_dims)
        self.reference_points = nn.Linear(embed_dims, 3)
        self.decoder = DetectionTransformerDecoder(embed_dims=embed_dims, **(decoder or {}))
        num_layers = len(self.decoder.layers)
        self.cls_branches = nn.ModuleList(cls_branch(embed_dims, num_classes) for _ in range(num_layers))
        self.reg_branches = nn.ModuleList(reg_branch(embed_dims, code_size) for _ in range(num_layers))

    def forward(self, bev_embed):
        """``bev_embed`` is the encoder output ``(bs, bev_h * bev_w, embed_dims)``.

        Returns every decoder layer's class logits ``(L, bs, num_query, num_classes)`` and box
        codes ``(L, bs, num_query, code_size)``, cx, cy and cz in metres.
        """
        bs = bev_embed.shape[0]
        query_pos, query = torch.split(self.query_embedding.weight, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        init_reference = self.reference_points(query_pos).sigmoid()

        hs, inter_references = self.decoder(
            query.permute(1, 0, 2),
            bev_embed.permute(1, 0, 2),
            query_pos.permute(1, 0, 2),
            init_reference,
            torch.tensor([[self.bev_h, self.bev_w]]),
            reg_branches=self.reg_branches,
        )
        hs = hs.permute(0, 2, 1, 3)

        pc_min = torch.tensor(self.pc_range[:3])
        pc_size = torch.tensor(self.pc_range[3:]) - pc_min
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            reference = inverse_sigmoid(init_reference if lvl == 0 else inter_references[lvl - 1])
            outputs_classes.append(self.cls_branches[lvl](hs[lvl]))

            box = self.reg_branches[lvl](hs[lvl])
            box[..., REG_XY] = (box[..., REG_XY] + reference[..., 0:2]).sigmoid() * pc_size[0:2] + pc_min[0:2]
            box[..., REG_Z] = (box[..., REG_Z] + reference[..., 2:3]).sigmoid() * pc_size[2:3] + pc_min[2:3]
            outputs_coords.append(box)

        return torch.stack(outputs_classes), torch.stack(outputs_coords)
