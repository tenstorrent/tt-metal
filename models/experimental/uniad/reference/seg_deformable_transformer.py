# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import math
import torch
import torch.nn as nn

from models.experimental.uniad.reference.detr_transformer_encoder import DetrTransformerEncoder
from models.experimental.uniad.reference.detr_transformer_decoder import DeformableDetrTransformerDecoder


class Transformer(nn.Module):
    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super().__init__()
        self.encoder = DetrTransformerEncoder()
        self.decoder = DeformableDetrTransformerDecoder(
            num_layers=6,
            embed_dim=256,
            num_heads=8,
        )
        self.embed_dims = self.encoder.embed_dims

    def forward(self, x, mask, query_embed, pos_embed):
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, h, w] -> [h*w, bs, c]
        pos_embed = pos_embed.view(bs, c, -1).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, h, w] -> [bs, h*w]
        memory = self.encoder(query=x, key=None, value=None, query_pos=pos_embed, query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)

        bs, c, h, w = x.shape
        out_dec = self.decoder(
            query=target, key=memory, value=memory, key_pos=pos_embed, query_pos=query_embed, key_padding_mask=mask
        )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, h, w)
        return out_dec, memory


class SegDeformableTransformer(Transformer):
    def __init__(self, as_two_stage=False, num_feature_levels=4, two_stage_num_proposals=300, **kwargs):
        super(SegDeformableTransformer, self).__init__(**kwargs)
        self.fp16_enabled = False
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dims = self.encoder.embed_dims
        self.init_layers()

    def init_layers(self):
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self, proposals, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def forward(
        self,
        mlvl_feats,
        mlvl_masks,
        query_embed,
        mlvl_pos_embeds,
        reg_branches=None,
        cls_branches=None,
        level_embeds=None,
        **kwargs,
    ):
        assert self.as_two_stage or query_embed is not None
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2)
            feat = feat.transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)

        spatial_shapes_prod = spatial_shapes.prod(1)
        spatial_shapes_cumsum = spatial_shapes_prod.cumsum(0)
        spatial_shapes_cumsum_excl_last = spatial_shapes_cumsum[:-1]
        zeros = spatial_shapes.new_zeros((1,))
        level_start_index = torch.cat((zeros, spatial_shapes_cumsum_excl_last))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        # query_embed N *(2C)
        query_pos, query = torch.split(query_embed, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs,
        )
        inter_references_out = inter_references

        return (
            (memory, lvl_pos_embed_flatten, mask_flatten, query_pos),
            inter_states,
            init_reference_out,
            inter_references_out,
            None,
            None,
        )
