# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from einops import rearrange
from models.experimental.uniad.reference.utils import bivariate_gaussian_activation
import copy


class PlanningHeadSingleMode(nn.Module):
    def __init__(
        self,
        bev_h=200,
        bev_w=200,
        embed_dims=256,
        planning_steps=6,
        with_adapter=True,
    ):
        super(PlanningHeadSingleMode, self).__init__()

        # Nuscenes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.navi_embed = nn.Embedding(3, embed_dims)
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, planning_steps * 2),
        )

        self.planning_steps = planning_steps

        #### planning head
        fuser_dim = 3
        attn_module_layer = nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims * 2, batch_first=False)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)

        self.mlp_fuser = nn.Sequential(
            nn.Linear(embed_dims * fuser_dim, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

        self.pos_embed = nn.Embedding(1, embed_dims)

        # TODO: reimplement it with down-scaled feature_map
        self.with_adapter = with_adapter
        if with_adapter:
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={}, command=None):
        sdc_traj_query = outs_motion["sdc_traj_query"]
        sdc_track_query = outs_motion["sdc_track_query"]
        bev_pos = outs_motion["bev_pos"]
        occ_mask = outs_occflow["seg_out"]

        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command)
        return outs_planning

    def forward(self, bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command):
        sdc_track_query = sdc_track_query.detach()
        sdc_traj_query = sdc_traj_query[-1]
        P = sdc_traj_query.shape[1]
        sdc_track_query = sdc_track_query[:, None].expand(-1, P, -1)

        navi_embed = self.navi_embed.weight[command]

        navi_embed = navi_embed[None].expand(-1, P, -1)
        plan_query = torch.cat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)

        plan_query = self.mlp_fuser(plan_query).max(1, keepdim=True)[
            0
        ]  # expand, then fuse  # [1, 6, 768] -> [1, 1, 256]
        plan_query = rearrange(plan_query, "b p c -> p b c")

        bev_pos = rearrange(bev_pos, "b c h w -> (h w) b c")

        bev_feat = bev_embed + bev_pos

        if self.with_adapter:
            bev_feat = rearrange(bev_feat, "(h w) b c -> b c h w", h=self.bev_h, w=self.bev_w)
            bev_feat = bev_feat + self.bev_adapter(bev_feat)  # residual connection
            bev_feat = rearrange(bev_feat, "b c h w -> (h w) b c")

        pos_embed = self.pos_embed.weight
        plan_query = plan_query + pos_embed[None]  # [1, 1, 256]

        plan_query = self.attn_module(plan_query, bev_feat)  # [1, 1, 256]

        sdc_traj_all = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))
        sdc_traj_all[..., :2] = torch.cumsum(sdc_traj_all[..., :2], dim=1)
        sdc_traj_all[0] = bivariate_gaussian_activation(sdc_traj_all[0])

        return dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
        )
