# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from einops import rearrange
from models.experimental.uniad.reference.utils import bivariate_gaussian_activation
from models.experimental.uniad.reference.utils import CollisionNonlinearOptimizer
import copy


class PlanningHeadSingleMode(nn.Module):
    def __init__(
        self,
        bev_h=200,
        bev_w=200,
        embed_dims=256,
        planning_steps=6,
        loss_planning=None,
        loss_collision=None,
        planning_eval=True,
        use_col_optim=True,
        col_optim_args=dict(
            occ_filter_range=5.0,
            sigma=1.0,
            alpha_collision=5.0,
        ),
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
        self.planning_eval = planning_eval

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

        self.use_col_optim = use_col_optim
        self.occ_filter_range = col_optim_args["occ_filter_range"]
        self.sigma = col_optim_args["sigma"]
        self.alpha_collision = col_optim_args["alpha_collision"]

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
        if self.use_col_optim and not self.training:
            # post process, only used when testing
            assert occ_mask is not None
            sdc_traj_all = self.collision_optimization(sdc_traj_all, occ_mask)

        return dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
        )

    def collision_optimization(self, sdc_traj_all, occ_mask):
        pos_xy_t = []
        valid_occupancy_num = 0

        if occ_mask.shape[2] == 1:
            occ_mask = occ_mask.squeeze(2)
        occ_horizon = occ_mask.shape[1]
        assert occ_horizon == 5

        for t in range(self.planning_steps):
            cur_t = min(t + 1, occ_horizon - 1)
            pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)
            pos_xy = pos_xy[:, [1, 0]]
            pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h // 2) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w // 2) * 0.5 + 0.25

            # filter the occupancy in range
            keep_index = (
                torch.sum((sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2]) ** 2, axis=-1) < self.occ_filter_range**2
            )
            pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())
            valid_occupancy_num += torch.sum(keep_index > 0)
        if valid_occupancy_num == 0:
            return sdc_traj_all

        col_optimizer = CollisionNonlinearOptimizer(
            self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t
        )
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())
        sol = col_optimizer.solve()
        sdc_traj_optim = torch.tensor(sol[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype)
        return sdc_traj_optim
