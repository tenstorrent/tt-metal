# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

import torch
from models.experimental.uniad.tt.ttnn_utils import bivariate_gaussian_activation_plan_head

from models.experimental.uniad.tt.ttnn_transformer_decoder_layer import TtTransformerDecoderLayer


class TtConv2d:
    def __init__(
        self,
        device,
        parameters,
        conv_pt,
        *,
        has_bias=True,
        act_block_h=None,
        reshard=False,
        deallocate=False,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_type="HS",
        dtype=ttnn.bfloat16,
        slice_type=None,
        num_slices=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> None:
        self.weights = parameters.conv.weight

        self.conv_pt = conv_pt
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = parameters.conv.bias

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.stride = conv_pt.stride
        self.padding = conv_pt.padding
        self.in_channels = conv_pt.in_channels
        self.out_channels = conv_pt.out_channels
        self.dilation = conv_pt.dilation
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.dtype = dtype
        self.slice_type = slice_type
        self.num_slices = num_slices
        self.device = device
        self.memory_config = memory_config

        if shard_type == "WS":
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        elif shard_type == "HS":
            self.shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif shard_type == "BS":
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        else:
            self.shard_layout = None

        self.deallocate = deallocate
        self.activation = activation

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape if self.has_bias else ''} {self.kernel_size}"

    def __call__(self, input_tensor, inp_h, inp_w):
        if self.slice_type is not None and self.num_slices is not None:
            slice_config = ttnn.Conv2dSliceConfig(
                slice_type=self.slice_type,
                num_slices=self.num_slices,
            )
        else:
            slice_config = None
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.dtype,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate,
            activation=self.activation,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,  # PCC drop for few cases
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [output_tensor, [out_h, out_w]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias if self.has_bias else None,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=1,
            input_height=inp_h,
            input_width=inp_w,
            conv_config=conv_config,
            compute_config=compute_config,
            slice_config=slice_config,
            return_output_dim=True,
            memory_config=self.memory_config,
        )
        return output_tensor, out_h, out_w


class TtPlanningHeadSingleMode:
    def __init__(
        self,
        device,
        parameters,
        conv_pt,
        bev_h=200,
        bev_w=200,
        embed_dims=256,
        planning_steps=6,
        with_adapter=True,
    ):
        self.device = device
        self.params = parameters
        self.eps = 1e-05

        # Nuscenes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.navi_embed = parameters.navi_embed.weight

        self.reg_branch = [
            ttnn.linear,
            ttnn.relu,
            ttnn.linear,
        ]

        self.planning_steps = planning_steps
        self.training = False

        # planning head
        self.attn_module = [
            TtTransformerDecoderLayer(
                parameters=parameters.attn_module.layers[0],
                device=device,
                d_model=embed_dims,
                nhead=8,
                dim_feedforward=embed_dims * 2,
                batch_first=False,
            ),
            TtTransformerDecoderLayer(
                parameters=parameters.attn_module.layers[1],
                device=device,
                d_model=embed_dims,
                nhead=8,
                dim_feedforward=embed_dims * 2,
                batch_first=False,
            ),
            TtTransformerDecoderLayer(
                parameters=parameters.attn_module.layers[2],
                device=device,
                d_model=embed_dims,
                nhead=8,
                dim_feedforward=embed_dims * 2,
                batch_first=False,
            ),
        ]

        self.mlp_fuser = [ttnn.linear, ttnn.layer_norm, ttnn.relu]
        self.pos_embed = parameters.pos_embed.weight

        # TODO: reimplement it with down-scaled feature_map
        self.with_adapter = with_adapter
        if with_adapter:
            self.bev_adapter = [
                TtConv2d(device, parameters=parameters.bev_adapter[0][0], conv_pt=conv_pt.bev_adapter[0][0]),
                TtConv2d(
                    device, parameters=parameters.bev_adapter[0][2], conv_pt=conv_pt.bev_adapter[0][2], activation=None
                ),
                TtConv2d(device, parameters=parameters.bev_adapter[1][0], conv_pt=conv_pt.bev_adapter[1][0]),
                TtConv2d(
                    device, parameters=parameters.bev_adapter[1][2], conv_pt=conv_pt.bev_adapter[1][2], activation=None
                ),
                TtConv2d(device, parameters=parameters.bev_adapter[2][0], conv_pt=conv_pt.bev_adapter[2][0]),
                TtConv2d(
                    device, parameters=parameters.bev_adapter[2][2], conv_pt=conv_pt.bev_adapter[2][2], activation=None
                ),
            ]

    def __call__(self, bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command=None):
        sdc_traj_query = sdc_traj_query[-1]
        P = sdc_traj_query.shape[1]
        sdc_track_query = ttnn.unsqueeze(sdc_track_query, 1)
        sdc_track_query = ttnn.expand(sdc_track_query, (-1, P, -1))

        navi_embed = self.navi_embed[:1, :]  # 1.0 if FP32 else 0.9999

        navi_embed = ttnn.unsqueeze(navi_embed, 0)
        navi_embed = ttnn.expand(navi_embed, (-1, P, -1))
        navi_embed = ttnn.to_layout(navi_embed, layout=ttnn.TILE_LAYOUT)
        plan_query = ttnn.concat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)

        # mlp_fuser
        plan_query = self.mlp_fuser[0](plan_query, self.params.mlp_fuser[0].weight, bias=self.params.mlp_fuser[0].bias)
        plan_query = self.mlp_fuser[1](
            plan_query, weight=self.params.mlp_fuser[1].weight, bias=self.params.mlp_fuser[1].bias, epsilon=self.eps
        )
        plan_query = self.mlp_fuser[2](plan_query)

        plan_query = ttnn.max(plan_query, dim=1)  # expand, then fuse  # [1, 6, 768] -> [1, 1, 256]
        plan_query = ttnn.unsqueeze(plan_query, 0)

        plan_query = ttnn.permute(plan_query, (1, 0, 2))  # rearrange(plan_query, 'b p c -> p b c')
        b, c, h, w = bev_pos.shape
        bev_pos = ttnn.reshape(bev_pos, (b, c, h * w))
        bev_pos = ttnn.permute(bev_pos, (2, 0, 1))

        bev_feat = bev_embed + bev_pos

        ##### Plugin adapter #####
        if self.with_adapter:
            bev_feat = ttnn.reshape(
                bev_feat, (self.bev_h, self.bev_w, bev_feat.shape[1], bev_feat.shape[-1])
            )  # '(h w) b c -> b c h w'
            bev_feat = ttnn.permute(bev_feat, (2, 0, 1, 3))  # b, h, w, c for ttnn conv input

            b, h, w, c = bev_feat.shape
            bev_feat = ttnn.reshape(bev_feat, (1, 1, b * h * w, c))
            bev_feat_copy = bev_feat

            bev_feat, out_h, out_w = self.bev_adapter[0](bev_feat, h, w)
            bev_feat, out_h, out_w = self.bev_adapter[1](bev_feat, out_h, out_w)
            bev_feat, out_h, out_w = self.bev_adapter[2](bev_feat, out_h, out_w)
            bev_feat, out_h, out_w = self.bev_adapter[3](bev_feat, out_h, out_w)
            bev_feat, out_h, out_w = self.bev_adapter[4](bev_feat, out_h, out_w)
            bev_feat, out_h, out_w = self.bev_adapter[5](bev_feat, out_h, out_w)

            bev_feat = bev_feat + bev_feat_copy  # residual connection # b, h, w, c
            bev_feat = ttnn.reshape(bev_feat, (bev_feat.shape[0], out_h, out_w, bev_feat.shape[-1]))

            bev_feat = ttnn.permute(bev_feat, (1, 2, 0, 3))
            bev_feat = ttnn.reshape(bev_feat, (out_h * out_w, bev_feat.shape[2], bev_feat.shape[-1]))

        pos_embed = ttnn.unsqueeze(self.pos_embed, 0)
        plan_query = plan_query + pos_embed  # [1, 1, 256]

        plan_query = self.attn_module[0](plan_query, bev_feat)
        plan_query = self.attn_module[1](plan_query, bev_feat)
        plan_query = self.attn_module[2](plan_query, bev_feat)

        sdc_traj_all = self.reg_branch[0](
            plan_query, self.params.reg_branch[0].weight, bias=self.params.reg_branch[0].bias
        )
        sdc_traj_all = self.reg_branch[1](sdc_traj_all)  # relu
        sdc_traj_all = self.reg_branch[2](
            sdc_traj_all, self.params.reg_branch[2].weight, bias=self.params.reg_branch[2].bias
        )

        sdc_traj_all = ttnn.reshape(sdc_traj_all, (-1, self.planning_steps, 2))
        sdc_traj_all = ttnn.cumsum(sdc_traj_all, dim=1)  # No need to slice sice last dim is 2

        sdc_traj_all = bivariate_gaussian_activation_plan_head(sdc_traj_all[0])
        sdc_traj_all = ttnn.unsqueeze(sdc_traj_all, 0)

        sdc_traj_all = ttnn.to_torch(sdc_traj_all, dtype=torch.float32)
        occ_mask = ttnn.to_torch(occ_mask, dtype=torch.float32)

        return dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
        )

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={}, command=None):
        sdc_traj_query = outs_motion["sdc_traj_query"]
        sdc_track_query = outs_motion["sdc_track_query"]
        bev_pos = outs_motion["bev_pos"]
        occ_mask = outs_occflow["seg_out"]

        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, command)
        return outs_planning
