# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from models.experimental.uniad.reference.modules import (
    BevFeatureSlicer,
    SimpleConv2d,
    Bottleneck,
    MLP,
    UpsamplingAdd,
    CVT_Decoder,
    DetrTransformerDecoder,
)
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class OccHead(nn.Module):
    def __init__(
        self,
        # General
        receptive_field=3,
        n_future=4,
        spatial_extent=(50, 50),
        ignore_index=255,
        # BEV
        grid_conf=None,
        bev_size=(50, 50),
        bev_emb_dim=256,
        bev_proj_dim=64,
        bev_proj_nlayers=1,
        # Query
        query_dim=256,
        query_mlp_layers=3,
        detach_query_pos=True,
        temporal_mlp_layer=2,
        # Transformer
        transformer_decoder=None,
        attn_mask_thresh=0.5,
        # Loss
        sample_ignore_mode="all_valid",
        aux_loss_weight=1.0,
        loss_mask=None,
        loss_dice=None,
        # Cfgs
        init_cfg=None,
        # Eval
        pan_eval=False,
        test_seg_thresh: float = 0.5,
        test_with_track_score=False,
    ):
        assert init_cfg is None, "To prevent abnormal initialization " "behavior, init_cfg is not allowed to be set"
        super(OccHead, self).__init__()
        self.receptive_field = receptive_field  # NOTE: Used by prepare_future_labels in E2EPredTransformer
        self.n_future = n_future
        self.spatial_extent = spatial_extent
        self.ignore_index = ignore_index

        bevformer_bev_conf = {
            "xbound": [-51.2, 51.2, 0.512],
            "ybound": [-51.2, 51.2, 0.512],
            "zbound": [-10.0, 10.0, 20.0],
        }
        self.bev_sampler = BevFeatureSlicer(bevformer_bev_conf, grid_conf)

        self.bev_size = bev_size
        self.bev_proj_dim = bev_proj_dim
        if bev_proj_nlayers == 0:
            self.bev_light_proj = nn.Sequential()
        else:
            self.bev_light_proj = SimpleConv2d(
                in_channels=bev_emb_dim,
                conv_channels=bev_emb_dim,
                out_channels=bev_proj_dim,
                num_conv=bev_proj_nlayers,
            )

        # Downscale bev_feat -> /4
        self.base_downscale = nn.Sequential(
            Bottleneck(in_channels=bev_proj_dim, downsample=True), Bottleneck(in_channels=bev_proj_dim, downsample=True)
        )

        self.transformer_decoder = DetrTransformerDecoder()

        # Future blocks with transformer
        self.n_future_blocks = self.n_future + 1

        # - transformer
        self.attn_mask_thresh = attn_mask_thresh

        # self.num_trans_layers = transformer_decoder.num_layers
        # assert self.num_trans_layers % self.n_future_blocks == 0

        # - temporal-mlps
        # query_out_dim = bev_proj_dim

        temporal_mlp = MLP(query_dim, query_dim, bev_proj_dim, num_layers=temporal_mlp_layer)
        self.temporal_mlps = _get_clones(temporal_mlp, self.n_future_blocks)

        # - downscale-convs
        downscale_conv = Bottleneck(in_channels=bev_proj_dim, downsample=True)
        self.downscale_convs = _get_clones(downscale_conv, self.n_future_blocks)

        # - upsampleAdds
        upsample_add = UpsamplingAdd(in_channels=bev_proj_dim, out_channels=bev_proj_dim)
        self.upsample_adds = _get_clones(upsample_add, self.n_future_blocks)

        # Decoder
        self.dense_decoder = CVT_Decoder(
            dim=bev_proj_dim,
            blocks=[bev_proj_dim, bev_proj_dim],
        )

        # Query
        self.mode_fuser = nn.Sequential(
            nn.Linear(query_dim, bev_proj_dim), nn.LayerNorm(bev_proj_dim), nn.ReLU(inplace=True)
        )
        self.multi_query_fuser = nn.Sequential(
            nn.Linear(query_dim * 3, query_dim * 2),
            nn.LayerNorm(query_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim * 2, bev_proj_dim),
        )

        self.detach_query_pos = detach_query_pos

        self.query_to_occ_feat = MLP(query_dim, query_dim, bev_proj_dim, num_layers=query_mlp_layers)
        self.temporal_mlp_for_mask = copy.deepcopy(self.query_to_occ_feat)

        self.pan_eval = pan_eval
        self.test_seg_thresh = test_seg_thresh

        self.test_with_track_score = test_with_track_score

    def forward(
        self,
        bev_feat,
        outs_dict,
        no_query=False,
        gt_segmentation=None,
        gt_instance=None,
        gt_img_is_valid=None,
    ):
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(
            gt_segmentation, gt_instance, gt_img_is_valid
        )
        out_dict = dict()
        out_dict["seg_gt"] = gt_segmentation[:, : 1 + self.n_future]  # [1, 5, 1, 200, 200]
        out_dict["ins_seg_gt"] = self.get_ins_seg_gt(gt_instance[:, : 1 + self.n_future])  # [1, 5, 200, 200]

        # output all zero results
        out_dict["seg_out"] = torch.zeros_like(out_dict["seg_gt"]).long()  # [1, 5, 1, 200, 200]
        out_dict["ins_seg_out"] = torch.zeros_like(out_dict["ins_seg_gt"]).long()  # [1, 5, 200, 200]
        return out_dict

    def get_ins_seg_gt(self, gt_instance):
        ins_gt_old = gt_instance  # Not consecutive, 0 for bg, otherwise ins_ind(start from 1)
        ins_gt_new = torch.zeros_like(ins_gt_old).to(ins_gt_old)  # Make it consecutive
        ins_inds_unique = torch.unique(ins_gt_old)
        new_id = 1
        for uni_id in ins_inds_unique:
            if uni_id.item() in [0, self.ignore_index]:  # ignore background_id
                continue
            ins_gt_new[ins_gt_old == uni_id] = new_id
            new_id += 1
        return ins_gt_new  # Consecutive

    def get_occ_labels(self, gt_segmentation, gt_instance, gt_img_is_valid):
        if not self.training:
            gt_segmentation = gt_segmentation[0]
            gt_instance = gt_instance[0]
            gt_img_is_valid = gt_img_is_valid[0]

        gt_segmentation = gt_segmentation[:, : self.n_future + 1].long().unsqueeze(2)
        gt_instance = gt_instance[:, : self.n_future + 1].long()
        gt_img_is_valid = gt_img_is_valid[:, : self.receptive_field + self.n_future]
        return gt_segmentation, gt_instance, gt_img_is_valid
