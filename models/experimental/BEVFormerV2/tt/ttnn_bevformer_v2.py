# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.BEVFormerV2.tt.ttnn_backbone import TtResNet50
from models.experimental.BEVFormerV2.tt.ttnn_fpn import TtFPN
from models.experimental.BEVFormerV2.tt.ttnn_bevformer_head import TtBEVFormerHead
from models.experimental.BEVFormerV2.reference.nms_free_coder import NMSFreeCoder


def bbox3d2result(bboxes, scores, labels):
    result_dict = dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
    return result_dict


class TtBevFormerV2:
    """TTNN implementation of BevFormerV2"""

    def __init__(
        self,
        device,
        params,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
    ):
        super(TtBevFormerV2, self).__init__()
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.params = params
        self.device = device

        self.video_test_mode = video_test_mode
        self.with_img_neck = True
        self.img_backbone = TtResNet50(params.conv_args["img_backbone"], params.img_backbone, device)
        self.img_neck = TtFPN(params.conv_args["img_neck"], params.img_neck, device)

        if pts_bbox_head is not None:
            bbox_coder_cfg = pts_bbox_head.get("bbox_coder") if isinstance(pts_bbox_head, dict) else None
            if bbox_coder_cfg is None:
                bbox_coder_obj = NMSFreeCoder(
                    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                    max_num=300,
                    voxel_size=[0.512, 0.512, 8],
                    num_classes=10,
                )
            else:
                bbox_coder_obj = NMSFreeCoder(**bbox_coder_cfg)
        else:
            bbox_coder_obj = None

        self.pts_bbox_head = TtBEVFormerHead(
            params=params,
            device=device,
            bbox_coder=bbox_coder_obj,
            with_box_refine=pts_bbox_head.get("with_box_refine", True)
            if isinstance(pts_bbox_head, dict) and pts_bbox_head
            else True,
            as_two_stage=pts_bbox_head.get("as_two_stage", False)
            if isinstance(pts_bbox_head, dict) and pts_bbox_head
            else False,
            transformer=True,
            num_cls_fcs=pts_bbox_head.get("num_cls_fcs", 2) if isinstance(pts_bbox_head, dict) and pts_bbox_head else 2,
            code_weights=pts_bbox_head.get("code_weights")
            if isinstance(pts_bbox_head, dict) and pts_bbox_head
            else None,
            bev_h=pts_bbox_head.get("bev_h", 100) if isinstance(pts_bbox_head, dict) and pts_bbox_head else 100,
            bev_w=pts_bbox_head.get("bev_w", 100) if isinstance(pts_bbox_head, dict) and pts_bbox_head else 100,
            num_query=pts_bbox_head.get("num_query", 900) if isinstance(pts_bbox_head, dict) and pts_bbox_head else 900,
            num_classes=pts_bbox_head.get("num_classes", 10)
            if isinstance(pts_bbox_head, dict) and pts_bbox_head
            else 10,
            embed_dims=pts_bbox_head.get("embed_dims", 256)
            if isinstance(pts_bbox_head, dict) and pts_bbox_head
            else 256,
            num_reg_fcs=pts_bbox_head.get("num_reg_fcs", 2) if isinstance(pts_bbox_head, dict) and pts_bbox_head else 2,
            encoder_num_layers=pts_bbox_head.get("encoder_num_layers", 6)
            if isinstance(pts_bbox_head, dict) and pts_bbox_head
            else 6,
            decoder_num_layers=pts_bbox_head.get("decoder_num_layers", 6)
            if isinstance(pts_bbox_head, dict) and pts_bbox_head
            else 6,
        )

    def extract_img_feat(self, img, img_metas, len_queue=None):
        if img is not None:
            if len(img.shape) == 5:
                B, N, C, H, W = img.shape
                img = ttnn.reshape(img, (B * N, C, H, W))
            elif len(img.shape) == 4:
                BN, C, H, W = img.shape
                B = 1
                N = BN
            else:
                raise ValueError(f"Unexpected img shape: {img.shape}")

            input_height, input_width = H, W

            img = ttnn.permute(img, (0, 2, 3, 1))
            batch_size, height, width, channels = img.shape
            img = ttnn.reshape(img, (1, 1, batch_size * height * width, channels))

            img_feats = self.img_backbone(
                img, batch_size=batch_size, input_height=input_height, input_width=input_width
            )

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats, batch_size=batch_size)

        img_feats_reshaped = []
        for i, img_feat in enumerate(img_feats):
            if img_feat.layout != ttnn.ROW_MAJOR_LAYOUT:
                img_feat = ttnn.to_layout(img_feat, layout=ttnn.ROW_MAJOR_LAYOUT)
            if img_feat.is_sharded():
                img_feat = ttnn.sharded_to_interleaved(img_feat)

            if len(img_feat.shape) == 4:
                BN, H_feat, W_feat, C_feat = img_feat.shape
                img_feat_permuted = ttnn.permute(img_feat, (0, 3, 1, 2))
                ttnn.deallocate(img_feat)
            else:
                raise ValueError(f"Unexpected img_feat shape after FPN: {img_feat.shape}")

            img_feat_reshaped = ttnn.reshape(img_feat_permuted, (B, N, C_feat, H_feat, W_feat))
            ttnn.deallocate(img_feat_permuted)
            img_feats_reshaped.append(img_feat_reshaped)

        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def __call__(self, return_loss=True, **kwargs):
        return self.forward_test(**kwargs)

    def forward_test(
        self,
        img_metas,
        img=None,
        prev_bev=None,
        **kwargs,
    ):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        while isinstance(img_metas, list) and len(img_metas) > 0 and isinstance(img_metas[0], list):
            img_metas = img_metas[0]

        if isinstance(img, list) and len(img) > 0:
            img = img[0]

        if "can_bus" not in img_metas[0]:
            img_metas[0]["can_bus"] = [0.0] * 18

        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas,
            img=img,
            prev_bev=None,
            **kwargs,
        )

        return bbox_results

    def simple_test(
        self,
        img_metas,
        img=None,
        prev_bev=None,
        points=None,
        rescale=False,
        **kwargs,
    ):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_list = self.simple_test_pts(
            img_feats,
            img_metas,
            prev_bev,
            rescale=rescale,
        )

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        prev_bev=None,
        rescale=False,
    ):
        x[0] = ttnn.to_layout(x[0], layout=ttnn.TILE_LAYOUT)
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bev_embed = outs["bev_embed"]
        if isinstance(bev_embed, ttnn.Tensor):
            bev_embed = ttnn.to_torch(bev_embed).float()
        else:
            bev_embed = bev_embed.float()
        outs["bev_embed"] = bev_embed

        outs["all_cls_scores"] = outs["all_cls_scores"].float()
        outs["all_bbox_preds"] = outs["all_bbox_preds"].float()

        import os

        save_path = "models/experimental/BEVFormerV2/tt/dumps"
        os.makedirs(save_path, exist_ok=True)
        keys_to_save = ["bev_embed", "all_cls_scores", "all_bbox_preds"]
        for key in keys_to_save:
            if key in outs:
                tensor = outs[key]
                torch.save(tensor, os.path.join(save_path, f"{key}.pt"))

        decoded_bbox_list = self.pts_bbox_head.bbox_coder.decode(outs)
        bbox_list = [
            dict(
                pts_bbox=dict(
                    boxes_3d=decoded_bbox_list[i]["bboxes"],
                    scores_3d=decoded_bbox_list[i]["scores"],
                    labels_3d=decoded_bbox_list[i]["labels"],
                )
            )
            for i in range(len(decoded_bbox_list))
        ]

        return outs["bev_embed"], bbox_list
