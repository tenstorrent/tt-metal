# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import copy
import ttnn

from models.experimental.MapTR.tt.ttnn_backbone import TtResNet50
from models.experimental.MapTR.tt.ttnn_fpn import TtFPN
from models.experimental.MapTR.tt.ttnn_head import TtMapTRHead
from models.experimental.MapTR.tt.ttnn_transformer import TtMapTRPerceptionTransformer
from models.experimental.MapTR.tt.ttnn_encoder import TtBEVFormerEncoder
from models.experimental.MapTR.tt.ttnn_decoder import TtMapTRDecoder


def pred2result(bboxes, scores, labels, pts, attrs=None):
    result_dict = dict(
        boxes_3d=bboxes.to("cpu"),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu(),
        pts_3d=pts.to("cpu"),
    )

    if attrs is not None:
        result_dict["attrs_3d"] = attrs.cpu()

    return result_dict


class TtMapTR:
    def __init__(
        self,
        device,
        params,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=True,
        pts_backbone=None,
        img_neck=True,
        pts_neck=None,
        pts_bbox_head=True,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        modality="vision",
        bev_h=200,
        bev_w=100,
        pc_range=None,
        num_vec=50,
        num_pts_per_vec=20,
        num_classes=3,
        embed_dims=256,
    ):
        super(TtMapTR, self).__init__()
        if pc_range is None:
            pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.params = params
        self.device = device
        self.modality = modality
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.embed_dims = embed_dims

        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        self.with_img_neck = img_neck is not None and img_neck

        if img_backbone:
            self.img_backbone = TtResNet50(
                params.conv_args["img_backbone"],
                params.img_backbone,
                device,
            )
        else:
            self.img_backbone = None

        if self.with_img_neck:
            img_neck_args = params.conv_args["img_neck"]
            lateral_conv_config = img_neck_args["lateral_convs"][0]
            fpn_conv_config = img_neck_args["fpn_convs"][0]
            self.img_neck = TtFPN(
                lateral_conv_config,
                fpn_conv_config,
                device,
            )
        else:
            self.img_neck = None

        self.transformer = None
        if pts_bbox_head and hasattr(params, "head") and hasattr(params.head, "transformer"):
            transformer_params = params.head.transformer

            encoder = None
            decoder = None

            try:
                encoder_params = transformer_params.encoder
                encoder = TtBEVFormerEncoder(
                    params=encoder_params,
                    device=device,
                    num_layers=1,
                    pc_range=pc_range,
                    embed_dims=embed_dims,
                )
            except (KeyError, AttributeError):
                pass

            try:
                decoder_params = transformer_params.decoder
                decoder = TtMapTRDecoder(
                    num_layers=6,
                    embed_dims=embed_dims,
                    num_heads=8,
                    params=decoder_params,
                    params_branches=params.head.branches,
                    device=device,
                )
            except (KeyError, AttributeError):
                pass

            if encoder is not None:
                self.transformer = TtMapTRPerceptionTransformer(
                    params=transformer_params,
                    device=device,
                    encoder=encoder,
                    decoder=decoder,
                    embed_dims=embed_dims,
                )

        if pts_bbox_head:
            self.pts_bbox_head = TtMapTRHead(
                params=params.head,
                device=device,
                transformer=self.transformer,
                positional_encoding=None,
                embed_dims=embed_dims,
                num_classes=num_classes,
                num_reg_fcs=2,
                code_size=2,
                bev_h=bev_h,
                bev_w=bev_w,
                pc_range=pc_range,
                num_vec=num_vec,
                num_pts_per_vec=num_pts_per_vec,
                num_decoder_layers=6,
                query_embed_type="instance_pts",
                transform_method="minmax",
                bev_encoder_type="BEVFormerEncoder",
                with_box_refine=True,
                as_two_stage=False,
            )
        else:
            self.pts_bbox_head = None

    def extract_img_feat(self, img, img_metas, len_queue=None):
        import logging

        logger = logging.getLogger(__name__)

        B = img.shape[0]
        logger.info(f"[TT] extract_img_feat input shape: {img.shape}")

        if img is not None:
            if img.shape[0] == 1 and len(img.shape) == 5:
                img = ttnn.squeeze(img, 0)
                logger.info(f"[TT] After squeeze 5D: {img.shape}")
            elif len(img.shape) == 4 and img.shape[0] > 1:
                B, N, C, H, W = img.shape
                img = ttnn.reshape(img, (B * N, C, H, W))
                logger.info(f"[TT] After reshape 4D: {img.shape}")

            img = ttnn.permute(img, (0, 2, 3, 1))
            N, H, W, C = img.shape
            batch_size = img.shape[0]
            img = ttnn.reshape(img, (1, 1, N * H * W, C))
            logger.info(f"[TT] Before backbone: {img.shape}")

            img_feats = self.img_backbone(img, batch_size=batch_size)
            logger.info(f"[TT] After backbone: type={type(img_feats)}")

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())

            for i, feat in enumerate(img_feats):
                feat_torch = ttnn.to_torch(feat)
                logger.info(
                    f"[TT] backbone[{i}] shape: {feat_torch.shape}, sample: {feat_torch.flatten()[:3].tolist()}"
                )
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            for i, feat in enumerate(img_feats):
                feat_torch = ttnn.to_torch(feat)
                logger.info(f"[TT] fpn[{i}] shape: {feat_torch.shape}, sample: {feat_torch.flatten()[:3].tolist()}")

        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = ttnn.unsqueeze(img_feat, 0)
            img_feat = ttnn.to_layout(img_feat, layout=ttnn.ROW_MAJOR_LAYOUT)
            img_feat = ttnn.sharded_to_interleaved(img_feat)
            img_feat = ttnn.reshape(img_feat, (6, 12, 20, img_feat.shape[-1]))
            img_feat = ttnn.permute(img_feat, (0, 3, 1, 2))
            BN, C, H, W = img_feat.shape
            if len_queue is not None:
                img_feat = ttnn.reshape(img_feat, (int(B / len_queue), len_queue, int(BN / B), C, H, W))
                img_feats_reshaped.append(img_feat)
            else:
                img_feat = ttnn.reshape(img_feat, (B, int(BN / B), C, H, W))
                img_feats_reshaped.append(img_feat)

        ttnn.deallocate(img_feats[0])
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def __call__(self, return_loss=True, **kwargs):
        return self.forward_test(**kwargs)

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        img = ttnn.unsqueeze(img[0][0], 0)
        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0],
            img=img,
            points=points,
            prev_bev=self.prev_frame_info["prev_bev"],
            **kwargs,
        )

        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev

        return bbox_results

    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        lidar_feat = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats,
            lidar_feat,
            img_metas,
            prev_bev=prev_bev,
            rescale=rescale,
            **kwargs,
        )

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox

        return new_prev_bev, bbox_list

    def simple_test_pts(self, x, lidar_feat, img_metas, prev_bev=None, rescale=False, **kwargs):
        x[0] = ttnn.to_layout(x[0], layout=ttnn.TILE_LAYOUT)
        outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=prev_bev)

        outs["bev_embed"] = ttnn.to_torch(outs["bev_embed"]).float()
        outs["all_cls_scores"] = ttnn.to_torch(outs["all_cls_scores"]).float()
        outs["all_bbox_preds"] = ttnn.to_torch(outs["all_bbox_preds"]).float()
        outs["all_pts_preds"] = ttnn.to_torch(outs["all_pts_preds"]).float()

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = [pred2result(bboxes, scores, labels, pts) for bboxes, scores, labels, pts in bbox_list]

        return outs["bev_embed"], bbox_results
