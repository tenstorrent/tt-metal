import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.experimental.maptr.reference.resnet import ResNet
from models.experimental.maptr.reference.fpn import FPN
from models.experimental.maptr.reference.head import MapTRHead
from models.experimental.maptr.reference.grid_mask import GridMask


class MapTR(nn.Module):
    """MapTR.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(
        self,
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
        modality="vision",
        lidar_encoder=None,
    ):
        super(MapTR, self).__init__()
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.with_img_neck = True

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        _dim_ = 256
        _pos_dim_ = _dim_ // 2
        bev_h_ = 80
        bev_w_ = 40
        self.modality = modality
        self.img_backbone = ResNet(
            depth=18,
            num_stages=4,
            out_indices=(3,),
            frozen_stages=-1,
            norm_eval=False,
            style="pytorch",
        )
        self.img_neck = FPN(
            in_channels=[512],
            out_channels=256,
            start_level=0,
            add_extra_convs="on_output",
            num_outs=1,
            relu_before_extra_convs=True,
        )
        self.pts_bbox_head = MapTRHead(
            bev_h=80,
            bev_w=40,
            num_query=900,
            num_vec=100,
            num_pts_per_vec=20,  # one bbox
            num_pts_per_gt_vec=20,
            dir_interval=1,
            query_embed_type="instance_pts",
            transform_method="minmax",
            gt_shift_pts_pattern="v2",
            num_classes=3,
            in_channels=256,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            code_size=2,
            code_weights=[1.0, 1.0, 1.0, 1.0],
            positional_encoding=dict(
                type="LearnedPositionalEncoding",
                num_feats=_pos_dim_,
                row_num_embed=bev_h_,
                col_num_embed=bev_w_,
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=0.0),
            loss_iou=dict(type="GIoULoss", loss_weight=0.0),
            loss_pts=dict(type="PtsL1Loss", loss_weight=5.0),
            loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.005),
            bbox_coder=dict(
                type="MapTRNMSFreeCoder",
                # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
                pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                max_num=50,
                voxel_size=[0.15, 0.15, 4],
                num_classes=3,
            ),
        )

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated."""
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]["prev_bev_exists"]:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(img_feats, None, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def extract_lidar_feat(self, points):
        feats, coords, sizes = self.voxelize(points)
        # voxel_features = self.lidar_modal_extractor["voxel_encoder"](feats, sizes, coords)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)

        return lidar_feat

    def forward(self, img_metas, img=None, points=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img
        points = [points] if points is None else points
        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        # print("cpu", "\n","img_metas:", img_metas[0], "\n", "img:", img[0], "\n", "points:", points[0], "\n", "prev_frame_info:", self.prev_frame_info['prev_bev'], "\n", "kwargs:", kwargs)

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], points[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )

        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            pts_3d=pts.to("cpu"),
        )

        if attrs is not None:
            result_dict["attrs_3d"] = attrs.cpu()

        return result_dict

    def simple_test_pts(self, x, lidar_feat, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = [self.pred2result(bboxes, scores, labels, pts) for bboxes, scores, labels, pts in bbox_list]
        return outs["bev_embed"], bbox_results

    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        lidar_feat = None
        if self.modality == "fusion":
            lidar_feat = self.extract_lidar_feat(points)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(img_feats, lidar_feat, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list


class MapTR_fp16(MapTR):
    """
    The default version BEVFormer currently can not support FP16.
    We provide this version to resolve this issue.
    """

    # @auto_fp16(apply_to=('img', 'prev_bev', 'points'))
    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        img_depth=None,
        img_mask=None,
        prev_bev=None,
    ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        # import pdb;pdb.set_trace()
        losses = dict()
        losses_pts = self.forward_pts_train(
            img_feats,
            gt_bboxes_3d,
            gt_labels_3d,
            img_metas,
            gt_bboxes_ignore,
            prev_bev=prev_bev,
        )
        losses.update(losses_pts)
        return losses

    def val_step(self, data, optimizer):
        """
        In BEVFormer_fp16, we use this `val_step` function to inference the `prev_pev`.
        This is not the standard function of `val_step`.
        """

        img = data["img"]
        img_metas = data["img_metas"]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        prev_bev = data.get("prev_bev", None)
        prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)
        return prev_bev
