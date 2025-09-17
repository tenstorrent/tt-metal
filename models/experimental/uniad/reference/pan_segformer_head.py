# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.experimental.uniad.reference.seg_deformable_transformer import SegDeformableTransformer
from models.experimental.uniad.reference.seg_mask_head import SegMaskHead
from models.experimental.uniad.reference.utils import inverse_sigmoid


def bbox_cxcywh_to_xyxy(bbox):
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def IOU(intputs, targets):
    numerator = (intputs * targets).sum(dim=1)
    denominator = intputs.sum(dim=1) + targets.sum(dim=1) - numerator
    loss = numerator / (denominator + 0.0000000000001)
    return loss.cpu(), numerator.cpu(), denominator.cpu()


class SinePositionalEncoding(nn.Module):
    def __init__(
        self, num_feats, temperature=10000, normalize=False, scale=2 * math.pi, eps=1e-6, offset=0.0, init_cfg=None
    ):
        super(SinePositionalEncoding, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set," "scale should be provided and in float or int type, " f"found {type(scale)}"
            )
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PansegformerHead(nn.Module):
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        canvas_size,
        pc_range,
        num_reg_fcs=2,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        quality_threshold_things=0.25,
        quality_threshold_stuff=0.25,
        overlap_threshold_things=0.4,
        overlap_threshold_stuff=0.2,
        thing_transformer_head=dict(
            type="TransformerHead", d_model=256, nhead=8, num_decoder_layers=6  # mask decoder for things
        ),
        stuff_transformer_head=dict(
            type="TransformerHead", d_model=256, nhead=8, num_decoder_layers=6  # mask decoder for stuff
        ),
        train_cfg=dict(
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="ClassificationCost", weight=1.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ),
            sampler=dict(type="PseudoSampler"),
        ),
        **kwargs,
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.canvas_size = canvas_size
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.quality_threshold_things = 0.1
        self.quality_threshold_stuff = quality_threshold_stuff
        self.overlap_threshold_things = overlap_threshold_things
        self.overlap_threshold_stuff = overlap_threshold_stuff
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        self.num_dec_things = 4
        self.num_dec_stuff = 6
        super(PansegformerHead, self).__init__()

        self.positional_encoding = SinePositionalEncoding(num_feats=128, normalize=True, scale=6.283185307179586)
        self.transformer = SegDeformableTransformer()
        self.as_two_stag = False
        self.embed_dims = 256
        self.cls_out_channels = 3
        self.num_reg_fcs = num_reg_fcs
        self.num_query = 300
        self.num_stuff_classes = 1
        self._init_layers()
        self.num_things_classes = 3
        self.things_mask_head = SegMaskHead(
            cfg=None,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=4,
            dim_feedforward=64,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
            self_attn=False,
        )
        self.stuff_mask_head = SegMaskHead(
            cfg=None,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=64,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
            self_attn=True,
        )

    def _init_layers(self):
        if not self.as_two_stag:
            self.bev_embedding = nn.Embedding(50 * 50, self.embed_dims)
            # self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        fc_cls = nn.Linear(self.embed_dims, self.cls_out_channels)
        fc_cls_stuff = nn.Linear(self.embed_dims, 1)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        self.transformer.decoder.num_layers = 6
        num_pred = (
            (self.transformer.decoder.num_layers + 1) if self.as_two_stage else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        self.stuff_query = nn.Embedding(self.num_stuff_classes, self.embed_dims * 2)
        self.reg_branches2 = _get_clones(reg_branch, self.num_dec_things)  # used in mask decoder
        self.cls_thing_branches = _get_clones(fc_cls, self.num_dec_things)  # used in mask decoder
        self.cls_stuff_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff)  # used in mask deocder

    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = float(-np.log((1 - 0.01) / 0.01))
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_thing_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_stuff_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        for m in self.reg_branches2:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)

        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(self, bev_embed, level_embeds=None):
        _, bs, _ = bev_embed.shape

        mlvl_feats = [torch.reshape(bev_embed, (bs, self.bev_h, self.bev_w, -1)).permute(0, 3, 1, 2)]
        img_masks = mlvl_feats[0].new_zeros((bs, self.bev_h, self.bev_w))

        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            img_masks_batched = img_masks[None]
            target_size = feat.shape[-2:]
            interpolated = F.interpolate(img_masks_batched, size=target_size)
            interpolated = interpolated.to(torch.bool)
            interpolated = interpolated.squeeze(0)
            mlvl_masks.append(interpolated)

            mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        out = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            level_embeds=level_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
        )

        (
            (memory, memory_pos, memory_mask, query_pos),
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord,
        ) = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            level_embeds=level_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
        )

        memory = memory.permute(1, 0, 2)
        query = hs[-1].permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        # we should feed these to mask deocder.
        args_tuple = [memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl]

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)

            outputs_class = self.cls_branches[lvl](hs[lvl])

            tmp = self.reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)

        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "bev_embed": None if self.as_two_stage else bev_embed,
            "outputs_classes": outputs_classes,
            "outputs_coords": outputs_coords,
            "enc_outputs_class": enc_outputs_class if self.as_two_stage else None,
            "enc_outputs_coord": enc_outputs_coord.sigmoid() if self.as_two_stage else None,
            "args_tuple": args_tuple,
            "reference": reference,
        }

        return outs

    def forward_test(
        self,
        pts_feats=None,
        gt_lane_labels=None,
        gt_lane_masks=None,
        img_metas=None,
        rescale=False,
        level_embeds=None,
    ):
        bbox_list = [dict() for i in range(len(img_metas))]

        pred_seg_dict = self(pts_feats, level_embeds=level_embeds)
        results = self.get_bboxes(
            pred_seg_dict["outputs_classes"],
            pred_seg_dict["outputs_coords"],
            pred_seg_dict["enc_outputs_class"],
            pred_seg_dict["enc_outputs_coord"],
            pred_seg_dict["args_tuple"],
            pred_seg_dict["reference"],
            img_metas,
            rescale=rescale,
        )

        with torch.no_grad():
            drivable_pred = results[0]["drivable"]
            drivable_gt = gt_lane_masks[0][0, -1]
            drivable_iou, drivable_intersection, drivable_union = IOU(
                drivable_pred.view(1, -1), drivable_gt.view(1, -1)
            )

            lane_pred = results[0]["lane"]
            lanes_pred = (results[0]["lane"].sum(0) > 0).int()
            lanes_gt = (gt_lane_masks[0][0][:-1].sum(0) > 0).int()
            lanes_iou, lanes_intersection, lanes_union = IOU(lanes_pred.view(1, -1), lanes_gt.view(1, -1))

            divider_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 0].sum(0) > 0).int()
            crossing_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 1].sum(0) > 0).int()
            contour_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 2].sum(0) > 0).int()
            divider_iou, divider_intersection, divider_union = IOU(lane_pred[0].view(1, -1), divider_gt.view(1, -1))
            crossing_iou, crossing_intersection, crossing_union = IOU(lane_pred[1].view(1, -1), crossing_gt.view(1, -1))
            contour_iou, contour_intersection, contour_union = IOU(lane_pred[2].view(1, -1), contour_gt.view(1, -1))

            ret_iou = {
                "drivable_intersection": drivable_intersection,
                "drivable_union": drivable_union,
                "lanes_intersection": lanes_intersection,
                "lanes_union": lanes_union,
                "divider_intersection": divider_intersection,
                "divider_union": divider_union,
                "crossing_intersection": crossing_intersection,
                "crossing_union": crossing_union,
                "contour_intersection": contour_intersection,
                "contour_union": contour_union,
                "drivable_iou": drivable_iou,
                "lanes_iou": lanes_iou,
                "divider_iou": divider_iou,
                "crossing_iou": crossing_iou,
                "contour_iou": contour_iou,
            }
        for result_dict, pts_bbox in zip(bbox_list, results):
            result_dict["pts_bbox"] = pts_bbox
            result_dict["ret_iou"] = ret_iou
            result_dict["args_tuple"] = pred_seg_dict["args_tuple"]
        return bbox_list

    def _get_bboxes_single(self, cls_score, bbox_pred, img_shape, scale_factor, rescale=False):
        assert len(cls_score) == len(bbox_pred)
        max_per_img = 100
        # exclude background
        self.loss_cls = True
        if self.loss_cls:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_things_classes

            bbox_index = indexes // self.num_things_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)
        return bbox_index, det_bboxes, det_labels

    def get_bboxes(
        self,
        all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        args_tuple,
        reference,
        img_metas,
        rescale=False,
    ):
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        seg_list = []
        stuff_score_list = []
        panoptic_list = []
        bbox_list = []
        labels_list = []
        drivable_list = []
        lane_list = []
        lane_score_list = []
        score_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            ori_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            scale_factor = 1

            index, bbox, labels = self._get_bboxes_single(cls_score, bbox_pred, img_shape, scale_factor, rescale)

            i = img_id
            thing_query = query[i : i + 1, index, :]
            thing_query_pos = query_pos[i : i + 1, index, :]
            joint_query = torch.cat([thing_query, self.stuff_query.weight[None, :, : self.embed_dims]], 1)

            stuff_query_pos = self.stuff_query.weight[None, :, self.embed_dims :]

            mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
                memory[i : i + 1],
                memory_mask[i : i + 1],
                None,
                joint_query[:, : -self.num_stuff_classes],
                None,
                None,
                hw_lvl=hw_lvl,
            )
            mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
                memory[i : i + 1],
                memory_mask[i : i + 1],
                None,
                joint_query[:, -self.num_stuff_classes :],
                None,
                stuff_query_pos,
                hw_lvl=hw_lvl,
            )

            attn_map = torch.cat([mask_things, mask_stuff], 1)
            attn_map = attn_map.squeeze(-1)  # BS, NQ, N_head,LEN

            stuff_query = query_inter_stuff[-1]
            scores_stuff = self.cls_stuff_branches[-1](stuff_query).sigmoid().reshape(-1)

            mask_pred = attn_map.reshape(-1, *hw_lvl[0])

            mask_pred = F.interpolate(mask_pred.unsqueeze(0), size=ori_shape[:2], mode="bilinear").squeeze(0)

            masks_all = mask_pred
            score_list.append(masks_all)
            drivable_list.append(masks_all[-1] > 0.5)
            masks_all = masks_all[: -self.num_stuff_classes]
            seg_all = masks_all > 0.5
            sum_seg_all = seg_all.sum((1, 2)).float() + 1

            scores_all = bbox[:, -1]
            bboxes_all = bbox
            labels_all = labels

            ## mask wise merging
            seg_scores = (masks_all * seg_all.float()).sum((1, 2)) / sum_seg_all
            seg_scores = seg_scores**2
            scores_all *= seg_scores

            scores_all, index = torch.sort(scores_all, descending=True)

            masks_all = masks_all[index]
            labels_all = labels_all[index]
            bboxes_all = bboxes_all[index]
            seg_all = seg_all[index]

            bboxes_all[:, -1] = scores_all

            # MDS: select things for instance segmeantion
            things_selected = labels_all < self.num_things_classes
            stuff_selected = labels_all >= self.num_things_classes
            bbox_th = bboxes_all[things_selected][:100]

            labels_th = labels_all[things_selected][:100]
            seg_th = seg_all[things_selected][:100]
            labels_st = labels_all[stuff_selected]
            scores_st = scores_all[stuff_selected]
            masks_st = masks_all[stuff_selected]

            stuff_score_list.append(scores_st)

            results = torch.zeros((2, *mask_pred.shape[-2:]), device=mask_pred.device).to(torch.long)
            id_unique = 1
            lane = torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]), device=mask_pred.device).to(torch.long)
            lane_score = torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]), device=mask_pred.device).to(
                mask_pred.dtype
            )
            for i, scores in enumerate(scores_all):
                # MDS: things and sutff have different threholds may perform a little bit better
                if labels_all[i] < self.num_things_classes and scores < self.quality_threshold_things:
                    continue
                elif labels_all[i] >= self.num_things_classes and scores < self.quality_threshold_stuff:
                    continue
                _mask = masks_all[i] > 0.5
                mask_area = _mask.sum().item()
                intersect = _mask & (results[0] > 0)
                intersect_area = intersect.sum().item()
                if labels_all[i] < self.num_things_classes:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area) > self.overlap_threshold_things:
                        continue
                else:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area) > self.overlap_threshold_stuff:
                        continue
                if intersect_area > 0:
                    _mask = _mask & (results[0] == 0)
                results[0, _mask] = labels_all[i]
                if labels_all[i] < self.num_things_classes:
                    lane[labels_all[i], _mask] = 1
                    lane_score[labels_all[i], _mask] = masks_all[i][_mask]
                    results[1, _mask] = id_unique
                    id_unique += 1

            file_name = img_metas[img_id]["pts_filename"].split("/")[-1].split(".")[0]
            panoptic_list.append((results.permute(1, 2, 0).cpu().numpy(), file_name, ori_shape))

            bbox_list.append(bbox_th)
            labels_list.append(labels_th)
            seg_list.append(seg_th)
            lane_list.append(lane)
            lane_score_list.append(lane_score)
        results = []
        for i in range(len(img_metas)):
            results.append(
                {
                    "bbox": bbox_list[i],
                    "segm": seg_list[i],
                    "labels": labels_list[i],
                    "panoptic": panoptic_list[i],
                    "drivable": drivable_list[i],
                    "score_list": score_list[i],
                    "lane": lane_list[i],
                    "lane_score": lane_score_list[i],
                    "stuff_score_list": stuff_score_list[i],
                }
            )
        return results
