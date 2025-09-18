# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import torch.nn as nn
import math
from models.experimental.uniad.tt.ttnn_seg_deformable_transformer import TtSegDeformableTransformer
from models.experimental.uniad.tt.ttnn_seg_mask_head import TtSegMaskHead


def IOU(intputs, targets):
    numerator = (intputs * targets).sum(dim=1)
    denominator = intputs.sum(dim=1) + targets.sum(dim=1) - numerator
    loss = numerator / (denominator + 0.0000000000001)
    return loss.cpu(), numerator.cpu(), denominator.cpu()


def inverse_sigmoid(x, eps: float = 1e-5):
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)
    if len(x.shape) == 3:
        x_temp = ttnn.ones(shape=[x.shape[0], x.shape[1], x.shape[2]], layout=ttnn.TILE_LAYOUT, device=x.device())
    else:
        x_temp = ttnn.ones(
            shape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]], layout=ttnn.TILE_LAYOUT, device=x.device()
        )
    x_temp = x_temp - x
    x2 = ttnn.clamp(x_temp, min=eps)
    return ttnn.log(ttnn.div(x1, x2))


def bbox_cxcywh_to_xyxy(bbox, device=None):
    bbox = ttnn.from_torch(bbox, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cx = bbox[..., 0:1]
    cy = bbox[..., 1:2]
    w = bbox[..., 2:3]
    h = bbox[..., 3:4]
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    out = ttnn.concat(bbox_new, dim=-1)
    out = ttnn.to_torch(out)
    return out


class TtSinePositionalEncoding(nn.Module):
    def __init__(
        self,
        num_feats,
        temperature=10000,
        normalize=False,
        scale=2 * math.pi,
        eps=1e-6,
        offset=0.0,
        init_cfg=None,
        device=None,
    ):
        super(TtSinePositionalEncoding, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set," "scale should be provided and in float or int type, " f"found {type(scale)}"
            )
        self.device = device
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        one_tensor = ttnn.ones(mask.shape, layout=ttnn.TILE_LAYOUT, device=self.device)
        not_mask = ttnn.subtract(one_tensor, mask)
        y_embed = ttnn.cumsum(not_mask, dim=1)
        x_embed = ttnn.cumsum(not_mask, dim=2)
        if self.normalize:
            y_embed = y_embed + self.offset
            norm_factor = y_embed[:, -1:, :] + self.eps
            y_embed = ttnn.div(y_embed, norm_factor)
            y_embed = y_embed * self.scale

            x_embed = x_embed + self.offset
            norm_factor = x_embed[:, :, -1:] + self.eps
            x_embed = ttnn.div(x_embed, norm_factor)
            x_embed = x_embed * self.scale

        dim_t = ttnn.arange(0, self.num_feats)
        dim_t = 2 * ttnn.to_torch(dim_t) // 2
        dim_t = ttnn.from_torch(dim_t, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        power = ttnn.div(dim_t, self.num_feats)
        dim_t = ttnn.pow(self.temperature, power)
        x_embed = ttnn.unsqueeze(x_embed, dim=-1)
        y_embed = ttnn.unsqueeze(y_embed, dim=-1)
        pos_x = ttnn.div(x_embed, dim_t)
        pos_y = ttnn.div(y_embed, dim_t)
        B, H, W = mask.shape[0], mask.shape[1], mask.shape[2]

        sin_part = ttnn.sin(pos_x[:, :, :, 0::2])
        cos_part = ttnn.cos(pos_x[:, :, :, 1::2])
        stacked = ttnn.stack((sin_part, cos_part), dim=4)
        pos_x = ttnn.reshape(stacked, (B, H, W, -1))

        sin_part = ttnn.sin(pos_y[:, :, :, 0::2])
        cos_part = ttnn.cos(pos_y[:, :, :, 1::2])
        stacked = ttnn.stack((sin_part, cos_part), dim=4)
        pos_y = ttnn.reshape(stacked, (B, H, W, -1))
        pos = ttnn.concat([pos_y, pos_x], dim=3)
        pos = ttnn.permute(pos, (0, 3, 1, 2))
        return pos


class TtPansegformerHead(nn.Module):
    def __init__(
        self,
        params,
        device,
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
        **kwargs,
    ):
        self.params = params
        self.device = device
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

        self.num_dec_things = 4
        self.num_dec_stuff = 6
        super(TtPansegformerHead, self).__init__()

        self.positional_encoding = TtSinePositionalEncoding(
            num_feats=128, normalize=True, scale=6.283185307179586, device=self.device
        )
        self.transformer = TtSegDeformableTransformer(
            device, params.transformer, params_branches=kwargs["parameters_branches"]
        )
        self.reg_branches = params.reg_branches
        self.as_two_stag = False
        self.embed_dims = 256
        self.cls_out_channels = 3
        self.num_reg_fcs = num_reg_fcs
        self.num_query = 300
        self.num_stuff_classes = 1
        self.num_things_classes = 3
        self.things_mask_head = TtSegMaskHead(
            params.things_mask_head,
            device,
            cfg=None,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=4,
            dim_feedforward=64,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            normalize_before=False,
            return_intermediate_dec=False,
            self_attn=False,
        )
        self.stuff_mask_head = TtSegMaskHead(
            params.stuff_mask_head,
            device,
            cfg=None,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=64,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            normalize_before=False,
            return_intermediate_dec=False,
            self_attn=True,
        )

    def _get_bboxes_single(self, cls_score, bbox_pred, img_shape, scale_factor, rescale=False):
        """ """
        max_per_img = 100
        # exclude background
        self.loss_cls = True
        if self.loss_cls:
            cls_score = cls_score.sigmoid()
            # TODO Raised issue fo - <https://github.com/tenstorrent/tt-metal/issues/26183>
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_things_classes
            bbox_index = indexes // self.num_things_classes
            bbox_pred = bbox_pred[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred, device=self.device)
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
            cls_score = ttnn.to_torch(cls_score)
            bbox_pred = ttnn.to_torch(bbox_pred)
            index, bbox, labels = self._get_bboxes_single(cls_score, bbox_pred, img_shape, scale_factor, rescale)

            i = img_id
            index = index.long()
            query = ttnn.to_torch(query)
            query_pos = ttnn.to_torch(query_pos)

            thing_query = query[i : i + 1, index, :]
            thing_query_pos = query_pos[i : i + 1, index, :]
            thing_query = ttnn.from_torch(thing_query, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            thing_query_pos = ttnn.from_torch(
                thing_query_pos, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )

            query_weight = self.params.stuff_query.weight
            query_weight = ttnn.unsqueeze(query_weight, dim=0)
            query_weight = query_weight[:, :, : self.embed_dims]
            query_weight = ttnn.to_layout(query_weight, ttnn.TILE_LAYOUT)

            joint_query = ttnn.concat([thing_query, query_weight], dim=1)

            stuff_query_pos = self.params.stuff_query.weight
            stuff_query_pos = ttnn.unsqueeze(stuff_query_pos, 0)
            stuff_query_pos = stuff_query_pos[:, :, self.embed_dims :]

            mask_things, mask_inter_things, query_inter_things = self.things_mask_head.forward(
                memory[i : i + 1],
                memory_mask[i : i + 1],
                None,
                joint_query[:, : -self.num_stuff_classes],
                None,
                None,
                hw_lvl=hw_lvl,
            )

            mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head.forward(
                memory[i : i + 1],
                memory_mask[i : i + 1],
                None,
                joint_query[:, -self.num_stuff_classes :],
                None,
                stuff_query_pos,
                hw_lvl=hw_lvl,
            )

            attn_map = ttnn.concat([mask_things, mask_stuff], dim=1)
            attn_map = ttnn.squeeze(attn_map, dim=-1)

            stuff_query = query_inter_stuff[-1]

            raw_scores = ttnn.linear(
                stuff_query, self.params.cls_stuff_branches[-1].weight, bias=self.params.cls_stuff_branches[-1].bias
            )
            scores_sigmoid = ttnn.sigmoid(raw_scores)
            scores_stuff = ttnn.reshape(scores_sigmoid, (-1,))

            mask_pred = ttnn.reshape(attn_map, (-1, *hw_lvl[0]))

            mask_pred = ttnn.unsqueeze(mask_pred, 0)
            target_size = (ori_shape[0], ori_shape[1])
            mask_pred = ttnn.to_layout(mask_pred, ttnn.ROW_MAJOR_LAYOUT)
            mask_pred = ttnn.upsample(mask_pred, scale_factor=1)
            mask_pred = ttnn.squeeze(mask_pred, 0)

            masks_all = mask_pred
            score_list.append(masks_all)
            out = masks_all[-1] > 0.5
            drivable_list.append(ttnn.to_torch(out))
            masks_all = masks_all[: -self.num_stuff_classes]
            seg_all = masks_all > 0.5
            sum_seg_all = ttnn.sum(seg_all, dim=(1, 2))
            sum_seg_all = ttnn.add(sum_seg_all, 1)

            scores_all = bbox[:, -1]
            bboxes_all = bbox
            labels_all = labels

            product = masks_all * seg_all
            summed = ttnn.sum(product, dim=(1, 2))
            seg_scores = ttnn.div(summed, sum_seg_all)

            seg_scores = ttnn.pow(seg_scores, 2)
            scores_all = ttnn.from_torch(scores_all, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

            scores_all = ttnn.mul(scores_all, seg_scores)

            scores_all = ttnn.unsqueeze(scores_all, dim=0)
            scores_all, index = ttnn.sort(scores_all, descending=True)
            scores_all = ttnn.squeeze(scores_all, dim=0)
            index = ttnn.squeeze(index, dim=0)
            index = ttnn.to_torch(index).to(torch.int64)
            masks_all = ttnn.to_torch(masks_all).to(torch.float32)
            labels_all = labels_all.to(torch.float32)
            seg_all = ttnn.to_torch(seg_all).to(torch.float32)
            scores_all = ttnn.to_torch(scores_all)

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
            results = ttnn.zeros(shape=[2, mask_pred.shape[-2], mask_pred.shape[-1]], device=self.device)

            id_unique = 1

            lane = ttnn.zeros(
                shape=[self.num_things_classes, mask_pred.shape[-2], mask_pred.shape[-1]], device=self.device
            )

            lane_score = ttnn.zeros(
                shape=[self.num_things_classes, mask_pred.shape[-2], mask_pred.shape[-1]], device=self.device
            )
            lane = ttnn.to_torch(lane).to(torch.int64)
            lane_score = ttnn.to_torch(lane_score).to(torch.float32)
            for i, scores in enumerate(scores_all):
                # MDS: things and sutff have different threholds may perform a little bit better
                if labels_all[i] < self.num_things_classes and scores < self.quality_threshold_things:
                    continue
                elif labels_all[i] >= self.num_things_classes and scores < self.quality_threshold_stuff:
                    continue
                _mask = masks_all[i] > 0.5
                mask_area = _mask.sum().item()
                intersect = _mask & ttnn.to_torch((results[0] > 0)).bool()
                intersect_area = intersect.sum().item()
                if labels_all[i] < self.num_things_classes:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area) > self.overlap_threshold_things:
                        continue
                else:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area) > self.overlap_threshold_stuff:
                        continue
                if intersect_area > 0:
                    _mask = _mask & ttnn.to_torch((results[0] == 0)).bool()
                results = ttnn.to_torch(results)
                results[0, _mask] = labels_all[i]
                labels_all = labels_all.long()
                if labels_all[i] < self.num_things_classes:
                    lane[labels_all[i], _mask] = 1
                    lane_score[labels_all[i], _mask] = masks_all[i][_mask]
                    results[1, _mask] = id_unique
                    id_unique += 1
                results = ttnn.from_torch(results, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

            file_name = img_metas[img_id]["pts_filename"].split("/")[-1].split(".")[0]
            panoptic_list.append((ttnn.permute(results, (1, 2, 0)), file_name, ori_shape))

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

    def forward(self, bev_embed, level_embeds=None):
        _, bs, _ = bev_embed.shape

        bev_embed = ttnn.reshape(bev_embed, (bs, self.bev_h, self.bev_w, -1))
        bev_embed = ttnn.permute(bev_embed, (0, 3, 1, 2))
        mlvl_feats = [bev_embed]

        img_masks = ttnn.zeros((bs, self.bev_h, self.bev_w), device=self.device)

        hw_lvl = []
        for feat_lvl in mlvl_feats:
            h, w = feat_lvl.shape[-2], feat_lvl.shape[-1]
            hw_lvl.append([h, w])

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            img_masks = ttnn.unsqueeze(img_masks, 0)
            out = ttnn.upsample(img_masks, scale_factor=1)
            out = ttnn.squeeze(out, 0)
            img_masks = ttnn.squeeze(img_masks, 0)
            mlvl_masks.append(out)
        out = self.positional_encoding(mlvl_masks[-1])
        mlvl_positional_encodings.append(out)

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.params.query_embedding.weight

        out = self.transformer.forward(
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
        ) = self.transformer.forward(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            level_embeds=level_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
        )

        memory = ttnn.permute(memory, (1, 0, 2))
        query = ttnn.permute(hs[-1], (1, 0, 2))
        query_pos = ttnn.permute(query_pos, (1, 0, 2))
        memory_pos = ttnn.permute(memory_pos, (1, 0, 2))

        args_tuple = [memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl]

        hs = ttnn.permute(hs, (0, 2, 1, 3))
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)

            outputs_class = ttnn.linear(
                hs[lvl], self.params.cls_branches[lvl].weight, bias=self.params.cls_branches[lvl].bias
            )

            tmp = ttnn.linear(
                hs[lvl], self.params.reg_branches[lvl][0].weight, bias=self.params.reg_branches[lvl][0].bias
            )
            tmp = ttnn.relu(tmp)
            tmp = ttnn.linear(tmp, self.params.reg_branches[lvl][2].weight, bias=self.params.reg_branches[lvl][2].bias)
            tmp = ttnn.relu(tmp)
            tmp = ttnn.linear(tmp, self.params.reg_branches[lvl][4].weight, bias=self.params.reg_branches[lvl][4].bias)

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                tmp = ttnn.to_torch(tmp)
                reference = ttnn.to_torch(reference)
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
                reference = ttnn.from_torch(reference, device=self.device, layout=ttnn.TILE_LAYOUT)
                tmp = ttnn.from_torch(tmp, device=self.device, layout=ttnn.TILE_LAYOUT)

            outputs_coord = ttnn.sigmoid(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = ttnn.stack(outputs_classes, dim=0)

        outputs_coords = ttnn.stack(outputs_coords, dim=0)

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
        self, pts_feats=None, gt_lane_labels=None, gt_lane_masks=None, img_metas=None, rescale=False, level_embeds=None
    ):
        bbox_list = [dict() for i in range(len(img_metas))]

        pred_seg_dict = self.forward(pts_feats, level_embeds=level_embeds)

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
            drivable_gt = ttnn.to_torch(drivable_gt)
            drivable_iou, drivable_intersection, drivable_union = IOU(
                drivable_pred.view(1, -1), drivable_gt.view(1, -1)
            )

            lane_pred = results[0]["lane"]
            lanes_pred = (results[0]["lane"].sum(0) > 0).int()
            lanes_gt = (ttnn.to_torch(gt_lane_masks[0][0][:-1]).sum(0) > 0).int()
            lanes_iou, lanes_intersection, lanes_union = IOU(lanes_pred.view(1, -1), lanes_gt.view(1, -1))

            divider_gt = (ttnn.to_torch(gt_lane_masks[0][0])[ttnn.to_torch(gt_lane_labels[0][0]) == 0].sum(0) > 0).int()
            crossing_gt = (
                ttnn.to_torch(gt_lane_masks[0][0])[ttnn.to_torch(gt_lane_labels[0][0]) == 1].sum(0) > 0
            ).int()
            contour_gt = (ttnn.to_torch(gt_lane_masks[0][0])[ttnn.to_torch(gt_lane_labels[0][0]) == 2].sum(0) > 0).int()
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
