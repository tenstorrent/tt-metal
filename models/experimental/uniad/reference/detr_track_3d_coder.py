# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.uniad.reference.utils import BaseBBoxCoder, denormalize_bbox


class DETRTrack3DCoder(BaseBBoxCoder):
    def __init__(
        self,
        pc_range,
        post_center_range=None,
        max_num=100,
        score_threshold=0.2,
        num_classes=7,
        with_nms=False,
        iou_thres=0.3,
    ):
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.with_nms = with_nms
        self.nms_iou_thres = iou_thres

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, track_scores, obj_idxes, with_mask=True, img_metas=None):
        max_num = self.max_num
        max_num = min(cls_scores.size(0), self.max_num)

        cls_scores = cls_scores.sigmoid()
        _, indexs = cls_scores.max(dim=-1)
        labels = indexs % self.num_classes

        _, bbox_index = track_scores.topk(max_num)

        labels = labels[bbox_index]
        bbox_preds = bbox_preds[bbox_index]
        track_scores = track_scores[bbox_index]
        obj_idxes = obj_idxes[bbox_index]

        scores = track_scores

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = track_scores
        final_preds = labels

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.with_nms:
            boxes_for_nms = xywhr2xyxyr(img_metas[0]["box_type_3d"](final_box_preds[:, :], 9).bev)
            nms_mask = boxes_for_nms.new_zeros(boxes_for_nms.shape[0]) > 0
            try:
                selected = nms_bev(boxes_for_nms, final_scores, thresh=self.nms_iou_thres)
                nms_mask[selected] = True
            except:
                nms_mask = boxes_for_nms.new_ones(boxes_for_nms.shape[0]) > 0
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask
            if not with_mask:
                mask = torch.ones_like(mask) > 0
            if self.with_nms:
                mask &= nms_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            track_scores = track_scores[mask]
            obj_idxes = obj_idxes[mask]
            predictions_dict = {
                "bboxes": boxes3d,
                "scores": scores,
                "labels": labels,
                "track_scores": track_scores,
                "obj_idxes": obj_idxes,
                "bbox_index": bbox_index,
                "mask": mask,
            }

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only " "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(self, preds_dicts, with_mask=True, img_metas=None):
        all_cls_scores = preds_dicts["cls_scores"]
        all_bbox_preds = preds_dicts["bbox_preds"]
        track_scores = preds_dicts["track_scores"]
        obj_idxes = preds_dicts["obj_idxes"]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        predictions_list.append(
            self.decode_single(all_cls_scores, all_bbox_preds, track_scores, obj_idxes, with_mask, img_metas)
        )

        return predictions_list
