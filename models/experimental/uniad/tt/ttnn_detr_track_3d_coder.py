# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.uniad.tt.ttnn_utils import TtBaseBBoxCoder, denormalize_bbox


class TtDETRTrack3DCoder(TtBaseBBoxCoder):
    def __init__(
        self,
        pc_range,
        post_center_range=None,
        max_num=100,
        score_threshold=0.2,
        num_classes=7,
        with_nms=False,
        iou_thres=0.3,
        device=None,
    ):
        self.device = device
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.with_nms = with_nms
        self.nms_iou_thres = iou_thres

        self.post_center_range = ttnn.from_torch(
            torch.tensor(self.post_center_range), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, track_scores, obj_idxes, with_mask=True, img_metas=None):
        max_num = self.max_num
        max_num = min(cls_scores.shape[0], self.max_num)

        cls_scores = ttnn.sigmoid(cls_scores)

        cls_scores = ttnn.to_torch(cls_scores)
        _, indexs = cls_scores.max(dim=-1)  # Low E2E PCC with ttnn argmax
        indexs = ttnn.from_torch(indexs, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)

        labels = ttnn.remainder(indexs, self.num_classes)
        labels = ttnn.to_torch(labels)

        track_scores = ttnn.to_torch(track_scores)
        _, bbox_index = track_scores.topk(max_num)

        obj_idxes = ttnn.to_torch(obj_idxes)
        bbox_preds = ttnn.to_torch(bbox_preds)

        # TODO Raised issue for this operation - <https://github.com/tenstorrent/tt-metal/issues/15553>
        labels = labels[bbox_index]
        bbox_preds = bbox_preds[bbox_index]
        track_scores = track_scores[bbox_index]
        obj_idxes = obj_idxes[bbox_index]

        scores = track_scores

        bbox_preds = ttnn.from_torch(bbox_preds, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range, device=self.device)

        final_scores = track_scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            # thresh_mask = ttnn.gt(final_scores, self.score_threshold) # ttnn.gt does not support float
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
            # self.post_center_range = torch.tensor(self.post_center_range, device=scores.device) # preprocess
            mask1 = ttnn.ge(final_box_preds[..., :3], self.post_center_range[:3])
            mask2 = ttnn.ge(self.post_center_range[3:], final_box_preds[..., :3])

            mask1 = ttnn.logical_and(ttnn.logical_and(mask1[:, 0], mask1[:, 1]), mask1[:, 2])
            mask2 = ttnn.logical_and(ttnn.logical_and(mask2[:, 0], mask2[:, 1]), mask2[:, 2])
            mask = ttnn.logical_and(mask1, mask2)

            mask = ttnn.to_torch(mask).bool()

            final_box_preds = ttnn.to_torch(final_box_preds)  # Indexing

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            track_scores = track_scores[mask]
            obj_idxes = obj_idxes[mask]

            boxes3d = ttnn.from_torch(boxes3d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            scores = ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            labels = ttnn.from_torch(labels, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=self.device)
            track_scores = ttnn.from_torch(
                track_scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            obj_idxes = ttnn.from_torch(obj_idxes, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=self.device)
            mask = ttnn.from_torch(mask, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=self.device)
            bbox_index = ttnn.from_torch(bbox_index, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=self.device)

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

        batch_size = 1
        predictions_list = []
        predictions_list.append(
            self.decode_single(all_cls_scores, all_bbox_preds, track_scores, obj_idxes, with_mask, img_metas)
        )
        return predictions_list
