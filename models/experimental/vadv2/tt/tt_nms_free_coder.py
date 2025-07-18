# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from models.experimental.vadv2.tt.tt_utils import tt_denormalize_2d_bbox, tt_denormalize_2d_pts


class TtMapNMSFreeCoder:
    def __init__(
        self, pc_range, voxel_size=None, post_center_range=None, max_num=100, score_threshold=None, num_classes=10
    ):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def decode_single(self, cls_scores, bbox_preds, pts_preds):
        max_num = self.max_num
        cls_scores = ttnn.to_layout(cls_scores, ttnn.TILE_LAYOUT)
        cls_scores = ttnn.sigmoid(cls_scores)
        cls_scores = ttnn.reshape(cls_scores, [cls_scores.shape[0] * cls_scores.shape[1]])
        print(cls_scores.shape, max_num)
        cls_scores = ttnn.to_layout(cls_scores, ttnn.TILE_LAYOUT)
        scores, indexs = ttnn.topk(cls_scores, max_num, dim=0)
        labels = ttnn.remainder(indexs, self.num_classes)
        bbox_index = ttnn.div(indexs, self.num_classes, dtype=ttnn.int64)
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]
        final_box_preds = tt_denormalize_2d_bbox(bbox_preds, self.pc_range)
        final_pts_preds = tt_denormalize_2d_pts(pts_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            self.post_center_range = ttnn.from_torch(self.post_center_range, dtype=ttnn.float32, device=scores.device)
            mask = (final_box_preds[..., :4] >= self.post_center_range[:4]).all(1)
            mask &= (final_box_preds[..., :4] <= self.post_center_range[4:]).all(1)

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            pts = final_pts_preds[mask]
            labels = final_preds[mask]
            predictions_dict = {
                "map_bboxes": boxes3d,
                "map_scores": scores,
                "map_labels": labels,
                "map_pts": pts,
            }

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only " "support post_center_range is not None for now!"
            )
        return predictions_dict

        return
