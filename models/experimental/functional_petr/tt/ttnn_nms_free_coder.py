# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.functional_petr.tt.utils import denormalize_bbox
from models.experimental.functional_petr.reference.utils import BaseBBoxCoder


class ttnn_NMSFreeCoder(BaseBBoxCoder):
    def __init__(
        self, pc_range, voxel_size=None, post_center_range=None, max_num=100, score_threshold=None, num_classes=10
    ):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        max_num = self.max_num
        device = cls_scores.device()

        cls_scores = ttnn.to_layout(cls_scores, layout=ttnn.TILE_LAYOUT)
        cls_scores = ttnn.sigmoid(cls_scores)

        temp_cls_scores = ttnn.reshape(cls_scores, (1, 1, 1, -1))
        temp_cls_scores = ttnn.to_torch(temp_cls_scores)

        scores, indexes = temp_cls_scores.topk(max_num)  # issue in ttnn topk

        # print("scores",scores.shape)
        # print("indexes",indexes.shape)

        # print("ttnn indexes",indexes)

        # .topk(max_num)

        labels = indexes % self.num_classes
        bbox_index = indexes // self.num_classes

        scores = ttnn.from_torch(scores, device=device)
        indexes = ttnn.from_torch(indexes, device=device)
        labels = ttnn.from_torch(labels, device=device)

        # print("bbox_preds",bbox_preds)
        # print("bbox_index",bbox_index)

        bbox_preds = ttnn.to_torch(bbox_preds)

        bbox_preds = bbox_preds.squeeze()
        bbox_index = bbox_index.squeeze()
        # print("ttnn bbox_preds",bbox_preds.shape)
        # print("ttnn bbox_index",bbox_index.shape)
        bbox_preds = bbox_preds[bbox_index]

        bbox_preds = ttnn.from_torch(bbox_preds, device=device)

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            final_scores = ttnn.to_torch(final_scores)
            final_preds = ttnn.to_torch(final_preds)
            final_box_preds = ttnn.to_torch(final_box_preds)

            # print("ttnn final_scores",final_box_preds.shape)
            # print("ttnn final_scores",final_scores.shape)
            # print("ttnn final_preds",final_preds.shape)

            final_scores = final_scores.squeeze()
            final_preds = final_preds.squeeze()
            final_box_preds = final_box_preds.squeeze()

            # print("ttnn after final_scores",final_box_preds.shape)
            # print("ttnn after final_scores",final_scores.shape)
            # print("ttnn  after final_preds",final_preds.shape)

            self.post_center_range = torch.tensor(self.post_center_range)

            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]

            # print("ttnn boxes3d",boxes3d.shape)
            # print("ttnn scores",scores.shape)
            # print("ttnn labels",labels.shape)
            predictions_dict = {"bboxes": boxes3d, "scores": scores, "labels": labels}

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only " "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(self, preds_dicts):
        device = preds_dicts["all_cls_scores"].device()
        all_cls_scores = ttnn.to_torch(preds_dicts["all_cls_scores"])[-1]
        all_bbox_preds = ttnn.to_torch(preds_dicts["all_bbox_preds"])[-1]

        all_cls_scores = ttnn.from_torch(all_cls_scores, device=device)
        all_bbox_preds = ttnn.from_torch(all_bbox_preds, device=device)

        # print("ttnn all_cls_scores",all_cls_scores.shape)
        # print("ttnn all_bbox_preds",all_bbox_preds.shape)

        batch_size = all_cls_scores.shape[0]
        predictions_list = []
        for i in range(batch_size):
            # print("all_cls_scores[i]",all_cls_scores[i:i+1].shape)
            # print("all_bbox_preds[i]",all_bbox_preds[i:i+1].shape)
            predictions_list.append(self.decode_single(all_cls_scores[i : i + 1], all_bbox_preds[i : i + 1]))
        return predictions_list
