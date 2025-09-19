# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


class TtOccHead:
    def __init__(
        self,
        device,
        ignore_index=255,
        n_future=4,
        receptive_field=3,
    ):
        self.device = device
        self.ignore_index = ignore_index
        self.n_future = n_future
        self.receptive_field = receptive_field

    def __call__(
        self,
        bev_feat,
        outs_dict=None,
        no_query=False,
        gt_segmentation=None,
        gt_instance=None,
        gt_img_is_valid=None,
    ):
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(
            gt_segmentation, gt_instance, gt_img_is_valid
        )

        out_dict = dict()

        out_dict["seg_gt"] = gt_segmentation[:, : 1 + self.n_future]
        seg_gt = ttnn.to_torch(gt_segmentation[:, : 1 + self.n_future])  # [1, 5, 1, 200, 200]
        ins_seg_gt = self.get_ins_seg_gt(gt_instance[:, : 1 + self.n_future])
        out_dict["ins_seg_gt"] = ttnn.from_torch(
            ins_seg_gt, device=self.device, dtype=ttnn.bfloat16
        )  # [1, 5, 200, 200]
        if no_query:
            # output all zero results
            out_dict["seg_out"] = ttnn.from_torch(
                torch.zeros_like(seg_gt).long(), device=self.device, dtype=ttnn.bfloat16
            )  # [1, 5, 1, 200, 200]
            out_dict["ins_seg_out"] = ttnn.from_torch(
                torch.zeros_like(ins_seg_gt).long(), device=self.device, dtype=ttnn.bfloat16
            )  # [1, 5, 200, 200]
        return out_dict

    def get_ins_seg_gt(self, gt_instance):
        ins_gt_old = ttnn.to_torch(gt_instance)  # Not consecutive, 0 for bg, otherwise ins_ind(start from 1)
        ins_gt_new = torch.zeros_like(ins_gt_old).to(ins_gt_old)  # Make it consecutive
        ins_inds_unique = torch.unique(
            ins_gt_old
        )  # TODO Raised issue for this operation - <https://github.com/tenstorrent/tt-metal/issues/26437>
        new_id = 1
        for uni_id in ins_inds_unique:
            if uni_id.item() in [0, self.ignore_index]:  # ignore background_id
                continue
            ins_gt_new[ins_gt_old == uni_id] = new_id
            new_id += 1
        return ins_gt_new  # Consecutive

    def get_occ_labels(self, gt_segmentation, gt_instance, gt_img_is_valid):
        gt_segmentation = gt_segmentation[0]
        gt_instance = gt_instance[0]
        gt_img_is_valid = gt_img_is_valid[0]

        gt_segmentation = gt_segmentation[:, : self.n_future + 1]
        gt_segmentation = ttnn.unsqueeze(gt_segmentation, dim=2)
        gt_instance = gt_instance[:, : self.n_future + 1]
        gt_instance = ttnn.add(gt_instance, 0.0, dtype=ttnn.int32)
        gt_img_is_valid = gt_img_is_valid[:, : self.receptive_field + self.n_future]
        return gt_segmentation, gt_instance, gt_img_is_valid
