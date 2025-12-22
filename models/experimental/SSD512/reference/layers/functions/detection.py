# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torchvision.ops import nms  # optional, if torchvision is available


class Detect(nn.Module):
    def __init__(self, num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh):
        super().__init__()  # <-- important
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError("nms_threshold must be non-negative.")
        self.conf_thresh = conf_thresh
        # self.variance = cfg['SSD{}'.format(size)]['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        loc_data: [batch, num_priors, 4] or [batch, num_priors*4]
        conf_data: [batch, num_priors, num_classes] or [batch * num_priors, num_classes]
        prior_data: [1, num_priors, 4] or [num_priors, 4]
        Returns:
            output: [batch, num_classes, top_k, 5] tensor
        """
        device = loc_data.device
        batch = loc_data.size(0)

        # normalize loc_data shape -> [batch, num_priors, 4]
        if loc_data.dim() == 2:
            num_priors = loc_data.size(1) // 4
            loc = loc_data.view(batch, num_priors, 4)
        else:
            loc = loc_data
            num_priors = loc.size(1)

        # Normalize conf_data -> [batch, num_priors, num_classes]
        if conf_data.dim() == 2:
            conf = conf_data.view(batch, num_priors, self.num_classes)
        else:
            conf = conf_data

        # Note: prior_data available if box decoding is needed in the future

        # Prepare output container
        output = torch.zeros(batch, self.num_classes, self.top_k, 5, device=device)

        # Per-batch, per-class NMS (skip background class 0)
        for i in range(batch):
            for cl in range(1, self.num_classes):
                scores = conf[i, :, cl]  # [num_priors]
                keep_mask = scores > self.conf_thresh
                if not keep_mask.any():  # still returns a tensor; must avoid .item()
                    continue
                boxes = loc[i][keep_mask]  # [k, 4]  (decode if necessary)
                scores_k = scores[keep_mask]  # [k]

                # If boxes are in center form, decode relative to priors here.
                # Example uses torchvision.nms if boxes are [x1,y1,x2,y2]:
                try:
                    keep_idx = nms(boxes, scores_k, self.nms_thresh)
                except Exception:
                    # fallback: no torchvision or different format - keep top scores
                    _, keep_idx = scores_k.topk(min(self.top_k, scores_k.numel()))
                    keep_idx = keep_idx.cpu()

                # cap top_k
                keep_idx = keep_idx[: self.top_k]

                K = self.top_k
                pad_idx = torch.zeros(K, dtype=torch.long, device=device)
                n = keep_idx.size(0)
                pad_idx[:n] = keep_idx[:K]
                scores_padded = scores_k[pad_idx]  # [K]
                boxes_padded = boxes[pad_idx]  # [K,4]
                valid_mask = torch.arange(K, device=device) < n
                # place into output in one go:
                output[i, cl, :, 0] = scores_padded * valid_mask.to(scores_padded.dtype)
                output[i, cl, :, 1:] = boxes_padded

        return output
