# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of detection layer for SSD test phase."""

import ttnn
import torch
from typing import Dict, List

from models.common.utility_functions import tt_to_torch_tensor


class TtDetect:
    """TTNN implementation of detection layer for test phase in SSD."""

    def __init__(self, num_classes: int, top_k: int, conf_thresh: float, nms_thresh: float, device=None):
        """Initialize detection layer.

        Args:
            num_classes: Number of object classes including background
            top_k: Keep top K detections per class
            conf_thresh: Confidence threshold for detections
            nms_thresh: Non-maximum suppression IoU threshold
            device: Device to place ops on
        """
        self.num_classes = num_classes
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.device = device

    def _decode_boxes(self, loc_data: ttnn.Tensor, priors: ttnn.Tensor) -> torch.Tensor:
        """Decode predicted loc/bbox to actual box coordinates using prior boxes.

        Center form: (cx, cy, w, h)

        Args:
            loc_data: Predicted box offsets [batch_size, num_priors, 4]
            priors: Prior box coordinates [num_priors, 4]

        Returns:
            Decoded box coordinates [batch_size, num_priors, 4] as torch.Tensor
        """
        # Convert TTNN tensors to torch for easier manipulation
        # TTNN tensors don't support item assignment, so we convert early
        loc_torch = tt_to_torch_tensor(loc_data)
        priors_torch = tt_to_torch_tensor(priors)

        # Get variances for decoding
        variances = [0.1, 0.2]

        # Ensure priors are broadcastable with loc_data
        # priors: [num_priors, 4] -> [1, num_priors, 4] to match loc_data: [batch_size, num_priors, 4]
        if priors_torch.dim() == 2:
            priors_torch = priors_torch.unsqueeze(0)  # [1, num_priors, 4]

        # Decode center coordinates (cx, cy)
        # boxes[..., :2] = priors[..., :2] + loc_data[..., :2] * variances[0] * priors[..., 2:]
        center_pred = priors_torch[..., :2] + loc_torch[..., :2] * variances[0] * priors_torch[..., 2:]

        # Decode width and height (w, h)
        # boxes[..., 2:] = priors[..., 2:] * exp(loc_data[..., 2:] * variances[1])
        size_pred = priors_torch[..., 2:] * torch.exp(loc_torch[..., 2:] * variances[1])

        # Combine center and size predictions
        boxes = torch.cat([center_pred, size_pred], dim=-1)  # [batch_size, num_priors, 4]

        # Convert to corner form (x1, y1, x2, y2)
        # First compute x1, y1 = center - size/2
        x1y1 = boxes[..., :2] - boxes[..., 2:] / 2
        # Then compute x2, y2 = x1, y1 + size (using the updated x1, y1)
        x2y2 = x1y1 + boxes[..., 2:]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size, num_priors, 4]

        return boxes

    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Apply non-maximum suppression.

        Args:
            boxes: Box coordinates [N, 4]
            scores: Box confidence scores [N]

        Returns:
            Indices of boxes to keep
        """
        # Convert to corner form if not already
        if boxes.size(1) == 4:
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

        area = (x2 - x1) * (y2 - y1)
        _, idx = scores.sort(0, descending=True)

        keep = []
        while idx.size(0) > 0:
            i = idx[0]
            keep.append(i)

            if idx.size(0) == 1:
                break

            # Compute IoU of remaining boxes with kept box
            xx1 = x1[idx[1:]].clamp(min=x1[i])
            yy1 = y1[idx[1:]].clamp(min=y1[i])
            xx2 = x2[idx[1:]].clamp(max=x2[i])
            yy2 = y2[idx[1:]].clamp(max=y2[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)

            inter = w * h
            ovr = inter / (area[i] + area[idx[1:]] - inter)

            # Keep boxes with IoU below threshold
            idx = idx[1:][ovr <= self.nms_thresh]

        return torch.tensor(keep, dtype=torch.long)

    def __call__(
        self, loc_data: ttnn.Tensor, conf_data: ttnn.Tensor, prior_data: ttnn.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Apply detection layer.

        Args:
            loc_data: Predicted box offsets [batch_size, num_priors, 4]
            conf_data: Class confidence scores [batch_size, num_priors, num_classes]
            prior_data: Prior box coordinates [num_priors, 4]

        Returns:
            List of dictionaries containing detected boxes, scores and labels
            for each image in the batch.
        """
        batch_size = loc_data.padded_shape[0]
        num_priors = loc_data.padded_shape[1]

        # Decode boxes (already returns torch.Tensor)
        boxes = self._decode_boxes(loc_data, prior_data)

        # Convert conf to PyTorch for NMS
        conf = tt_to_torch_tensor(conf_data)

        # Lists to store output detections for batch
        output = []

        # Apply NMS per class for each image
        conf_preds = conf.view(batch_size, num_priors, self.num_classes)

        for i in range(batch_size):
            decoded_boxes = boxes[i]
            conf_scores = conf_preds[i]

            # Lists to store detections for this image
            box_list = []
            score_list = []
            label_list = []

            # Skip j = 0 since background class
            for j in range(1, self.num_classes):
                # Get confidence scores for this class
                scores = conf_scores[:, j].clone()

                # Filter by confidence threshold
                mask = scores > self.conf_thresh
                if not mask.any():
                    continue

                scores = scores[mask]
                boxes_j = decoded_boxes[mask]

                # Apply NMS
                keep = self._nms(boxes_j, scores)

                # Keep top k results
                if keep.size(0) > self.top_k:
                    keep = keep[: self.top_k]

                box_list.append(boxes_j[keep])
                score_list.append(scores[keep])
                label_list.extend([j] * keep.size(0))

            if len(box_list) > 0:
                output.append(
                    {
                        "boxes": torch.cat(box_list, 0),
                        "scores": torch.cat(score_list, 0),
                        "labels": torch.tensor(label_list),
                    }
                )
            else:
                # No detections above threshold
                output.append(
                    {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}
                )

        return output
