# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""RetinaNet postprocessing, split between device and host.

The device selects the ``topk_candidates`` best (anchor, class) pairs of every pyramid level and
decodes their boxes, so the host receives a few thousand candidates instead of every anchor's
class logits. The host keeps the candidates above the score threshold and runs NMS, whose output
size depends on the data. Together they compute torchvision's ``RetinaNet.postprocess_detections``,
which thresholds before its top-k: the score is monotonic in the logit, so the top-k of the
candidates above the threshold are the top-k candidates that are above it.
"""

import math

import torch
import ttnn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import boxes as box_ops

NUM_CLASSES = 91
NUM_ANCHORS = 9
NUM_LEVELS = 5
# torchvision's BoxCoder clamp, which keeps exp() of the width and height deltas finite.
BBOX_XFORM_CLIP = math.log(1000.0 / 16)


def level_sizes(image_size):
    """(H, W) of P3-P7: stride 8 for P3, and every later level halves, rounding up."""
    height, width = (math.ceil(size / 8) for size in image_size)
    sizes = []
    for _ in range(NUM_LEVELS):
        sizes.append((height, width))
        height, width = math.ceil(height / 2), math.ceil(width / 2)
    return sizes


def anchor_generator():
    """torchvision's RetinaNet anchor generator: three sizes per octave from 32 px on P3, and
    three aspect ratios."""
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)


def retinanet_anchors(image_size):
    """torchvision's RetinaNet anchors for ``image_size``, one [H*W*A, 4] xyxy tensor per level."""
    sizes = level_sizes(image_size)
    images = ImageList(torch.zeros(1, 3, *image_size), [tuple(image_size)])
    features = [torch.zeros(1, 1, height, width) for height, width in sizes]
    anchors = anchor_generator()(images, features)[0]
    return list(anchors.split([height * width * NUM_ANCHORS for height, width in sizes]))


class TTDetectionDecoder:
    """Per level: top-k over the flattened (anchor, class) logits, then the boxes of the selected
    anchors decoded with torchvision's BoxCoder (unit weights) and clipped to the image."""

    def __init__(self, device, image_size=(512, 512), topk_candidates=1000):
        self.device = device
        self.image_size = tuple(image_size)
        anchors_per_level = retinanet_anchors(self.image_size)
        self.anchors_per_level = [len(anchors) for anchors in anchors_per_level]
        self.topk = [min(topk_candidates, count * NUM_CLASSES) for count in self.anchors_per_level]

        # Boxes are decoded with one coordinate per row and one anchor per column, so the gather
        # runs along the last dimension and every row pair shares the same elementwise ops.
        self.anchor_centers = []
        self.anchor_sizes = []
        for anchors in anchors_per_level:
            sizes = anchors[:, 2:] - anchors[:, :2]
            centers = anchors[:, :2] + 0.5 * sizes
            self.anchor_centers.append(self._to_device(centers.T))
            self.anchor_sizes.append(self._to_device(sizes.T))

        height, width = self.image_size
        self.upper_bound = self._to_device(torch.tensor([[width], [height]], dtype=torch.float32))

    def _to_device(self, tensor):
        return ttnn.from_torch(
            tensor.reshape(1, 1, *tensor.shape),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, classification, regression):
        """``classification`` [1, anchors, 91] and ``regression`` [1, anchors, 4], the heads'
        outputs over all levels. Returns per level the boxes [1, 1, 4, k] (rows x1, y1, x2, y2),
        scores [1, 1, 1, k] and labels [1, 1, 1, k], still on device."""
        num_anchors = sum(self.anchors_per_level)
        assert classification.shape[1] == num_anchors and regression.shape[1] == num_anchors, (
            f"The heads' outputs have {classification.shape[1]} and {regression.shape[1]} anchors; "
            f"a {self.image_size} image has {num_anchors}"
        )
        boxes, scores, labels = [], [], []
        start = 0
        for level, count in enumerate(self.anchors_per_level):
            end = start + count
            logits = ttnn.slice(classification, [0, start, 0], [1, end, NUM_CLASSES])
            deltas = ttnn.slice(regression, [0, start, 0], [1, end, 4])
            level_boxes, level_scores, level_labels = self._decode_level(level, logits, deltas, count)
            boxes.append(level_boxes)
            scores.append(level_scores)
            labels.append(level_labels)
            start = end
        return boxes, scores, labels

    def _decode_level(self, level, logits, deltas, count):
        k = self.topk[level]
        logits = ttnn.reshape(logits, (1, 1, 1, count * NUM_CLASSES))
        # Sigmoid is monotonic, so it runs on the k selected logits only.
        values, indices = ttnn.topk(logits, k, dim=-1, largest=True, sorted=True)
        scores = ttnn.sigmoid(ttnn.typecast(values, ttnn.float32))

        indices = ttnn.typecast(indices, ttnn.int32)
        anchor_index = ttnn.div(indices, NUM_CLASSES, rounding_mode="floor")
        labels = ttnn.subtract(indices, ttnn.multiply(anchor_index, NUM_CLASSES))

        gather_index = ttnn.repeat(ttnn.typecast(anchor_index, ttnn.uint32), (1, 1, 2, 1))
        deltas = ttnn.typecast(deltas, ttnn.float32)
        deltas = ttnn.permute(ttnn.reshape(deltas, (1, 1, count, 4)), (0, 1, 3, 2))
        delta_centers = ttnn.gather(ttnn.slice(deltas, [0, 0, 0, 0], [1, 1, 2, count]), -1, gather_index)
        delta_sizes = ttnn.gather(ttnn.slice(deltas, [0, 0, 2, 0], [1, 1, 4, count]), -1, gather_index)
        anchor_centers = ttnn.gather(self.anchor_centers[level], -1, gather_index)
        anchor_sizes = ttnn.gather(self.anchor_sizes[level], -1, gather_index)

        centers = ttnn.add(ttnn.multiply(delta_centers, anchor_sizes), anchor_centers)
        sizes = ttnn.multiply(ttnn.exp(ttnn.clamp(delta_sizes, max=BBOX_XFORM_CLIP)), anchor_sizes)
        half_sizes = ttnn.multiply(sizes, 0.5)
        top_left = self._clip(ttnn.subtract(centers, half_sizes))
        bottom_right = self._clip(ttnn.add(centers, half_sizes))
        boxes = ttnn.concat([top_left, bottom_right], dim=2)
        return boxes, scores, labels

    def _clip(self, corners):
        return ttnn.clamp(ttnn.minimum(corners, self.upper_bound), min=0.0)


def select_detections(boxes, scores, labels, score_thresh=0.05, nms_thresh=0.5, detections_per_img=300):
    """Host side of the postprocessing for the device candidates of ``TTDetectionDecoder``: the
    score threshold and class-wise NMS. Returns torchvision's detection dict for the image."""
    boxes = torch.cat([ttnn.to_torch(level).reshape(4, -1).T for level in boxes]).float()
    scores = torch.cat([ttnn.to_torch(level).reshape(-1) for level in scores]).float()
    labels = torch.cat([ttnn.to_torch(level).reshape(-1) for level in labels]).long()

    keep = scores > score_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)[:detections_per_img]
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
