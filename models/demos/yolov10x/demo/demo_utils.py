# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch

from models.demos.utils.common_demo_utils import Results, non_max_suppression, scale_boxes


def postprocess(preds, img, orig_imgs, batch, names):
    nc = 80
    max_det = 300
    args = {"conf": 0.5, "iou": 0.7, "agnostic_nms": False, "max_det": 300, "classes": None}
    preds = preds.permute(0, 2, 1)
    batch_size, anchors, _ = preds.shape
    boxes, scores = preds.split([4, nc], dim=-1)
    index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
    boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
    scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
    scores, index = scores.flatten(1).topk(min(max_det, anchors))
    i = torch.arange(batch_size)[..., None]
    preds = torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)
    preds = non_max_suppression(
        preds,
        args["conf"],
        args["iou"],
        agnostic=args["agnostic_nms"],
        max_det=args["max_det"],
        classes=args["classes"],
    )

    results = []
    for pred, orig_img, img_path in zip(preds, orig_imgs, batch[0]):
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(Results(orig_img, path=img_path, names=names, boxes=pred))

    return results
