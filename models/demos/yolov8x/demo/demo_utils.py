# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from models.demos.utils.common_demo_utils import Results, non_max_suppression, scale_boxes


def postprocess(preds, img, orig_imgs, batch, names):
    args = {"conf": 0.25, "iou": 0.7, "agnostic_nms": False, "max_det": 300, "classes": None}

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
