# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import numpy as np

# from mmdet.evaluation.functional.bbox_overlaps import bbox_overlaps
from models.experimental.vadv2.demo.utils.tpfp_chamfer import (
    vec_iou,
    convex_iou,
    rbbox_iou,
    polyline_score,
    custom_polyline_score,
)

# from vecmapnet_ops.ops.iou import convex_iou


def tpfp_bbox(det_bboxes, gt_bboxes, gt_bbox_masks, threshold=0.5):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """

    num_dets = len(det_bboxes)
    num_gts = len(gt_bboxes)

    # tp and fp
    tp = np.zeros((num_dets), dtype=np.float32)
    fp = np.zeros((num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    # XXX
    if num_gts == 0:
        fp[...] = 1
        return tp, fp

    if num_dets == 0:
        return tp, fp

    # # distance matrix: n x m
    bbox_p = det_bboxes[:, :-1].reshape(num_dets, -1, 2)
    bbox_g = gt_bboxes.reshape(num_gts, -1, 2)
    bbox_gm = gt_bbox_masks.reshape(num_gts, -1, 2)
    matrix = convex_iou(bbox_p, bbox_g, bbox_gm)

    # for each det, the max iou with all gts
    matrix_max = matrix.max(axis=1)
    # for each det, which gt overlaps most with it
    matrix_argmax = matrix.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])

    gt_covered = np.zeros(num_gts, dtype=bool)

    # tp = 0 and fp = 0 means ignore this detected bbox,
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            matched_gt = matrix_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp


def tpfp_rbbox(det_bboxes, gt_bboxes, gt_bbox_masks, threshold=0.5):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """

    num_dets = len(det_bboxes)
    num_gts = len(gt_bboxes)

    # tp and fp
    tp = np.zeros((num_dets), dtype=np.float32)
    fp = np.zeros((num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    # XXX
    if num_gts == 0:
        fp[...] = 1
        return tp, fp

    if num_dets == 0:
        return tp, fp

    # # distance matrix: n x m
    bbox_p = det_bboxes[:, :-1].reshape(num_dets, -1, 2)
    bbox_g = gt_bboxes.reshape(num_gts, -1, 2)
    bbox_gm = gt_bbox_masks.reshape(num_gts, -1, 2)
    matrix = rbbox_iou(bbox_p, bbox_g, bbox_gm)

    # for each det, the max iou with all gts
    matrix_max = matrix.max(axis=1)
    # for each det, which gt overlaps most with it
    matrix_argmax = matrix.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])

    gt_covered = np.zeros(num_gts, dtype=bool)

    # tp = 0 and fp = 0 means ignore this detected bbox,
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            matched_gt = matrix_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp


def tpfp_det(det_bboxes, gt_bboxes, threshold=0.5):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]

    # tp and fp
    tp = np.zeros((num_dets), dtype=np.float32)
    fp = np.zeros((num_dets), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    # XXX
    if num_gts == 0:
        fp[...] = 1
        return tp, fp

    if num_dets == 0:
        return tp, fp

    # # distance matrix: n x m
    matrix = vec_iou(det_bboxes[:, :-1].reshape(num_dets, -1, 2), gt_bboxes.reshape(num_gts, -1, 2))
    # for each det, the max iou with all gts
    matrix_max = matrix.max(axis=1)
    # for each det, which gt overlaps most with it
    matrix_argmax = matrix.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])

    gt_covered = np.zeros(num_gts, dtype=bool)

    # tp = 0 and fp = 0 means ignore this detected bbox,
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            matched_gt = matrix_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp


def tpfp_gen(gen_lines, gt_lines, threshold=0.5, metric="POR"):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """

    num_gens = gen_lines.shape[0]
    num_gts = gt_lines.shape[0]

    # tp and fp
    tp = np.zeros((num_gens), dtype=np.float32)
    fp = np.zeros((num_gens), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        fp[...] = 1
        return tp, fp

    if num_gens == 0:
        return tp, fp

    gen_scores = gen_lines[:, -1]  # n
    # distance matrix: n x m

    # matrix = custom_polyline_score(
    #         gen_lines[:,:-1].reshape(num_gens,-1,2),
    #         gt_lines.reshape(num_gts,-1,2),linewidth=2.,metric=metric)

    # TODO MAY bug here
    matrix = polyline_score(
        gen_lines[:, :-1].reshape(num_gens, -1, 2), gt_lines.reshape(num_gts, -1, 2), linewidth=2.0, metric=metric
    )
    # for each det, the max iou with all gts
    matrix_max = matrix.max(axis=1)
    # for each det, which gt overlaps most with it
    matrix_argmax = matrix.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-gen_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)

    # tp = 0 and fp = 0 means ignore this detected bbox,
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            matched_gt = matrix_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp


def custom_tpfp_gen(gen_lines, gt_lines, threshold=0.5, metric="chamfer"):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Default: False.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
        each array is (num_scales, m).
    """
    print("det type:", type(gen_lines), "gt type:", type(gt_lines))
    if metric == "chamfer":
        if threshold > 0:
            threshold = -threshold
    # else:
    #     raise NotImplementedError

    # import pdb;pdb.set_trace()
    num_gens = gen_lines.shape[0]
    num_gts = gt_lines.shape[0]

    # tp and fp
    tp = np.zeros((num_gens), dtype=np.float32)
    fp = np.zeros((num_gens), dtype=np.float32)

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        fp[...] = 1
        return tp, fp

    if num_gens == 0:
        return tp, fp

    gen_scores = gen_lines[:, -1]  # n
    # distance matrix: n x m

    matrix = custom_polyline_score(
        gen_lines[:, :-1].reshape(num_gens, -1, 2), gt_lines.reshape(num_gts, -1, 2), linewidth=2.0, metric=metric
    )
    # for each det, the max iou with all gts
    matrix_max = matrix.max(axis=1)
    # for each det, which gt overlaps most with it
    matrix_argmax = matrix.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-gen_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)

    # tp = 0 and fp = 0 means ignore this detected bbox,
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            matched_gt = matrix_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    return tp, fp
