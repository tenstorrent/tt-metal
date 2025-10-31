# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# from ..chamfer_dist import ChamferDistance
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from scipy.spatial import distance
import similaritymeasures

# def chamfer_distance(pred_bbox, gt_bbox):

#     cd_dist_func = ChamferDistance.vec_cd_dist(
#         pred, pred_mask, tgt, tgt_mask)()


def vec_iou(pred_lines, gt_lines):
    """
    each line with 1 meter width
    pred_lines: num_preds, npts, 2
    gt_lines: num_gts, npts, 2
    """

    num_preds = pred_lines.shape[0]
    num_gts = gt_lines.shape[0]

    pred_lines_shapely = [
        LineString(i).buffer(1.0, cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round) for i in pred_lines
    ]
    gt_lines_shapely = [
        LineString(i).buffer(1.0, cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round) for i in gt_lines
    ]

    # construct tree
    tree = STRtree(gt_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(gt_lines_shapely))

    iou_matrix = np.zeros((num_preds, num_gts))

    for i, pline in enumerate(pred_lines_shapely):
        for o in tree.query(pline):
            if o.intersects(pline):
                gt_id = index_by_id[id(o)]

                inter = o.intersection(pline).area
                union = o.union(pline).area
                iou_matrix[i, gt_id] = inter / union

    return iou_matrix


def convex_iou(pred_lines, gt_lines, gt_mask):
    """
    each line with 1 meter width
    pred_lines: num_preds, List [npts, 2]
    gt_lines: num_gts, npts, 2
    gt_mask: num_gts, npts, 2
    """

    num_preds = len(pred_lines)
    num_gts = len(gt_lines)

    pred_lines_shapely = [Polygon(i).convex_hull for i in pred_lines]
    gt_lines_shapely = [Polygon(i[m].reshape(-1, 2)).convex_hull for i, m in zip(gt_lines, gt_mask)]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))

    iou_matrix = np.zeros((num_preds, num_gts))

    for i, pline in enumerate(gt_lines_shapely):
        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]

                inter = o.intersection(pline).area
                union = o.union(pline).area
                iou_matrix[pred_id, i] = inter / union

    return iou_matrix


def rbbox_iou(pred_lines, gt_lines, gt_mask):
    """
    each line with 1 meter width
    pred_lines: num_preds, List [npts, 2]
    gt_lines: num_gts, npts, 2
    gt_mask: num_gts, npts, 2
    """

    num_preds = len(pred_lines)
    num_gts = len(gt_lines)

    pred_lines_shapely = [Polygon(i).minimum_rotated_rectangle for i in pred_lines]
    gt_lines_shapely = [Polygon(i[m].reshape(-1, 2)) for i, m in zip(gt_lines, gt_mask)]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))

    iou_matrix = np.zeros((num_preds, num_gts))

    for i, pline in enumerate(gt_lines_shapely):
        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]

                inter = o.intersection(pline).area
                union = o.union(pline).area
                iou_matrix[pred_id, i] = inter / union

    return iou_matrix


def polyline_score(pred_lines, gt_lines, linewidth=1.0, metric="POR"):
    """
    each line with 1 meter width
    pred_lines: num_preds, List [npts, 2]
    gt_lines: num_gts, npts, 2
    gt_mask: num_gts, npts, 2
    """
    positive_threshold = 1.0
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)
    line_length = pred_lines.shape[1]

    # gt_lines = gt_lines + np.array((1.,1.))

    pred_lines_shapely = [
        LineString(i).buffer(linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for i in pred_lines
    ]
    gt_lines_shapely = [
        LineString(i).buffer(linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for i in gt_lines
    ]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))

    if metric == "POR":
        iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float64)
    elif metric == "frechet":
        iou_matrix = np.full((num_preds, num_gts), -100.0)
    elif metric == "chamfer":
        iou_matrix = np.full((num_preds, num_gts), -100.0)
    elif metric == "chamfer_v2":
        iou_matrix = np.full((num_preds, num_gts), -100.0)

    for i, pline in enumerate(gt_lines_shapely):
        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]

                if metric == "POR":
                    dist_mat = distance.cdist(pred_lines[pred_id], gt_lines[i], "euclidean")

                    valid_ab = (dist_mat.min(-1) < positive_threshold).sum()
                    valid_ba = (dist_mat.min(-2) < positive_threshold).sum()

                    iou_matrix[pred_id, i] = min(valid_ba, valid_ab) / line_length
                    # iou_matrix[pred_id, i] = ((valid_ba+valid_ab)/2) / line_length
                    # assert iou_matrix[pred_id, i] <= 1. and iou_matrix[pred_id, i] >= 0.
                elif metric == "frechet":
                    fdistance_1 = -similaritymeasures.frechet_dist(pred_lines[pred_id], gt_lines[i])
                    fdistance_2 = -similaritymeasures.frechet_dist(pred_lines[pred_id][::-1], gt_lines[i])
                    fdistance = max(fdistance_1, fdistance_2)
                    iou_matrix[pred_id, i] = fdistance

                elif metric == "chamfer":
                    dist_mat = distance.cdist(pred_lines[pred_id], gt_lines[i], "euclidean")

                    valid_ab = dist_mat.min(-1).sum()
                    valid_ba = dist_mat.min(-2).sum()

                    iou_matrix[pred_id, i] = -(valid_ba + valid_ab) / (2 * line_length)
                    # if iou_matrix[pred_id, i] == 0:
                    #     import ipdb; ipdb.set_trace()
                elif metric == "chamfer_v2":
                    dist_mat = distance.cdist(pred_lines[pred_id], gt_lines[i], "euclidean")

                    valid_ab = dist_mat.min(-1).sum()
                    valid_ba = dist_mat.min(-2).sum()

                    iou_matrix[pred_id, i] = (
                        -(valid_ba / pred_lines[pred_id].shape[0] + valid_ab / gt_lines[i].shape[0]) / 2
                    )
                    # if iou_matrix[pred_id, i] == 0:
                    #     import ipdb; ipdb.set_trace()

    # if True:
    #     import matplotlib.pyplot as plt
    #     print('pred num', num_preds)
    #     print('gt num', num_gts)
    #     for i in range(num_preds):
    #         plt.plot(pred_lines[i][:,0],pred_lines[i][:,1],'-',color='red',alpha=0.5)
    #     for i in range(num_gts):
    #         plt.plot(gt_lines[i][:,0],gt_lines[i][:,1],'-',color='blue',alpha=0.5)
    #     plt.savefig('test.png')
    #     plt.close()
    return iou_matrix


def custom_polyline_score(pred_lines, gt_lines, linewidth=1.0, metric="chamfer"):
    """
    each line with 1 meter width
    pred_lines: num_preds, List [npts, 2]
    gt_lines: num_gts, npts, 2
    gt_mask: num_gts, npts, 2
    """
    if metric == "iou":
        linewidth = 1.0
    positive_threshold = 1.0
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)
    line_length = pred_lines.shape[1]

    # gt_lines = gt_lines + np.array((1.,1.))

    pred_lines_shapely = [
        LineString(i).buffer(linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for i in pred_lines
    ]
    gt_lines_shapely = [
        LineString(i).buffer(linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre) for i in gt_lines
    ]

    # construct tree
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(pred_lines_shapely))

    if metric == "chamfer":
        iou_matrix = np.full((num_preds, num_gts), -100.0)
    elif metric == "iou":
        iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float64)
    else:
        raise NotImplementedError

    for i, pline in enumerate(gt_lines_shapely):
        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]

                if metric == "chamfer":
                    dist_mat = distance.cdist(pred_lines[pred_id], gt_lines[i], "euclidean")
                    # import pdb;pdb.set_trace()
                    valid_ab = dist_mat.min(-1).mean()
                    valid_ba = dist_mat.min(-2).mean()

                    iou_matrix[pred_id, i] = -(valid_ba + valid_ab) / 2
                elif metric == "iou":
                    inter = o.intersection(pline).area
                    union = o.union(pline).area
                    iou_matrix[pred_id, i] = inter / union

    return iou_matrix


if __name__ == "__main__":
    import torch

    line1 = torch.tensor([[1, 5], [3, 5], [5, 5]])

    line0 = torch.tensor([[3, 6], [4, 8], [5, 6]])

    line2 = torch.tensor([[1, 4], [3, 4], [5, 4]])

    line3 = torch.tensor([[4, 4], [3, 3], [5, 3]])

    gt = torch.stack((line2, line3), dim=0).type(torch.float32)
    pred = torch.stack((line0, line1), dim=0).type(torch.float32)

    # import ipdb; ipdb.set_trace()

    # with mmcv.Timer():
    #     gt = upsampler(gt, pts=10)
    #     pred = upsampler(pred, pts=10)

    import matplotlib.pyplot as plt
    from shapely.geometry import LineString
    from descartes import PolygonPatch

    iou_matrix = vec_iou(pred, gt)
    print(iou_matrix)
    # import pdb;pdb.set_trace()
    score_matrix = custom_polyline_score(pred, gt, linewidth=1.0, metric="chamfer")
    print(score_matrix)
    fig, ax = plt.subplots()
    for i in gt:
        i = i.numpy()
        plt.plot(i[:, 0], i[:, 1], "o", color="red")
        plt.plot(i[:, 0], i[:, 1], "-", color="red")

        dilated = LineString(i).buffer(1, cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round)
        patch1 = PolygonPatch(dilated, fc="red", ec="red", alpha=0.5, zorder=-1)
        ax.add_patch(patch1)

    for i in pred:
        i = i.numpy()
        plt.plot(i[:, 0], i[:, 1], "o", color="blue")
        plt.plot(i[:, 0], i[:, 1], "-", color="blue")

        dilated = LineString(i).buffer(1, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
        patch1 = PolygonPatch(dilated, fc="blue", ec="blue", alpha=0.5, zorder=-1)
        ax.add_patch(patch1)

    ax.axis("equal")

    plt.savefig("test3.png")
