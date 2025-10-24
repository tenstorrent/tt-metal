# Copyright (c) Facebook, Inc. and its affiliates.

""" Helper functions for calculating 2D and 3D bounding box IoU.

Collected and written by Charles R. Qi
Last modified: Apr 2021 by Ishan Misra
"""
import torch
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from models.experimental.detr3d.reference.utils.misc import to_list_1d, to_list_3d

try:
    from models.experimental.detr3d.source.detr3d.utils.box_intersection import box_intersection
except ImportError:
    print("Could not import cythonized box intersection. Consider compiling box_intersection.pyx for faster training.")
    box_intersection = None


def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    """pc: (N,3), box3d: (8,3)"""
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def polygon_clip(subjectPolygon, clipPolygon):
    """Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList


def poly_area(x, y):
    """Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    """Compute area of two convex hull's intersection area.
    p1,p2 are a list of (x,y) tuples of hull vertices.
    return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_vol(corners):
    """corners: (8,3) no assumption on axis direction"""
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def is_clockwise(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) > 0


def box3d_iou(corners1, corners2):
    """Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    """
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def box2d_iou(box1, box2):
    """Compute 2D bounding box IoU.

    Input:
        box1: tuple of (xmin,ymin,xmax,ymax)
        box2: tuple of (xmin,ymin,xmax,ymax)
    Output:
        iou: 2D IoU scalar
    """
    return get_iou(
        {"x1": box1[0], "y1": box1[1], "x2": box1[2], "y2": box1[3]},
        {"x1": box2[0], "y1": box2[1], "x2": box2[2], "y2": box2[3]},
    )


# -----------------------------------------------------------
# Convert from box parameters to
# -----------------------------------------------------------
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def get_3d_box(box_size, heading_angle, center):
    """box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
    output (8,3) array for 3D box cornders
    Similar to utils/compute_orientation_3d
    """
    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def flip_axis_to_camera_np(pc):
    """Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = pc.copy()
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2


def get_3d_box_batch_np(box_size, angle, center):
    input_shape = angle.shape
    R = roty_batch(angle)
    l = np.expand_dims(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = np.expand_dims(box_size[..., 1], -1)
    h = np.expand_dims(box_size[..., 2], -1)
    corners_3d = np.zeros(tuple(list(input_shape) + [8, 3]))
    corners_3d[..., :, 0] = np.concatenate((l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1)
    corners_3d[..., :, 1] = np.concatenate((h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1)
    corners_3d[..., :, 2] = np.concatenate((w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d


def flip_axis_to_camera_tensor(pc):
    """Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = torch.clone(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2


def roty_batch_tensor(t):
    input_shape = t.shape
    output = torch.zeros(tuple(list(input_shape) + [3, 3]), dtype=torch.float32, device=t.device)
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def get_3d_box_batch_tensor(box_size, angle, center):
    assert isinstance(box_size, torch.Tensor)
    assert isinstance(angle, torch.Tensor)
    assert isinstance(center, torch.Tensor)

    reshape_final = False
    if angle.ndim == 2:
        assert box_size.ndim == 3
        assert center.ndim == 3
        bsize = box_size.shape[0]
        nprop = box_size.shape[1]
        box_size = box_size.reshape(-1, box_size.shape[-1])
        angle = angle.reshape(-1)
        center = center.reshape(-1, 3)
        reshape_final = True

    input_shape = angle.shape
    R = roty_batch_tensor(angle)
    l = torch.unsqueeze(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = torch.unsqueeze(box_size[..., 1], -1)
    h = torch.unsqueeze(box_size[..., 2], -1)
    corners_3d = torch.zeros(tuple(list(input_shape) + [8, 3]), device=box_size.device, dtype=torch.float32)
    corners_3d[..., :, 0] = torch.cat((l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1)
    corners_3d[..., :, 1] = torch.cat((h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1)
    corners_3d[..., :, 2] = torch.cat((w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = torch.matmul(corners_3d, R.permute(tlist))
    corners_3d += torch.unsqueeze(center, -2)
    if reshape_final:
        corners_3d = corners_3d.reshape(bsize, nprop, 8, 3)
    return corners_3d


def get_3d_box_batch(box_size, angle, center):
    """box_size: [x1,x2,...,xn,3]
        angle: [x1,x2,...,xn]
        center: [x1,x2,...,xn,3]
    Return:
        [x1,x3,...,xn,8,3]
    """
    input_shape = angle.shape
    R = roty_batch(angle)
    l = np.expand_dims(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = np.expand_dims(box_size[..., 1], -1)
    h = np.expand_dims(box_size[..., 2], -1)
    corners_3d = np.zeros(tuple(list(input_shape) + [8, 3]))
    corners_3d[..., :, 0] = np.concatenate((l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1)
    corners_3d[..., :, 1] = np.concatenate((h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1)
    corners_3d[..., :, 2] = np.concatenate((w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d


####### GIoU related operations. Differentiable #############


def helper_computeIntersection(cp1: torch.Tensor, cp2: torch.Tensor, s: torch.Tensor, e: torch.Tensor):
    dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
    dp = [s[0] - e[0], s[1] - e[1]]
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    # return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
    return torch.stack([(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3])


def helper_inside(cp1: torch.Tensor, cp2: torch.Tensor, p: torch.Tensor):
    ineq = (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])
    return ineq.item()


def polygon_clip_unnest(subjectPolygon: torch.Tensor, clipPolygon: torch.Tensor):
    """Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """
    outputList = [subjectPolygon[x] for x in range(subjectPolygon.shape[0])]
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList.copy()
        outputList.clear()
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if helper_inside(cp1, cp2, e):
                if not helper_inside(cp1, cp2, s):
                    outputList.append(helper_computeIntersection(cp1, cp2, s, e))
                outputList.append(e)
            elif helper_inside(cp1, cp2, s):
                outputList.append(helper_computeIntersection(cp1, cp2, s, e))
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            # return None
            break
    return outputList


def box3d_vol_tensor(corners):
    EPS = 1e-6
    reshape = False
    B, K = corners.shape[0], corners.shape[1]
    if len(corners.shape) == 4:
        # batch x prop x 8 x 3
        reshape = True
        corners = corners.view(-1, 8, 3)
    a = torch.sqrt((corners[:, 0, :] - corners[:, 1, :]).pow(2).sum(dim=1).clamp(min=EPS))
    b = torch.sqrt((corners[:, 1, :] - corners[:, 2, :]).pow(2).sum(dim=1).clamp(min=EPS))
    c = torch.sqrt((corners[:, 0, :] - corners[:, 4, :]).pow(2).sum(dim=1).clamp(min=EPS))
    vols = a * b * c
    if reshape:
        vols = vols.view(B, K)
    return vols


def enclosing_box3d_vol(corners1, corners2):
    """
    volume of enclosing axis-aligned box
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners2.shape[2] == 8
    assert corners2.shape[3] == 3
    EPS = 1e-6

    corners1 = corners1.clone()
    corners2 = corners2.clone()
    # flip Y axis, since it is negative
    corners1[:, :, :, 1] *= -1
    corners2[:, :, :, 1] *= -1

    al_xmin = torch.min(
        torch.min(corners1[:, :, :, 0], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 0], dim=2).values[:, None, :],
    )
    al_ymin = torch.max(
        torch.max(corners1[:, :, :, 1], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 1], dim=2).values[:, None, :],
    )
    al_zmin = torch.min(
        torch.min(corners1[:, :, :, 2], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 2], dim=2).values[:, None, :],
    )
    al_xmax = torch.max(
        torch.max(corners1[:, :, :, 0], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 0], dim=2).values[:, None, :],
    )
    al_ymax = torch.min(
        torch.min(corners1[:, :, :, 1], dim=2).values[:, :, None],
        torch.min(corners2[:, :, :, 1], dim=2).values[:, None, :],
    )
    al_zmax = torch.max(
        torch.max(corners1[:, :, :, 2], dim=2).values[:, :, None],
        torch.max(corners2[:, :, :, 2], dim=2).values[:, None, :],
    )

    diff_x = torch.abs(al_xmax - al_xmin)
    diff_y = torch.abs(al_ymax - al_ymin)
    diff_z = torch.abs(al_zmax - al_zmin)
    vol = diff_x * diff_y * diff_z
    return vol


def generalized_box3d_iou_tensor(
    corners1: torch.Tensor,
    corners2: torch.Tensor,
    nums_k2: torch.Tensor,
    rotated_boxes: bool = True,
    return_inter_vols_only: bool = False,
):
    """
    Input:
        corners1: torch Tensor (B, K1, 8, 3), assume up direction is negative Y
        corners2: torch Tensor (B, K2, 8, 3), assume up direction is negative Y
        Assumes that the box is only rotated along Z direction
    Returns:
        B x K1 x K2 matrix of generalized IOU by approximating the boxes to be axis aligned
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == corners2.shape[2]
    assert corners1.shape[3] == corners2.shape[3]

    B, K1 = corners1.shape[0], corners1.shape[1]
    _, K2 = corners2.shape[0], corners2.shape[1]

    # # box height. Y is negative, so max is torch.min
    ymax = torch.min(corners1[:, :, 0, 1][:, :, None], corners2[:, :, 0, 1][:, None, :])
    ymin = torch.max(corners1[:, :, 4, 1][:, :, None], corners2[:, :, 4, 1][:, None, :])
    height = (ymax - ymin).clamp(min=0)
    EPS = 1e-8

    idx = torch.arange(start=3, end=-1, step=-1, device=corners1.device)
    idx2 = torch.tensor([0, 2], dtype=torch.int64, device=corners1.device)
    rect1 = corners1[:, :, idx, :]
    rect2 = corners2[:, :, idx, :]
    rect1 = rect1[:, :, :, idx2]
    rect2 = rect2[:, :, :, idx2]

    lt = torch.max(rect1[:, :, 1][:, :, None, :], rect2[:, :, 1][:, None, :, :])
    rb = torch.min(rect1[:, :, 3][:, :, None, :], rect2[:, :, 3][:, None, :, :])
    wh = (rb - lt).clamp(min=0)
    non_rot_inter_areas = wh[:, :, :, 0] * wh[:, :, :, 1]
    non_rot_inter_areas = non_rot_inter_areas.view(B, K1, K2)
    if nums_k2 is not None:
        for b in range(B):
            non_rot_inter_areas[b, :, nums_k2[b] :] = 0

    enclosing_vols = enclosing_box3d_vol(corners1, corners2)

    # vols of boxes
    vols1 = box3d_vol_tensor(corners1).clamp(min=EPS)
    vols2 = box3d_vol_tensor(corners2).clamp(min=EPS)

    sum_vols = vols1[:, :, None] + vols2[:, None, :]

    # filter malformed boxes
    good_boxes = (enclosing_vols > 2 * EPS) * (sum_vols > 4 * EPS)

    if rotated_boxes:
        inter_areas = torch.zeros((B, K1, K2), dtype=torch.float32)
        rect1 = rect1.cpu()
        rect2 = rect2.cpu()
        nums_k2_np = to_list_1d(nums_k2)
        non_rot_inter_areas_np = to_list_3d(non_rot_inter_areas)
        for b in range(B):
            for k1 in range(K1):
                for k2 in range(K2):
                    if nums_k2 is not None and k2 >= nums_k2_np[b]:
                        break
                    if non_rot_inter_areas_np[b][k1][k2] == 0:
                        continue
                    ##### compute volume of intersection
                    inter = polygon_clip_unnest(rect1[b, k1], rect2[b, k2])
                    if len(inter) > 0:
                        xs = torch.stack([x[0] for x in inter])
                        ys = torch.stack([x[1] for x in inter])
                        inter_areas[b, k1, k2] = torch.abs(
                            torch.dot(xs, torch.roll(ys, 1)) - torch.dot(ys, torch.roll(xs, 1))
                        )
        inter_areas.mul_(0.5)
    else:
        inter_areas = non_rot_inter_areas

    inter_areas = inter_areas.to(corners1.device)
    ### gIOU = iou - (1 - sum_vols/enclose_vol)
    inter_vols = inter_areas * height
    if return_inter_vols_only:
        return inter_vols

    union_vols = (sum_vols - inter_vols).clamp(min=EPS)
    ious = inter_vols / union_vols
    giou_second_term = -(1 - union_vols / enclosing_vols)
    gious = ious + giou_second_term
    gious *= good_boxes
    if nums_k2 is not None:
        mask = torch.zeros((B, K1, K2), device=height.device, dtype=torch.float32)
        for b in range(B):
            mask[b, :, : nums_k2[b]] = 1
        gious *= mask
    return gious


generalized_box3d_iou_tensor_jit = torch.jit.script(generalized_box3d_iou_tensor)


def generalized_box3d_iou_cython(
    corners1: torch.Tensor,
    corners2: torch.Tensor,
    nums_k2: torch.Tensor,
    rotated_boxes: bool = True,
    return_inter_vols_only: bool = False,
):
    """
    Input:
        corners1: torch Tensor (B, K1, 8, 3), assume up direction is negative Y
        corners2: torch Tensor (B, K2, 8, 3), assume up direction is negative Y
        Assumes that the box is only rotated along Z direction
    Returns:
        B x K1 x K2 matrix of generalized IOU by approximating the boxes to be axis aligned
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape) == 4
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == corners2.shape[2]
    assert corners1.shape[3] == corners2.shape[3]

    B, K1 = corners1.shape[0], corners1.shape[1]
    _, K2 = corners2.shape[0], corners2.shape[1]

    # # box height. Y is negative, so max is torch.min
    ymax = torch.min(corners1[:, :, 0, 1][:, :, None], corners2[:, :, 0, 1][:, None, :])
    ymin = torch.max(corners1[:, :, 4, 1][:, :, None], corners2[:, :, 4, 1][:, None, :])
    height = (ymax - ymin).clamp(min=0)
    EPS = 1e-8

    idx = torch.arange(start=3, end=-1, step=-1, device=corners1.device)
    idx2 = torch.tensor([0, 2], dtype=torch.int64, device=corners1.device)
    rect1 = corners1[:, :, idx, :]
    rect2 = corners2[:, :, idx, :]
    rect1 = rect1[:, :, :, idx2]
    rect2 = rect2[:, :, :, idx2]

    lt = torch.max(rect1[:, :, 1][:, :, None, :], rect2[:, :, 1][:, None, :, :])
    rb = torch.min(rect1[:, :, 3][:, :, None, :], rect2[:, :, 3][:, None, :, :])
    wh = (rb - lt).clamp(min=0)
    non_rot_inter_areas = wh[:, :, :, 0] * wh[:, :, :, 1]
    non_rot_inter_areas = non_rot_inter_areas.view(B, K1, K2)
    if nums_k2 is not None:
        for b in range(B):
            non_rot_inter_areas[b, :, nums_k2[b] :] = 0

    enclosing_vols = enclosing_box3d_vol(corners1, corners2)

    # vols of boxes
    vols1 = box3d_vol_tensor(corners1).clamp(min=EPS)
    vols2 = box3d_vol_tensor(corners2).clamp(min=EPS)

    sum_vols = vols1[:, :, None] + vols2[:, None, :]

    # filter malformed boxes
    good_boxes = (enclosing_vols > 2 * EPS) * (sum_vols > 4 * EPS)

    if rotated_boxes:
        inter_areas = np.zeros((B, K1, K2), dtype=np.float32)
        rect1 = rect1.cpu().numpy().astype(np.float32)
        rect2 = rect2.cpu().numpy().astype(np.float32)
        nums_k2_np = nums_k2.cpu().detach().numpy().astype(np.int32)
        non_rot_inter_areas_np = non_rot_inter_areas.cpu().detach().numpy().astype(np.float32)
        box_intersection(rect1, rect2, non_rot_inter_areas_np, nums_k2_np, inter_areas, True)
        inter_areas = torch.from_numpy(inter_areas)
    else:
        inter_areas = non_rot_inter_areas

    inter_areas = inter_areas.to(corners1.device)
    ### gIOU = iou - (1 - sum_vols/enclose_vol)
    inter_vols = inter_areas * height
    if return_inter_vols_only:
        return inter_vols

    union_vols = (sum_vols - inter_vols).clamp(min=EPS)
    ious = inter_vols / union_vols
    giou_second_term = -(1 - union_vols / enclosing_vols)
    gious = ious + giou_second_term
    gious *= good_boxes
    if nums_k2 is not None:
        mask = torch.zeros((B, K1, K2), device=height.device, dtype=torch.float32)
        for b in range(B):
            mask[b, :, : nums_k2[b]] = 1
        gious *= mask
    return gious


def generalized_box3d_iou(
    corners1: torch.Tensor,
    corners2: torch.Tensor,
    nums_k2: torch.Tensor,
    rotated_boxes: bool = True,
    return_inter_vols_only: bool = False,
    needs_grad: bool = False,
):
    if needs_grad is True or box_intersection is None:
        context = torch.enable_grad if needs_grad else torch.no_grad
        with context():
            return generalized_box3d_iou_tensor_jit(corners1, corners2, nums_k2, rotated_boxes, return_inter_vols_only)

    else:
        # Cythonized implementation of GIoU
        with torch.no_grad():
            return generalized_box3d_iou_cython(corners1, corners2, nums_k2, rotated_boxes, return_inter_vols_only)
