# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import cv2


def resize_pad(img):
    size0 = img.shape
    if size0[0] >= size0[1]:
        h1 = 256
        w1 = 256 * size0[1] // size0[0]
        padh = 0
        padw = 256 - w1
        scale = size0[1] / w1
    else:
        h1 = 256 * size0[0] // size0[1]
        w1 = 256
        padh = 256 - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh // 2
    padh2 = padh // 2 + padh % 2
    padw1 = padw // 2
    padw2 = padw // 2 + padw % 2
    img1 = cv2.resize(img, (w1, h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0, 0)))
    pad = (int(padh1 * scale), int(padw1 * scale))
    img2 = cv2.resize(img1, (128, 128))
    return img1, img2, scale, pad


def blazeblock(x, in_channel, out_channel, kernel_size, stride, padding, skip_proj, parameters, i):
    channel_pad = out_channel - in_channel
    if stride == 2:
        if kernel_size == 3:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
        else:
            h = F.pad(x, (1, 2, 1, 2), "constant", 0)
        max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        x = max_pool(x)
    else:
        h = x
    if skip_proj:
        conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        conv.weight = parameters[i].skip_proj.weight
        conv.bias = parameters[i].skip_proj.bias
        x = conv(x)
    elif channel_pad > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, channel_pad), "constant", 0)
    conv1 = nn.Conv2d(
        in_channels=in_channel,
        out_channels=in_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=in_channel,
    )
    conv1.weight = parameters[i].convs[0].weight
    conv1.bias = parameters[i].convs[0].bias

    h = conv1(h)

    conv2 = nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=1,
        stride=1,
        padding=0,
    )
    conv2.weight = parameters[i].convs[1].weight
    conv2.bias = parameters[i].convs[1].bias

    h = conv2(h)
    act = nn.ReLU(inplace=True)
    return act(h + x)


def blazepose(x, parameters):
    detection2roi_method = "alignment"
    kp1 = 2
    kp2 = 3
    theta0 = 90 * np.pi / 180
    dscale = 1.5
    dy = 0.0
    b = x.shape[0]
    backbone = nn.Conv2d(
        in_channels=3,
        out_channels=48,
        kernel_size=5,
        stride=1,
        padding=2,
    )
    backbone.weight = parameters.backbone1[0].weight
    backbone.bias = parameters.backbone1[0].bias
    x = backbone(x)
    relu = nn.ReLU(inplace=True)
    x = relu(x)
    in_channel = [48, 48, 48, 48, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128]
    out_channel = [48, 48, 48, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128, 128]
    i = 2
    for i in range(2, 24):
        if i > 1:
            if i == 5 or i == 9 or i == 16:
                x = blazeblock(x, in_channel[i - 2], out_channel[i - 2], 5, 2, 0, True, parameters.backbone1, i)
            else:
                x = blazeblock(x, in_channel[i - 2], out_channel[i - 2], 5, 1, 2, False, parameters.backbone1, i)
        i += 1
    i = 0

    for i in range(6):
        if i == 0:
            h = blazeblock(x, 128, 256, 5, 2, 0, True, parameters.backbone2, i)
        else:
            h = blazeblock(h, 256, 256, 5, 1, 2, False, parameters.backbone2, i)
        i += 1

    class8 = nn.Conv2d(128, 2, 1)
    class8.weight = parameters.classifier_8.weight
    class8.bias = parameters.classifier_8.bias
    c1 = class8(x)
    c1 = c1.permute(0, 2, 3, 1)
    c1 = c1.reshape(b, -1, 1)

    class16 = nn.Conv2d(256, 6, 1)
    class16.weight = weight = parameters.classifier_16.weight
    class16.bias = parameters.classifier_16.bias
    c2 = class16(h)
    c2 = c2.permute(0, 2, 3, 1)
    c2 = c2.reshape(b, -1, 1)
    c = torch.cat((c1, c2), dim=1)

    regressor_8 = nn.Conv2d(128, 24, 1)
    regressor_8.weight = parameters.regressor_8.weight
    regressor_8.bias = parameters.regressor_8.bias
    r1 = regressor_8(x)
    r1 = r1.permute(0, 2, 3, 1)
    r1 = r1.reshape(b, -1, 12)

    regressor_16 = nn.Conv2d(256, 72, 1)
    regressor_16.weight = parameters.regressor_16.weight
    regressor_16.bias = parameters.regressor_16.bias
    r2 = regressor_16(h)
    r2 = r2.permute(0, 2, 3, 1)
    r2 = r2.reshape(b, -1, 12)
    r = torch.cat((r1, r2), dim=1)
    return [r, c]


def decode_boxes(raw_boxes, anchors):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    boxes = torch.zeros_like(raw_boxes)
    x_scale = 128.0
    y_scale = 128.0
    h_scale = 128.0
    w_scale = 128.0
    num_keypoints = 4
    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.0  # ymin
    boxes[..., 1] = x_center - w / 2.0  # xmin
    boxes[..., 2] = y_center + h / 2.0  # ymax
    boxes[..., 3] = x_center + w / 2.0  # xmax

    for k in range(num_keypoints):
        offset = 4 + k * 2
        keypoint_x = raw_boxes[..., offset] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y
    return boxes


def tensors_to_detections(raw_box_tensor, raw_score_tensor, anchors):
    num_anchors = 896
    assert anchors.ndimension() == 2
    assert anchors.shape[0] == num_anchors
    assert anchors.shape[1] == 4

    num_coords = 12
    num_classes = 1
    assert raw_box_tensor.ndimension() == 3
    assert raw_box_tensor.shape[1] == num_anchors
    assert raw_box_tensor.shape[2] == num_coords

    assert raw_score_tensor.ndimension() == 3
    assert raw_score_tensor.shape[1] == num_anchors
    assert raw_score_tensor.shape[2] == num_classes

    assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

    detection_boxes = decode_boxes(raw_box_tensor, anchors)
    score_clipping_thresh = 100.0
    thresh = score_clipping_thresh
    raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
    detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)

    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    min_score_thresh = 0.75
    mask = detection_scores >= min_score_thresh

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    output_detections = []
    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
        output_detections.append(torch.cat((boxes, scores), dim=-1))

    return output_detections


def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2),
    )
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2),
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)


def weighted_non_max_suppression(detections):
    if len(detections) == 0:
        return []

    output_detections = []
    num_coords = 12
    # Sort the detections from highest to lowest score.
    remaining = torch.argsort(detections[:, num_coords], descending=True)

    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        min_suppression_threshold = 0.3
        mask = ious > min_suppression_threshold
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection.clone()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :num_coords]
            scores = detections[overlapping, num_coords : num_coords + 1]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(dim=0) / total_score
            weighted_detection[:num_coords] = weighted
            weighted_detection[num_coords] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections


def predict_on_batch(x, anchors, parameters):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).permute((0, 3, 1, 2))
    x_scale = 128.0
    y_scale = 128.0
    assert x.shape[1] == 3
    assert x.shape[2] == y_scale
    assert x.shape[3] == x_scale

    x = x.float() / 255.0

    # 2. Run the neural network:
    with torch.no_grad():
        out = blazepose(x, parameters)
    detections = tensors_to_detections(out[0], out[1], anchors)
    num_coords = 12
    filtered_detections = []
    for i in range(len(detections)):
        faces = weighted_non_max_suppression(detections[i])
        faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, num_coords + 1))
        filtered_detections.append(faces)

    return filtered_detections


def predict_on_image(img, parameters, anchors):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).permute((2, 0, 1))
    return predict_on_batch(img.unsqueeze(0), anchors, parameters)[0]


def denormalize_detections_ref(detections, scale, pad):
    detections[:, 0] = detections[:, 0] * scale * 256 - pad[0]
    detections[:, 1] = detections[:, 1] * scale * 256 - pad[1]
    detections[:, 2] = detections[:, 2] * scale * 256 - pad[0]
    detections[:, 3] = detections[:, 3] * scale * 256 - pad[1]

    detections[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]
    detections[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]
    return detections


def detection2roi(detection):
    detection2roi_method = "alignment"
    kp1 = 2
    kp2 = 3
    theta0 = 90 * np.pi / 180
    if detection2roi_method == "box":
        # compute box center and scale
        # use mediapipe/calculators/util/detections_to_rects_calculator.cc
        xc = (detection[:, 1] + detection[:, 3]) / 2
        yc = (detection[:, 0] + detection[:, 2]) / 2
        scale = detection[:, 3] - detection[:, 1]  # assumes square boxes

    elif detection2roi_method == "alignment":
        # compute box center and scale
        # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
        xc = detection[:, 4 + 2 * kp1]
        yc = detection[:, 4 + 2 * kp1 + 1]
        x1 = detection[:, 4 + 2 * kp2]
        y1 = detection[:, 4 + 2 * kp2 + 1]
        scale = ((xc - x1) ** 2 + (yc - y1) ** 2).sqrt() * 2

    dscale = 1.5
    dy = 0.0
    yc += dy * scale
    scale *= dscale

    # compute box rotation
    x0 = detection[:, 4 + 2 * kp1]
    y0 = detection[:, 4 + 2 * kp1 + 1]
    x1 = detection[:, 4 + 2 * kp2]
    y1 = detection[:, 4 + 2 * kp2 + 1]
    # theta = np.arctan2(y0-y1, x0-x1) - self.theta0
    theta = torch.atan2(y0 - y1, x0 - x1) - theta0
    return xc, yc, scale, theta
