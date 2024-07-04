import numpy as np
import torch
import ttnn
from models.experimental.functional_blazepose.tt.blazepose_model import blazepose


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
    assert len(anchors.shape()) == 2
    assert anchors.shape[0] == num_anchors
    assert anchors.shape[1] == 4

    num_coords = 12
    num_classes = 1
    assert len(raw_box_tensor.shape()) == 3
    assert raw_box_tensor.shape[1] == num_anchors
    assert raw_box_tensor.shape[2] == num_coords

    assert len(raw_score_tensor.shape()) == 3
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


def predict_on_batch(x, anchors, parameters, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).permute((0, 3, 1, 2))
    x_scale = 128.0
    y_scale = 128.0
    assert x.shape[1] == 3
    assert x.shape[2] == y_scale
    assert x.shape[3] == x_scale

    x = x.float() / 255.0

    # 2. Run the neural network:
    x = torch.permute(x, (0, 2, 3, 1))
    print("Shape of x in  utils:", x.shape)
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)  # , layout = ttnn.TILE_LAYOUT)
    with torch.no_grad():
        out = blazepose(x, parameters, device)
        # out[0] = ttnn.to_torch(out[0])
        # out[1] = ttnn.to_torch(out[1])
        return out
    detections = tensors_to_detections(out[0], out[1], anchors)
    num_coords = 12
    filtered_detections = []
    for i in range(len(detections)):
        faces = weighted_non_max_suppression(detections[i])
        faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, num_coords + 1))
        filtered_detections.append(faces)

    return filtered_detections


def predict_on_image(img, parameters, anchors, device):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).permute((2, 0, 1))
    return predict_on_batch(img.unsqueeze(0), anchors, parameters, device)
