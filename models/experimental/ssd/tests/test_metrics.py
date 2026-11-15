# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.ssd.reference.utils.metrics import calculate_ap


def test_one_detection_on_one_gt():
    """
    Hand-computed case 1: One detection on one GT -> AP 1.0
    - GT: 1 object at [10, 10, 20, 20], class 1 (num_ground_truth = 1)
    - Pred: 1 detection at [10, 10, 20, 20], class 1, conf 0.9
    - IoU = 1.0 >= 0.5, class matches -> TP = 1, FP = 0
    - cum_tp = [1], cum_fp = [0]
    - recall = [1.0], precision = [1.0]
    - At all 11 recall thresholds t in [0.0, 0.1, ..., 1.0], recall >= t is True.
    - Max precision at recall >= t is 1.0 for all 11 points.
    - AP = 11 * (1.0 / 11) = 1.0
    """
    predictions = [{"class": 1, "confidence": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]}]
    ground_truth = [[{"class": 1, "bbox": [10.0, 10.0, 20.0, 20.0]}]]
    ap = calculate_ap(predictions, ground_truth)
    assert ap == pytest.approx(1.0)


def test_duplicate_detections_change_ap():
    """
    Hand-computed case 2: Image with two GT objects A and B of class 1,
    with a duplicate detection on A that lowers precision before B is detected.
    - GT A: [10, 10, 20, 20], class 1
    - GT B: [50, 50, 20, 20], class 1
      Total GT objects = 2 (num_ground_truth = 2)
    - Detections:
      1. on A: conf 0.9, bbox [10, 10, 20, 20], class 1
      2. duplicate on A: conf 0.85, bbox [10, 10, 20, 20], class 1
      3. on B: conf 0.8, bbox [50, 50, 20, 20], class 1
    Per-detection:
      - Det 1 (conf 0.9): TP=1, FP=0 -> cum_tp=1, cum_fp=0 -> r = 1/2 = 0.5, p = 1/1 = 1.0
      - Det 2 (conf 0.85): duplicate on A -> TP=0, FP=1 -> cum_tp=1, cum_fp=1 -> r = 1/2 = 0.5, p = 1/2 = 0.5
      - Det 3 (conf 0.8): TP on B -> TP=1, FP=0 -> cum_tp=2, cum_fp=1 -> r = 2/2 = 1.0, p = 2/3
    11-point interpolated precision:
      - For t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] (6 points): max precision at r >= t is 1.0
      - For t in [0.6, 0.7, 0.8, 0.9, 1.0] (5 points): max precision at r >= t is 2/3
      - AP = (6 * 1.0 + 5 * (2/3)) / 11 = (28/3) / 11 = 28/33 = 0.848484...
    """
    ground_truth = [
        [
            {"class": 1, "bbox": [10.0, 10.0, 20.0, 20.0]},
            {"class": 1, "bbox": [50.0, 50.0, 20.0, 20.0]},
        ]
    ]
    predictions = [
        {"class": 1, "confidence": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]},
        {"class": 1, "confidence": 0.85, "bbox": [10.0, 10.0, 20.0, 20.0]},
        {"class": 1, "confidence": 0.8, "bbox": [50.0, 50.0, 20.0, 20.0]},
    ]
    ap = calculate_ap(predictions, ground_truth)
    assert ap == pytest.approx(28 / 33)


def test_two_identical_detections_on_one_gt():
    """
    Single-duplicate case: One GT with two identical detections -> AP 1.0
    - GT: 1 object at [10, 10, 20, 20], class 1 (num_ground_truth = 1)
    - Pred 1: [10, 10, 20, 20], class 1, conf 0.9 -> matches GT (TP=1, FP=0), GT is consumed
    - Pred 2: [10, 10, 20, 20], class 1, conf 0.9 -> duplicate (TP=0, FP=1)
    - Per-detection:
      - After detection 1: recall 1.0, precision 1.0
      - After detection 2 (duplicate FP): recall 1.0, precision 0.5
    - In standard VOC 11-point AP, precision/recall are taken after EACH detection.
    - Interpolated precision at every threshold t in [0.0, 0.1, ..., 1.0] is the max precision
      at recall >= t, which is 1.0 (from detection 1).
    - AP = 1.0
    """
    predictions = [
        {"class": 1, "confidence": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]},
        {"class": 1, "confidence": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]},
    ]
    ground_truth = [[{"class": 1, "bbox": [10.0, 10.0, 20.0, 20.0]}]]
    ap = calculate_ap(predictions, ground_truth)
    assert ap == pytest.approx(1.0)


def test_detection_wrong_class():
    """
    Hand-computed case 3: Detection on the wrong class -> AP 0.0
    - GT: 1 object at [10, 10, 20, 20], class 1 (num_ground_truth = 1)
    - Pred: 1 detection at [10, 10, 20, 20], class 2, conf 0.9
    - IoU = 1.0 >= 0.5, but class 2 != class 1 -> TP = 0, FP = 1
    - cum_tp = [0], cum_fp = [1]
    - recall = [0.0], precision = [0.0]
    - At t = 0.0, precision is 0.0. For t > 0.0, no points have recall >= t.
    - AP = 0.0
    """
    predictions = [{"class": 2, "confidence": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]}]
    ground_truth = [[{"class": 1, "bbox": [10.0, 10.0, 20.0, 20.0]}]]
    ap = calculate_ap(predictions, ground_truth)
    assert ap == pytest.approx(0.0)


def test_two_images_one_gt_each_one_correct_detection_each():
    """
    Hand-computed case 4: Two images with one GT each and one correct detection each -> AP 1.0
    - Image 0: GT at [10, 10, 20, 20], class 1. Pred at [10, 10, 20, 20], class 1, conf 0.9 -> TP
    - Image 1: GT at [50, 50, 20, 20], class 1. Pred at [50, 50, 20, 20], class 1, conf 0.8 -> TP
    - Total GT objects = 2 (summed across both images, num_ground_truth = 2)
    - Sorted by confidence:
      * Pred 0 (conf 0.9, img 0): cum_tp = 1, cum_fp = 0 -> rec = 1/2 = 0.5, prec = 1/1 = 1.0
      * Pred 1 (conf 0.8, img 1): cum_tp = 2, cum_fp = 0 -> rec = 2/2 = 1.0, prec = 2/2 = 1.0
    - For t in [0.0, ..., 0.5]: max precision for rec >= t is 1.0
    - For t in [0.6, ..., 1.0]: max precision for rec >= t is 1.0
    - At all 11 recall thresholds t in [0.0, 0.1, ..., 1.0], max precision is 1.0.
    - AP = 11 * (1.0 / 11) = 1.0
    """
    predictions = [
        {"class": 1, "confidence": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]},
        {"class": 1, "confidence": 0.8, "bbox": [50.0, 50.0, 20.0, 20.0]},
    ]
    ground_truth = [
        [{"class": 1, "bbox": [10.0, 10.0, 20.0, 20.0]}],
        [{"class": 1, "bbox": [50.0, 50.0, 20.0, 20.0]}],
    ]
    ap = calculate_ap(predictions, ground_truth)
    assert ap == pytest.approx(1.0)


def test_miss():
    """
    Hand-computed case 5: A miss (IoU < 0.5 threshold) -> AP 0.0
    - GT: 1 object at [10, 10, 20, 20], class 1
    - Pred: 1 detection at [100, 100, 20, 20], class 1, conf 0.9
    - IoU = 0.0 < 0.5 -> TP = 0, FP = 1
    - cum_tp = [0], cum_fp = [1]
    - recall = [0.0], precision = [0.0]
    - AP = 0.0
    """
    predictions = [{"class": 1, "confidence": 0.9, "bbox": [100.0, 100.0, 20.0, 20.0]}]
    ground_truth = [[{"class": 1, "bbox": [10.0, 10.0, 20.0, 20.0]}]]
    ap = calculate_ap(predictions, ground_truth)
    assert ap == pytest.approx(0.0)


def test_explicit_image_id():
    """
    Predictions carrying explicit image_id match within their specified image,
    even when out of order or with varied confidences.
    """
    predictions = [
        {"class": 1, "confidence": 0.7, "bbox": [50.0, 50.0, 20.0, 20.0], "image_id": 1},
        {"class": 1, "confidence": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0], "image_id": 0},
    ]
    ground_truth = [
        [{"class": 1, "bbox": [10.0, 10.0, 20.0, 20.0]}],
        [{"class": 1, "bbox": [50.0, 50.0, 20.0, 20.0]}],
    ]
    ap = calculate_ap(predictions, ground_truth)
    assert ap == pytest.approx(1.0)


def test_cross_image_prediction_cannot_match_other_image_gt():
    """
    A detection for image 1 pointing at image 0's bbox cannot match image 0's GT.
    """
    predictions = [
        {"class": 1, "confidence": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0], "image_id": 1},
    ]
    ground_truth = [
        [{"class": 1, "bbox": [10.0, 10.0, 20.0, 20.0]}],  # image 0 has bbox at 10,10
        [{"class": 1, "bbox": [50.0, 50.0, 20.0, 20.0]}],  # image 1 has bbox at 50,50
    ]
    ap = calculate_ap(predictions, ground_truth)
    assert ap == pytest.approx(0.0)


def test_empty_predictions_or_ground_truth():
    """
    Edge cases: empty predictions or empty ground truth return 0.0.
    """
    assert calculate_ap([], [[{"class": 1, "bbox": [10.0, 10.0, 20.0, 20.0]}]]) == 0.0
    assert calculate_ap([{"class": 1, "confidence": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]}], []) == 0.0
    assert calculate_ap([{"class": 1, "confidence": 0.9, "bbox": [10.0, 10.0, 20.0, 20.0]}], [[]]) == 0.0
