# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np


def check_aspect(crop_range, aspect_min):
    xy_aspect = np.min(crop_range[:2]) / np.max(crop_range[:2])
    xz_aspect = np.min(crop_range[[0, 2]]) / np.max(crop_range[[0, 2]])
    yz_aspect = np.min(crop_range[1:]) / np.max(crop_range[1:])
    return (xy_aspect >= aspect_min) or (xz_aspect >= aspect_min) or (yz_aspect >= aspect_min)


class RandomCuboid(object):
    """
    RandomCuboid augmentation from DepthContrast [https://arxiv.org/abs/2101.02691]
    We slightly modify this operation to account for object detection.
    This augmentation randomly crops a cuboid from the input and
    ensures that the cropped cuboid contains at least one bounding box
    """

    def __init__(
        self,
        min_points,
        aspect=0.8,
        min_crop=0.5,
        max_crop=1.0,
        box_filter_policy="center",
    ):
        self.aspect = aspect
        self.min_crop = min_crop
        self.max_crop = max_crop
        self.min_points = min_points
        self.box_filter_policy = box_filter_policy

    def __call__(self, point_cloud, target_boxes, per_point_labels=None):
        range_xyz = np.max(point_cloud[:, 0:3], axis=0) - np.min(point_cloud[:, 0:3], axis=0)

        for _ in range(100):
            crop_range = self.min_crop + np.random.rand(3) * (self.max_crop - self.min_crop)
            if not check_aspect(crop_range, self.aspect):
                continue

            sample_center = point_cloud[np.random.choice(len(point_cloud)), 0:3]

            new_range = range_xyz * crop_range / 2.0

            max_xyz = sample_center + new_range
            min_xyz = sample_center - new_range

            upper_idx = np.sum((point_cloud[:, 0:3] <= max_xyz).astype(np.int32), 1) == 3
            lower_idx = np.sum((point_cloud[:, 0:3] >= min_xyz).astype(np.int32), 1) == 3

            new_pointidx = (upper_idx) & (lower_idx)

            if np.sum(new_pointidx) < self.min_points:
                continue

            new_point_cloud = point_cloud[new_pointidx, :]

            # filtering policy is the only modification from DepthContrast
            if self.box_filter_policy == "center":
                # remove boxes whose center does not lie within the new_point_cloud
                new_boxes = target_boxes
                if target_boxes.sum() > 0:  # ground truth contains no bounding boxes. Common in SUNRGBD.
                    box_centers = target_boxes[:, 0:3]
                    new_pc_min_max = np.min(new_point_cloud[:, 0:3], axis=0), np.max(new_point_cloud[:, 0:3], axis=0)
                    keep_boxes = np.logical_and(
                        np.all(box_centers >= new_pc_min_max[0], axis=1),
                        np.all(box_centers <= new_pc_min_max[1], axis=1),
                    )
                    if keep_boxes.sum() == 0:
                        # current data augmentation removes all boxes in the pointcloud. fail!
                        continue
                    new_boxes = target_boxes[keep_boxes]
                if per_point_labels is not None:
                    new_per_point_labels = [x[new_pointidx] for x in per_point_labels]
                else:
                    new_per_point_labels = None
                # if we are here, all conditions are met. return boxes
                return new_point_cloud, new_boxes, new_per_point_labels

        # fallback
        return point_cloud, target_boxes, per_point_labels
