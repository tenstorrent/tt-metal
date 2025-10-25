# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.uniad.reference.utils import Instances

from models.experimental.uniad.reference.utils import denormalize_bbox


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.5, filter_score_thresh=0.4, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances, iou_thre=None):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                if iou_thre is not None and track_instances.pred_boxes[track_instances.obj_idxes >= 0].shape[0] != 0:
                    iou3ds = iou_3d(
                        denormalize_bbox(track_instances.pred_boxes[i].unsqueeze(0), None)[..., :7],
                        denormalize_bbox(track_instances.pred_boxes[track_instances.obj_idxes >= 0], None)[..., :7],
                    )
                    if iou3ds.max() > iou_thre:
                        continue
                # new track
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    track_instances.obj_idxes[i] = -1
