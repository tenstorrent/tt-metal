# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.uniad.tt.ttnn_utils import Instances


class TtRuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.5, filter_score_thresh=0.4, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances, iou_thre=None):
        # Port of reference RuntimeTrackerBase.update. The new-track id
        # assignment allocates consecutive ids from a running counter
        # (self.max_obj_id), which is sequential host state and not
        # trace-friendly, so this small per-frame post-processing step runs on
        # host. The caller invokes it outside any trace region (it already
        # reads obj_idxes back to host around this call), so the device->host
        # transfer here is not on the trace-replay hot path. obj_idxes /
        # disappear_time are written back as device tensors because downstream
        # ops (active-track selection, the -2 ego-query mask) consume them as
        # ttnn tensors.
        #
        # The reference also has an optional IoU gate (skip the new-track id if
        # the box overlaps an existing track above iou_thre). Every UniAD caller
        # passes iou_thre=None, so that branch is not ported; fail loudly rather
        # than silently over-allocate ids if it is ever enabled.
        if iou_thre is not None:
            raise NotImplementedError(
                "TtRuntimeTrackerBase.update does not implement the iou_thre IoU gate; "
                "all UniAD callers pass iou_thre=None."
            )

        device = track_instances.obj_idxes.device()
        obj_layout = track_instances.obj_idxes.layout
        dt_layout = track_instances.disappear_time.layout

        obj_idxes = ttnn.to_torch(track_instances.obj_idxes).to(torch.int64)
        scores = ttnn.to_torch(track_instances.scores).float()
        disappear_time = ttnn.to_torch(track_instances.disappear_time).float()
        obj_shape = obj_idxes.shape
        dt_shape = disappear_time.shape

        obj_flat = obj_idxes.flatten()
        scores_flat = scores.flatten()
        dt_flat = disappear_time.flatten()

        confident = scores_flat >= self.score_thresh

        # Confident detections reset their disappear timer.
        dt_flat[confident] = 0

        # Brand-new tracks (unmatched and confident) are assigned consecutive
        # ids. Vectorised equivalent of the reference's per-element counter:
        # the n-th new track in index order gets max_obj_id + n. Entries that
        # are not -1 (e.g. the -2 ego/SDC query marker, or already-tracked
        # objects) are left untouched.
        new_mask = (obj_flat == -1) & confident
        num_new = int(new_mask.sum().item())
        if num_new > 0:
            obj_flat[new_mask] = torch.arange(self.max_obj_id, self.max_obj_id + num_new, dtype=obj_flat.dtype)
            self.max_obj_id += num_new

        # Existing tracks that drop below the filter threshold age out; once
        # they have been missing for miss_tolerance frames their id is released
        # (set back to -1). The two branches are mutually exclusive with the
        # new-track branch (that one requires score >= score_thresh, this one
        # score < filter_score_thresh <= score_thresh).
        ageing = (obj_flat >= 0) & (scores_flat < self.filter_score_thresh)
        dt_flat[ageing] += 1
        obj_flat[ageing & (dt_flat >= self.miss_tolerance)] = -1

        track_instances.obj_idxes = ttnn.from_torch(
            obj_flat.reshape(obj_shape).to(torch.int32), device=device, layout=obj_layout, dtype=ttnn.int32
        )
        track_instances.disappear_time = ttnn.from_torch(dt_flat.reshape(dt_shape), device=device, layout=dt_layout)
