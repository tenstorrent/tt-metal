# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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
        track_instances.disappear_time = ttnn.where(
            track_instances.scores >= self.score_thresh,
            ttnn.zeros_like(track_instances.disappear_time),
            track_instances.disappear_time,
        )
        ## This below should not be called but due to ttnn pipeline it is called because of mismatch values, Hence keeping it here
        # track_instances.obj_idxes = ttnn.to_torch(track_instances.obj_idxes)
        # track_instances.disappear_time = ttnn.to_torch( track_instances.disappear_time)

        # for i in range(track_instances.disappear_time.shape[0]):
        #     if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
        #         # new track
        #         track_instances.obj_idxes[i] = self.max_obj_id
        #         self.max_obj_id += 1
        #     elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
        #         # sleep time ++
        #         track_instances.disappear_time[i] += 1
        #         if track_instances.disappear_time[i] >= self.miss_tolerance:
        #             # mark deaded tracklets: Set the obj_id to -1.
        #             # TODO: remove it by following functions
        #             # Then this track will be removed by TrackEmbeddingLayer.
        #             track_instances.obj_idxes[i] = -1
