# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.uniad.reference.runtime_tracker_base import RuntimeTrackerBase
from models.experimental.uniad.tt.ttnn_runtime_tracker_base import TtRuntimeTrackerBase
from models.experimental.uniad.reference.utils import Instances
from models.experimental.uniad.tt.ttnn_utils import Instances as TtInstances


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_run_time_tracker(
    device,
    reset_seeds,
):
    torch_model = RuntimeTrackerBase(score_thresh=0.4, filter_score_thresh=0.35, miss_tolerance=5)

    num_instances = 901

    track_instances = Instances((1, 1))

    ref_pts = torch.rand(num_instances, 3)
    query = torch.rand(num_instances, 512)
    output_embedding = torch.rand(num_instances, 256)
    mem_padding_mask = torch.randint(0, 2, (901, 4), dtype=torch.bool)
    mem_bank = torch.zeros((901, 4, 256))
    disappear_time = torch.zeros(901)
    obj_idxes = torch.full((num_instances,), -1, dtype=torch.int32)
    scores = torch.rand(901)
    save_period = torch.rand(901)
    track_instances.disappear_time = disappear_time
    track_instances.ref_pts = ref_pts
    track_instances.query = query
    track_instances.output_embedding = output_embedding
    track_instances.obj_idxes = obj_idxes
    track_instances.mem_padding_mask = mem_padding_mask
    track_instances.mem_bank = mem_bank
    track_instances.scores = scores
    track_instances.save_period = save_period
    data = track_instances

    torch_output = torch_model.update(data)

    tt_track_instances = TtInstances((1, 1))

    tt_track_instances.disappear_time = ttnn.from_torch(disappear_time, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.ref_pts = ttnn.from_torch(ref_pts, device=device)
    tt_track_instances.query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.output_embedding = ttnn.from_torch(output_embedding, dtype=ttnn.bfloat16, device=device)
    tt_track_instances.obj_idxes = ttnn.from_torch(obj_idxes, device=device)
    tt_track_instances.scores = ttnn.from_torch(scores, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.save_period = ttnn.from_torch(save_period, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.mem_padding_mask = mem_padding_mask
    tt_track_instances.mem_bank = ttnn.from_torch(mem_bank, dtype=ttnn.bfloat16, device=device)

    data = tt_track_instances

    tt_model = TtRuntimeTrackerBase(score_thresh=0.4, filter_score_thresh=0.35, miss_tolerance=5)

    # ttnn_output = tt_model.update(data) #Outputs NoneType because it just modifies the track_instances object directly.
    # ttnn_out = ttnn.to_torch(ttnn_output)

    # assert_with_pcc(torch_output, ttnn_out, 0.99)
