# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.uniad.reference.runtime_tracker_base import RuntimeTrackerBase
from models.experimental.uniad.tt.ttnn_runtime_tracker_base import TtRuntimeTrackerBase
from models.experimental.uniad.reference.utils import Instances
from models.experimental.uniad.tt.ttnn_utils import Instances as TtInstances
from tests.ttnn.utils_for_testing import assert_with_pcc


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

    # Hand-craft the first few entries so the update() exercises EVERY branch,
    # not just new-track assignment (the rest stay unmatched -1 with random
    # scores, which alone only ever hits the new-track path). The TT port is
    # compared exactly against the reference below, so the reference is the
    # oracle — we just need inputs that reach each branch:
    #   idx 0: new track       obj=-1, score>=score_thresh         -> assigned a fresh id
    #   idx 1: ego sentinel    obj=-2                              -> left untouched (-2)
    #   idx 2: ageing          obj>=0, score<filter, dt small      -> dt += 1, id kept
    #   idx 3: ID release      obj>=0, score<filter, dt=tol-1      -> dt hits tol -> id = -1
    #   idx 4: confident track obj>=0, score>=score_thresh         -> dt reset to 0, id kept
    #   idx 5: limbo band      obj>=0, filter<=score<score_thresh  -> untouched
    obj_idxes[0] = -1
    scores[0] = 0.90
    obj_idxes[1] = -2
    obj_idxes[2] = 10
    scores[2] = 0.10
    disappear_time[2] = 1.0
    obj_idxes[3] = 11
    scores[3] = 0.10
    disappear_time[3] = 4.0  # miss_tolerance - 1
    obj_idxes[4] = 12
    scores[4] = 0.90
    obj_idxes[5] = 13
    scores[5] = 0.37  # filter_score_thresh (0.35) <= score < score_thresh (0.4)
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

    # Snapshot the fields update() mutates in place BEFORE running it. update()
    # writes obj_idxes / disappear_time through the shared tensors, so if the tt
    # path below read the post-update tensors it would be handed the reference's
    # answer and the comparison would be tautological.
    obj_idxes_pre = obj_idxes.clone()
    disappear_time_pre = disappear_time.clone()

    # update() mutates obj_idxes (new-track id assignment + dead-track
    # removal) and disappear_time in place and returns None, so compare the
    # mutated fields rather than the return value.
    torch_model.update(data)

    # Sanity-check that the crafted scenario actually reached each branch in the
    # reference (the oracle), so the TT comparison below is not silently only
    # covering new-track assignment if a threshold ever changes.
    ref_obj = track_instances.obj_idxes
    ref_dt = track_instances.disappear_time
    assert ref_obj[0] >= 0, "idx0 should have been assigned a new track id"
    assert ref_obj[1] == -2, "idx1 ego sentinel must be preserved"
    assert ref_obj[2] == 10 and ref_dt[2] == 2.0, "idx2 should age (dt+1), id kept"
    assert ref_obj[3] == -1 and ref_dt[3] == 5.0, "idx3 should be released (dt hit miss_tolerance)"
    assert ref_obj[4] == 12 and ref_dt[4] == 0.0, "idx4 confident track: id kept, dt reset"
    assert ref_obj[5] == 13, "idx5 limbo band: id untouched"

    tt_track_instances = TtInstances((1, 1))

    tt_track_instances.disappear_time = ttnn.from_torch(disappear_time_pre, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.ref_pts = ttnn.from_torch(ref_pts, device=device)
    tt_track_instances.query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.output_embedding = ttnn.from_torch(output_embedding, dtype=ttnn.bfloat16, device=device)
    tt_track_instances.obj_idxes = ttnn.from_torch(obj_idxes_pre, device=device)
    tt_track_instances.scores = ttnn.from_torch(scores, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.save_period = ttnn.from_torch(save_period, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.mem_padding_mask = mem_padding_mask
    tt_track_instances.mem_bank = ttnn.from_torch(mem_bank, dtype=ttnn.bfloat16, device=device)

    data = tt_track_instances

    tt_model = TtRuntimeTrackerBase(score_thresh=0.4, filter_score_thresh=0.35, miss_tolerance=5)

    # update() returns None and mutates `tt_track_instances` in place.
    tt_model.update(data)

    # obj_idxes are integer track IDs, so compare them exactly — PCC is the
    # wrong metric here: comp_pcc returns 1.0 when one side has zero variance,
    # and the unported tt path leaves obj_idxes a constant (-1), which would
    # spuriously "pass" a PCC check against the reference's assigned IDs.
    assert torch.equal(
        track_instances.obj_idxes.to(torch.int64),
        ttnn.to_torch(tt_track_instances.obj_idxes).to(torch.int64),
    ), "tt obj_idxes do not match the reference (device-side id assignment not ported)"
    assert_with_pcc(
        track_instances.disappear_time.float(),
        ttnn.to_torch(tt_track_instances.disappear_time).float(),
        0.99,
    )
