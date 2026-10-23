# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.uniad.reference.query_interaction_module import QueryInteractionModule
from models.experimental.uniad.tt.ttnn_query_interaction import TtQueryInteractionModule
from models.experimental.uniad.reference.utils import Instances
from models.experimental.uniad.tt.ttnn_utils import Instances as TtInstances
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.uniad.tt.model_preprocessing_encoder import (
    create_uniad_model_parameters_encoder,
)
from models.experimental.uniad.tests.common import load_torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_uniad_query_interaction(device, reset_seeds, model_location_generator):
    torch_model = QueryInteractionModule()

    torch_model = load_torch_model(
        torch_model=torch_model, layer="query_interact", model_location_generator=model_location_generator
    )

    num_instances = 901

    track_instances = Instances((1, 1))

    ref_pts = torch.rand(num_instances, 3)
    query = torch.rand(num_instances, 512)
    # Non-zero output_embedding: it feeds the self-attention value and the FFN
    # in _update_track_embedding, so zeros would drive the query_feat half of
    # the output to a per-instance constant and make the PCC check below
    # degenerate. Random input exercises the attention/FFN path for real.
    output_embedding = torch.rand(num_instances, 256)
    # _select_active_tracks keeps only `obj_idxes >= 0`, and only those active
    # tracks pass through _update_track_embedding. With every obj_idx == -1 the
    # active set is empty, so the QIM was a pure no-op and the assertions below
    # compared the (unchanged) input query/ref_pts to itself — vacuously
    # passing. Make a deterministic ~half active so the attention/FFN update
    # actually runs and the filter (>= 0) itself is exercised.
    obj_idxes = torch.full((num_instances,), -1, dtype=torch.long)
    obj_idxes[: num_instances // 2] = torch.arange(num_instances // 2, dtype=torch.long)
    track_instances.ref_pts = ref_pts
    track_instances.query = query
    track_instances.output_embedding = output_embedding
    track_instances.obj_idxes = obj_idxes

    data = {"init_track_instances": track_instances, "track_instances": track_instances}

    torch_output = torch_model(data)

    tt_track_instances = TtInstances((1, 1), ttnn_device=device)

    tt_track_instances.ref_pts = ttnn.from_torch(ref_pts, device=device)
    tt_track_instances.query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.output_embedding = ttnn.from_torch(output_embedding, device=device)
    # Force ttnn.int32 — without an explicit dtype, ttnn.from_torch picks
    # UINT32 for torch.long inputs, and `obj_idxes >= 0` then evaluates as
    # always-True (uint32 is always non-negative), collapsing the "active"
    # filter. Matches the int32 dtype used by TtUniAD when it constructs
    # this field via ttnn.full(..., dtype=ttnn.int32, ...).
    tt_track_instances.obj_idxes = ttnn.from_torch(obj_idxes, device=device, dtype=ttnn.int32)
    data = {"init_track_instances": tt_track_instances, "track_instances": tt_track_instances}

    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)

    tt_model = TtQueryInteractionModule(
        params=parameter,
        device=device,
    )

    ttnn_output = tt_model(data)
    ttnn_query = ttnn.to_torch(ttnn_output.query)
    ttnn_ref_pts = ttnn.to_torch(ttnn_output.ref_pts)
    assert_with_pcc(torch_output.query, ttnn_query, 0.99)
    assert_with_pcc(torch_output.ref_pts, ttnn_ref_pts, 0.99)
