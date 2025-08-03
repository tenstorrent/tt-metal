# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.uniad.reference.query_interaction import QueryInteractionModule
from models.experimental.uniad.tt.ttnn_query_interaction import TtQueryInteractionModule
from models.experimental.uniad.reference.utils import Instances
from models.experimental.uniad.tt.ttnn_utils import Instances as TtInstances
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.uniad.tt.model_preprocessing_encoder import (
    create_uniad_model_parameters_encoder,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_uniad_query_interaction(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"

    torch_model = QueryInteractionModule()

    torch_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    torch_dict = torch_dict["state_dict"]

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("query_interact"))}

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    num_instances = 901

    track_instances = Instances((1, 1))

    ref_pts = torch.rand(num_instances, 3)
    query = torch.rand(num_instances, 512)
    output_embedding = torch.zeros(num_instances, 256)
    obj_idxes = torch.full((num_instances,), -1, dtype=torch.long)
    track_instances.ref_pts = ref_pts
    track_instances.query = query
    track_instances.output_embedding = output_embedding
    track_instances.obj_idxes = obj_idxes

    data = {"init_track_instances": track_instances, "track_instances": track_instances}

    torch_output = torch_model(data)

    tt_track_instances = TtInstances((1, 1))

    tt_track_instances.ref_pts = ttnn.from_torch(ref_pts, device=device)
    tt_track_instances.query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.output_embedding = ttnn.from_torch(output_embedding, device=device)
    tt_track_instances.obj_idxes = ttnn.from_torch(obj_idxes, device=device)
    data = {"init_track_instances": tt_track_instances, "track_instances": tt_track_instances}

    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)

    tt_model = TtQueryInteractionModule(
        params=parameter,
        device=device,
    )

    ttnn_output = tt_model(data)
    ttnn_query = ttnn.to_torch(ttnn_output.query)
    assert_with_pcc(torch_output.query, ttnn_query, 0.99)
