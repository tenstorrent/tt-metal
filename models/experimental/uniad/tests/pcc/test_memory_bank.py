# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.uniad.reference.memory_bank import MemoryBank
from models.experimental.uniad.tt.ttnn_memory_bank import TtMemoryBank
from models.experimental.uniad.reference.utils import Instances
from models.experimental.uniad.tt.ttnn_utils import Instances as TtInstances
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.uniad.tt.model_preprocessing_encoder import (
    create_uniad_model_parameters_encoder,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_uniad_memory_bank(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"

    torch_model = MemoryBank()

    torch_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    torch_dict = torch_dict["state_dict"]

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("memory_bank"))}

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    num_instances = 901

    track_instances = Instances((1, 1))

    ref_pts = torch.rand(num_instances, 3)
    query = torch.rand(num_instances, 512)
    output_embedding = torch.zeros(num_instances, 256)
    mem_padding_mask = torch.randint(0, 2, (901, 4), dtype=torch.bool)
    mem_bank = torch.zeros((901, 4, 256))
    obj_idxes = torch.full((num_instances,), -1, dtype=torch.long)
    scores = torch.rand(901)
    save_period = torch.rand(901)
    track_instances.ref_pts = ref_pts
    track_instances.query = query
    track_instances.output_embedding = output_embedding
    track_instances.obj_idxes = obj_idxes
    track_instances.mem_padding_mask = mem_padding_mask
    track_instances.mem_bank = mem_bank
    track_instances.scores = scores
    track_instances.save_period = save_period
    data = track_instances

    torch_output = torch_model(data)
    tt_track_instances = TtInstances((1, 1))

    tt_track_instances.ref_pts = ttnn.from_torch(ref_pts, device=device)
    tt_track_instances.query = ttnn.from_torch(query, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.output_embedding = ttnn.from_torch(output_embedding, dtype=ttnn.bfloat16, device=device)
    tt_track_instances.obj_idxes = ttnn.from_torch(obj_idxes, device=device)
    tt_track_instances.scores = ttnn.from_torch(scores, device=device)
    tt_track_instances.save_period = ttnn.from_torch(save_period, device=device, layout=ttnn.TILE_LAYOUT)
    tt_track_instances.mem_padding_mask = mem_padding_mask
    tt_track_instances.mem_bank = ttnn.from_torch(mem_bank, dtype=ttnn.bfloat16, device=device)

    data = tt_track_instances

    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)

    tt_model = TtMemoryBank(
        params=parameter,
        device=device,
    )

    ttnn_output = tt_model.forward(data)
    ttnn_output_embedding = ttnn_output.output_embedding
    ttnn_mem_bank = ttnn.to_torch(ttnn_output.mem_bank)
    ttnn_save_period = ttnn.to_torch(ttnn_output.save_period)

    assert_with_pcc(torch_output.output_embedding, ttnn_output_embedding, 0.99)
    assert_with_pcc(torch_output.mem_bank, ttnn_mem_bank, 0.99)
    assert_with_pcc(torch_output.save_period, ttnn_save_period, 0.99)
