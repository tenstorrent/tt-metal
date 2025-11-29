# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from tests.ttnn.utils_for_testing import comp_pcc

from models.experimental.pointpillars.common import load_torch_model


def test_reference(model_location_generator, reset_seeds):
    reference_model = load_torch_model(model_location_generator)
    batch_inputs_dict = torch.load(
        "models/experimental/pointpillars/inputs_weights/batch_inputs_dict_orig.pth", weights_only=False
    )
    batch_data_samples_modified = torch.load(
        "models/experimental/pointpillars/inputs_weights/batch_data_samples_orig.pth", weights_only=False
    )

    output = reference_model(batch_inputs_dict, batch_data_samples_modified)

    for i, out_list in enumerate(output):
        for j, tensor in enumerate(out_list):
            orig = torch.load(f"/home/ubuntu/harini_pointpillars/mmdetection3d/outs_{i}_{j}.pt")
            print(comp_pcc(orig, tensor, 1.0))
