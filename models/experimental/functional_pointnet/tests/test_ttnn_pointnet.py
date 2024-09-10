# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from torch.autograd import Variable
from models.experimental.functional_pointnet.reference.PointNetDenseCls import PointNetDenseCls
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_pointnet_model(device, reset_seeds):
    input = torch.randn(32, 3, 2500, requires_grad=True)
    reference_model = PointNetDenseCls(k=3)

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    ds_state_dict = {k: v for k, v in reference_model.state_dict().items()}
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    output, _, _ = reference_model(input)
