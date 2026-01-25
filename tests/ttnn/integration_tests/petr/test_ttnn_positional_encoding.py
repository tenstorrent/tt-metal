# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from models.experimental.functional_petr.reference.positional_encoding import SinePositionalEncoding3D
from models.experimental.functional_petr.tt.ttnn_positional_encoding import ttnn_SinePositionalEncoding3D
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_SinePositionalEncoding3D(device):
    input = torch.randint(0, 2, (1, 6, 20, 50))
    ttnn_input = ttnn.from_torch(input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    torch_model = SinePositionalEncoding3D(num_feats=128, normalize=True)
    ttnn_model = ttnn_SinePositionalEncoding3D(num_feats=128, normalize=True)

    output = torch_model(input)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(output, ttnn_output, pcc=0.99)  # 0.6691
