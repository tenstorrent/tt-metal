# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from models.experimental.detr3d.reference.model_3detr import GenericMLP as ref_model
from models.experimental.detr3d.source.detr3d.models.helpers import GenericMLP as org_model
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_dim,hidden_dims,output_dim,norm_fn_name,activation,use_conv,"
    "hidden_use_bias,output_use_bias,output_use_activation,output_use_norm,"
    "weight_init_name,dropout,x_shape",
    [
        (256, [256], 256, "bn1d", "relu", True, False, False, True, True, None, None, (1, 256, 1024)),
        (256, [256], 256, None, "relu", True, True, True, True, False, None, None, (1, 256, 128)),
        (256, [256, 256], 12, "bn1d", "relu", True, False, True, False, False, None, None, (8, 256, 128)),
    ],
)
def test_generic_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    norm_fn_name,
    activation,
    use_conv,
    hidden_use_bias,
    output_use_bias,
    output_use_activation,
    output_use_norm,
    weight_init_name,
    dropout,
    x_shape,
):
    print("nefwfubr", output_use_norm)
    org_module = org_model(
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name,
        activation,
        use_conv,
        dropout,
        hidden_use_bias,
        output_use_bias,
        output_use_activation,
        output_use_norm,
        weight_init_name,
    ).to(torch.bfloat16)
    org_module.eval()
    ref_module = ref_model(
        input_dim,
        hidden_dims,
        output_dim,
        norm_fn_name,
        activation,
        use_conv,
        dropout,
        hidden_use_bias,
        output_use_bias,
        output_use_activation,
        output_use_norm,
        weight_init_name,
    ).to(torch.bfloat16)
    ref_module.eval()
    ref_module.load_state_dict(org_module.state_dict())
    x = torch.randn(x_shape, dtype=torch.bfloat16)
    org_out = org_module(x)
    ref_out = ref_module(x)
    assert_with_pcc(org_out, ref_out, 1.0)
