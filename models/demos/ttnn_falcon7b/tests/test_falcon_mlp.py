# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

from models.demos.ttnn_falcon7b.tt.falcon_mlp import TtFalconMLP
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config, get_tt_cache_path
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
import transformers

torch.manual_seed(0)


def run_test_FalconMLP_inference(
    device,
    model_name,
    batch,
    seq_len,
    pcc,
    model_config,
):
    configuration = transformers.FalconConfig.from_pretrained(model_name)
    mlp_input = (torch.rand(batch, 1, seq_len, configuration.hidden_size) * 2) - 1
    model = transformers.models.falcon.modeling_falcon.FalconMLP(configuration).eval()
    torch_output = model(mlp_input)
    tt_cache_path = get_tt_cache_path(model_name)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=create_custom_preprocessor(model_config, tt_cache_path=tt_cache_path, device=device),
    )
    tt_FalconMLP_model = TtFalconMLP(
        device,
        model_config,
        parameters,
    )

    tt_mlp_input = ttnn.from_torch(
        mlp_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
    )

    tt_out = tt_FalconMLP_model(tt_mlp_input)
    tt_out = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_output, tt_out.to(torch_output.dtype), pcc)


@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        (
            "tiiuae/falcon-7b-instruct",
            1,
            128,
            0.98,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_FalconMLP_inference(
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    device,
):
    model_config = get_model_config(model_config_str)

    run_test_FalconMLP_inference(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
    )
