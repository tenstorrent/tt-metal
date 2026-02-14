# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger

from ttnn.device import is_wormhole_b0

from models.experimental.functional_mole.tt import mole_torch, mole_ttnn
from models.experimental.functional_mole.tt.model_preprocessing import create_mole_input_tensors

from models.experimental.functional_mole.tests.common import (
    verify_with_pcc,
    verify_with_errlimit,
    MOLE_L1_SMALL_REGION_SIZE,
    MOLE_FULL_MODEL_PCC,
    MOLE_RTOL_LIMIT,
    MOLE_ATOL_LIMIT,
)

model_torch_dict = {"dlinear": mole_torch.Mole, "rmlp": mole_torch.Rmlp, "rlinear": mole_torch.Rlinear}

model_ttnn_dict = {
    "dlinear": mole_ttnn.Mole,
    "rmlp": mole_ttnn.Rmlp,
    "rlinear": mole_ttnn.Rlinear,
}


def run_mole_model(seq_len, pred_len, t_dim, enc_in, batch_size, model_name, device):
    logger.info("enter mole_model.")
    "------------- internal model configs(with fixed value) ---------------"
    kernel_size = 25
    stride = 1
    time_features = 4

    torch_dtype = torch.float32
    ttnn_dtype = ttnn.float32

    "------------- create input tensors ---------------"
    torch_input_tensor, torch_input_mark_tensor, ttnn_input_tensor, ttnn_input_mark_tensor = create_mole_input_tensors(
        batch_size=batch_size,
        seq_len=seq_len,
        enc_in=enc_in,
        time_features=time_features,
        torch_dtype=torch_dtype,
        ttnn_dtype=ttnn_dtype,
        device=device,
    )
    # print("ttnn_input_tensor: {}\nttnn_input_mark_tensor: {}".format(ttnn_input_tensor, ttnn_input_mark_tensor))

    "------------- define the mole torch model ---------------"
    mole_torch_model = model_torch_dict[model_name].from_random_weights(
        seq_len, pred_len, enc_in, t_dim, time_features, kernel_size, stride
    )
    logger.info("mole_torch_model: {}", model_name)

    "------------- run the mole torch model ---------------"
    output_torch_tensor = None
    with torch.no_grad():
        output_torch_tensor = mole_torch_model(torch_input_tensor, torch_input_mark_tensor)
    # print("output_torch_tensor: ", output_torch_tensor.shape)

    "------------- define the mole ttnn model ---------------"
    mole_ttnn_model = model_ttnn_dict[model_name](seq_len, pred_len, enc_in, t_dim, kernel_size, stride, device)

    "------------- initialize the weights/bias ---------------"
    mole_ttnn_model.set_parameters(mole_torch_model)

    "------------- run the mole ttnn model ---------------"
    output_ttnn_tensor = mole_ttnn_model(ttnn_input_tensor, ttnn_input_mark_tensor)
    # print("output_ttnn_tensor: ", output_ttnn_tensor.shape)
    output_ttnn_tensor = ttnn.to_torch(output_ttnn_tensor)

    "------------- verify the results ---------------"
    verify_with_pcc(output_torch_tensor, output_ttnn_tensor, MOLE_FULL_MODEL_PCC)

    # verify_with_errlimit(output_torch_tensor, output_ttnn_tensor, MOLE_RTOL_LIMIT, MOLE_ATOL_LIMIT)

    logger.info("end of mole_model.")
    return


@pytest.mark.parametrize("seq_len", [336])
@pytest.mark.parametrize("pred_len", [96, 192, 336, 720])
@pytest.mark.parametrize("t_dim", [1, 2, 4, 8])
@pytest.mark.parametrize("enc_in", [321])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("model_name", ["dlinear", "rmlp", "rlinear"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": MOLE_L1_SMALL_REGION_SIZE}], indirect=True)
def test_mole_model(seq_len, pred_len, t_dim, enc_in, batch_size, model_name, device, reset_seeds):
    if not is_wormhole_b0(device):
        pytest.skip(f"mole only support wormhole platform!")

    run_mole_model(seq_len, pred_len, t_dim, enc_in, batch_size, model_name, device)
