import torch
import pytest
from loguru import logger

import tt_lib
from tests.python_api_testing.models.falcon.reference.hf_falcon_model import (
    RWForCausalLM,
)
from tests.python_api_testing.models.falcon.falcon_mlp import TtFalconMLP

from tests.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class PytorchFalconMLPModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.mlp = hf_reference_model.transformer.h[layer_num].mlp

        # Disable dropout
        self.mlp.eval()

    def forward(self, x):
        result = self.mlp(x)
        return result


def run_test_FalconMLP_inference(device, model_version, batch, seq_len, on_weka, pcc):
    # torch.bfloat16 input hangs in first Linear for MLP in hf reference model
    hugging_face_reference_model = RWForCausalLM.from_pretrained(model_version)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    mlp_input = (torch.rand(batch, 1, seq_len, configuration.hidden_size) * 2) - 1
    layer_num = 0
    base_url = "transformer.h"

    # PyTorch output --------------------------------------------------------------------
    pytorch_FalconMLP_model = PytorchFalconMLPModel(
        hugging_face_reference_model, layer_num
    )
    pytorch_out = pytorch_FalconMLP_model(mlp_input)

    # TT hardware execution -------------------------------------------------------------
    tt_FalconMLP_model = TtFalconMLP(
        device,
        state_dict,
        base_url,
        layer_num,
        configuration.hidden_size,
    )

    tt_mlp_input = torch2tt_tensor(mlp_input, device)

    tt_out = tt_FalconMLP_model(tt_mlp_input)
    tt_out = tt2torch_tensor(tt_out)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon MLP output Passed!")
    else:
        logger.warning("Falcon MLP output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, batch, seq_len, on_weka, pcc",
    (
        (
            "tiiuae/falcon-7b-instruct",
            1,
            128,
            False,
            0.98,
        ),
    ),
)
def test_FalconMLP_inference(model_version, batch, seq_len, on_weka, pcc):
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    run_test_FalconMLP_inference(device, model_version, batch, seq_len, on_weka, pcc)
    tt_lib.device.CloseDevice(device)
