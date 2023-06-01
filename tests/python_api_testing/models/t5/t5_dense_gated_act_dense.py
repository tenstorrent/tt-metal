from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import numpy as np
from torch import nn
import tt_lib
from loguru import logger

from transformers import T5Model
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from tt_lib.utils import print_diff_argmax
from python_api_testing.models.t5.t5_utils import torch2tt_tensor, tt2torch_tensor
from tt_lib.fused_ops.linear import Linear as TtLinear


class TtT5DenseGatedActDense(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        enc_dec = "decoder" if config["is_decoder"] else "encoder"
        state_dict = hugging_face_reference_model.state_dict()

        self.wi_0_weights = torch2tt_tensor(
            state_dict[f"{base_address}.wi_0.weight"], tt_lib.device.GetHost()
        )
        self.wi_1_weights = torch2tt_tensor(
            state_dict[f"{base_address}.wi_1.weight"], tt_lib.device.GetHost()
        )
        self.wo_weights = torch2tt_tensor(
            state_dict[f"{base_address}.wo.weight"], tt_lib.device.GetHost()
        )

        self.wi_0 = TtLinear(
            config.d_model,
            config.d_ff,
            weight=self.wi_0_weights.data(),
            bias=None,
            device=device,
        )
        self.wi_1 = TtLinear(
            config.d_model,
            config.d_ff,
            weight=self.wi_1_weights.data(),
            bias=None,
            device=device,
        )
        self.wo = TtLinear(
            config.d_ff,
            config.d_model,
            weight=self.wo_weights.data(),
            bias=None,
            device=device,
        )

        # self.dropout = nn.Dropout(config["dropout_rate"])
        self.act = tt_lib.tensor.relu

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        # if (
        #    isinstance(self.wo.weight, torch.Tensor)
        #    and hidden_states.dtype != self.wo.weight.dtype
        #    and self.wo.weight.dtype != torch.int8
        # ):
        #    hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


def test_T5DenseGatedActDense_inference(device):
    hugging_face_reference_model = T5Model.from_pretrained("t5-small")
    hugging_face_reference_model.eval()

    config = json.loads(hugging_face_reference_model.config.to_json_string())
    config["is_decoder"] = True

    if config["is_decoder"]:
        hf_reference_module = (
            hugging_face_reference_model.decoder.block[0].layer[2].DenseReluDense
        )
        base_address = f"decoder.block.0.layer.2.DenseReluDense"
    else:
        hf_reference_module = (
            hugging_face_reference_model.encoder.block[0].layer[1].DenseReluDense
        )
        base_address = f"encoder.block.0.layer.1.DenseReluDense"

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(1, 1, 2048, 512) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input)[0].unsqueeze(1)

    tt_model = TtT5DenseGatedActDense(
        config, hugging_face_reference_model.state_dict(), base_address, device
    )
    tt_out = tt_model(torch2tt_tensor(test_input, device))
    tt_out = tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    if does_pass:
        logger.info("test_T5DenseGatedActDense_inference Passed!")
    else:
        logger.warning("test_T5DenseGatedActDense_inference Failed!")


if __name__ == "__main__":
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    host = tt_lib.device.GetHost()
    test_T5DenseGatedActDense_inference(device)
    tt_lib.device.CloseDevice(device)
