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
from libs import tt_lib as ttm
from loguru import logger

from transformers import T5Model
from utility_functions import print_diff_argmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.t5.t5_utils import torch2tt_tensor, tt2torch_tensor, read_model_config, print_corr_coef
from python_api_testing.models.t5.test_t5_layer_norm import TtT5LayerNorm
from python_api_testing.models.t5.test_t5_dense_act_dense import TtT5DenseActDense
from python_api_testing.models.t5.t5_dense_gated_act_dense import TtT5DenseGatedActDense


# class T5LayerFF(nn.Module):
#    def __init__(self, config: T5Config):
#        super().__init__()
#        if config.is_gated_act:
#            self.DenseReluDense = T5DenseGatedActDense(config)
#        else:
#            self.DenseReluDense = T5DenseActDense(config)

#        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
#        self.dropout = nn.Dropout(config.dropout_rate)

#    def forward(self, hidden_states):
#        forwarded_states = self.layer_norm(hidden_states)
#        forwarded_states = self.DenseReluDense(forwarded_states)
#        hidden_states = hidden_states + self.dropout(forwarded_states)
#        return hidden_states


class TtT5LayerFF(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        if "is_gated_act" in config and config["is_gated_act"]:
            self.DenseReluDense = TtT5DenseGatedActDense(config, state_dict, f"{base_address}.DenseReluDense", device)
        else:
            self.DenseReluDense = TtT5DenseActDense(config, state_dict, f"{base_address}.DenseReluDense", device)

        self.layer_norm = TtT5LayerNorm(config, state_dict, f"{base_address}.layer_norm", device)
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        #hidden_states = hidden_states + self.dropout(forwarded_states)
        hidden_states = ttm.tensor.add(hidden_states, forwarded_states)
        return hidden_states


def run_test_T5LayerFF_inference(device):
    hf_reference_model = T5Model.from_pretrained("t5-small")
    hf_reference_model.eval()

    model_json_config = "tests/python_api_testing/models/t5/t5-small.json"
    config = read_model_config(model_json_config)

    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder.block[0].layer[2]
        base_address = f"decoder.block.0.layer.2"
    else:
        hf_reference_module = hf_reference_model.encoder.block[0].layer[1]
        base_address = f"encoder.block.0.layer.1"

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(1, 1, 2048, 512) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input)[0].unsqueeze(1)

    # T5-small config file: https://huggingface.co/t5-small/resolve/main/config.json
    tt_model = TtT5LayerFF(config, hf_reference_model.state_dict(), base_address, device)
    tt_out = tt_model(torch2tt_tensor(test_input, device))
    tt_out = tt2torch_tensor(tt_out)

    print(pt_out[0, 0, 1:10, 1:10])
    print(tt_out[0, 0, 1:10, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("test_T5LayerFF_inference Passed!")
    else:
        logger.warning("test_T5LayerFF_inference Failed!")


def test_T5LayerFF_inference():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_test_T5LayerFF_inference(device)
    ttm.device.CloseDevice(device)
