from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch
from transformers import BertForQuestionAnswering
import numpy as np

from gpai import gpai
from python_api_testing.fused_ops.add_and_norm import AddAndNorm
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, print_diff_argmax

class TtAddAndNormModel(torch.nn.Module):
    def __init__(self, state_dict, device, lnorm_type):
        super().__init__()

        if lnorm_type == "attention":
            gamma = tilize_to_list(pad_weight(state_dict["bert.encoder.layer.0.attention.output.LayerNorm.weight"]))
            beta = tilize_to_list(pad_weight(state_dict["bert.encoder.layer.0.attention.output.LayerNorm.bias"]))
        elif lnorm_type == "ffn":
            gamma = tilize_to_list(pad_weight(state_dict["bert.encoder.layer.0.output.LayerNorm.weight"]))
            beta = tilize_to_list(pad_weight(state_dict["bert.encoder.layer.0.output.LayerNorm.bias"]))
        else:
            assert False, "Invalid lnorm_type"

        self.add_and_norm = AddAndNorm(gamma, beta, 1e-12, 128, 128, device)

    def forward(self, a, b):
        return self.add_and_norm(a, b)

class PytorchAddAndNormModel(torch.nn.Module):
    def __init__(self, hugging_face_reference_model, lnorm_type):
        super().__init__()
        if lnorm_type == "attention":
            self.layernorm = hugging_face_reference_model.bert.encoder.layer[0].attention.output.LayerNorm
        elif lnorm_type == "ffn":
            self.layernorm = hugging_face_reference_model.bert.encoder.layer[0].output.LayerNorm
        else:
            assert False, "Invalid lnorm_type"

    def forward(self, a, b):
        out = self.layernorm(a + b)
        return out

def run_add_and_norm_inference():
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained("prajjwal1/bert-tiny", torchscript=False)
    tt_add_and_norm_model = TtAddAndNormModel(hugging_face_reference_model.state_dict(), device, "attention")
    pytorch_add_and_norm_model = PytorchAddAndNormModel(hugging_face_reference_model, "attention")

    # Prepare input
    torch.manual_seed(0)
    add_and_norm_inputa = (torch.rand(1, 1, 128, 128) * 2) - 1
    add_and_norm_inputb = (torch.rand(1, 1, 128, 128) * 2) - 1

    pytorch_out = pytorch_add_and_norm_model(add_and_norm_inputa, add_and_norm_inputb)

    tt_add_and_norm_input_a = tilize_to_list(pad_activation(add_and_norm_inputa))
    tt_add_and_norm_input_a = gpai.tensor.Tensor(tt_add_and_norm_input_a, add_and_norm_inputa.shape, gpai.tensor.DataFormat.FLOAT32,  gpai.tensor.Layout.TILE, device)
    tt_add_and_norm_input_b = tilize_to_list(pad_activation(add_and_norm_inputb))
    tt_add_and_norm_input_b = gpai.tensor.Tensor(tt_add_and_norm_input_b, add_and_norm_inputb.shape, gpai.tensor.DataFormat.FLOAT32,  gpai.tensor.Layout.TILE, device)

    tt_out = tt_add_and_norm_model(tt_add_and_norm_input_a, tt_add_and_norm_input_b).to(host)
    tt_out = untilize(torch.Tensor(tt_out.data()).reshape(*pytorch_out.shape))
    assert np.allclose(pytorch_out.detach().numpy(), tt_out.numpy(), 1e-5, 0.17)

if __name__ == "__main__":
    # Initialize the device
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()
    run_add_and_norm_inference()
    gpai.device.CloseDevice(device)
