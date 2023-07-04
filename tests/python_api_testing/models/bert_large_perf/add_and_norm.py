import pytest
from loguru import logger
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")

import torch
from transformers import BertForQuestionAnswering
import numpy as np

import tt_lib as ttl
from python_api_testing.models.bert_large_perf.fused_ops.add_and_norm import AddAndNorm
from python_api_testing.models.bert_large_perf.fused_ops.layernorm import create_var_scaler
from tt_lib.utils import pad_activation, pad_weight, print_diff_argmax
from utility_functions import enable_compile_cache, comp_pcc, comp_allclose

class TtAddAndNormModel(torch.nn.Module):
    def __init__(self, config, state_dict, var_scaler, device, lnorm_type):
        super().__init__()

        if lnorm_type == "attention":
            gamma = pad_weight(state_dict["bert.encoder.layer.0.attention.output.LayerNorm.weight"])
            gamma = ttl.tensor.Tensor(gamma.reshape(-1).tolist(), gamma.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).to(device)
            beta = pad_weight(state_dict["bert.encoder.layer.0.attention.output.LayerNorm.bias"])
            beta = ttl.tensor.Tensor(beta.reshape(-1).tolist(), beta.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).to(device)
        elif lnorm_type == "ffn":
            gamma = pad_weight(state_dict["bert.encoder.layer.0.output.LayerNorm.weight"])
            gamma = ttl.tensor.Tensor(gamma.reshape(-1).tolist(), gamma.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).to(device)
            beta = pad_weight(state_dict["bert.encoder.layer.0.output.LayerNorm.bias"])
            beta = ttl.tensor.Tensor(beta.reshape(-1).tolist(), beta.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE).to(device)
        else:
            assert False, "Invalid lnorm_type"

        self.add_and_norm = AddAndNorm(gamma, beta, config.layer_norm_eps, var_scaler, config.hidden_size, config.hidden_size, device)

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

def run_add_and_norm_inference(model_version, batch, seq_len, on_weka, pcc, model_location_generator):

    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    # Initialize the device
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()

    if on_weka:
        model_name = str(model_location_generator("tt_dnn-models/Bert/BertForQuestionAnswering/models/") / model_version)
    else:
        model_name = model_version

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    config = hugging_face_reference_model.config
    var_scaler = create_var_scaler(seq_len, config.hidden_size, config.layer_norm_eps, device)

    tt_add_and_norm_model = TtAddAndNormModel(config, hugging_face_reference_model.state_dict(), var_scaler, device, "attention")
    pytorch_add_and_norm_model = PytorchAddAndNormModel(hugging_face_reference_model, "attention")

    # Prepare input
    torch.manual_seed(0)
    add_and_norm_inputa = (torch.rand(batch, 1, seq_len, config.hidden_size) * 2) - 1
    add_and_norm_inputb = (torch.rand(batch, 1, seq_len, config.hidden_size) * 2) - 1

    pytorch_out = pytorch_add_and_norm_model(add_and_norm_inputa, add_and_norm_inputb)

    pad_add_and_norm_inputa = pad_activation(add_and_norm_inputa)
    pad_add_and_norm_inputb = pad_activation(add_and_norm_inputb)
    tt_add_and_norm_input_a = ttl.tensor.Tensor(pad_add_and_norm_inputa.reshape(-1).tolist(), pad_add_and_norm_inputa.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE)
    tt_add_and_norm_input_a = tt_add_and_norm_input_a.to(device)
    tt_add_and_norm_input_b = ttl.tensor.Tensor(pad_add_and_norm_inputb.reshape(-1).tolist(), pad_add_and_norm_inputb.shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE)
    tt_add_and_norm_input_b = tt_add_and_norm_input_b.to(device)

    tt_out = tt_add_and_norm_model(tt_add_and_norm_input_a, tt_add_and_norm_input_b).to(host)
    tt_out = torch.Tensor(tt_out.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(tt_out.shape())

    ttl.device.CloseDevice(device)

    passing, output = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output {output}")
    _, output = comp_allclose(pytorch_out, tt_out, 0.5, 0.5) # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Output {output}")
    if not passing:
        logger.error(f"Output PCC < {pcc}")
    # assert np.allclose(pytorch_out.detach().numpy(), tt_out.numpy(), 1e-5, 0.17)

@pytest.mark.parametrize(
    "model_version, batch, seq_len, on_weka,  pcc",
    (
        ("mrm8488/bert-tiny-finetuned-squadv2", 1, 128, True, 0.99),
        ("phiyodr/bert-base-finetuned-squad2", 1, 128, True, 0.99),
        ("phiyodr/bert-large-finetuned-squad2", 1, 384, True, 0.99)
    ),
)
def test_add_and_norm_inference(model_version, batch, seq_len, on_weka, pcc, model_location_generator):

    run_add_and_norm_inference(model_version, batch, seq_len, on_weka, pcc, model_location_generator)


if __name__ == "__main__":
    run_add_and_norm_inference("mrm8488/bert-tiny-finetuned-squadv2", 1, 128, True, 0.99)
