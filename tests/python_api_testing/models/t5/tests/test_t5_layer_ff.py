import torch
import json
import tt_lib
from loguru import logger

from transformers import T5Model
from tt_models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from tt_models.utility_functions import comp_pcc
from tt_models.t5.tt.t5_layer_ff import TtT5LayerFF


def run_test_T5LayerFF_inference(device, model_name, input_h, input_w):
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())
    config["is_decoder"] = False

    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder.block[0].layer[2]
        base_address = f"decoder.block.0.layer.2"
    else:
        hf_reference_module = hf_reference_model.encoder.block[0].layer[1]
        base_address = f"encoder.block.0.layer.1"

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(1, 1, input_h, input_w) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input)[0].unsqueeze(1)

    # T5-small config file: https://huggingface.co/t5-small/resolve/main/config.json
    tt_model = TtT5LayerFF(
        config, hf_reference_model.state_dict(), base_address, device
    )
    tt_out = tt_model(torch2tt_tensor(test_input, device))
    tt_out = tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)
    logger.info(pcc_message)

    if does_pass:
        logger.info(f"test_T5LayerFF_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5LayerFF_inference {model_name} Failed!")

    assert does_pass


def test_T5LayerFF_inference_t5_small():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_test_T5LayerFF_inference(device, "t5-small", 64, 512)
    tt_lib.device.CloseDevice(device)


def test_T5LayerFF_inference_flan_t5_small():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_test_T5LayerFF_inference(device, "google/flan-t5-small", 64, 512)
    tt_lib.device.CloseDevice(device)


def test_T5LayerFF_inference_t5_base():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_test_T5LayerFF_inference(device, "t5-base", 64, 768)
    tt_lib.device.CloseDevice(device)
