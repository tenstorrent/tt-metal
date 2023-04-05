from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import copy
import torch
from torch import nn
from libs import tt_lib as ttm
from loguru import logger

from transformers import AutoTokenizer, T5Tokenizer, T5Model
from utility_functions import print_diff_argmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.t5.t5_utils import torch2tt_tensor, tt2torch_tensor, read_model_config
from python_api_testing.models.t5.t5_model import TtT5Model



def run_test_T5Model_inference(device):
    #tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=32)
    hf_reference_model = T5Model.from_pretrained("t5-small")
    hf_reference_model.eval()

    # T5-small config file: https://huggingface.co/t5-small/resolve/main/config.json
    model_json_config = "tests/python_api_testing/models/t5/t5-small.json"
    config = read_model_config(model_json_config)

    # Prepare input
    input_sentance = "Studies have been shown that owning a dog is good for you"
    input_ids = tokenizer(input_sentance, padding="max_length", max_length=32, return_tensors="pt").input_ids  # Batch size 1

    decoder_input_sentence = "Studies show that"
    decoder_input_ids = tokenizer(decoder_input_sentence, padding="max_length", max_length=32, return_tensors="pt").input_ids  # Batch size 1

    # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
    # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
    decoder_input_ids = hf_reference_model._shift_right(decoder_input_ids)

    # PyTorch forward pass
    pt_out = hf_reference_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    pt_out = pt_out.last_hidden_state
    pt_out = pt_out.unsqueeze(0)

    hf_reference_model = T5Model.from_pretrained("t5-small", torch_dtype=torch.float16)
    hf_reference_model.eval()

    tt_model = TtT5Model(config, hf_reference_model.state_dict(), device)
    tt_model_outputs = tt_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    tt_out = tt2torch_tensor(tt_model_outputs[0])

    print(pt_out[0, 0, 0:3, 1:10])
    print(tt_out[0, 0, 0:3, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    print("Pt decoded output:")
    print(tokenizer.decode(pt_out[0][0].softmax(0).argmax(1)))

    print("Tt decoded output:")
    print(tokenizer.decode(tt_out[0][0].softmax(0).argmax(1)))

    if does_pass:
        logger.info("test_T5Model_inference Passed!")
    else:
        logger.warning("test_T5Model_inference Failed!")


def test_T5Model_inference():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_test_T5Model_inference(device)
    ttm.device.CloseDevice(device)
