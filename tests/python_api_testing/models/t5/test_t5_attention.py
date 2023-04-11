from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import json
from libs import tt_lib as ttm

from transformers import T5Model
from utility_functions import print_diff_argmax, comp_allclose, comp_pcc

from loguru import logger
from python_api_testing.fused_ops.softmax import softmax as tt_softmax
from python_api_testing.models.t5.t5_utils import torch2tt_tensor, tt2torch_tensor
from python_api_testing.models.t5.t5_attention import TtT5Attention, t5_unshape_pt, t5_unshape_tt, t5_shape_tt, t5_shape_pt



def run_test_t5_shape(device):

    batch_size = 32
    n_heads = 8
    key_value_proj_dim = 64 #

    torch.manual_seed(0)
    test_input = (torch.rand(1, 1, 2048, 512) * 2) - 1

    pt_out = t5_shape_pt(test_input, batch_size, n_heads, key_value_proj_dim)
    tt_out = t5_shape_tt(torch2tt_tensor(test_input, device), batch_size, n_heads, key_value_proj_dim, device)
    tt_out = tt2torch_tensor(tt_out)

    # print(tt_out.shape)
    # print(pt_out.shape)
    assert(tt_out.shape == pt_out.shape)

    # print(pt_out[0, 0, 1:10, 1:10])
    # print(tt_out[0, 0, 1:10, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("t5_shape test Passed!")
    else:
        logger.warning("t5_shape test Failed!")


def run_test_t5_unshape(device):
    torch.manual_seed(0)
    test_input = (torch.rand(32, 8, 128, 64) * 2) - 1

    batch_size = 32
    inner_dim = 512

    pt_out = t5_unshape_pt(test_input, batch_size, inner_dim)
    tt_out = t5_unshape_tt(torch2tt_tensor(test_input, device), batch_size, inner_dim, device)
    tt_out = tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Test unshape Passed!")
    else:
        logger.warning("Test unshape Failed!")


def run_test_transpose(device):
    torch.manual_seed(0)
    test_input = (torch.rand(1, 1, 2048, 512) * 2) - 1

    pt_out = test_input.transpose(3, 2)
    tt_out = ttm.tensor.transpose(torch2tt_tensor(test_input, device))
    tt_out = tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Test traspose Passed!")
    else:
        logger.warning("Test traspose Failed!")


def run_test_matmul(device):
    torch.manual_seed(0)
    test_input1 = ((torch.rand(32, 8, 128, 64) * 2) - 1)
    test_input2 = ((torch.rand(32, 8, 64, 128) * 2) - 1)

    pt_out = torch.matmul(test_input1, test_input2)
    tt_out = ttm.tensor.bmm(torch2tt_tensor(test_input1, device), torch2tt_tensor(test_input2, device))
    tt_out = tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Test matmul Passed!")
    else:
        logger.warning("Test matmul Failed!")


def run_test_softmax(device):
    torch.manual_seed(0)
    test_input  = ((torch.rand(32, 8, 128, 128) * 2) - 1)

    pt_out = torch.nn.functional.softmax(test_input.float(), dim=-1).type_as(test_input)

    tt_out = tt_softmax(torch2tt_tensor(test_input, device))
    tt_out = tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Test softmax Passed!")
    else:
        logger.warning("Test softmax Failed!")


def run_test_T5Attention_inference(device, block, use_mask):
    hugging_face_reference_model = T5Model.from_pretrained("t5-small") #, torch_dtype=torch.float16)
    hugging_face_reference_model.eval()

    # Input is (batch_size, seq_length, dim)
    torch.manual_seed(0)
    test_input = ((torch.rand(1, 32, 512) * 2) - 1) / 512

    # "/ 2" is added beacuse of Tt device precision. Not to hit limits of float16
    mask = -65504.0 * torch.cat([torch.zeros(7), torch.ones(25)]) / 2
    mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    mask = mask if use_mask else None

    config = json.loads(hugging_face_reference_model.config.to_json_string())
    config["is_decoder"] = False
    has_relative_attention_bias = bool(block == 0)

    # Module to test
    if config["is_decoder"]:
        hf_reference_module = hugging_face_reference_model.decoder.block[block].layer[0].SelfAttention
        base_address = f"decoder.block.{block}.layer.0.SelfAttention"
    else:
        hf_reference_module = hugging_face_reference_model.encoder.block[block].layer[0].SelfAttention
        base_address = f"encoder.block.{block}.layer.0.SelfAttention"

    pytorch_model = hf_reference_module
    # pytorch_model = T5Attention(config, hf_reference_module)
    pt_out = pytorch_model(hidden_states=test_input, mask=mask)[0].unsqueeze(0)

    test_input = test_input.unsqueeze(0)
    tt_test_input = torch2tt_tensor(test_input, device)

    tt_model = TtT5Attention(config, hugging_face_reference_model.state_dict(), base_address, device, has_relative_attention_bias)
    tt_out = tt_model(hidden_states=tt_test_input, mask=mask)[0]
    tt_out = tt2torch_tensor(tt_out)

    print(pt_out[0, 0, 1:10, 1:10])
    print(tt_out[0, 0, 1:10, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("test_T5Attention_inference Passed!")
    else:
        logger.warning("test_T5Attention_inference Failed!")


def test_t5_shape():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_test_t5_shape(device)
    ttm.device.CloseDevice(device)


def test_transpose():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_test_transpose(device)
    ttm.device.CloseDevice(device)


def test_matmul():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_test_matmul(device)
    ttm.device.CloseDevice(device)


def test_softmax():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_test_softmax(device)
    ttm.device.CloseDevice(device)


def test_t5_unshape():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_test_t5_unshape(device)
    ttm.device.CloseDevice(device)


def test_T5Attention_block_0_no_mask():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_test_T5Attention_inference(device, block=0, use_mask=False)
    ttm.device.CloseDevice(device)


def test_T5Attention_block_2_no_mask():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_test_T5Attention_inference(device, block=2, use_mask=False)
    ttm.device.CloseDevice(device)


def test_T5Attention_block_0_with_mask():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_test_T5Attention_inference(device, block=0, use_mask=True)
    ttm.device.CloseDevice(device)


if __name__ == "__main__":
    test_T5Attention_block_0_with_mask()
