from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from libs import tt_lib as ttm

from transformers import BloomForCausalLM
from utility_functions import print_diff_argmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger

import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.bloom_attention as bloom_attention
import python_api_testing.models.bloom.bloom_block as bloom_block


def run_bloom_block_test(device):

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    hugging_bloom_reference_model.eval()

    print(hugging_bloom_reference_model.config)

    block = 1
    hidden_size = hugging_bloom_reference_model.config.hidden_size # 1024
    n_head = hugging_bloom_reference_model.config.n_head
    layer_norm_epsilon = hugging_bloom_reference_model.config.layer_norm_epsilon
    apply_residual_connection_post_layernorm = hugging_bloom_reference_model.config.apply_residual_connection_post_layernorm
    hidden_dropout = hugging_bloom_reference_model.config.hidden_dropout
    beta = 0 # ??

    tt_bloom_block = bloom_block.TtBloomBlock(device, "transformer.h", block, hugging_bloom_reference_model, hidden_size, n_head, layer_norm_epsilon, apply_residual_connection_post_layernorm, hidden_dropout, beta)
    pt_bloom_block = hugging_bloom_reference_model.transformer.h[block]

    torch.manual_seed(0)

    hidden_states = ((torch.rand(1, 64, hidden_size) * 2) - 1) / hidden_size
    residual = ((torch.rand(1, 64, hidden_size) * 2) - 1) / hidden_size
    alibi = ((torch.rand(n_head, 64, 64) * 2) - 1) / (64 * 64)
    attention_mask = torch.randint(0, 2, (1, 1, 64, 64))

    #must be binary

    pt_out = pt_bloom_block.forward(hidden_states, alibi, attention_mask)[0]
    print("PT finished")

    tt_out = tt_bloom_block.forward(device, hidden_states, alibi, attention_mask)
    print("TT finished")

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)
    tt_out_converted = tt_out_converted.squeeze()

    print_diff_argmax(pt_out, tt_out_converted)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.94)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_block: Passed!")
    else:
        logger.warning("bloom_block: Failed!")

    assert does_pass


def test_bloom_block():
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_bloom_block_test(device)
    ttm.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_block()
