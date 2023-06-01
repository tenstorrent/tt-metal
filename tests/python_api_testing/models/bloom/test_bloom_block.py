from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib

from transformers import BloomForCausalLM
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from loguru import logger
import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.bloom_block as bloom_block


def run_bloom_block_test(device):
    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-560m", torchscript=False
    )
    hugging_bloom_reference_model.eval()

    do_all_blocks_pass = True
    min_pcc = 1.0

    for block in range(24):
        config = hugging_bloom_reference_model.config
        state_dict = hugging_bloom_reference_model.state_dict()
        base_address = f"transformer.h.{block}"
        hidden_size = config.hidden_size
        n_head = config.n_head

        tt_bloom_block = bloom_block.TtBloomBlock(
            config, state_dict, base_address, device
        )
        pt_bloom_block = hugging_bloom_reference_model.transformer.h[block]

        torch.manual_seed(0)
        seq_len = 62

        hidden_states = ((torch.rand(1, seq_len, hidden_size) * 2) - 1) / hidden_size
        alibi = ((torch.rand(n_head, seq_len, seq_len) * 2) - 1) / (seq_len * seq_len)
        attention_mask = torch.randint(0, 2, (1, 1, seq_len, seq_len))

        pt_out = pt_bloom_block.forward(hidden_states, alibi, attention_mask)[0]

        hidden_states = bloom_utils.torch2tt_tensor(hidden_states, device)
        alibi = bloom_utils.torch2tt_tensor(alibi, device)

        tt_out = tt_bloom_block.forward(device, hidden_states, alibi, attention_mask)[0]

        tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)
        tt_out_converted = tt_out_converted.squeeze()

        does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.98)
        logger.info(pcc_message)

        if does_pass:
            logger.info(f"bloom_block {block}: Passed!")
        else:
            do_all_blocks_pass = False
            logger.warning(f"bloom_block {block}: Failed!")

    assert do_all_blocks_pass


def test_bloom_block():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_bloom_block_test(device)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_block()
