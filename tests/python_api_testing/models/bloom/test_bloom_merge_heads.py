from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import tt_lib
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.bloom_merge_heads as bloom_attention_merge_heads


def run_bloom_merge_heads_test(device, num_heads, hidden_size, num_attention_heads):
    torch.manual_seed(0)
    test_in = torch.rand(4096, 128, 32)

    pt_out = bloom_attention_merge_heads.merge_heads(
        test_in, num_heads, hidden_size, num_attention_heads
    )
    tt_out = bloom_attention_merge_heads.tt_merge_heads(
        test_in, num_heads, hidden_size, num_attention_heads, device
    )

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("bloom_attention_merge_heads: Passed!")
    else:
        logger.warning("bloom_attention_merge_heads: Failed!")

    assert does_pass


def test_bloom_merge_heads():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_bloom_merge_heads_test(device, 32, 1024, 32)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_bloom_merge_heads()
