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
import python_api_testing.models.bloom.baddbmm as baddbmm


def run_baddbmm_test(device):
    torch.manual_seed(0)

    dim1 = 32
    dim2 = 62

    input = torch.rand(dim1, dim2, dim2)
    batch1 = torch.rand(dim1, dim2, dim1)
    batch2 = torch.rand(dim1, dim1, dim2)

    alpha = 0.25
    beta = 0.5
    tt_alpha = bloom_utils.tt_const_tensor(alpha, [1, dim1, dim2, dim2], device)
    tt_beta = bloom_utils.tt_const_tensor(beta, [1, dim1, dim2, dim2], device)

    pt_out = torch.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)
    pt_out_size = list(pt_out.shape)

    while len(pt_out_size) < 4:
        pt_out_size.insert(0, 1)

    pt_out = torch.reshape(pt_out, pt_out_size)

    input = bloom_utils.torch2tt_tensor(input, device)
    batch1 = bloom_utils.torch2tt_tensor(batch1, device)
    batch2 = bloom_utils.torch2tt_tensor(batch2, device)

    tt_out = baddbmm.tt_baddbmm(
        device, input, batch1, batch2, beta=tt_beta, alpha=tt_alpha
    )
    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.99)
    logger.info(pcc_message)

    if does_pass:
        logger.info("baddbmm: Passed!")
    else:
        logger.warning("baddbmm: Failed!")

    assert does_pass


def test_baddbmm():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_baddbmm_test(device)
    tt_lib.device.CloseDevice(device)


if __name__ == "__main__":
    test_baddbmm()
