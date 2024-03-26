# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
from torchvision import models
import pytest
import tt_lib

from models.utility_functions import is_e75, skip_for_wormhole_b0

from models.demos.resnet.tt.metalResnetBlock50 import ResNet, Bottleneck
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    get_atol_rtol_pcc,
    comp_pcc,
)

# golden pcc is ordered fidelity, weight dtype, activation dtype
golden_pcc = {
    8: {
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.990804,  # Max ATOL Delta: 1.607335090637207, Max RTOL Delta: 115.62200164794922, PCC: 0.9908042840544742
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.986301,  # Max ATOL Delta: 1.5697126388549805, Max RTOL Delta: 21.3042049407959, PCC: 0.9863013351442654
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.973763,  # Max ATOL Delta: 2.455164909362793, Max RTOL Delta: inf, PCC: 0.9737631427307492
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.978099,  # Max ATOL Delta: 1.955164909362793, Max RTOL Delta: inf, PCC: 0.9780993165966628
        (
            tt_lib.tensor.MathFidelity.HiFi2,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.983400,  # Max ATOL Delta: 1.7310011386871338, Max RTOL Delta: 369.5689392089844, PCC: 0.9834004200555363
        (
            tt_lib.tensor.MathFidelity.HiFi2,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.984828,  # Max ATOL Delta: 1.6054553985595703, Max RTOL Delta: 59.124324798583984, PCC: 0.9848281996919587
        (
            tt_lib.tensor.MathFidelity.HiFi2,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.934073,  # Max ATOL Delta: 4.330164909362793, Max RTOL Delta: inf, PCC: 0.9340735819578696
        (
            tt_lib.tensor.MathFidelity.HiFi2,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.944435,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9444350983635019
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.938909,  # Max ATOL Delta: 3.861414909362793, Max RTOL Delta: 240.63145446777344, PCC: 0.9389092547575272
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.959609,  # Max ATOL Delta: 3.205164909362793, Max RTOL Delta: 141.7057342529297, PCC: 0.9596095155046113
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.854903,  # Max ATOL Delta: 7.830164909362793, Max RTOL Delta: inf, PCC: 0.8549035869182201
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.884609,  # Max ATOL Delta: 6.455164909362793, Max RTOL Delta: inf, PCC: 0.8846098380419433
    },
    16: {
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.978099,  # Max ATOL Delta: 1.955164909362793, Max RTOL Delta: inf, PCC: 0.9780993165966632
        (
            tt_lib.tensor.MathFidelity.HiFi2,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.944435,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9444350983635021
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.884609,  # Max ATOL Delta: 6.455164909362793, Max RTOL Delta: inf, PCC: 0.8846098380419435
    },
    20: {
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.978099,  # Max ATOL Delta: 1.955164909362793, Max RTOL Delta: inf, PCC: 0.9780993165966628
        (
            tt_lib.tensor.MathFidelity.HiFi2,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.944435,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9444350983635021
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.884609,  # Max ATOL Delta: 6.455164909362793, Max RTOL Delta: inf, PCC: 0.8846098380419433
    },
}


@skip_for_wormhole_b0("This test is not supported on WHB0, please use the TTNN version.")
@pytest.mark.parametrize("batch_size", [1, 2, 8, 16, 20], ids=["batch_1", "batch_2", "batch_8", "batch_16", "batch_20"])
@pytest.mark.parametrize(
    "weights_dtype",
    [tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B],
    ids=["weights_BFLOAT16", "weights_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B],
    ids=["activations_BFLOAT16", "activations_BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "math_fidelity",
    [tt_lib.tensor.MathFidelity.HiFi4, tt_lib.tensor.MathFidelity.HiFi2, tt_lib.tensor.MathFidelity.LoFi],
    ids=["HiFi4", "HiFi2", "LoFi"],
)
def test_run_resnet50_inference(
    device, use_program_cache, batch_size, weights_dtype, activations_dtype, math_fidelity, imagenet_sample_input
):
    if is_e75(device):
        pytest.skip("Resnet50 is not supported on E75")

    if batch_size > 8 and (
        activations_dtype != tt_lib.tensor.DataType.BFLOAT8_B or weights_dtype != tt_lib.tensor.DataType.BFLOAT8_B
    ):
        pytest.skip("Batch > 8 must be run fully bfp8")
    if batch_size <= 2:
        pytest.skip("batch 1 and 2 are not supported with sharded data")
    image1 = imagenet_sample_input
    image = image1
    model_config = {
        "MATH_FIDELITY": math_fidelity,
        "WEIGHTS_DTYPE": weights_dtype,
        "ACTIVATIONS_DTYPE": activations_dtype,
    }
    for i in range(batch_size - 1):
        image = torch.cat((image, image1), dim=0)
    with torch.no_grad():
        torch.manual_seed(1234)

        tt_lib.device.EnableMemoryReports()

        torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch_resnet50.eval()

        state_dict = torch_resnet50.state_dict()
        storage_in_dram = False
        sharded = False
        if batch_size >= 8:
            sharded = True
        # run once to compile ops
        tt_resnet50 = ResNet(
            Bottleneck,
            [3, 4, 6, 3],
            device=device,
            state_dict=state_dict,
            base_address="",
            fold_batchnorm=True,
            storage_in_dram=storage_in_dram,
            batch_size=batch_size,
            model_config=model_config,
            sharded=sharded,
        )

        torch_output = torch_resnet50(image).unsqueeze(1).unsqueeze(1)
        tt_image = tt_resnet50.preprocessing(image)
        tt_output = tt_resnet50(tt_image)
        tt_output = tt_output.cpu().to_torch().to(torch.float)

        # # run again to measure end to end perf
        # start_time = datetime.now()
        # tt_output = tt_resnet50(image)
        # end_time = datetime.now()
        # diff = end_time - start_time
        # logger.info("End to end time (microseconds))", diff.microseconds)
        # throughput_fps = (float) (1000000 / diff.microseconds)
        # logger.info("Throughput (fps)", throughput_fps)

        _, _, _, info = get_atol_rtol_pcc(torch_output, tt_output)
        logger.info(info)

        valid_pcc = 1.0
        if batch_size >= 8:
            valid_pcc = golden_pcc[batch_size][
                (model_config["MATH_FIDELITY"], model_config["WEIGHTS_DTYPE"], model_config["ACTIVATIONS_DTYPE"])
            ]
        else:
            if model_config["ACTIVATIONS_DTYPE"] == tt_lib.tensor.DataType.BFLOAT8_B:
                if model_config["MATH_FIDELITY"] == tt_lib.tensor.MathFidelity.LoFi:
                    valid_pcc = 0.87
                else:
                    valid_pcc = 0.94
            else:
                if model_config["MATH_FIDELITY"] == tt_lib.tensor.MathFidelity.LoFi:
                    valid_pcc = 0.93
                else:
                    valid_pcc = 0.982
        passing_pcc, _ = comp_pcc(torch_output, tt_output, pcc=valid_pcc)
        assert passing_pcc
        # assert passing # fails because of torch.allclose
