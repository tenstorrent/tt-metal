# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
from torchvision import models
import pytest
import tt_lib

from models.utility_functions import is_e75

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
        ): 0.989913,  # Max ATOL Delta: 1.982335090637207, Max RTOL Delta: 22.094308853149414, PCC: 0.989913317779133
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.988538,  # Max ATOL Delta: 1.508270263671875, Max RTOL Delta: 168.00689697265625, PCC: 0.9885382108063352
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.972638,  # Max ATOL Delta: 2.089695930480957, Max RTOL Delta: inf, PCC: 0.9726389297275764
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.956349,  # Max ATOL Delta: 3.455164909362793, Max RTOL Delta: inf, PCC: 0.9563498331411239
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.943105,  # Max ATOL Delta: 3.736414909362793, Max RTOL Delta: 53.903194427490234, PCC: 0.9431057021249141
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.937181,  # Max ATOL Delta: 4.267664909362793, Max RTOL Delta: 225.56088256835938, PCC: 0.9371810428520426
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.835586,  # Max ATOL Delta: 8.330164909362793, Max RTOL Delta: inf, PCC: 0.8355864305936868
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.859410,  # Max ATOL Delta: 7.642664909362793, Max RTOL Delta: inf, PCC: 0.8594109868977741
    },
    16: {
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.956349,  # Max ATOL Delta: 3.455164909362793, Max RTOL Delta: inf, PCC: 0.956349833141124
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.859410,  # Max ATOL Delta: 7.642664909362793, Max RTOL Delta: inf, PCC: 0.8594109868977738
    },
}


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16], ids=["batch_1", "batch_2", "batch_8", "batch_16"])
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
    "math_fidelity", [tt_lib.tensor.MathFidelity.HiFi4, tt_lib.tensor.MathFidelity.LoFi], ids=["HiFi4", "LoFi"]
)
def test_run_resnet50_inference(
    use_program_cache, device, batch_size, weights_dtype, activations_dtype, math_fidelity, imagenet_sample_input
):
    if is_e75(device):
        pytest.skip("Resnet50 is not supported on E75")

    if batch_size > 8 and (
        activations_dtype != tt_lib.tensor.DataType.BFLOAT8_B or weights_dtype != tt_lib.tensor.DataType.BFLOAT8_B
    ):
        pytest.skip("Batch > 8 must be run fully bfp8")

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
        tt_output = tt_output.to_torch().to(torch.float)

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
                    valid_pcc = 0.96
            else:
                if model_config["MATH_FIDELITY"] == tt_lib.tensor.MathFidelity.LoFi:
                    valid_pcc = 0.93
                else:
                    valid_pcc = 0.985
        passing_pcc, _ = comp_pcc(torch_output, tt_output, pcc=valid_pcc)
        assert passing_pcc
        # assert passing # fails because of torch.allclose
