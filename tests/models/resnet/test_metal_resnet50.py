# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

from loguru import logger
import torch
from torchvision import models
import pytest
import tt_lib
from datetime import datetime

from tests.models.resnet.metalResnetBlock50 import ResNet, Bottleneck
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    get_atol_rtol_pcc,
    comp_pcc,
)
from models.utility_functions import is_e75

@pytest.mark.parametrize("batch_size", [1, 2, 8], ids=["batch_1", "batch_2", "batch_8"])
@pytest.mark.parametrize("weights_dtype", [tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B], ids=["BFLOAT16", "BFLOAT8_B"])
@pytest.mark.parametrize("math_fidelity", [tt_lib.tensor.MathFidelity.HiFi4, tt_lib.tensor.MathFidelity.LoFi], ids=["HiFi4", "LoFi"])
def test_run_resnet50_inference(use_program_cache, device, batch_size, weights_dtype, math_fidelity, imagenet_sample_input):
    if is_e75(device):
        pytest.skip("Resnet50 is not supported on E75")

    image1 = imagenet_sample_input
    image = image1
    model_config = {
        "MATH_FIDELITY": math_fidelity,
        "WEIGHTS_DTYPE": weights_dtype,
        "ACTIVATIONS_DTYPE": tt_lib.tensor.DataType.BFLOAT16,
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
        if batch_size == 8:
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
        tt_output = tt_resnet50(image)

        # # run again to measure end to end perf
        # start_time = datetime.now()
        # tt_output = tt_resnet50(image)
        # end_time = datetime.now()
        # diff = end_time - start_time
        # print("End to end time (microseconds))", diff.microseconds)
        # throughput_fps = (float) (1000000 / diff.microseconds)
        # print("Throughput (fps)", throughput_fps)

        _, _, _, info = get_atol_rtol_pcc(torch_output, tt_output)
        logger.info(info)
        if batch_size == 8:
            if model_config["MATH_FIDELITY"] == tt_lib.tensor.MathFidelity.LoFi:
                golden_pcc = 0.9446551761853834
            else:
                golden_pcc = 0.9899485705112977
        else:
            if model_config["MATH_FIDELITY"] == tt_lib.tensor.MathFidelity.LoFi:
                golden_pcc = 0.93
            else:
                golden_pcc = 0.985
        passing_pcc, _ = comp_pcc(torch_output, tt_output, pcc=golden_pcc)
        assert passing_pcc
        # assert passing # fails because of torch.allclose
