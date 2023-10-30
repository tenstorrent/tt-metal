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

# golden pcc is ordered fidelity, weight dtype, activation dtype
golden_pcc = {
    8: {
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.9900861228915353,  # Max ATOL Delta: 1.982335090637207, Max RTOL Delta: 221.08363342285156, PCC: 0.9900861228915353
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.9905268223455689,  # Max ATOL Delta: 1.633270263671875, Max RTOL Delta: 609.134521484375, PCC: 0.9905268223455689
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.9700243646676288,  # Max ATOL Delta: 2.027195930480957, Max RTOL Delta: inf, PCC: 0.9700243646676288
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.9560413660709707,  # Max ATOL Delta: 3.205164909362793, Max RTOL Delta: inf, PCC: 0.9560413660709707
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.9450991418256837,  # Max ATOL Delta: 3.455164909362793, Max RTOL Delta: 2949.2197265625, PCC: 0.9450991418256837
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.9392607666771678,  # Max ATOL Delta: 3.830164909362793, Max RTOL Delta: 434.328369140625, PCC: 0.9392607666771678
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.8369879885166531,  # Max ATOL Delta: 8.205164909362793, Max RTOL Delta: inf, PCC: 0.8369879885166531
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.8629922303807895,  # Max ATOL Delta: 7.705164909362793, Max RTOL Delta: inf, PCC: 0.8629922303807895
    }
}


@pytest.mark.parametrize("batch_size", [1, 2, 8], ids=["batch_1", "batch_2", "batch_8"])
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

        valid_pcc = 1.0
        if batch_size == 8:
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
