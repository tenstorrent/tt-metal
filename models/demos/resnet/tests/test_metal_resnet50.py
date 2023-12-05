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
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_conv import TTPyConv
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_untilize_with_halo import TTPyUntilizeWithHalo

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
        ): 0.985777,  # Max ATOL Delta: 2.232335090637207, Max RTOL Delta: 143.95509338378906, PCC: 0.9857770896722845
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.972452,  # Max ATOL Delta: 2.330164909362793, Max RTOL Delta: inf, PCC: 0.9724525986614849
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.976795,  # Max ATOL Delta: 2.705164909362793, Max RTOL Delta: inf, PCC: 0.9767951799869169
        (
            tt_lib.tensor.MathFidelity.HiFi2,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.983090,  # Max ATOL Delta: 1.9497511386871338, Max RTOL Delta: 224.39161682128906, PCC: 0.9830904625520384
        (
            tt_lib.tensor.MathFidelity.HiFi2,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.984897,  # Max ATOL Delta: 1.7310011386871338, Max RTOL Delta: 25.013704299926758, PCC: 0.984897263218172
        (
            tt_lib.tensor.MathFidelity.HiFi2,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.930658,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9306583851557794
        (
            tt_lib.tensor.MathFidelity.HiFi2,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.947387,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9473877801237898
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.943105,  # Max ATOL Delta: 3.736414909362793, Max RTOL Delta: 53.903194427490234, PCC: 0.9431057021249141
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT16,
        ): 0.962400,  # Max ATOL Delta: 2.830164909362793, Max RTOL Delta: 159.69110107421875, PCC: 0.9624007409443723
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT16,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.844913,  # Max ATOL Delta: 8.142664909362793, Max RTOL Delta: inf, PCC: 0.8449133550359712
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.894135,  # Max ATOL Delta: 6.205164909362793, Max RTOL Delta: inf, PCC: 0.8941357301670331
    },
    16: {
        (
            tt_lib.tensor.MathFidelity.HiFi4,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.976795,  # Max ATOL Delta: 2.705164909362793, Max RTOL Delta: inf, PCC: 0.9767951799869171
        (
            tt_lib.tensor.MathFidelity.HiFi2,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.947387,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9473877801237898
        (
            tt_lib.tensor.MathFidelity.LoFi,
            tt_lib.tensor.DataType.BFLOAT8_B,
            tt_lib.tensor.DataType.BFLOAT8_B,
        ): 0.884609,  # Max ATOL Delta: 6.455164909362793, Max RTOL Delta: inf, PCC: 0.8846098380419435
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
    "math_fidelity",
    [tt_lib.tensor.MathFidelity.HiFi4, tt_lib.tensor.MathFidelity.HiFi2, tt_lib.tensor.MathFidelity.LoFi],
    ids=["HiFi4", "HiFi2", "LoFi"],
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

        TTPyConv.static_kernel_configs_cache_map = {}
        TTPyUntilizeWithHalo.static_kernel_configs_cache_map = {}
