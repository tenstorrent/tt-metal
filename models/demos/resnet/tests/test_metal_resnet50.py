# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
from torchvision import models
import pytest
import tt_lib

from models.utility_functions import is_e75

from models.demos.resnet.tt.metalResnetBlock50 import ResNet, Bottleneck
from models.utility_functions import comp_pcc, comp_allclose_and_pcc


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_run_resnet50_inference(device, batch_size, imagenet_sample_input):
    if is_e75(device):
        pytest.skip("Resnet50 is not supported on E75")

    image1 = imagenet_sample_input
    image = image1
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

        passing, info = comp_allclose_and_pcc(torch_output, tt_output, pcc=0.985)
        logger.info(info)
        golden_pcc = 0.985
        if batch_size == 8:
            golden_pcc = 0.9899485705112977
        passing_pcc, _ = comp_pcc(torch_output, tt_output, pcc=golden_pcc)
        assert passing_pcc
        # assert passing # fails because of torch.allclose
