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
    comp_allclose_and_pcc,
    comp_pcc,
)


@pytest.mark.parametrize(
    "batch_size",
    [
        1,
        2,
        8,
    ],
)
def test_run_resnet50_inference(use_program_cache, batch_size, imagenet_sample_input):
    image1 = imagenet_sample_input
    image = image1
    for i in range(batch_size - 1):
        image = torch.cat((image, image1), dim=0)
    with torch.no_grad():
        torch.manual_seed(1234)

        # Initialize the device
        device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
        tt_lib.device.InitializeDevice(device)
        tt_lib.device.SetDefaultDevice(device)
        tt_lib.device.EnableMemoryReports()

        torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch_resnet50.eval()

        state_dict = torch_resnet50.state_dict()
        storage_in_dram = False
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
        )

        # torch_output = torch_resnet50(image).unsqueeze(1).unsqueeze(1)
        # tt_output = tt_resnet50(image)
        torch_conv1_output = torch_resnet50.conv1(image)
        torch_conv1_output = torch_resnet50.bn1(torch_conv1_output)
        torch_conv1_output = torch_resnet50.relu(torch_conv1_output)
        torch_max_pool_output = torch_resnet50.maxpool(torch_conv1_output)

        (
            tt_conv1_output_utilize_out,
            tt_max_pool_output_with_untilize_out,
        ) = tt_resnet50.run_first_part(image)

        print("Compare pytorch conv1 output with TT conv1 + untilize")
        passing, info = comp_allclose_and_pcc(
            torch_conv1_output, tt_conv1_output_utilize_out, pcc=0.985
        )
        logger.info(info)
        golden_pcc = 0.985
        passing_pcc_conv1, _ = comp_pcc(
            torch_conv1_output, tt_conv1_output_utilize_out, pcc=golden_pcc
        )

        print("Compare pytorch maxpool output with TT conv1 + untilize + maxpool")
        passing, info = comp_allclose_and_pcc(
            torch_max_pool_output, tt_max_pool_output_with_untilize_out, pcc=0.985
        )
        logger.info(info)
        golden_pcc = 0.985
        passing_pcc_maxpool, _ = comp_pcc(
            torch_max_pool_output, tt_max_pool_output_with_untilize_out, pcc=golden_pcc
        )

        # print("Comparing output of max pool w/ input in DRAM vs max pool w/ input in L1")
        # if (not torch.equal(tt_max_pool_output_with_untilize_out_in_dram, tt_max_pool_output_with_untilize_out_in_l1)):
        #     print("Printing mismatch values and locations")
        #     num_errors = 0
        #     num_matches = 0
        #     for n in range(8):
        #         for c in range(64):
        #             for h in range(56):
        #                 for w in range(56):
        #                     if tt_max_pool_output_with_untilize_out_in_dram[n][c][h][w].item() != tt_max_pool_output_with_untilize_out_in_l1[n][c][h][w].item():
        #                         #if (num_errors < 100):
        #                         # print("Mismatch at (nchw) - ", n, c, h, w)
        #                         # print("Maxpool output with untilize out in dram - ", tt_max_pool_output_with_untilize_out_in_dram[n][c][h][w].item())
        #                         # print("Maxpool output with untilize out in l1 - ", tt_max_pool_output_with_untilize_out_in_l1[n][c][h][w].item())
        #                         num_errors += 1
        #                     else:
        #                         # print("Match at (nchw) - ", n, c, h, w)
        #                         # print("Maxpool output with untilize out in dram - ", tt_max_pool_output_with_untilize_out_in_dram[n][c][h][w].item())
        #                         # print("Maxpool output with untilize out in l1 - ", tt_max_pool_output_with_untilize_out_in_l1[n][c][h][w].item())
        #                         num_matches += 1

        #     print(tt_max_pool_output_with_untilize_out_in_l1)
        #     print("Total number of mismatches=", num_errors)
        #     print("Total number of matches=", num_matches)

        # passing, info = comp_allclose_and_pcc(torch_output, tt_output, pcc=0.985)
        # logger.info(info)
        # tt_lib.device.CloseDevice(device)
        # golden_pcc = 0.985
        # passing_pcc, _ = comp_pcc(torch_output, tt_output, pcc=golden_pcc)

        assert passing_pcc_conv1
        assert passing_pcc_maxpool
        # assert passing_pcc
