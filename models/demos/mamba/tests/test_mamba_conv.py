# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from typing import Optional
import ttnn
from models.demos.mamba.tt.full_model import TtTensorLoader
from models.demos.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.mamba.tt.mamba_block import TtMambaBlock
from models.demos.mamba.tt import model_config
from models.demos.mamba.tt.mamba_conv import MambaConv

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "model_version, seqlen, pcc",
    (
        (
            "state-spaces/mamba-2.8b",
            32,
            0.99,
        ),
    ),
)
def test_mamba_conv(
    device: ttnn.Device,
    use_program_cache,
    model_version: MambaPretrainedModelName,
    seqlen: int,
    pcc: float,
):
    torch.manual_seed(0)

    LAYER_NUM = 0
    batch = 1

    reference_model = MambaDecode.from_pretrained(model_version)
    reference_model.args.batch_size = batch
    reference_model.args.seq_len = seqlen

    loader = TtTensorLoader(reference_model.state_dict(), device)

    d_model = reference_model.args.d_model
    input = torch.rand(batch, seqlen, 2 * d_model)
    torch_input = input.permute(0, 2, 1)
    torch_depthwise_conv = torch.nn.Conv1d(2 * d_model, 2 * d_model, 4, 1, 3, groups=2 * d_model, bias=False)
    torch_depthwise_conv.weight.data = loader.get_tensor_loader(LAYER_NUM)("mixer.conv1d.weight", return_as_torch=True)
    reference_output = torch_depthwise_conv(torch_input)
    reference_output = reference_output[:, :, :-3]
    reference_output = torch_depthwise_conv(reference_output)
    reference_output = reference_output[:, :, :-3]
    reference_output = reference_output.squeeze(0).permute(1, 0).unsqueeze(0).unsqueeze(0)

    # model = TtMambaBlock(reference_model.args, device, config, loader.get_tensor_loader(LAYER_NUM))
    tt_depthwise_conv = MambaConv(
        device, reference_model.args, loader.get_tensor_loader(LAYER_NUM), "mixer.conv1d.weight"
    )

    tt_input = input.view(1, seqlen, 1, 2 * d_model)
    tt_input = ttnn.from_torch(tt_input, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device)
    tt_output1 = tt_depthwise_conv(tt_input)
    ttnn.deallocate(tt_input)
    print("first conv done")
    tt_output1 = ttnn.to_memory_config(tt_output1, ttnn.DRAM_MEMORY_CONFIG)
    tt_output1 = ttnn.to_layout(tt_output1, ttnn.ROW_MAJOR_LAYOUT)
    tt_output1 = ttnn.to_dtype(tt_output1, ttnn.bfloat16)
    tt_output2 = tt_depthwise_conv(tt_output1)
    ttnn.deallocate(tt_output1)
    tt_output = ttnn.from_device(tt_output2)
    tt_output = ttnn.to_torch(tt_output)

    print(tt_output[:, :, :, 0])
    print(reference_output[:, :, :, 0])
    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba block output failed")
        assert does_pass, f"PCC value is lower than {pcc}"
