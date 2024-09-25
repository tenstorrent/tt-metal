# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.wormhole.mamba.tt.mamba_model import TtTensorLoader
from models.demos.wormhole.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.wormhole.mamba.tt import model_config
from models.demos.wormhole.mamba.tt.mamba_conv import MambaConv, MambaConvConfig
from models.demos.wormhole.mamba.reference.args import ModelMode


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
            1027,
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

    reference_model = MambaDecode.from_pretrained(model_version, batch_size=batch)
    loader = TtTensorLoader(reference_model.state_dict(), device)
    mamba_conv_config = MambaConvConfig(
        input_length=seqlen,
        weights_dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.HiFi4,
    )
    torch_input = torch.randn(batch, mamba_conv_config.input_channels, seqlen)
    torch_depthwise_conv = torch.nn.Conv1d(
        mamba_conv_config.input_channels,
        mamba_conv_config.input_channels,
        mamba_conv_config.output_channels,
        mamba_conv_config.stride,
        mamba_conv_config.padding,
        groups=mamba_conv_config.groups,
        bias=False,
    )
    torch_depthwise_conv.weight.data = loader.get_tensor_loader(LAYER_NUM)("mixer.conv1d.weight", return_as_torch=True)
    reference_output = torch_depthwise_conv(torch_input)

    tt_depthwise_conv = MambaConv(device, loader.get_tensor_loader(LAYER_NUM), mamba_conv_config)
    torch_input = torch_input.permute(0, 2, 1)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    with torch.no_grad():
        tt_output = tt_depthwise_conv(tt_input)
    ttnn.deallocate(tt_input)
    tt_output = ttnn.to_torch(tt_output).squeeze(0)
    tt_output = tt_output.permute(0, 2, 1)
    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba Conv output failed")
        assert does_pass, f"PCC value is lower than {pcc}"
