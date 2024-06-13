# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from typing import Optional

import ttnn

from models.demos.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.mamba.reference.prefill_model import MambaPrefill
from models.demos.mamba.tt.full_model import TtTensorLoader
from models.demos.mamba.tt.mamba_one_step_ssm import TtMambaSSM
from models.demos.mamba.tt import model_config
from models.demos.mamba.tt.types import ModelMode

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


class PytorchMambaSSM(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.block = hf_reference_model.layers[layer_num].mixer
        self.block.eval()

    def forward(self, x):
        result = self.block.ssm(x)
        return result


@pytest.mark.parametrize(
    "model_version, batch, pcc",
    (
        (
            "state-spaces/mamba-2.8b",
            32,
            0.99,
        ),
    ),
)
def test_mamba_ssm_inference(
    device: ttnn.Device,
    use_program_cache,
    model_version: MambaPretrainedModelName,
    batch: int,
    pcc: float,
):
    torch.manual_seed(0)

    LAYER_NUM = 0

    reference_model = MambaDecode.from_pretrained(model_version)
    reference_model.args.batch_size = batch

    d_in = reference_model.args.d_model * reference_model.args.expand
    input = torch.rand(batch, 1, d_in)

    reference_output = PytorchMambaSSM(reference_model, LAYER_NUM)(input)

    residual_block = reference_model.layers[LAYER_NUM]
    assert not isinstance(residual_block, torch.Tensor), "Expected torch.Module"

    config = model_config.create_model_config(batch, reference_model.args.d_model)

    loader = TtTensorLoader(reference_model.state_dict(), device)

    model = TtMambaSSM(reference_model.args, device, config, loader.get_tensor_loader(LAYER_NUM))
    tt_input = input.view(1, 1, batch, d_in)
    tt_input = ttnn.to_device(
        ttnn.from_torch(tt_input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output = model(tt_input)
    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.view(batch, 1, -1)
    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba SSM output failed")
        assert does_pass, f"PCC value is lower than {pcc}"


class PytorchMambaPrefillSSM(torch.nn.Module):
    def __init__(self, hf_reference_model: MambaPrefill, layer_num: int):
        super().__init__()
        self.block = hf_reference_model.layers[layer_num].mixer
        self.block.eval()

    def forward(self, x):
        result = self.block.ssm(x)  # (B, L, D_IN)
        return result


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
def test_mamba_ssm_prefill(
    device: ttnn.Device,
    model_version: MambaPretrainedModelName,
    seqlen: int,
    pcc: float,
    use_program_cache,
):
    torch.manual_seed(0)

    layer_num = 0
    batch = 1

    logger.info(f"Creating prefill reference model on CPU (model_version='{model_version}')")
    reference_model = MambaPrefill.from_pretrained(model_version)

    d_in = reference_model.args.d_model * reference_model.args.expand
    input = torch.rand(batch, seqlen, d_in)  # (B, L, D_IN)

    logger.info(f"Running reference model to get expected output (input={input.shape})")
    expected = PytorchMambaPrefillSSM(reference_model, layer_num)(input).unsqueeze(0)  # (1, B, L, D_IN)

    logger.info(f"Creating prefill reference model on device (model_version='{model_version}')")
    config = model_config.create_model_config(batch, reference_model.args.d_model)
    loader = TtTensorLoader(reference_model.state_dict(), device)
    model = TtMambaSSM(reference_model.args, device, config, loader.get_tensor_loader(layer_num))

    input = input.view(1, batch, seqlen, d_in)
    input = ttnn.to_device(
        ttnn.from_torch(input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    logger.info(f"Running model on device (input={input.shape})")
    actual = model(input, mode=ModelMode.PREFILL)  # (1, B, L, D_IN)
    actual = ttnn.to_torch(actual)

    does_pass, output_pcc = comp_pcc(expected, actual, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba SSM prefill output failed")
        assert does_pass, f"PCC value is lower than {pcc}"
