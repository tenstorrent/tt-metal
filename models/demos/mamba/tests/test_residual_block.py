# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from typing import Optional
import ttnn
from models.demos.mamba.tt.full_model import TtTensorLoader
from models.demos.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.mamba.reference.prefill_model import MambaPrefill
from models.demos.mamba.tt.residual_block import TtResidualBlock
from models.demos.mamba.tt import model_config
from models.demos.mamba.tt.types import ModelMode
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


class PytorchResidualBlock(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.block = hf_reference_model.layers[layer_num]
        self.block.eval()

    def forward(self, x):
        result = self.block(x)
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
def test_mamba_residual_block_inference(
    device: ttnn.Device,
    use_program_cache,
    model_version: MambaPretrainedModelName,
    batch: int,
    pcc: float,
):
    torch.manual_seed(0)

    LAYER_NUM = 0

    reference_model = MambaDecode.from_pretrained(model_version, batch_size=batch)
    reference_model.args.batch_size = batch

    d_model = reference_model.args.d_model
    input = torch.rand(batch, 1, d_model)

    reference_output = PytorchResidualBlock(reference_model, LAYER_NUM)(input)

    residual_block = reference_model.layers[LAYER_NUM]
    assert not isinstance(residual_block, torch.Tensor), "Expected torch.Module"

    config = model_config.create_model_config(batch, d_model)

    loader = TtTensorLoader(reference_model.state_dict(), device)

    model = TtResidualBlock(reference_model.args, device, config, loader.get_tensor_loader(LAYER_NUM))
    tt_input = input.view(1, 1, batch, d_model)
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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "model_version, seq_len, pcc",
    (
        (
            "state-spaces/mamba-2.8b",
            32,
            0.99,
        ),
    ),
)
def test_mamba_residual_block_prefill(
    device: ttnn.Device,
    use_program_cache,
    model_version: MambaPretrainedModelName,
    seq_len: int,
    pcc: float,
):
    torch.manual_seed(0)

    LAYER_NUM = 0
    batch = 1

    reference_model = MambaPrefill.from_pretrained(model_version)
    reference_model.args.batch_size = batch
    reference_model.args.seq_len = seq_len
    reference_model.args.mode = ModelMode.PREFILL

    d_model = reference_model.args.d_model
    input = torch.rand(1, seq_len, d_model)

    reference_output = PytorchResidualBlock(reference_model, LAYER_NUM)(input)

    residual_block = reference_model.layers[LAYER_NUM]
    assert not isinstance(residual_block, torch.Tensor), "Expected torch.Module"

    config = model_config.create_model_config(seq_len, d_model)

    loader = TtTensorLoader(reference_model.state_dict(), device)

    model = TtResidualBlock(reference_model.args, device, config, loader.get_tensor_loader(LAYER_NUM))
    tt_input = input.view(1, 1, seq_len, d_model)
    tt_input = ttnn.to_device(
        ttnn.from_torch(tt_input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output = model(tt_input)
    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.squeeze(0)
    # tt_output = tt_output.view(batch, 1, -1)
    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba SSM output failed")
        assert does_pass, f"PCC value is lower than {pcc}"
