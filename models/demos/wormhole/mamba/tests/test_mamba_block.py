# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import time
import pytest
from loguru import logger
import ttnn
from models.demos.wormhole.mamba.tt.mamba_model import TtTensorLoader
from models.demos.wormhole.mamba.reference.prefill_decode_model import Mamba, MambaPretrainedModelName
from models.demos.wormhole.mamba.reference.args import ModelMode
from models.demos.wormhole.mamba.tt.mamba_block import TtMambaBlock
from models.demos.wormhole.mamba.tt import model_config
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


class PytorchMambaBlock(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.block = hf_reference_model.layers[layer_num].mixer
        self.block.eval()

    def forward(self, x):
        result = self.block(x)
        return result


@pytest.mark.parametrize("layer", [0])
@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True, False],
    ids=[
        "pretrained_weight_true",
        "pretrained_weight_false",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "model_version, mode, batch, seq_len, pcc",
    (
        (
            "state-spaces/mamba-2.8b",
            ModelMode.PREFILL,
            1,
            128,
            0.985,
        ),
        (
            "state-spaces/mamba-2.8b",
            ModelMode.DECODE,
            32,
            1,
            0.975,
        ),
    ),
)
def test_mamba_block_inference(
    model_version: MambaPretrainedModelName,
    mode: ModelMode,
    batch: int,
    seq_len: int,
    pcc: float,
    use_pretrained_weights: bool,
    layer: int,
    device: ttnn.Device,
    use_program_cache,
    reset_seeds,
):
    load_reference_model = Mamba.from_pretrained if use_pretrained_weights else Mamba.from_random
    reference_model = load_reference_model(model_version, batch_size=batch)
    reference_model.args.mode = mode

    d_model = reference_model.args.d_model
    input = torch.rand(batch, seq_len, d_model)

    reference_output = PytorchMambaBlock(reference_model, layer)(input)

    config = model_config.create_model_config(batch, reference_model.args.d_model, mode=mode, seq_len=seq_len)

    loader = TtTensorLoader(reference_model.state_dict(), device)

    logger.info(f"Initalizing Mamba block from layer {layer}")
    start = time.time()
    model = TtMambaBlock(reference_model.args, device, config, loader.get_tensor_loader(layer))
    logger.info(f"Finished initializing Mamba block (took {time.time() - start:.3f} sec)")

    tt_input = input.view(1, 1, config["outer_dim"], d_model)
    tt_input = ttnn.to_device(
        ttnn.from_torch(tt_input, layout=ttnn.TILE_LAYOUT, dtype=config["dtype"]["activations"]),
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tt_output = model(tt_input)
    tt_output = ttnn.to_torch(tt_output)
    tt_output = tt_output.view(batch, seq_len, -1)
    logger.info(comp_allclose(reference_output, tt_output))

    does_pass, output_pcc = comp_pcc(reference_output, tt_output, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if not does_pass:
        logger.warning("Mamba Block output failed")
        assert does_pass, f"PCC value is lower than {pcc}"
