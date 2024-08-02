# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.common.mlp import MLP
from models.demos.wormhole.mistral7b.tt.model_config import TtModelArgs
from models.demos.wormhole.mistral7b.reference.model import FeedForward
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "seq_len",
    (
        4096,
        128,
        32,
    ),
)
def test_mistral_mlp_inference(device, seq_len, use_program_cache, reset_seeds):
    dtype = ttnn.bfloat8_b
    model_args = TtModelArgs(device=device)
    state_dict = torch.load(model_args.consolidated_weights_path)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {k[22:]: v for k, v in state_dict.items() if (k.startswith("layers.0.feed_forward"))}

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = FeedForward(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    tt_model = MLP(
        device=device,
        state_dict=state_dict,
        state_dict_prefix="layers.0.feed_forward",
        weight_cache_path=model_args.weight_cache_path(dtype),
        activation=ttnn.UnaryOpType.SILU,
        w1w3_dtype=ttnn.bfloat4_b,
    )

    torch_input = torch.randn(1, 1, seq_len, 4096)
    reference_output = reference_model(torch_input)
    tt_input = ttnn.from_torch(
        torch_input, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )

    logger.info("Compilation pass for Mistral_MLP")
    tt_output = tt_model(tt_input)

    logger.info("Performance pass for Mistral_MLP")
    tt_output = tt_model(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_MLP Passed!")
    else:
        logger.warning("Mistral_MLP Failed!")

    assert passing, f"Mistral_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
