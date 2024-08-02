# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.common.mlp import MLP
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull


class TorchMLP(torch.nn.Module):
    def __init__(self, dim, hidden_dim, act=torch.nn.functional.silu):
        super().__init__()

        self.w1 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, hidden_dim, bias=False)
        self.act = act

    def forward(self, x):
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class RefModel(torch.nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.mlp = TorchMLP(dim, hidden_dim)

    def forward(self, x):
        return self.mlp(x)


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    ("dim", "hidden_dim", "dtype"),
    (
        (4096, 14336, ttnn.bfloat4_b),  # Mistral 7b
        (4096, 14336, ttnn.bfloat8_b),  # Llama 3.1 8b
    ),
)
def test_mlp(device, dim, hidden_dim, dtype, use_program_cache, reset_seeds):
    reference_model = RefModel(dim, hidden_dim)
    state_dict = reference_model.state_dict()

    tt_model = MLP(
        device=device,
        state_dict=state_dict,
        state_dict_prefix="mlp",
        activation=ttnn.UnaryOpType.SILU,
        w1w3_dtype=dtype,
    )

    passing = True
    for seq_len in [32, 128, 1024, 4096]:
        torch_input = torch.randn(1, 1, seq_len, dim)
        reference_output = reference_model(torch_input)

        # Assume DRAM inputs for prefill
        mem = ttnn.L1_MEMORY_CONFIG if seq_len <= 32 else ttnn.DRAM_MEMORY_CONFIG
        tt_input = ttnn.from_torch(
            torch_input,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        tt_output = tt_model(tt_input)
        tt_output_torch = ttnn.to_torch(tt_output)

        pcc_required = 0.99
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)
        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"seq_len: {seq_len} - {pcc_message})")
        if not passing:
            break

    if passing:
        logger.info("Passed!")
    else:
        logger.warning("Failed!")

    assert passing, f"MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
