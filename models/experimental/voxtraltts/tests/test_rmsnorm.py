# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.functional import rms_norm as reference_rms_norm
from models.experimental.voxtraltts.tt.rmsnorm import VoxtralTTRMSNorm
from models.tt_transformers.tt.common import Mode


@torch.no_grad()
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("hidden_size", [3072])
@pytest.mark.parametrize("mode", [Mode.PREFILL, Mode.DECODE])
def test_voxtral_rmsnorm_pcc(device, reset_seeds, batch, seq_len, hidden_size, mode):
    eps = 1e-5

    torch_input = torch.randn(batch, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    torch_weight = torch.randn(hidden_size, dtype=torch.bfloat16)

    # Reference path uses [B, S, H].
    reference_output = reference_rms_norm(torch_input.squeeze(1), torch_weight, eps=eps)

    tt_model = VoxtralTTRMSNorm(
        device=device,
        dim=hidden_size,
        state_dict={"test_norm.weight": torch_weight},
        weight_key="test_norm",
        eps=eps,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = tt_model(tt_input, mode=mode)
    tt_output_torch = ttnn.to_torch(tt_output).squeeze(1)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc=0.999)
    assert passing, f"Voxtral RMSNorm PCC failed: {pcc_message}"
