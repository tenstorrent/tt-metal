# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.functional import rms_norm as reference_rms_norm
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.utils.test_common import (
    load_acoustic_fm_layer_weights_or_skip,
    resolve_voxtral_model_name_or_skip,
)
from models.experimental.voxtraltts.tt.rmsnorm import VoxtralTTRMSNorm
from models.tt_transformers.tt.common import Mode


@torch.no_grad()
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("mode", [Mode.PREFILL, Mode.DECODE])
def test_voxtral_rmsnorm_pcc(device, reset_seeds, batch, seq_len, mode):
    """RMSNorm PCC at production FM dims with checkpoint ``attention_norm`` weights."""
    model_name = resolve_voxtral_model_name_or_skip()
    layer_weights = load_acoustic_fm_layer_weights_or_skip(0)
    ac_cfg = load_voxtral_config(model_name).audio_model_args.acoustic_transformer_args

    torch_weight = layer_weights["attention_norm.weight"]
    hidden_size = int(torch_weight.numel())
    eps = float(ac_cfg.sigma)

    torch.manual_seed(0)
    torch_input = torch.randn(batch, 1, seq_len, hidden_size, dtype=torch.bfloat16)

    reference_output = reference_rms_norm(torch_input.squeeze(1), torch_weight, eps=eps)

    tt_model = VoxtralTTRMSNorm(
        device=device,
        dim=hidden_size,
        state_dict={"attention_norm.weight": torch_weight},
        weight_key="attention_norm",
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
