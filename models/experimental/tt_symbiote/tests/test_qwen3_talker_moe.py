#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests TTNN Qwen3-Omni talker MoE block against PyTorch implementation."""

import os

import pytest
import torch
import ttnn

from transformers import Qwen3OmniMoeForConditionalGeneration

from models.experimental.tt_symbiote.modules.moe import TTNNQwen3TalkerMoE
from models.experimental.tt_symbiote.utils.device_management import set_device


REAL_WEIGHTS_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

_MESH_DEVICE_ENV = "MESH_DEVICE"
if _MESH_DEVICE_ENV not in os.environ:
    os.environ[_MESH_DEVICE_ENV] = "T3K"
MESH_DEVICE = os.environ.get(_MESH_DEVICE_ENV, "T3K")


@pytest.mark.parametrize(
    "device_params",
    [
        {"l1_small_size": 245760, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING},
    ],
    indirect=True,
)
def test_qwen3_talker_moe_ttnn_matches_torch(mesh_device):
    """Verify TTNNQwen3TalkerMoE matches PyTorch Qwen3OmniMoeTalkerTextSparseMoeBlock (via PCC)."""
    torch.manual_seed(0)

    # Load full Qwen3-Omni-MoE model and grab a talker text MoE block with real config.
    full_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        REAL_WEIGHTS_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    full_model.eval()

    # Take the first talker text MoE block: talker.model.layers[0].mlp
    hf_block = full_model.talker.model.layers[0].mlp
    hf_block.eval()

    # Infer hidden size from the shared expert MLP.
    hidden_size = hf_block.shared_expert.hidden_size

    batch_size, seq_len = 1, 32
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        hf_out = hf_block(x)

    # Create TTNN implementation from the same HF block.
    ttnn_block = TTNNQwen3TalkerMoE.from_torch(hf_block)
    set_device(ttnn_block, mesh_device)

    with torch.no_grad():
        ttnn_out = ttnn_block(x)

    # Sanity check: shapes must match.
    assert hf_out.shape == ttnn_out.shape

    # Explicit PCC check for clarity between PyTorch and TTNN paths.
    hf_flat = hf_out.flatten().to(torch.float32)
    # Ensure we compare on plain torch tensors (avoid TorchTTNNTensor dispatch quirks).
    if hasattr(ttnn_out, "to_torch"):
        ttnn_out_torch = ttnn_out.to_torch
    else:
        ttnn_out_torch = ttnn_out
    ttnn_flat = ttnn_out_torch.flatten().to(torch.float32)
    pcc = torch.corrcoef(torch.stack([hf_flat, ttnn_flat]))[0, 1]
    print(f"PCC TTNNQwen3TalkerMoE_vs_Qwen3OmniMoeTalkerTextSparseMoeBlock: {pcc.item()}")
    assert not torch.isnan(pcc), "PCC is NaN between HF and TTNN Qwen3 talker MoE outputs"
    assert pcc > 0.99, f"PCC too low between HF and TTNN Qwen3 talker MoE outputs: {pcc.item()}"
