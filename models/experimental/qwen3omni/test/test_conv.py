# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for qwen3omni TT conv modules (CausalConvNet, CausalTransConvNet, ConvNeXtBlock).
Loads checkpoints the same way as test_patchembed.py (direct safetensors path + load_file).
ConvNeXtBlock weights live in the final shard (model-00015-of-00015.safetensors).
"""

import os

import torch
import pytest
import ttnn
from safetensors.torch import load_file

from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.qwen3omni.tt.conv import TtQwen3OmniMoeConvNeXtBlock


MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
# CHECKPOINT_PATH = "/home/ubuntu/tt-metal/models/experimental/qwen3omni/checkpoints/model-00001-of-00015.safetensors"
# code2wav ConvNeXtBlock / upsample weights are in the last shard (not in 00001)
CODE2WAV_CHECKPOINT_PATH = (
    "/home/ubuntu/tt-metal/models/experimental/qwen3omni/checkpoints/model-00015-of-00015.safetensors"
)

# ConvNeXtBlock weights in the model live under code2wav.upsample.<i>.1 (i=0,1,...)
CONV_NEXT_PREFIX = "code2wav.upsample."
CONV_NEXT_SUFFIX = ".1."  # block index 1 in each upsample ModuleList is ConvNeXtBlock

# Code2Wav config (Qwen3OmniMoeCode2WavConfig)
CODE2WAV_HIDDEN_SIZE = 1024


# ------------------------------------------------------------
# Checkpoint sufficiency (same style as test_patchembed: explicit paths + load_file)
# ------------------------------------------------------------
def test_conv_checkpoints_sufficient():
    """
    ConvNeXtBlock weights are in model-00015-of-00015.safetensors (not shard 00001).
    Vision / patch-embed tests use CHECKPOINT_PATH (00001); conv tests need CODE2WAV_CHECKPOINT_PATH.
    """
    assert os.path.isfile(CODE2WAV_CHECKPOINT_PATH), (
        f"Conv tests need {CODE2WAV_CHECKPOINT_PATH} for {MODEL_NAME}. "
        "Shard 00001 (CHECKPOINT_PATH) does not contain code2wav.upsample ConvNeXt weights. "
        "Download model-00015-of-00015.safetensors (see models/experimental/qwen3omni/download.py)."
    )


# ------------------------------------------------------------
# Load one ConvNeXtBlock state from checkpoint
# ------------------------------------------------------------
def load_conv_next_block_state(block_index=0):
    """
    Load state dict for a single ConvNeXtBlock from checkpoint.
    block_index 0 -> code2wav.upsample.0.1.*, block_index 1 -> code2wav.upsample.1.1.*, etc.
    """
    if not os.path.isfile(CODE2WAV_CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint not found: {CODE2WAV_CHECKPOINT_PATH}")
    state = load_file(CODE2WAV_CHECKPOINT_PATH)
    prefix = f"{CONV_NEXT_PREFIX}{block_index}{CONV_NEXT_SUFFIX}"
    block_state = {k: v for k, v in state.items() if k.startswith(prefix)}
    assert block_state, f"No keys with prefix {prefix} in {CODE2WAV_CHECKPOINT_PATH}"
    return block_state, prefix


# ------------------------------------------------------------
# PyTorch reference (ConvNeXtBlock)
# ------------------------------------------------------------
def get_torch_conv_next_block(dim: int):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeConvNeXtBlock,
    )

    return Qwen3OmniMoeConvNeXtBlock(dim)


def _tt_tensor_to_torch(tt_tensor, mesh_device):
    """
    Read TT tensor to host torch. Avoid create_submesh(1,1): it can leave L1_SMALL banks at 0 B.
    On multi-device, ttnn.to_torch(mesh_composer=...) may not bind; use per-device shard [0] like gpt_oss/sdxl.
    """
    n = getattr(mesh_device, "get_num_devices", lambda: 1)()
    if n <= 1:
        with ttnn.distribute(ttnn.ConcatMeshToTensor(mesh_device, dim=0)):
            return ttnn.to_torch(tt_tensor).float()
    shards = ttnn.get_device_tensors(tt_tensor)
    return ttnn.to_torch(shards[0]).float()


# ------------------------------------------------------------
# Test: ConvNeXtBlock with real weights (TT vs PyTorch)
# ------------------------------------------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
def test_qwen_conv_next_block_real_weights(mesh_device):
    """Compare TtQwen3OmniMoeConvNeXtBlock vs PyTorch ConvNeXtBlock with real checkpoint weights (PCC)."""
    torch.manual_seed(0)

    conv_device = mesh_device
    block_state, prefix = load_conv_next_block_state(block_index=0)
    dim = CODE2WAV_HIDDEN_SIZE

    # PyTorch reference: match bfloat16 input (Conv1d/LayerNorm/Linear must all be bf16)
    torch_block = get_torch_conv_next_block(dim)
    torch_state = {k[len(prefix) :]: v for k, v in block_state.items()}
    torch_block.load_state_dict(torch_state, strict=True)
    torch_block.to(torch.bfloat16)

    tt_block = TtQwen3OmniMoeConvNeXtBlock(device=conv_device, dim=dim)
    tt_block.load_state_dict(block_state, strict=True, prefix=prefix)
    tt_block.preprocess_weights()
    tt_block.move_weights_to_device()

    # TT block uses (B, L, C). HuggingFace Code2Wav passes (B, C, L) into upsample blocks
    # (see modeling: hidden = hidden.permute(0, 2, 1) before the upsample loop).
    batch_size, seq_len = 1, 64
    hidden_states_torch = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)

    # PyTorch forward: (B, C, L) in / out
    with torch.no_grad():
        torch_in = hidden_states_torch.permute(0, 2, 1).contiguous()
        torch_out_bcl = torch_block(torch_in)
    torch_out = torch_out_bcl.permute(0, 2, 1).contiguous()  # (B, L, C) for PCC vs TT

    # TT forward: (B, L, C) on device — replicate on multi-device so dwconv's host round-trip sees full shards
    if conv_device.get_num_devices() > 1:
        with ttnn.distribute(ttnn.ReplicateTensorToMesh(conv_device)):
            hidden_states_tt = ttnn.from_torch(
                hidden_states_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=conv_device,
            )
    else:
        hidden_states_tt = ttnn.from_torch(
            hidden_states_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=conv_device,
        )
    tt_out = tt_block.forward(hidden_states_tt)
    if isinstance(tt_out, TorchTTNNTensor) and getattr(tt_out, "ttnn_tensor", None) is not None:
        tt_out = tt_out.ttnn_tensor
    tt_out_torch = _tt_tensor_to_torch(tt_out, mesh_device)
    if tt_out_torch.shape[0] != torch_out.shape[0]:
        tt_out_torch = tt_out_torch[: torch_out.shape[0]]

    # PCC
    pcc = torch.corrcoef(torch.stack([torch_out.flatten().float(), tt_out_torch.flatten()]))[0, 1].item()
    print(f"ConvNeXtBlock PCC: {pcc}")
    assert pcc > 0.99, f"ConvNeXtBlock mismatch with real weights (PCC={pcc})"
