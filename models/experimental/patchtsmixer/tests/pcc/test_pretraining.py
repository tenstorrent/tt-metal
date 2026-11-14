# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from models.experimental.patchtsmixer.reference.pytorch_patchtsmixer import PatchTSMixerForPretraining
from models.experimental.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerForPretraining
from models.experimental.patchtsmixer.tt.model_processing import (
    preprocess_block,
    preprocess_embedding_proj,
    preprocess_model,
    preprocess_pretrain_head,
    preprocess_positional_encoding,
)
from tt_py_test_utils_common.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mode", ["common_channel", "mix_channel"])
@pytest.mark.parametrize("use_gated_attn", [False, True])
def test_patchtsmixer_pretraining(device, reset_seeds, mode, use_gated_attn):
    """Test PatchTSMixer pre-training with different channel modes."""
    # Model config
    batch_size = 2
    context_length = 512
    patch_length = 8
    patch_stride = 8
    num_channels = 7
    d_model = 64
    num_layers = 2
    expansion = 2

    # Create PyTorch reference model
    torch_model = PatchTSMixerForPretraining(
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        num_channels=num_channels,
        d_model=d_model,
        num_layers=num_layers,
        mode=mode,
        expansion=expansion,
        dropout=0.0,
        use_gated_attn=use_gated_attn,
        head_dropout=0.0,
    )
    torch_model.eval()

    # Create input
    past_values = torch.randn(batch_size, context_length, num_channels)

    # PyTorch forward
    with torch.no_grad():
        torch_output, _ = torch_model(past_values)

    # Preprocess parameters
    state_dict = torch_model.state_dict()
    parameters = {}

    # Helper to extract sub-state_dict
    def extract_substate(prefix):
        if prefix:
            return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        return state_dict

    # Embedding
    embed_sd = extract_substate("embedding.")
    parameters.update(preprocess_embedding_proj(embed_sd, "embedding", device=device))

    # Positional encoding
    pos_enc_sd = extract_substate("pos_encoder.")
    parameters.update(preprocess_positional_encoding(pos_enc_sd, "pos_encoder", device=device))

    # Encoder
    encoder_sd = extract_substate("encoder.")
    parameters.update(
        preprocess_block(encoder_sd, "encoder", device, num_layers=num_layers, mode=mode, use_gated_attn=use_gated_attn)
    )

    # Head
    head_sd = extract_substate("head.")
    parameters.update(preprocess_pretrain_head(head_sd, "head", device=device))

    # Create TT model
    tt_model = TtPatchTSMixerForPretraining(
        device=device,
        base_address="",
        parameters=parameters,
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        num_channels=num_channels,
        d_model=d_model,
        num_layers=num_layers,
        mode=mode,
        expansion=expansion,
        use_gated_attn=use_gated_attn,
    )

    # TT forward
    tt_past_values = ttnn.from_torch(past_values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_output, _ = tt_model(tt_past_values)
    tt_torch_output = ttnn.to_torch(tt_output)

    # Expected shape: (B, C, Np, patch_length)
    num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1
    expected_shape = (batch_size, num_channels, num_patches, patch_length)

    # Validate shapes
    assert torch_output.shape == expected_shape
    assert tt_output.shape == expected_shape

    # Validate PCC
    assert_with_pcc(torch_output, tt_torch_output, pcc=0.99)


@pytest.mark.parametrize("mask_type", ["random", "forecast"])
def test_patchtsmixer_pretraining_masked(device, reset_seeds, mask_type):
    """Pre-training as HF runs it: per-window scaling, no positional encoding, patches masked before the encoder."""
    batch_size = 4
    context_length = 512
    patch_length = 16
    patch_stride = 16
    num_channels = 7
    d_model = 48
    num_layers = 2
    num_patches = context_length // patch_stride

    torch_model = PatchTSMixerForPretraining(
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        num_channels=num_channels,
        d_model=d_model,
        num_layers=num_layers,
        expansion=3,
        dropout=0.0,
        use_gated_attn=True,
        head_dropout=0.0,
        scaling="std",
        use_positional_encoding=False,
        mask_type=mask_type,
    ).eval()

    loc, scale = torch.linspace(-20, 40, num_channels), torch.linspace(0.5, 8, num_channels)
    past_values = torch.randn(batch_size, context_length, num_channels) * scale + loc
    with torch.no_grad():
        torch_output, mask = torch_model(past_values)

    # random: half the patches of every channel hidden, the same ones across channels; forecast: the last 2 patches
    assert mask.shape == (batch_size, num_channels, num_patches)
    masked_per_channel = num_patches - num_patches // 2 if mask_type == "random" else 2
    assert torch.equal(mask.sum(-1), torch.full((batch_size, num_channels), masked_per_channel))
    assert torch.equal(mask, mask[:, :1].expand_as(mask))
    if mask_type == "forecast":
        assert mask[..., -2:].all()

    parameters = preprocess_model(
        torch_model.state_dict(), "", device, num_layers=num_layers, use_gated_attn=True, task="pretraining"
    )
    tt_model = TtPatchTSMixerForPretraining(
        device=device,
        base_address="",
        parameters=parameters,
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        num_channels=num_channels,
        d_model=d_model,
        num_layers=num_layers,
        expansion=3,
        use_gated_attn=True,
        scaling="std",
        use_positional_encoding=False,
        mask_type=mask_type,
    )

    tt_past_values = ttnn.from_torch(past_values, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_output, tt_mask = tt_model(tt_past_values, mask=mask)

    assert tt_mask is mask
    assert_with_pcc(torch_output, ttnn.to_torch(tt_output), pcc=0.99)
