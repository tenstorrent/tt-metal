import pytest
import torch

import ttnn

from models.experimental.patchtsmixer.reference.pytorch_patchtsmixer import PatchTSMixerForPretraining
from models.experimental.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerForPretraining
from models.experimental.patchtsmixer.tt.model_processing import (
    preprocess_block,
    preprocess_embedding_proj,
    preprocess_pretrain_head,
    preprocess_positional_encoding,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


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
        torch_output = torch_model(past_values)

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
    tt_output = tt_model(tt_past_values)
    tt_torch_output = ttnn.to_torch(tt_output)

    # Expected shape: (B, C, Np, patch_length)
    num_patches = (max(context_length, patch_length) - patch_length) // patch_stride + 1
    expected_shape = (batch_size, num_channels, num_patches, patch_length)

    # Validate shapes
    assert torch_output.shape == expected_shape
    assert tt_output.shape == expected_shape

    # Validate PCC
    assert_with_pcc(torch_output, tt_torch_output, pcc=0.99)
