import pytest
import torch

import ttnn

from models.experimental.patchtsmixer.reference.pytorch_patchtsmixer import PatchTSMixerForRegression
from models.experimental.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerForRegression
from models.experimental.patchtsmixer.tt.model_processing import (
    preprocess_block,
    preprocess_embedding_proj,
    preprocess_linear_head,
    preprocess_positional_encoding,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("head_aggregation", ["avg_pool", "max_pool", "use_last"])
def test_patchtsmixer_regression_aggregations(device, reset_seeds, head_aggregation):
    """Test PatchTSMixer regression with different aggregation modes."""
    # Model config
    batch_size = 2
    context_length = 512
    patch_length = 8
    patch_stride = 8
    num_channels = 7
    d_model = 64
    num_layers = 2
    num_targets = 3
    mode = "common_channel"
    expansion = 2

    # Create PyTorch reference model
    torch_model = PatchTSMixerForRegression(
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        num_channels=num_channels,
        d_model=d_model,
        num_layers=num_layers,
        num_targets=num_targets,
        mode=mode,
        expansion=expansion,
        dropout=0.0,
        use_gated_attn=False,
        head_dropout=0.0,
    )
    # Update head aggregation
    torch_model.head.head_aggregation = head_aggregation
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
        preprocess_block(encoder_sd, "encoder", device, num_layers=num_layers, mode=mode, use_gated_attn=False)
    )

    # Head
    head_sd = extract_substate("head.")
    parameters.update(preprocess_linear_head(head_sd, "head", device=device))

    # Create TT model
    tt_model = TtPatchTSMixerForRegression(
        device=device,
        base_address="",
        parameters=parameters,
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        num_channels=num_channels,
        d_model=d_model,
        num_layers=num_layers,
        num_targets=num_targets,
        mode=mode,
        expansion=expansion,
        use_gated_attn=False,
        head_aggregation=head_aggregation,
    )

    # TT forward
    tt_past_values = ttnn.from_torch(past_values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_output = tt_model(tt_past_values)
    tt_torch_output = ttnn.to_torch(tt_output)
    tt_torch_output = tt_torch_output.squeeze(dim=(0, 1))  # [1, 1, X, Y] -> torch [X, Y]

    # Validate PCC
    assert_with_pcc(torch_output, tt_torch_output, pcc=0.99)
