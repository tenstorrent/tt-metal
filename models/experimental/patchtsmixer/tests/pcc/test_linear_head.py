import torch
import pytest

import ttnn
from models.experimental.patchtsmixer.reference.pytorch_patchtsmixer import PatchTSMixerLinearHead
from models.experimental.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerLinearHead
from models.experimental.patchtsmixer.tt.model_processing import preprocess_linear_head
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("head_aggregation", [None, "use_last", "max_pool", "avg_pool"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_patchtsmixer_linear_head(device, head_aggregation):
    """Test LinearHead for classification/regression."""

    # Test configuration
    B = 2
    C = 7
    Np = 8
    D = 16
    num_targets = 5

    # PyTorch model
    torch_head = PatchTSMixerLinearHead(
        d_model=D,
        num_channels=C,
        num_patches=Np,
        num_targets=num_targets,
        head_agregation=head_aggregation,
        output_range=None,
        head_dropout=0.0,  # dropping out would introduce undeterministic results.
    ).eval()

    x = torch.randn(B, C, Np, D)

    with torch.no_grad():
        torch_output = torch_head(x)  # (B, num_targets)

    # Preprocess parameters for TTNN
    base = "head"

    parameters = preprocess_linear_head(torch_head.state_dict(), base, device=device)

    # Create TTNN model
    tt_head = TtPatchTSMixerLinearHead(
        device=device,
        base_address=base,
        parameters=parameters,
        num_targets=num_targets,
        head_aggregation=head_aggregation,
        output_range=None,
    )

    # Convert input to TTNN
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Get TTNN output
    tt_output = tt_head(x_tt)  # Returns torch tensor (B, num_targets)

    # Compare outputs
    print(f"\n=== LinearHead Test (aggregation={head_aggregation}) ===")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {torch_output.shape}")
    print(f"PyTorch output range: [{torch_output.min():.4f}, {torch_output.max():.4f}]")
    print(f"TTNN output range: [{tt_output.min():.4f}, {tt_output.max():.4f}]")

    assert_with_pcc(torch_output, tt_output, 0.99)
