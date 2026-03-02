"""
PCC test: PyTorch RelPositionalEncoding vs TTNN RelPositionalEncodingTTNN.

Validates that the TTNN relative positional encoding matches the reference
(same sinusoidal formula, 2*T-1 output length, correct shape for MHA).
"""

import pytest
import torch
import ttnn

from models.experimental.parakeet.reference.pytorch_conf_layer import RelPositionalEncoding
from models.experimental.parakeet.tt.tt_pos_encode import RelPositionalEncodingTTNN
from tests.ttnn.utils_for_testing import check_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_relpos_encoding_ttnn_vs_ref_pcc(device):
    """TTNN RelPositionalEncoding output matches PyTorch reference (PCC)."""
    torch.manual_seed(0)
    d_model = 256
    max_len = 512
    batch_size = 1
    T = 32

    ref_enc = RelPositionalEncoding(d_model=d_model, max_len=max_len, return_two_values=False)
    tt_enc = RelPositionalEncodingTTNN(
        device=device,
        d_model=d_model,
        max_len=max_len,
        return_two_values=False,
    )

    x = torch.randn(batch_size, T, d_model)
    with torch.no_grad():
        ref_pos = ref_enc(x)  # (2*T-1, 1, d_model)
    tt_pos = tt_enc(x)  # TTNN tensor (1, 2*T-1, d_model)
    tt_pos_torch = ttnn.to_torch(tt_pos).float()

    # Flatten to (2*T-1, d_model) for comparison
    ref_flat = ref_pos.squeeze(0)  # (2*T-1, d_model)
    tt_flat = tt_pos_torch.squeeze(0)  # (1, 2*T-1, d_model) -> (2*T-1, d_model)

    passed, msg = check_with_pcc(ref_flat, tt_flat, pcc=0.99)
    print(f"RelPositionalEncoding PCC: {msg}")
    assert passed, f"PCC failed: {msg}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_relpos_encoding_ttnn_shape(device):
    """TTNN RelPositionalEncoding returns correct shape (1, 2*T-1, d_model)."""
    d_model = 128
    max_len = 256
    T = 16
    x = torch.randn(1, T, d_model)

    tt_enc = RelPositionalEncodingTTNN(device=device, d_model=d_model, max_len=max_len)
    pos_emb = tt_enc(x)
    pos_torch = ttnn.to_torch(pos_emb)

    expected_len = 2 * T - 1
    assert pos_torch.dim() == 3, "pos_emb should be 3D (batch, seq, d_model)"
    assert pos_torch.shape[0] == 1, "batch dim should be 1"
    assert pos_torch.shape[1] == expected_len, f"seq len should be 2*T-1={expected_len}"
    assert pos_torch.shape[2] == d_model, f"feature dim should be d_model={d_model}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_relpos_encoding_ttnn_return_two_values(device):
    """TTNN RelPositionalEncoding with return_two_values returns (x, pos_emb)."""
    d_model = 64
    max_len = 128
    T = 8
    x = torch.randn(1, T, d_model)

    tt_enc = RelPositionalEncodingTTNN(
        device=device,
        d_model=d_model,
        max_len=max_len,
        return_two_values=True,
    )
    out_x, pos_emb = tt_enc(x)

    assert out_x is x, "First return should be the input x"
    pos_torch = ttnn.to_torch(pos_emb)
    assert pos_torch.shape == (1, 2 * T - 1, d_model), "pos_emb shape (1, 2*T-1, d_model)"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("T", [8, 32, 64])
def test_relpos_encoding_ttnn_various_lengths(device, T):
    """TTNN RelPositionalEncoding matches reference for different sequence lengths."""
    torch.manual_seed(0)
    d_model = 128
    max_len = 256
    batch_size = 1

    ref_enc = RelPositionalEncoding(d_model=d_model, max_len=max_len)
    tt_enc = RelPositionalEncodingTTNN(device=device, d_model=d_model, max_len=max_len)

    x = torch.randn(batch_size, T, d_model)
    with torch.no_grad():
        ref_pos = ref_enc(x)
    tt_pos = tt_enc(x)
    tt_pos_torch = ttnn.to_torch(tt_pos).float()

    ref_flat = ref_pos.squeeze(0)
    tt_flat = tt_pos_torch.squeeze(0)
    passed, msg = check_with_pcc(ref_flat, tt_flat, pcc=0.99)
    assert passed, f"T={T}: PCC failed: {msg}"
