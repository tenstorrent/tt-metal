"""
Tests for DLinear model
"""
import torch
import pytest
from models.dlinear import DLinear, DLinearTTNN, SeriesDecomp


def test_dlinear_forward():
    """Test DLinear forward pass"""
    batch_size = 8
    seq_len = 96
    pred_len = 96
    enc_in = 7
    
    model = DLinear(seq_len, pred_len, enc_in, individual=False)
    x = torch.randn(batch_size, seq_len, enc_in)
    
    output = model(x)
    
    assert output.shape == (batch_size, pred_len, enc_in)
    print(f"✓ DLinear forward pass: {x.shape} -> {output.shape}")


def test_dlinear_individual():
    """Test DLinear with individual linear layers"""
    batch_size = 8
    seq_len = 96
    pred_len = 96
    enc_in = 7
    
    model = DLinear(seq_len, pred_len, enc_in, individual=True)
    x = torch.randn(batch_size, seq_len, enc_in)
    
    output = model(x)
    
    assert output.shape == (batch_size, pred_len, enc_in)
    print(f"✓ DLinear individual: {x.shape} -> {output.shape}")


def test_series_decomp():
    """Test series decomposition"""
    batch_size = 8
    seq_len = 96
    channels = 7
    
    decomp = SeriesDecomp(kernel_size=25)
    x = torch.randn(batch_size, seq_len, channels)
    
    seasonal, trend = decomp(x)
    
    assert seasonal.shape == x.shape
    assert trend.shape == x.shape
    assert torch.allclose(x, seasonal + trend, atol=1e-5)
    print(f"✓ Series decomposition: {x.shape} -> seasonal {seasonal.shape}, trend {trend.shape}")


def test_dlinear_gradient():
    """Test gradient flow in DLinear"""
    model = DLinear(96, 96, 7, individual=False)
    x = torch.randn(4, 96, 7, requires_grad=True)
    target = torch.randn(4, 96, 7)
    
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    # Check that gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed"
    print("✓ DLinear gradient flow works")


if __name__ == '__main__':
    test_dlinear_forward()
    test_dlinear_individual()
    test_series_decomp()
    test_dlinear_gradient()
    print("\n✓ All DLinear tests passed!")
