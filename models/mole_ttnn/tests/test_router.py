"""
Tests for Router model
"""
import torch
import pytest
from models.router import Router, TopKRouter


def test_router_forward():
    """Test Router forward pass"""
    batch_size = 8
    seq_len = 96
    enc_in = 7
    num_experts = 4
    
    router = Router(seq_len, enc_in, num_experts)
    x = torch.randn(batch_size, seq_len, enc_in)
    
    weights = router(x)
    
    assert weights.shape == (batch_size, num_experts)
    # Check softmax property
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    assert (weights >= 0).all() and (weights <= 1).all()
    print(f"✓ Router forward: {x.shape} -> weights {weights.shape}")


def test_topk_router():
    """Test TopK Router"""
    batch_size = 8
    seq_len = 96
    enc_in = 7
    num_experts = 8
    top_k = 2
    
    router = TopKRouter(seq_len, enc_in, num_experts, top_k=top_k)
    x = torch.randn(batch_size, seq_len, enc_in)
    
    weights, aux_loss = router(x)
    
    assert weights.shape == (batch_size, num_experts)
    
    # Check top-k sparsity
    nonzero_counts = (weights > 0).sum(dim=1)
    assert (nonzero_counts == top_k).all()
    
    # Check normalization
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-4)
    print(f"✓ TopK Router: top_k={top_k}, aux_loss={aux_loss.item():.4f}")


def test_router_feature_extraction():
    """Test feature extraction"""
    batch_size = 8
    seq_len = 96
    enc_in = 7
    num_experts = 4
    
    router = Router(seq_len, enc_in, num_experts)
    x = torch.randn(batch_size, seq_len, enc_in)
    
    features = router.extract_features(x)
    
    expected_dim = enc_in * 4  # mean, std, min, max
    assert features.shape == (batch_size, expected_dim)
    print(f"✓ Feature extraction: {x.shape} -> features {features.shape}")


def test_router_gradient():
    """Test gradient flow in Router"""
    router = Router(96, 7, 4)
    x = torch.randn(4, 96, 7, requires_grad=True)
    
    weights = router(x)
    loss = weights.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in router.parameters())
    assert has_grad, "No gradients computed"
    print("✓ Router gradient flow works")


if __name__ == '__main__':
    test_router_forward()
    test_topk_router()
    test_router_feature_extraction()
    test_router_gradient()
    print("\n✓ All Router tests passed!")
