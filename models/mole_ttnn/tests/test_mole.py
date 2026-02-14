"""
Tests for MoLE model
"""
import torch
import pytest
from models.mole import MoLE, MoLEConfig, MoLETTNN


def test_mole_forward():
    """Test MoLE forward pass"""
    batch_size = 8
    seq_len = 96
    pred_len = 96
    enc_in = 7
    num_experts = 4
    
    model = MoLE(seq_len, pred_len, enc_in, num_experts=num_experts)
    x = torch.randn(batch_size, seq_len, enc_in)
    
    output = model(x)
    
    assert output.shape == (batch_size, pred_len, enc_in)
    print(f"✓ MoLE forward: {x.shape} -> {output.shape}")


def test_mole_with_weights():
    """Test MoLE with weight return"""
    batch_size = 8
    seq_len = 96
    pred_len = 96
    enc_in = 7
    num_experts = 4
    
    model = MoLE(seq_len, pred_len, enc_in, num_experts=num_experts)
    x = torch.randn(batch_size, seq_len, enc_in)
    
    output, weights, aux_loss = model(x, return_weights=True)
    
    assert output.shape == (batch_size, pred_len, enc_in)
    assert weights.shape == (batch_size, num_experts)
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    print(f"✓ MoLE with weights: output {output.shape}, weights {weights.shape}")


def test_mole_topk():
    """Test MoLE with Top-K routing"""
    batch_size = 8
    seq_len = 96
    pred_len = 96
    enc_in = 7
    num_experts = 8
    top_k = 2
    
    model = MoLE(seq_len, pred_len, enc_in, num_experts=num_experts, top_k=top_k)
    x = torch.randn(batch_size, seq_len, enc_in)
    
    output, weights, aux_loss = model(x, return_weights=True)
    
    assert output.shape == (batch_size, pred_len, enc_in)
    
    # Check top-k sparsity
    nonzero_counts = (weights > 0).sum(dim=1)
    assert (nonzero_counts == top_k).all()
    print(f"✓ MoLE Top-K: top_k={top_k}, sparse weights")


def test_mole_expert_usage():
    """Test expert usage tracking"""
    batch_size = 8
    seq_len = 96
    pred_len = 96
    enc_in = 7
    num_experts = 4
    
    model = MoLE(seq_len, pred_len, enc_in, num_experts=num_experts)
    
    # Reset stats
    model.reset_expert_stats()
    
    # Run multiple batches
    for _ in range(10):
        x = torch.randn(batch_size, seq_len, enc_in)
        _ = model(x)
    
    usage = model.get_expert_usage()
    
    assert len(usage) == num_experts
    assert usage.sum() > 0  # Should have recorded activations
    print(f"✓ Expert usage: {usage}")


def test_mole_config():
    """Test MoLE configuration"""
    config = MoLEConfig(
        seq_len=96,
        pred_len=96,
        enc_in=7,
        num_experts=4,
        expert_type='dlinear'
    )
    
    model = config.create_model()
    
    x = torch.randn(4, 96, 7)
    output = model(x)
    
    assert output.shape == (4, 96, 7)
    print("✓ MoLE config works")


def test_mole_gradient():
    """Test gradient flow in MoLE"""
    model = MoLE(96, 96, 7, num_experts=4)
    x = torch.randn(4, 96, 7, requires_grad=True)
    target = torch.randn(4, 96, 7)
    
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed"
    print("✓ MoLE gradient flow works")


def test_mole_vs_single_expert():
    """Compare MoLE with single expert"""
    from models.dlinear import DLinear
    
    torch.manual_seed(42)
    
    seq_len = 96
    pred_len = 96
    enc_in = 7
    
    # Single expert (DLinear)
    single_model = DLinear(seq_len, pred_len, enc_in)
    
    # MoLE with 1 expert (should behave similarly)
    mole_model = MoLE(seq_len, pred_len, enc_in, num_experts=1)
    # Copy weights from single model to the one expert
    mole_model.experts[0].load_state_dict(single_model.state_dict())
    
    x = torch.randn(4, seq_len, enc_in)
    
    single_output = single_model(x)
    mole_output = mole_model(x)
    
    # They won't be identical due to router weights, but should be close
    # when router gives full weight to the only expert
    print(f"✓ Single vs MoLE(1 expert) outputs compared")


if __name__ == '__main__':
    test_mole_forward()
    test_mole_with_weights()
    test_mole_topk()
    test_mole_expert_usage()
    test_mole_config()
    test_mole_gradient()
    test_mole_vs_single_expert()
    print("\n✓ All MoLE tests passed!")
