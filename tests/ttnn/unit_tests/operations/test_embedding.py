import pytest
import torch
import ttnn

def test_embedding_basic():
    """Test basic embedding functionality."""
    # Create input tensor
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    
    # Create embedding weight
    vocab_size = 10
    embed_dim = 4
    weight = torch.randn(vocab_size, embed_dim)
    
    # Convert to TTNN tensors
    ttnn_input_ids = ttnn.from_torch(input_ids, device=ttnn.open_device(0))
    ttnn_weight = ttnn.from_torch(weight, device=ttnn.open_device(0))
    
    # Perform embedding
    output = ttnn.embedding(ttnn_input_ids, ttnn_weight)
    
    # Convert back to torch for comparison
    torch_output = ttnn.to_torch(output)
    expected = torch.nn.functional.embedding(input_ids, weight)
    
    assert torch_output.shape == expected.shape
    torch.testing.assert_close(torch_output, expected, rtol=1e-3, atol=1e-3)

def test_embedding_1d_input():
    """Test embedding with 1D input."""
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    
    vocab_size = 5
    embed_dim = 3
    weight = torch.randn(vocab_size, embed_dim)
    
    ttnn_input_ids = ttnn.from_torch(input_ids, device=ttnn.open_device(0))
    ttnn_weight = ttnn.from_torch(weight, device=ttnn.open_device(0))
    
    output = ttnn.embedding(ttnn_input_ids, ttnn_weight)
    torch_output = ttnn.to_torch(output)
    expected = torch.nn.functional.embedding(input_ids, weight)
    
    assert torch_output.shape == expected.shape
    torch.testing.assert_close(torch_output, expected, rtol=1e-3, atol=1e-3)

def test_embedding_2d_input():
    """Test embedding with 2D input."""
    input_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    
    vocab_size = 5
    embed_dim = 3
    weight = torch.randn(vocab_size, embed_dim)
    
    ttnn_input_ids = ttnn.from_torch(input_ids, device=ttnn.open_device(0))
    ttnn_weight = ttnn.from_torch(weight, device=ttnn.open_device(0))
    
    output = ttnn.embedding(ttnn_input_ids, ttnn_weight)
    torch_output = ttnn.to_torch(output)
    expected = torch.nn.functional.embedding(input_ids, weight)
    
    assert torch_output.shape == expected.shape
    torch.testing.assert_close(torch_output, expected, rtol=1e-3, atol=1e-3)

def test_embedding_invalid_3d_input():
    """Test that 3D input raises an error."""
    # Create 3D input tensor
    input_ids = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.long)
    
    vocab_size = 10
    embed_dim = 4
    weight = torch.randn(vocab_size, embed_dim)
    
    ttnn_input_ids = ttnn.from_torch(input_ids, device=ttnn.open_device(0))
    ttnn_weight = ttnn.from_torch(weight, device=ttnn.open_device(0))
    
    with pytest.raises(RuntimeError, match="Input tensor rank must be <= 2"):
        ttnn.embedding(ttnn_input_ids, ttnn_weight)

def test_embedding_invalid_4d_input():
    """Test that 4D input raises an error."""
    # Create 4D input tensor
    input_ids = torch.tensor([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], dtype=torch.long)
    
    vocab_size = 10
    embed_dim = 4
    weight = torch.randn(vocab_size, embed_dim)
    
    ttnn_input_ids = ttnn.from_torch(input_ids, device=ttnn.open_device(0))
    ttnn_weight = ttnn.from_torch(weight, device=ttnn.open_device(0))
    
    with pytest.raises(RuntimeError, match="Input tensor rank must be <= 2"):
        ttnn.embedding(ttnn_input_ids, ttnn_weight)

def test_embedding_invalid_5d_input():
    """Test that 5D input raises an error."""
    # Create 5D input tensor
    input_ids = torch.tensor([[[[[1]]]]], dtype=torch.long)
    
    vocab_size = 10
    embed_dim = 4
    weight = torch.randn(vocab_size, embed_dim)
    
    ttnn_input_ids = ttnn.from_torch(input_ids, device=ttnn.open_device(0))
    ttnn_weight = ttnn.from_torch(weight, device=ttnn.open_device(0))
    
    with pytest.raises(RuntimeError, match="Input tensor rank must be <= 2"):
        ttnn.embedding(ttnn_input_ids, ttnn_weight)

def test_embedding_edge_case_large_rank():
    """Test embedding with very high dimensional input (should fail)."""
    # Create 6D input tensor
    shape = [2] * 6
    input_ids = torch.ones(shape, dtype=torch.long)
    
    vocab_size = 10
    embed_dim = 4
    weight = torch.randn(vocab_size, embed_dim)
    
    ttnn_input_ids = ttnn.from_torch(input_ids, device=ttnn.open_device(0))
    ttnn_weight = ttnn.from_torch(weight, device=ttnn.open_device(0))
    
    with pytest.raises(RuntimeError, match="Input tensor rank must be <= 2"):
        ttnn.embedding(ttnn_input_ids, ttnn_weight)

def test_embedding_scalar_input():
    """Test embedding with scalar input (0D tensor)."""
    input_ids = torch.tensor(2, dtype=torch.long)  # scalar tensor
    
    vocab_size = 5
    embed_dim = 3
    weight = torch.randn(vocab_size, embed_dim)
    
    ttnn_input_ids = ttnn.from_torch(input_ids, device=ttnn.open_device(0))
    ttnn_weight = ttnn.from_torch(weight, device=ttnn.open_device(0))
    
    output = ttnn.embedding(ttnn_input_ids, ttnn_weight)
    torch_output = ttnn.to_torch(output)
    expected = torch.nn.functional.embedding(input_ids, weight)
    
    assert torch_output.shape == expected.shape
    torch.testing.assert_close(torch_output, expected, rtol=1e-3, atol=1e-3)

def test_embedding_out_of_bounds():
    """Test embedding with out of bounds indices."""
    input_ids = torch.tensor([0, 9], dtype=torch.long)  # index 9 is out of bounds for vocab_size=5
    
    vocab_size = 5
    embed_dim = 3
    weight = torch.randn(vocab_size, embed_dim)
    
    ttnn_input_ids = ttnn.from_torch(input_ids, device=ttnn.open_device(0))
    ttnn_weight = ttnn.from_torch(weight, device=ttnn.open_device(0))
    
    with pytest.raises(RuntimeError):
        ttnn.embedding(ttnn_input_ids, ttnn_weight)

def test_embedding_negative_indices():
    """Test embedding with negative indices (should fail)."""
    input_ids = torch.tensor([-1, 2], dtype=torch.long)
    
    vocab_size = 5
    embed_dim = 3
    weight = torch.randn(vocab_size, embed_dim)
    
    ttnn_input_ids = ttnn.from_torch(input_ids, device=ttnn.open_device(0))
    ttnn_weight = ttnn.from_torch(weight, device=ttnn.open_device(0))
    
    with pytest.raises(RuntimeError):
        ttnn.embedding(ttnn_input_ids, ttnn_weight)