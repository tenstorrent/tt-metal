"""
Common utility functions used across PI-Zero PyTorch modules.

This module provides shared helper functions that don't belong to a specific
component but are used by multiple modules (suffix, denoise, etc.).

Use Case:
    - Provides device-aware dtype selection
    - Implements timestep positional encoding for flow matching
    - Provides noise and time sampling functions for training
    - Handles safe tensor operations with dtype conversion
"""

import math

import torch
from torch import Tensor


def get_safe_dtype(target_dtype, device_type):
    """
    Get a safe dtype for the given device type.
    
    Args:
        target_dtype: Desired dtype (e.g., torch.bfloat16)
        device_type: Device type string ("cpu" or "cuda")
    
    Returns:
        Safe dtype that works on the specified device
    
    Use Case:
        Ensures CPU compatibility by converting bfloat16 to float32,
        since CPU doesn't support bfloat16 operations.
    """
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """
    Computes sine-cosine positional embedding vectors for scalar positions.
    
    This is used for timestep encoding in flow matching, where we need to
    represent continuous time values as embeddings.
    
    Args:
        time: Tensor of shape (batch_size,) containing timestep values
        dimension: Embedding dimension (must be divisible by 2)
        min_period: Minimum period for sinusoidal encoding
        max_period: Maximum period for sinusoidal encoding
        device: Device to create tensors on
    
    Returns:
        Tensor of shape (batch_size, dimension) with sinusoidal embeddings
    
    Use Case:
        Encodes continuous timestep values (0-1) into high-dimensional
        embeddings for the flow matching denoising process. Used in
        suffix embedding to condition actions on the current timestep.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    """
    Sample from a Beta distribution.
    
    Args:
        alpha: Alpha parameter of Beta distribution
        beta: Beta parameter of Beta distribution
        bsize: Batch size
        device: Device to create tensors on
    
    Returns:
        Tensor of shape (bsize,) with Beta-distributed samples
    
    Use Case:
        Used for sampling timesteps during training. The Beta distribution
        provides a good prior for timestep sampling in flow matching.
    """
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def sample_noise(shape, device):
    """
    Sample Gaussian noise for flow matching.
    
    Args:
        shape: Shape tuple for the noise tensor
        device: Device to create tensors on
    
    Returns:
        Tensor of specified shape with standard normal noise
    
    Use Case:
        Generates random noise that serves as the starting point for
        the denoising process during inference, or as noise to add
        to actions during training.
    """
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )


def sample_time(bsize, device):
    """
    Sample timesteps from a Beta distribution for training.
    
    Args:
        bsize: Batch size
        device: Device to create tensors on
    
    Returns:
        Tensor of shape (bsize,) with timesteps in [0.001, 0.999]
    
    Use Case:
        Samples random timesteps during training to learn the velocity
        field at different points in the flow matching process.
    """
    time_beta = sample_beta(1.5, 1.0, bsize, device)
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32, device=device)


def compute_position_ids(pad_masks):
    """
    Compute position IDs from padding masks.
    
    Args:
        pad_masks: Boolean tensor of shape (batch_size, seq_len) where
                   True indicates valid tokens
    
    Returns:
        Tensor of shape (batch_size, seq_len) with position indices
    
    Use Case:
        Creates position IDs for transformer models by computing cumulative
        sums of padding masks. This ensures padding tokens don't get position
        embeddings and valid tokens get sequential positions.
    """
    return torch.cumsum(pad_masks, dim=1) - 1


def safe_cat(tensors, dim):
    """
    Safely concatenate tensors with dtype handling.
    
    Args:
        tensors: List of tensors to concatenate
        dim: Dimension along which to concatenate
    
    Returns:
        Concatenated tensor
    
    Use Case:
        Concatenates tensors that may have different dtypes, ensuring
        compatibility before concatenation. Used when combining embeddings
        from different sources (images, language, state, actions).
    """
    # Convert all tensors to the same dtype (use the first tensor's dtype)
    if len(tensors) > 0:
        target_dtype = tensors[0].dtype
        tensors = [t.to(dtype=target_dtype) if t.dtype != target_dtype else t for t in tensors]
    return torch.cat(tensors, dim=dim)

