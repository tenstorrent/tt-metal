# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Denoising module for TTNN PI0 implementation.

This module implements the flow matching denoising process for action generation:
    - Euler integration for ODE solving
    - KV cache management for efficient prefix reuse
    - Multi-step denoising loop

Flow Matching:
    - Start with noise x_T ~ N(0, I)
    - Integrate: dx/dt = v_θ(x_t, t)
    - Using Euler: x_{t+1} = x_t + dt * v_θ(x_t, t)
    - After T steps: x_0 = clean action prediction
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

try:
    import ttnn
    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None


@dataclass
class DenoiseConfig:
    """Configuration for denoising."""
    num_steps: int = 10
    noise_scale: float = 1.0
    action_dim: int = 32
    action_horizon: int = 50


class DenoisingModuleTorch:
    """
    Flow matching denoising module (PyTorch).
    
    Implements Euler integration to denoise actions.
    """
    
    def __init__(
        self,
        config: DenoiseConfig,
        forward_fn: Callable,
    ):
        """
        Initialize denoising module.
        
        Args:
            config: Denoising configuration
            forward_fn: Function to compute velocity v_θ(x_t, t)
                       Takes (suffix_embs, time, kv_cache) and returns velocity
        """
        self.config = config
        self.forward_fn = forward_fn
    
    def sample_noise(
        self,
        batch_size: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample initial noise for denoising.
        
        Args:
            batch_size: Number of samples
            device: Device to create tensor on
            dtype: Data type
        
        Returns:
            Noise tensor (batch_size, action_horizon, action_dim)
        """
        shape = (batch_size, self.config.action_horizon, self.config.action_dim)
        return torch.randn(shape, device=device, dtype=dtype) * self.config.noise_scale
    
    def get_timesteps(
        self,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Get timesteps for denoising.
        
        Returns linearly spaced timesteps from 1.0 to 0.0.
        
        Args:
            device: Device
            dtype: Data type
        
        Returns:
            Timesteps tensor (num_steps + 1,)
        """
        return torch.linspace(1.0, 0.0, self.config.num_steps + 1, device=device, dtype=dtype)
    
    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        prefix_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        **forward_kwargs,
    ) -> torch.Tensor:
        """
        Single Euler denoising step.
        
        Args:
            x_t: Current noisy actions (batch, horizon, action_dim)
            t: Current timestep (scalar)
            dt: Time step size (negative for forward integration)
            prefix_kv_cache: Cached K, V from prefix
            **forward_kwargs: Additional arguments for forward_fn
        
        Returns:
            Updated actions x_{t+dt}
        """
        # Compute velocity
        v_t = self.forward_fn(x_t, t, prefix_kv_cache, **forward_kwargs)
        
        # Euler step: x_{t+dt} = x_t + dt * v_t
        x_next = x_t + dt * v_t
        
        return x_next
    
    def sample_actions(
        self,
        batch_size: int,
        prefix_kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        **forward_kwargs,
    ) -> torch.Tensor:
        """
        Full denoising loop to sample actions.
        
        Args:
            batch_size: Number of action sequences to sample
            prefix_kv_cache: Cached K, V from prefix (for efficient reuse)
            device: Device
            dtype: Data type
            **forward_kwargs: Additional arguments for forward_fn
        
        Returns:
            Denoised actions (batch_size, action_horizon, action_dim)
        """
        # Sample initial noise
        x_t = self.sample_noise(batch_size, device, dtype)
        
        # Get timesteps
        timesteps = self.get_timesteps(device, dtype)
        
        # Denoising loop (host-side control)
        for i in range(self.config.num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t  # Negative since we're going from 1 to 0
            
            x_t = self.denoise_step(
                x_t,
                t.expand(batch_size),
                dt,
                prefix_kv_cache,
                **forward_kwargs,
            )
        
        return x_t


class KVCacheManager:
    """
    Manages KV cache for efficient prefix reuse.
    
    During denoising, the prefix (images + language) is processed once,
    and its KV cache is reused for all denoising steps.
    """
    
    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize KV cache.
        
        Args:
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            num_kv_heads: Number of KV heads (1 for MQA)
            head_dim: Dimension per head
            device: Device
            dtype: Data type
        """
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        self.cache = None
        self.seq_len = 0
    
    def initialize(self, batch_size: int):
        """
        Initialize empty cache for given batch size.
        
        Args:
            batch_size: Batch size
        """
        self.cache = []
        for _ in range(self.num_layers):
            k = torch.zeros(
                batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
                device=self.device, dtype=self.dtype,
            )
            v = torch.zeros(
                batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim,
                device=self.device, dtype=self.dtype,
            )
            self.cache.append((k, v))
        self.seq_len = 0
    
    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ):
        """
        Update cache with new K, V values.
        
        Args:
            layer_idx: Layer index
            new_k: New key tensor (batch, num_kv_heads, new_len, head_dim)
            new_v: New value tensor
        """
        new_len = new_k.shape[2]
        k, v = self.cache[layer_idx]
        
        k[:, :, self.seq_len:self.seq_len + new_len, :] = new_k
        v[:, :, self.seq_len:self.seq_len + new_len, :] = new_v
    
    def get(
        self,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached K, V for a layer.
        
        Args:
            layer_idx: Layer index
        
        Returns:
            Tuple of (k, v) tensors with actual sequence length
        """
        k, v = self.cache[layer_idx]
        return k[:, :, :self.seq_len, :], v[:, :, :self.seq_len, :]
    
    def increment_seq_len(self, delta: int):
        """Increment the current sequence length."""
        self.seq_len += delta


class DenoisingModuleTTNN:
    """
    Flow matching denoising module using TTNN.
    
    Uses TTNN for forward pass while keeping control flow on host.
    """
    
    def __init__(
        self,
        config: DenoiseConfig,
        forward_fn: Callable,
        device: "ttnn.Device",
    ):
        """
        Initialize denoising module.
        
        Args:
            config: Denoising configuration
            forward_fn: TTNN forward function
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.config = config
        self.forward_fn = forward_fn
        self.device = device
    
    def sample_noise(
        self,
        batch_size: int,
    ) -> "ttnn.Tensor":
        """
        Sample initial noise (on device).
        
        Args:
            batch_size: Number of samples
        
        Returns:
            TTNN noise tensor
        """
        shape = (batch_size, self.config.action_horizon, self.config.action_dim)
        noise_torch = torch.randn(shape, dtype=torch.float32) * self.config.noise_scale
        
        return ttnn.from_torch(
            noise_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    
    def denoise_step(
        self,
        x_t: "ttnn.Tensor",
        t: "ttnn.Tensor",
        dt: float,
        prefix_kv_cache: Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = None,
        **forward_kwargs,
    ) -> "ttnn.Tensor":
        """
        Single Euler denoising step using TTNN.
        
        Args:
            x_t: Current noisy actions
            t: Current timestep
            dt: Time step size
            prefix_kv_cache: Cached KV
            **forward_kwargs: Additional arguments
        
        Returns:
            Updated actions
        """
        # Compute velocity on device
        v_t = self.forward_fn(x_t, t, prefix_kv_cache, **forward_kwargs)
        
        # Euler step on device
        dt_tensor = ttnn.full_like(v_t, dt)
        scaled_v = ttnn.multiply(v_t, dt_tensor)
        x_next = ttnn.add(x_t, scaled_v)
        
        return x_next
    
    def sample_actions(
        self,
        batch_size: int,
        prefix_kv_cache: Optional[List[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = None,
        **forward_kwargs,
    ) -> "ttnn.Tensor":
        """
        Full denoising loop using TTNN.
        
        Args:
            batch_size: Number of samples
            prefix_kv_cache: Cached KV from prefix
            **forward_kwargs: Additional arguments
        
        Returns:
            Denoised actions
        """
        # Sample initial noise
        x_t = self.sample_noise(batch_size)
        
        # Get timesteps (on host for control flow)
        timesteps = torch.linspace(1.0, 0.0, self.config.num_steps + 1)
        
        # Denoising loop
        for i in range(self.config.num_steps):
            t = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t
            
            # Create timestep tensor on device
            t_tensor = ttnn.from_torch(
                torch.full((batch_size,), t, dtype=torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            
            x_t = self.denoise_step(
                x_t,
                t_tensor,
                dt,
                prefix_kv_cache,
                **forward_kwargs,
            )
        
        return x_t


class KVCacheManagerTTNN:
    """
    KV cache manager using TTNN tensors.
    
    Optimized for Tenstorrent hardware with paged attention support.
    """
    
    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: "ttnn.Device",
    ):
        """
        Initialize TTNN KV cache.
        
        Args:
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            num_kv_heads: Number of KV heads
            head_dim: Dimension per head
            device: TTNN device
        """
        if not TTNN_AVAILABLE:
            raise RuntimeError("TTNN not available")
        
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        
        self.cache = None
        self.seq_len = 0
    
    def initialize(self, batch_size: int):
        """
        Initialize cache on device.
        
        Args:
            batch_size: Batch size
        """
        self.cache = []
        for _ in range(self.num_layers):
            k_shape = (batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim)
            v_shape = (batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim)
            
            # Create zero-initialized tensors on device
            k = ttnn.from_torch(
                torch.zeros(k_shape, dtype=torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            v = ttnn.from_torch(
                torch.zeros(v_shape, dtype=torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.cache.append((k, v))
        self.seq_len = 0
    
    def update(
        self,
        layer_idx: int,
        new_k: "ttnn.Tensor",
        new_v: "ttnn.Tensor",
    ):
        """
        Update cache using TTNN fill_cache operation.
        
        Args:
            layer_idx: Layer index
            new_k: New key tensor
            new_v: New value tensor
        """
        k, v = self.cache[layer_idx]
        
        # Use fill_cache to update in-place
        ttnn.fill_cache(k, new_k, self.seq_len)
        ttnn.fill_cache(v, new_v, self.seq_len)
    
    def get(
        self,
        layer_idx: int,
    ) -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
        """
        Get cached K, V for a layer.
        
        Args:
            layer_idx: Layer index
        
        Returns:
            Tuple of (k, v) TTNN tensors
        """
        k, v = self.cache[layer_idx]
        # Slice to actual sequence length
        return k[:, :, :self.seq_len, :], v[:, :, :self.seq_len, :]
    
    def increment_seq_len(self, delta: int):
        """Increment the current sequence length."""
        self.seq_len += delta
    
    def deallocate(self):
        """Free device memory."""
        if self.cache:
            for k, v in self.cache:
                ttnn.deallocate(k)
                ttnn.deallocate(v)
            self.cache = None


# Default exports
DenoisingModule = DenoisingModuleTorch

