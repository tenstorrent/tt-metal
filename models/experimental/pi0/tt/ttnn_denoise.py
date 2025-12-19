# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Denoising module - TTNN Implementation.

This module implements the flow matching denoising process using TTNN:
    - Euler integration for ODE solving
    - KV cache management for efficient prefix reuse
    - Multi-step denoising loop with device acceleration

Flow Matching:
    - Start with noise x_T ~ N(0, I)
    - Integrate: dx/dt = v_θ(x_t, t)
    - Using Euler: x_{t+1} = x_t + dt * v_θ(x_t, t)
    - After T steps: x_0 = clean action prediction
"""

from typing import Callable, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0.common.configs import DenoiseConfig


class TtDenoisingModule:
    """
    Flow matching denoising module using TTNN.
    
    Uses TTNN for forward pass while keeping control flow on host.
    """
    
    def __init__(
        self,
        config: DenoiseConfig,
        forward_fn: Callable,
        device: ttnn.Device,
    ):
        """
        Initialize denoising module.
        
        Args:
            config: Denoising configuration
            forward_fn: TTNN forward function
            device: TTNN device
        """
        self.config = config
        self.forward_fn = forward_fn
        self.device = device
    
    def sample_noise(self, batch_size: int) -> ttnn.Tensor:
        """Sample initial noise (on device)."""
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
        x_t: ttnn.Tensor,
        t: ttnn.Tensor,
        dt: float,
        prefix_kv_cache: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        **forward_kwargs,
    ) -> ttnn.Tensor:
        """
        Single Euler denoising step using TTNN.
        
        Args:
            x_t: Current noisy actions
            t: Current timestep
            dt: Time step size
            prefix_kv_cache: Cached KV
        
        Returns:
            Updated actions
        """
        v_t = self.forward_fn(x_t, t, prefix_kv_cache, **forward_kwargs)
        
        # Euler step on device
        dt_tensor = ttnn.from_torch(
            torch.tensor([dt], dtype=torch.bfloat16),
            device=self.device,
        )
        scaled_v = ttnn.multiply(v_t, dt_tensor)
        x_next = ttnn.add(x_t, scaled_v)
        
        return x_next
    
    def sample_actions(
        self,
        batch_size: int,
        prefix_kv_cache: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        **forward_kwargs,
    ) -> ttnn.Tensor:
        """
        Full denoising loop to sample actions.
        
        Args:
            batch_size: Number of action sequences to sample
            prefix_kv_cache: Cached K, V from prefix
        
        Returns:
            Denoised actions (TTNN tensor)
        """
        x_t = self.sample_noise(batch_size)
        
        # Timesteps (on host for control flow)
        timesteps = torch.linspace(1.0, 0.0, self.config.num_steps + 1)
        
        for i in range(self.config.num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = float(t_next - t)
            
            # Create timestep tensor on device
            t_tensor = ttnn.from_torch(
                torch.full((batch_size,), t, dtype=torch.bfloat16),
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


class TtKVCacheManager:
    """Manages KV cache on TTNN device for efficient prefix reuse."""
    
    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: ttnn.Device,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.cache = None
        self.seq_len = 0
    
    def initialize(self, batch_size: int):
        """Initialize empty cache for given batch size."""
        self.cache = []
        for _ in range(self.num_layers):
            k = ttnn.zeros(
                (batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            v = ttnn.zeros(
                (batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            self.cache.append((k, v))
        self.seq_len = 0
    
    def get(self, layer_idx: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get cached K, V for a layer."""
        return self.cache[layer_idx]
    
    def increment_seq_len(self, delta: int):
        """Increment the current sequence length."""
        self.seq_len += delta

