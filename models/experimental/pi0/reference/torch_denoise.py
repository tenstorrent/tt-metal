# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Denoising module - PyTorch Reference Implementation.

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

from typing import Callable, List, Optional, Tuple

import torch

from models.experimental.pi0.common.configs import DenoiseConfig


class DenoisingModule:
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
        """
        self.config = config
        self.forward_fn = forward_fn

    def sample_noise(
        self,
        batch_size: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample initial noise for denoising."""
        shape = (batch_size, self.config.action_horizon, self.config.action_dim)
        return torch.randn(shape, device=device, dtype=dtype) * self.config.noise_scale

    def get_timesteps(
        self,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get timesteps for denoising (linearly spaced from 1.0 to 0.0)."""
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
            x_t: Current noisy actions
            t: Current timestep
            dt: Time step size
            prefix_kv_cache: Cached K, V from prefix

        Returns:
            Updated actions x_{t+dt}
        """
        v_t = self.forward_fn(x_t, t, prefix_kv_cache, **forward_kwargs)
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
            prefix_kv_cache: Cached K, V from prefix
            device: Device
            dtype: Data type

        Returns:
            Denoised actions (batch_size, action_horizon, action_dim)
        """
        x_t = self.sample_noise(batch_size, device, dtype)
        timesteps = self.get_timesteps(device, dtype)

        for i in range(self.config.num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            x_t = self.denoise_step(
                x_t,
                t.expand(batch_size),
                dt,
                prefix_kv_cache,
                **forward_kwargs,
            )

        return x_t


class KVCacheManager:
    """Manages KV cache for efficient prefix reuse during denoising."""

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.cache = None
        self.seq_len = 0

    def initialize(self, batch_size: int):
        """Initialize empty cache for given batch size."""
        self.cache = []
        for _ in range(self.num_layers):
            k = torch.zeros(
                batch_size,
                self.num_kv_heads,
                self.max_seq_len,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            v = torch.zeros(
                batch_size,
                self.num_kv_heads,
                self.max_seq_len,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            self.cache.append((k, v))
        self.seq_len = 0

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """Update cache with new K, V values."""
        new_len = new_k.shape[2]
        k, v = self.cache[layer_idx]
        k[:, :, self.seq_len : self.seq_len + new_len, :] = new_k
        v[:, :, self.seq_len : self.seq_len + new_len, :] = new_v

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached K, V for a layer."""
        k, v = self.cache[layer_idx]
        return k[:, :, : self.seq_len, :], v[:, :, : self.seq_len, :]

    def increment_seq_len(self, delta: int):
        """Increment the current sequence length."""
        self.seq_len += delta
