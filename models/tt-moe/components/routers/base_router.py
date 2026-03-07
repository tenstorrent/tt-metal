# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Base router interface for unified MoE implementation."""

from abc import ABC, abstractmethod

import ttnn


class BaseRouter(ABC):
    """
    Abstract base class for MoE routers.

    All router implementations must inherit from this class and implement
    the forward method which returns expert indices and weights.
    """

    @abstractmethod
    def __init__(self, config: dict, mesh_device: ttnn.MeshDevice):
        """
        Initialize the router with configuration and device.

        Args:
            config: Router configuration dictionary from JSON
            mesh_device: TTNN mesh device for tensor placement
        """

    @abstractmethod
    def forward(self, x: ttnn.Tensor, mode: str = "decode"):
        """
        Forward pass through the router.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim] or flattened
            mode: "decode" or "prefill" mode

        Returns:
            Tuple of:
                - indices: Expert indices for each token
                - weights: Normalized weights for selected experts
        """

    @abstractmethod
    def load_weights(self, state_dict: dict, weight_path: str = None):
        """
        Load router weights from state dict or cached path.

        Args:
            state_dict: Dictionary containing router weights
            weight_path: Optional path for cached weights
        """
