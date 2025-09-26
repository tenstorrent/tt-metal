"""
Minimal mesh configuration for MoE models

Handles parallelization strategies:
- TP (Tensor Parallel): Split tensors across devices
- DP (Data Parallel): Split data across devices
- EP (Expert Parallel): Split experts across devices
"""

from .config import MeshConfig

__all__ = ["MeshConfig"]
