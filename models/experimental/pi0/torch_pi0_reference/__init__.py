"""
PI-Zero PyTorch Reference Implementation.

This package provides a modular reference implementation of the PI-Zero model
for PyTorch. The implementation is organized into separate modules for better
understanding and maintainability.

Main Components:
    - torch_pi0: Main model orchestrator
    - torch_paligemma: Vision-language backbone wrapper
    - torch_prefix: Prefix embedding (images + language)
    - torch_suffix: Suffix embedding (state + actions + time)
    - torch_denoise: Denoising logic for inference
    - torch_attention: Attention mask utilities
    - common: Shared utility functions
"""

from .torch_pi0 import PI0Pytorch

__all__ = ["PI0Pytorch"]

