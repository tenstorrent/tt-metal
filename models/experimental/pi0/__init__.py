# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Model Implementation for Tenstorrent.

This package provides both PyTorch reference implementations and TTNN
implementations for the PI0 robotics model.

Structure:
    - common/: Shared configs and utilities
    - reference/: Pure PyTorch implementations (torch_*.py)
    - tt/: TTNN implementations (ttnn_*.py)
    - tests/pcc/: PCC tests comparing TTNN vs PyTorch
    - tests/perf/: Performance benchmarks
    - ttnn_pi0_reference/: Original combined implementation (legacy)

Usage:
    # PyTorch reference
    from models.experimental.pi0.reference import GemmaBlock, SigLIPVisionTower

    # TTNN implementation
    from models.experimental.pi0.tt import TtGemmaBlock, TtSigLIPVisionTower

    # Configs
    from models.experimental.pi0.common import GemmaConfig, PI0ModelConfig

Model Architecture:
    PI0 is a vision-language-action model with:
    - SigLIP Vision Tower: Processes images (27 transformer blocks)
    - Gemma 2B VLM: Language model backbone (18 transformer blocks)
    - Gemma 300M Expert: Action generation (18 transformer blocks)
    - Flow matching denoiser: Generates actions over 10 steps
"""

__version__ = "0.1.0"
