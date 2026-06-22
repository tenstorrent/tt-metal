# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference modules for PCC tests (not a runnable demo)."""

from .e2e_model import run_torch_denoise_loop
from .full_pipeline import AceStepV15TorchPipeline

__all__ = [
    "AceStepV15TorchPipeline",
    "run_torch_denoise_loop",
]
