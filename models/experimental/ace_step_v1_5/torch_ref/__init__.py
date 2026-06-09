# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference modules for PCC tests (not a runnable demo)."""

from .full_pipeline import AceStepV15TorchPipeline

__all__ = [
    "AceStepV15TorchPipeline",
]
