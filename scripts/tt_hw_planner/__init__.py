# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
tt_hw_planner — pre-flight memory planner for Tenstorrent hardware.

Given a HuggingFace model ID, this package answers:

    "Which Tenstorrent box (N150 / N300 / T3K / QB2 / Galaxy) can run this
     model, at what dtype, at what mesh shape, and with how much headroom?"

The model is purely *static* (no hardware required, no weights downloaded).
It walks the HuggingFace Hub metadata, dispatches to an architecture-specific
memory model (Dense / MLA / SlidingWindow / SSM / MoE), and applies a
hardware database with named overhead constants — replacing the single
hand-tuned safety factor used by earlier iterations.

Public entry point:
    python -m scripts.tt_hw_planner <model_id> [options]

See cli.py for the CLI surface.
"""

__version__ = "0.2.0"

from .probe import ModelProbe, probe_model  # noqa: F401
from .verdict import FitVerdict, FitRow, Tightness  # noqa: F401
from .hardware import Box, HARDWARE  # noqa: F401
from .architecture import MemoryModel  # noqa: F401
