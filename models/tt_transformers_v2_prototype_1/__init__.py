# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TT Transformers v2 (TTTv2) - A modular, composable library for transformer models

This library provides core building blocks for implementing transformer models
with minimal dependencies, following semantic versioning for stable releases.
"""

__version__ = "2.0.0"

from .core import *  # noqa
from .interfaces import *  # noqa
from .config import *  # noqa
