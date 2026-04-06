# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 expert configuration.

Re-exports Gemma4ExpertConfig from __init__.py for convenience.
"""

from models.demos.gemma4.tt.experts import Gemma4ExpertConfig

__all__ = ["Gemma4ExpertConfig"]
