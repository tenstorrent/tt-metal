# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 attention configuration.

Re-exports Gemma4AttentionConfig from __init__.py for convenience.
Additional attention-specific configuration helpers can be added here.
"""

from models.demos.gemma4.tt.attention import Gemma4AttentionConfig

__all__ = ["Gemma4AttentionConfig"]
