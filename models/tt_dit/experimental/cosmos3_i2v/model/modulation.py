# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Timestep + FPS modulation notes.

Phase 1: runs on PyTorch via tt-symbiote fallback (sinusoidal embedding is
small; not a hot path). No replacement module needed.
"""

from __future__ import annotations
