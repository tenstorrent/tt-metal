# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for the Kimi-K3 vision tower (MoonViT2), the golden for `tt/`.

Weights reach a module as a host-side dataclass from `weights.py`, so how they are sourced
(checkpoint, cache, stream, synthetic) never touches a module.
"""
