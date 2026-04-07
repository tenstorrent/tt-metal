# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim — DeepSeek **pack/fuse** implementation is in ``weights.fusion_runtime``.

:class:`TensorCache` imports ``create_overlapped_tensor`` from this module to avoid pulling
model code into generic cache modules at package import time; the implementation delegates
to :mod:`models.demos.deepseek_v3_b1.weights.fusion_runtime`.
"""

from __future__ import annotations

from models.demos.deepseek_v3_b1.weights.fusion_runtime import create_overlapped_tensor

__all__ = ["create_overlapped_tensor"]
