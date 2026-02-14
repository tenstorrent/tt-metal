# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Informer time-series forecasting model implementation using TTNN."""

from .ttnn_informer import DistilConfig, InformerConfig, InformerModel, TILE_SIZE, to_torch

__all__ = ["DistilConfig", "InformerConfig", "InformerModel", "TILE_SIZE", "to_torch"]
__version__ = "0.3.0"
