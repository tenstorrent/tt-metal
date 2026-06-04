# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Oobleck VAE decoder.

Module-by-module mapping to ``torch_ref/vae/oobleck_decoder.py``:

  ``Snake1d``             -> ``TtSnake1d``
  ``OobleckResidualUnit`` -> ``TtOobleckResidualUnit``
  ``OobleckDecoderBlock`` -> ``TtOobleckDecoderBlock``
  ``OobleckDecoder``      -> ``TtOobleckDecoder``

The decoder is device-pure: weights are uploaded once during ``__init__`` and
all forward operations run on device. Inputs and outputs are TTNN tensors.
"""

from .snake import TtSnake1d
from .residual import TtOobleckResidualUnit
from .block import TtOobleckDecoderBlock
from .decoder import TtOobleckDecoder
from .weight_utils import fuse_weight_norm, fused_oobleck_decoder_weights

__all__ = [
    "TtSnake1d",
    "TtOobleckResidualUnit",
    "TtOobleckDecoderBlock",
    "TtOobleckDecoder",
    "fuse_weight_norm",
    "fused_oobleck_decoder_weights",
]
