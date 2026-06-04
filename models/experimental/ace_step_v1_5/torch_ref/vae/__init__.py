# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for the ACE-Step VAE (diffusers' ``AutoencoderOobleck``).

Mirrors the diffusers Oobleck decoder verbatim so that PCC tests can run without
pulling in the entire ``diffusers`` package. The forward pass and weight layout
are identical, so a state dict produced by ``AutoencoderOobleck.state_dict()``
loads into the modules defined here.
"""

from .oobleck_decoder import (
    OobleckDecoder,
    OobleckDecoderBlock,
    OobleckDecoderConfig,
    OobleckResidualUnit,
    Snake1d,
)

__all__ = [
    "OobleckDecoder",
    "OobleckDecoderBlock",
    "OobleckDecoderConfig",
    "OobleckResidualUnit",
    "Snake1d",
]
