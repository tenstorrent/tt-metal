# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wormhole data-transfer blocks.

The architecture itself is :class:`..arch.WormholeArch` -- a set of units, not a
subclass. This module only re-exports the factory, so existing imports keep
working.
"""

from ..data_transfer_blocks import WormholeDataTransferBlocks

__all__ = ["WormholeDataTransferBlocks"]
