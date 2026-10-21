# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Blackhole datacopy."""

from ...data_transfer_blocks import BlackholeDataTransferBlocks
from ..datacopy import DataCopyGolden


class BlackholeDataCopyGolden(DataCopyGolden):
    """datacopy on Blackhole. The chain is the shared one; only the blocks differ."""

    blocks_class = BlackholeDataTransferBlocks
