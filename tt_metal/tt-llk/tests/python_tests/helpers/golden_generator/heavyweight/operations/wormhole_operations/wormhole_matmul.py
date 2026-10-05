# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wormhole matmul."""

from ...data_transfer_blocks.wormhole_data_transfer import WormholeDataTransferBlocks
from ..matmul import MatmulGolden


class WormholeMatmulGolden(MatmulGolden):
    """matmul on Wormhole. The chain is the shared one; only the blocks differ."""

    blocks_class = WormholeDataTransferBlocks
