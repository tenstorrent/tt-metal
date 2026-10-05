# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Blackhole matmul."""

from ...data_transfer_blocks.blackhole_data_transfer import BlackholeDataTransferBlocks
from ..matmul import MatmulGolden


class BlackholeMatmulGolden(MatmulGolden):
    """matmul on Blackhole. The chain is the shared one; only the blocks differ."""

    blocks_class = BlackholeDataTransferBlocks
