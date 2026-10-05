# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Blackhole eltwise."""

from ...data_transfer_blocks.blackhole_data_transfer import BlackholeDataTransferBlocks
from ..eltwise import EltwiseBinaryGolden


class BlackholeEltwiseBinaryGolden(EltwiseBinaryGolden):
    """eltwise on Blackhole. The chain is the shared one; only the blocks differ."""

    blocks_class = BlackholeDataTransferBlocks
