# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Quasar datacopy."""

from ...data_transfer_blocks.quasar_data_transfer import QuasarDataTransferBlocks
from ..datacopy import DataCopyGolden


class QuasarDataCopyGolden(DataCopyGolden):
    """Datacopy on Quasar. The chain is the shared one; only the blocks differ."""

    blocks_class = QuasarDataTransferBlocks
