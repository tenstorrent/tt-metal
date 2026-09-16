# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Quasar element-wise binary with Dest reuse."""

from ...data_transfer_blocks.quasar_data_transfer import QuasarDataTransferBlocks
from ..reuse_dest import EltwiseBinaryReuseDestGolden


class QuasarEltwiseBinaryReuseDestGolden(EltwiseBinaryReuseDestGolden):
    """Dest-reuse element-wise binary on Quasar."""

    blocks_class = QuasarDataTransferBlocks
    MANTISSA_SPLIT = (7, 7)
