# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wormhole eltwise."""

from ...data_transfer_blocks.wormhole_data_transfer import WormholeDataTransferBlocks
from ..eltwise import EltwiseBinaryGolden


class WormholeEltwiseBinaryGolden(EltwiseBinaryGolden):
    """Element-wise binary on Wormhole.

    The chain is the shared one and only the blocks differ, with one exception
    that matters: ``MANTISSA_SPLIT`` is unset here, so there is no per-phase
    mantissa split. The multiply runs as a single exact product and
    ``math_fidelity`` is ignored entirely -- including at HiFi4, which on this
    architecture is not the exact product either. Constructing one warns
    (:func:`.fidelity.warn_unmodelled_split`). The lightweight golden does model
    these masks; see the README's "does not model" list.
    """

    blocks_class = WormholeDataTransferBlocks
