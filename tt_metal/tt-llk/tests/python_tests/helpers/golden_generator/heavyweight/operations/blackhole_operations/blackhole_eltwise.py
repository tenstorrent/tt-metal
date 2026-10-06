# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Blackhole eltwise."""

from ...data_transfer_blocks.blackhole_data_transfer import BlackholeDataTransferBlocks
from ..eltwise import EltwiseBinaryGolden


class BlackholeEltwiseBinaryGolden(EltwiseBinaryGolden):
    """Element-wise binary on Blackhole.

    The chain is the shared one and only the blocks differ, with one exception
    that matters: ``MANTISSA_SPLIT`` is unset here, so there is no per-phase
    mantissa split. The multiply runs as a single exact product and
    ``math_fidelity`` is ignored entirely -- including at HiFi4, which on this
    architecture is not the exact product either. Constructing one warns
    (:func:`.fidelity.warn_unmodelled_split`). The lightweight golden does model
    these masks; see the README's "does not model" list.
    """

    blocks_class = BlackholeDataTransferBlocks
