# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""L1 <-> register-file data-transfer blocks.

:class:`DataTransferBlocks` is the façade; the transfers live on the units in
:mod:`.units`, and an architecture (:mod:`.arch`) is a set of those units.
"""

from .arch import ArchGolden, BlackholeArch, QuasarArch, WormholeArch
from .data_transfer_blocks import (
    BlackholeDataTransferBlocks,
    DataTransferBlocks,
    QuasarDataTransferBlocks,
    WormholeDataTransferBlocks,
)
from .l1_codec import pack_to_l1, unpack_from_l1
from .l1_support import L1Buffer, L1Support
from .pack_effects import (
    EdgeMaskMode,
    PackEdgeMask,
    apply_pack_effects,
    apply_relu,
    is_deterministic,
)
from .storage import DEST_STORAGE_FORMATS, SRC_STORAGE_FORMATS
from .units import Fpu, Pack, Unpack
from .units.pack import UnmodelledHardwareWarning

__all__ = [
    "DEST_STORAGE_FORMATS",
    "SRC_STORAGE_FORMATS",
    "ArchGolden",
    "BlackholeArch",
    "BlackholeDataTransferBlocks",
    "DataTransferBlocks",
    "EdgeMaskMode",
    "Fpu",
    "L1Buffer",
    "L1Support",
    "Pack",
    "PackEdgeMask",
    "QuasarArch",
    "QuasarDataTransferBlocks",
    "Unpack",
    "UnmodelledHardwareWarning",
    "WormholeArch",
    "WormholeDataTransferBlocks",
    "apply_pack_effects",
    "apply_relu",
    "is_deterministic",
    "pack_to_l1",
    "unpack_from_l1",
]
