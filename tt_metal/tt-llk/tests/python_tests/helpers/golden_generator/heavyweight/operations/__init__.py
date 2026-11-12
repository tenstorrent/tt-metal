# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Operations, each declaring a chain of data-transfer blocks.

The operations live in :mod:`.pipelines`. Everything here builds and runs them::

    get_golden(Op.Matmul, QUASAR, MathFidelity.HiFi4)

See `README.md`.
"""

from .arch import BLACKHOLE, QUASAR, WORMHOLE, ArchProfile, Op, get_golden
from .chain import Chain, Registers, StageRecord, Step
from .golden import Golden, OpConfig
from .pipelines.datacopy import DataCopyGolden
from .pipelines.eltwise import EltwiseBinaryGolden
from .pipelines.matmul import MatmulGolden
from .pipelines.reuse_dest import EltwiseBinaryReuseDestGolden

__all__ = [
    "BLACKHOLE",
    "QUASAR",
    "WORMHOLE",
    "ArchProfile",
    "Chain",
    "DataCopyGolden",
    "EltwiseBinaryGolden",
    "EltwiseBinaryReuseDestGolden",
    "Golden",
    "MatmulGolden",
    "OpConfig",
    "Registers",
    "Op",
    "StageRecord",
    "Step",
    "get_golden",
]
