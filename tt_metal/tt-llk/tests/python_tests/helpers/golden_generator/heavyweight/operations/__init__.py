# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Operations, each declaring a chain of data-transfer blocks.

Use an architecture's golden — ``quasar_operations``, ``blackhole_operations``,
``wormhole_operations``. The classes here are the architecture-independent
pipelines they share.
"""

from .chain import Chain, Registers, StageRecord, Step
from .datacopy import DataCopyGolden
from .eltwise import EltwiseBinaryGolden
from .golden import Golden, OpConfig
from .matmul import MatmulGolden

__all__ = [
    "Chain",
    "DataCopyGolden",
    "EltwiseBinaryGolden",
    "Golden",
    "MatmulGolden",
    "OpConfig",
    "Registers",
    "StageRecord",
    "Step",
]
