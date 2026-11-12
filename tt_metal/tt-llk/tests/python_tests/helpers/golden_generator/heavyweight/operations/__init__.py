# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Operations, each declaring a chain of data-transfer blocks.

The operations live in :mod:`.pipelines`. Everything here is the machinery
that builds and runs them.

Ask the factory for the golden you want, naming the operation and the
architecture::

    get_golden(Op.Matmul, QUASAR, MathFidelity.HiFi4)

One function, not one per operation and not one per operation and architecture
pair, so a test that resolves either at run time hands it straight over. The
pipeline classes are exported too, for a caller that wants to subclass one or
to pass ``blocks=`` directly.
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
