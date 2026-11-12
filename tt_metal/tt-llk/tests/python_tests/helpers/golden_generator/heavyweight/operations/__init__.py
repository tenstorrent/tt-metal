# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Operations, each declaring a chain of data-transfer blocks.

The operations live in :mod:`.pipelines`. Everything here is the machinery
that builds and runs them.

Ask a factory for the golden you want, naming the architecture first::

    get_matmul_golden(QUASAR, MathFidelity.HiFi4)

There is one factory per operation, not one per operation and architecture, so
a test that resolves its own architecture at run time hands the profile over
rather than picking a name. The pipeline classes are exported too, for a caller
that wants to subclass one or to pass ``blocks=`` directly.
"""

from .arch import (
    BLACKHOLE,
    QUASAR,
    WORMHOLE,
    ArchProfile,
    get_data_copy_golden,
    get_eltwise_binary_golden,
    get_eltwise_binary_reuse_dest_golden,
    get_matmul_golden,
)
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
    "StageRecord",
    "Step",
    "get_data_copy_golden",
    "get_eltwise_binary_golden",
    "get_eltwise_binary_reuse_dest_golden",
    "get_matmul_golden",
]
