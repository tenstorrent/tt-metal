# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Operations, each declaring a chain of data-transfer blocks.

The operations live in :mod:`.pipelines`; everything here is the machinery
that builds and runs them.

Use an architecture's golden -- ``QuasarMatmulGolden``,
``WormholeEltwiseBinaryGolden`` and so on, all exported from here. Each is the
architecture-independent pipeline bound to an ``ArchProfile`` from
:mod:`.arch`; the classes themselves take the architecture as an argument.
"""

from .arch import (
    BLACKHOLE,
    QUASAR,
    WORMHOLE,
    ArchProfile,
    BlackholeDataCopyGolden,
    BlackholeEltwiseBinaryGolden,
    BlackholeMatmulGolden,
    QuasarDataCopyGolden,
    QuasarEltwiseBinaryGolden,
    QuasarEltwiseBinaryReuseDestGolden,
    QuasarMatmulGolden,
    WormholeDataCopyGolden,
    WormholeEltwiseBinaryGolden,
    WormholeMatmulGolden,
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
    "BlackholeDataCopyGolden",
    "BlackholeEltwiseBinaryGolden",
    "BlackholeMatmulGolden",
    "QuasarDataCopyGolden",
    "QuasarEltwiseBinaryGolden",
    "QuasarEltwiseBinaryReuseDestGolden",
    "QuasarMatmulGolden",
    "WormholeDataCopyGolden",
    "WormholeEltwiseBinaryGolden",
    "WormholeMatmulGolden",
    "Chain",
    "DataCopyGolden",
    "EltwiseBinaryGolden",
    "Golden",
    "EltwiseBinaryReuseDestGolden",
    "MatmulGolden",
    "OpConfig",
    "Registers",
    "StageRecord",
    "Step",
]
