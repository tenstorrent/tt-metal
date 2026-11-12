# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Operations, each declaring a chain of data-transfer blocks.

The operations live in :mod:`.pipelines`. Everything here is the machinery
that builds and runs them.

Use an architecture's golden, such as ``get_quasar_matmul_golden``,
``get_wormhole_eltwise_binary_golden`` and so on, all exported from here. Each is the
architecture-independent pipeline bound to an ``ArchProfile`` from
:mod:`.arch`. The classes themselves take the architecture as an argument.
"""

from .arch import (
    BLACKHOLE,
    QUASAR,
    WORMHOLE,
    ArchProfile,
    get_blackhole_data_copy_golden,
    get_blackhole_eltwise_binary_golden,
    get_blackhole_matmul_golden,
    get_quasar_data_copy_golden,
    get_quasar_eltwise_binary_golden,
    get_quasar_eltwise_binary_reuse_dest_golden,
    get_quasar_matmul_golden,
    get_wormhole_data_copy_golden,
    get_wormhole_eltwise_binary_golden,
    get_wormhole_matmul_golden,
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
    "get_blackhole_data_copy_golden",
    "get_blackhole_eltwise_binary_golden",
    "get_blackhole_matmul_golden",
    "get_quasar_data_copy_golden",
    "get_quasar_eltwise_binary_golden",
    "get_quasar_eltwise_binary_reuse_dest_golden",
    "get_quasar_matmul_golden",
    "get_wormhole_data_copy_golden",
    "get_wormhole_eltwise_binary_golden",
    "get_wormhole_matmul_golden",
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
