# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Collective Communication Layer (CCL) building block.

This module provides distributed tensor operations including all_reduce,
all_gather, and distributed normalization for multi-device configurations.
Each operation follows the fractal API design with its own Spec and ImplConfig.
"""

# Import submodules
from . import all_reduce
from . import all_gather
from . import distributed_norm
from . import manager

# Manager for semaphore handling
from .manager import CCLManager

# All-reduce operation
from .all_reduce import (
    AllReduceSpec,
    AllReduceImplConfig,
    get_default_impl_config as get_all_reduce_default_impl_config,
    all_reduce_forward,
    prefill_forward as all_reduce_prefill_forward,
    decode_forward as all_reduce_decode_forward,
)

# All-gather operation
from .all_gather import (
    AllGatherSpec,
    AllGatherImplConfig,
    get_default_impl_config as get_all_gather_default_impl_config,
    all_gather_forward,
    prefill_forward as all_gather_prefill_forward,
    decode_forward as all_gather_decode_forward,
)

# Distributed RMS normalization
from .distributed_norm import (
    DistributedRMSNormSpec,
    DistributedRMSNormImplConfig,
    get_default_impl_config as get_distributed_rmsnorm_default_impl_config,
    distributed_rmsnorm_forward,
    prefill_forward as distributed_rmsnorm_prefill_forward,
    decode_forward as distributed_rmsnorm_decode_forward,
)

__all__ = [
    # Submodules
    "all_reduce",
    "all_gather",
    "distributed_norm",
    "manager",
    # Manager
    "CCLManager",
    # All-reduce
    "AllReduceSpec",
    "AllReduceImplConfig",
    "get_all_reduce_default_impl_config",
    "all_reduce_forward",
    "all_reduce_prefill_forward",
    "all_reduce_decode_forward",
    # All-gather
    "AllGatherSpec",
    "AllGatherImplConfig",
    "get_all_gather_default_impl_config",
    "all_gather_forward",
    "all_gather_prefill_forward",
    "all_gather_decode_forward",
    # Distributed RMS norm
    "DistributedRMSNormSpec",
    "DistributedRMSNormImplConfig",
    "get_distributed_rmsnorm_default_impl_config",
    "distributed_rmsnorm_forward",
    "distributed_rmsnorm_prefill_forward",
    "distributed_rmsnorm_decode_forward",
]
