# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generic overlap primitives for fusing multiple tensors into a single L1 buffer."""

from models.demos.deepseek_v3_b1.weights.overlap.packing import (
    OverlapEntry,
    OverlappedTensor,
    overlap_tensors,
)
from models.demos.deepseek_v3_b1.weights.overlap.spec import (
    OverlappedTensorSpec,
    max_shard_bytes,
)

__all__ = [
    "OverlapEntry",
    "OverlappedTensor",
    "OverlappedTensorSpec",
    "max_shard_bytes",
    "overlap_tensors",
]
