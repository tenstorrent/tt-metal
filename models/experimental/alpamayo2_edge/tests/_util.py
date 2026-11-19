# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the test modules."""

import pytest
import torch

from models.experimental.alpamayo2_edge.reference.config import num_layers_override
from models.experimental.alpamayo2_edge.reference.text import KVCache

# Bring-up floor for every PCC gate; measured values are 0.9976 to 0.99999 (README). Per-block floors closer to the
# measurements are set in #66402.
PCC = 0.99

full_depth_only = pytest.mark.skipif(
    num_layers_override() is not None,
    reason="pins a full-depth result; ALPAMAYO_NUM_LAYERS builds a shallower model",
)


def pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    """Pearson correlation in fp64, for logging; gates use `assert_quality`, which also rejects NaN."""
    stacked = torch.stack([golden.flatten().double(), actual.flatten().double()])
    return torch.corrcoef(stacked)[0, 1].item()


def reference_cache(prefix: dict) -> KVCache:
    """A fresh fp32 reference cache over the bf16 K/V of the `prefix` fixture."""
    kv = KVCache(len(prefix["k"]))
    kv.k = [k.float() for k in prefix["k"]]
    kv.v = [v.float() for v in prefix["v"]]
    return kv
