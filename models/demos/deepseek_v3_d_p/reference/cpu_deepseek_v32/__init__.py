# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3.2 / GLM-5.1 sparse-MLA CPU reference — public API.

The only supported entry points are re-exported here. Internals (``ModelArgs``, ``MLACPU``,
``IndexerCPU``, weight loaders, RoPE helpers) are implementation detail — import from this package,
not its submodules. See ``API_SPEC.md``.
"""

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.reference import (
    CANONICAL_WEIGHT_NAMES,
    SparseMLAConfig,
    SparseMLAReference,
    Weights,
    pretrained_mla_weights,
    random_mla_weights,
)

__all__ = [
    "SparseMLAReference",
    "SparseMLAConfig",
    "Weights",
    "random_mla_weights",
    "pretrained_mla_weights",
    "CANONICAL_WEIGHT_NAMES",
]
