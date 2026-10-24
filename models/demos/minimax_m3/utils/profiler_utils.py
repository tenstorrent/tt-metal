# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 zone profiling: the model's ZoneSpec and its ``zone`` / ``read_profiler``.

The mechanism lives in models/demos/common/prefill/profiling (shared with GPT-OSS); this module only
says what is M3-specific — the signpost prefix, the env vars, the dense/sparse layer classes and which
zones are communication vs KV-cache memory — and re-exports the API the model code uses::

    from models.demos.minimax_m3.utils.profiler_utils import zone

    with zone("indexer"):
        block_scores = ttnn.experimental.indexer_score_msa(...)

Off by default: ``M3_PROFILE_ZONES=1`` arms the zones, ``M3_PROFILE_LEVEL=1|2|3`` picks the detail (read
once, when this module is imported, so the harness sets them before importing the model). See
tests/perf/README_profiling.md.
"""

from __future__ import annotations

from models.demos.common.prefill.profiling.spec import LayerClass, ZoneSpec
from models.demos.common.prefill.profiling.zones import COARSE, FINE, MEDIUM, ZoneProfiler

# MiniMax-M3: layers 0-2 are dense (dense attention + dense MLP), 3-59 sparse (MSA + MoE); tt/layer.py
# tags each layer zone `layerNN_{dense|sparse}`. The MSA cache read is the ag_kv + ag_index_k gathers.
SPEC = ZoneSpec(
    model_name="MiniMax-M3",
    signpost_prefix="M3_ZONE",
    env_prefix="M3_PROFILE",
    host_zone_scope="minimax_m3",
    layer_classes=(
        LayerClass("dense", "Dense layer", 3),
        LayerClass("sparse", "Sparse layer", 57),
    ),
    comm_keys=(
        "ccl_out_allreduce",
        "ccl_out_allgather",
        "tp_allreduce",
        "tp_allgather",
        "ag_kv",
        "ag_index_k",
        "dispatch",
        "combine",
        "moe_reduce",
        "pre_dispatch_allgather",
    ),
    mem_keys=("kv_write", "index_k_write"),
)

PROFILER = ZoneProfiler(SPEC)
zone = PROFILER.zone
read_profiler = PROFILER.read_profiler
ZONES_ENABLED = PROFILER.enabled
LEVEL = PROFILER.level
ZONE_START_PREFIX = SPEC.zone_start
ZONE_END_PREFIX = SPEC.zone_end

__all__ = [
    "COARSE",
    "MEDIUM",
    "FINE",
    "LEVEL",
    "PROFILER",
    "SPEC",
    "ZONES_ENABLED",
    "ZONE_END_PREFIX",
    "ZONE_START_PREFIX",
    "read_profiler",
    "zone",
]
