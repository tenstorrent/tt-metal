# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS zone profiling: the model's ZoneSpec and its ``zone`` / ``read_profiler``.

The mechanism lives in models/demos/common/prefill/profiling (shared with MiniMax-M3); this module only
says what is GPT-OSS-specific — the signpost prefix, the env vars, the two layer classes and which
zones are communication vs KV-cache memory — and re-exports the API the model code uses::

    from models.demos.gpt_oss_d_p.utils.profiler_utils import COARSE, FINE, zone

    with zone("ring_joint_sdpa"):
        out = dense_sp_attention(...)

Off by default: ``GPTOSS_PROFILE_ZONES=1`` arms the zones, ``GPTOSS_PROFILE_LEVEL=1|2|3`` picks the detail
(read once, when this module is imported, so the harness sets them before importing the model). See
tests/perf/README_profiling.md.
"""

from __future__ import annotations

from models.demos.common.prefill.profiling.spec import LayerClass, ZoneSpec
from models.demos.common.prefill.profiling.zones import COARSE, FINE, MEDIUM, ZoneProfiler

# GPT-OSS-120B: 36 layers alternating sliding-window (even) / full-causal (odd) attention, 18 of each;
# the MLP is MoE on every layer, so the attention class is the only thing distinguishing layers
# (tt/layer.py tags each layer zone `layerNN_{sliding|full}`).
#
# ring_joint_sdpa is deliberately NOT a comm key. The cache-backed ring SDPA fuses the SP ring-rotation
# CCL with the attention compute in one device op, so its comm share cannot be split out; it is reported
# as compute. The one-shot path's ag_qkv / sdpa_reduce_scatter ARE separate ops, which is what makes the
# CACHE=0 capture the comm/compute reference point for attention.
SPEC = ZoneSpec(
    model_name="GPT-OSS",
    signpost_prefix="GPTOSS_ZONE",
    env_prefix="GPTOSS_PROFILE",
    host_zone_scope="gpt_oss_d_p",
    layer_classes=(
        LayerClass("sliding", "Sliding-attention layer", 18),
        LayerClass("full", "Full-attention layer", 18),
    ),
    comm_keys=(
        "ccl_out_allreduce",
        "ccl_out_allgather",
        "ag_qkv",
        "sdpa_reduce_scatter",
        "tp_allgather",
        "dispatch",
        "combine",
        "moe_reduce",
        "pre_dispatch_allgather",
    ),
    mem_keys=("kv_write", "defrag_move"),
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
