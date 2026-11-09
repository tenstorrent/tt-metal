# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The validated 8x4 Galaxy configuration of the persistent-stack port (l2-34, 2026-11-06 .. 08): TP32 x EP32 over the
32 chips, the sequence-sharded DSA pool (1 M x 4 and 128 K x 16 users resident), the TP prefill with the DSA indexer,
the hybrid expert layout, the pooled layer intermediates. `apply(mesh_shape, seq, isl)` sets every knob below with
os.environ.setdefault when the mesh is the 8x4 (an explicit environment value always wins) and the per-context terms
-- the trace region, the prefill buckets, the TP prefill's sparse window ladder -- from the cache length `seq`.

Measured (45 layers, OSL 128, tok/s per user / TTFT): 32 K x 4 37.1 / 35 s, 32 K x 16 14.6 / 142 s, 128 K x 8 23.1 /
290 s, 128 K x 16 14.0 / 583 s, 512 K x 4 34.6 / 626 s, 1 M x 1 51.7 / 345 s, 1 M x 4 35.4 / 1,379 s
(models/demos/glm53_flash/doc/perf/shard_8x4_design.md section 6)."""
from __future__ import annotations

import os

GALAXY = (8, 4)
KNOBS = {
    "GLM53_FABRIC": "1d",  # the 1D fabric: the EP all-to-all and the shard's streams (tt/mesh.py)
    "GLM53_MK_EP": "1",  # the routed experts EP-sharded over the chips (tt/ep_experts.py)
    "GLM53_STACK_DSA": "1",  # the DSA indexer in the decode stack (sparse attention past 2,051 rows)
    "GLM53_STACK_PREFILL": "tp",  # the TP prefill (tt/prefill_tp.py) ...
    "GLM53_PF_DSA": "1",  # ... with the DSA indexer (the sharded pool's writer)
    "GLM53_DSA_SHARD": "1",  # the sequence-sharded DSA pool: unit u of a slot on chip u % 32 (tt/tp_dsa_shard.py)
    "GLM53_DSA_SHARD_LANES_T4": "1",  # token lanes for the t <= 4 streams too
    "GLM53_DSA_SHARD_SHARE_SYNC": "1",  # one stream sync set per row count, shared by the layers
    "GLM53_EP_KEEP_PERMUTED": "1",  # the permuted gate_up copies stay under GLM53_FUSED_EXPERTS_CACHE
    "GLM53_DSA_POOL": "1",  # the sparse_dsa layers' intermediates from one AllocPool (stack_model._dsa_pools)
    "GLM53_MOE_SHARE_M": "1",  # one shared-expert intermediate per stack (mk_moe.MkMoE)
}
# the TP prefill's sparse window ladder (GLM53_PF_SPARSE_T) per cache length, as validated 4 K .. 1 M
SPARSE_T = {
    8192: "4096,8192",
    65536: "16384,32768,49152,65536",
    262144: "32768,65536,98304,131072,163840",
    524288: "131072,262144,393216,524288",
    1048576: "262144,524288,786432,1048576",
}


def trace_mb(seq: int) -> int:
    """the trace region (MiB) the decode + prefill traces of a cache length need (validated 2048 .. 8192)"""
    return 2048 if seq <= 65536 else 3072 if seq <= 262144 else 6144 if seq <= 524288 else 8192


def sparse_t(seq: int) -> str:
    return SPARSE_T.get(int(seq)) or ",".join(str(max(4096, int(seq) // 4) * k) for k in range(1, 5))


def apply(mesh_shape, seq: int, isl: int | None = None) -> dict:
    """-> {knob: value} of what this call set (nothing on other meshes, nothing already in the environment)"""
    if tuple(int(v) for v in mesh_shape) != GALAXY:
        return {}
    want = dict(KNOBS)
    want["GLM53_MODEL_TRACE_MB"] = str(trace_mb(int(seq)))
    want["GLM53_PF_SPARSE_T"] = sparse_t(int(seq))
    if (
        isl is None or int(isl) >= 2048
    ):  # the 2 K / 8 K prefill buckets (a 127-token prompt keeps the 256 / 512 / 2048 defaults)
        want["GLM53_PREFILL_BUCKETS"] = "2048,8192"
    out = {}
    for k, v in want.items():
        if os.environ.get(k) is None:
            os.environ[k] = v
            out[k] = v
    return out
