# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The validated 8x4 Galaxy configuration of the persistent-stack port: TP32 x EP32 over the 32 chips, the
sequence-sharded DSA pool (1 M x 4 and 128 K x 16 users resident), the TP prefill with the DSA indexer, the hybrid
expert layout, the pooled layer intermediates. `apply(mesh_shape, seq, isl)` sets every knob below with
os.environ.setdefault when the mesh is the 8x4 (an explicit environment value always wins) and the per-context terms
-- the trace region, the prefill buckets, the TP prefill's sparse window ladder -- from the cache length `seq`. The
validated rows: models/demos/glm53_flash/doc/perf/shard_8x4_design.md section 6."""
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
    "GLM53_DSA_SHARD_KPH": "auto",  # a merge helper holds 32 / helpers chips' lists: 4 token lanes at t 4, 8 at t 8-16
    "GLM53_EP_KEEP_PERMUTED": "1",  # the permuted gate_up copies stay under GLM53_FUSED_EXPERTS_CACHE
    "GLM53_DSA_POOL": "1",  # the sparse_dsa layers' intermediates from one AllocPool (stack_model._dsa_pools)
    "GLM53_MOE_SHARE_M": "1",  # one shared-expert intermediate per stack (mk_moe.MkMoE)
}
# the TP prefill's sparse window ladder (GLM53_PF_SPARSE_T) per cache length, as validated 4 K .. 1 M
SPARSE_T = {
    8192: "4096,8192",
    65536: "16384,32768,49152,65536",
    262144: "32768,65536,98304,131072,163840,262144",  # through the cache length: a prompt past 163840 rows needs it
    524288: "131072,262144,393216,524288",
    1048576: "262144,524288,786432,1048576",
}


def trace_mb(seq: int) -> int:
    """the trace region (MiB) the decode + prefill traces of a cache length need (validated 2048 .. 8192)"""
    return 2048 if seq <= 65536 else 3072 if seq <= 262144 else 6144 if seq <= 524288 else 8192


def sparse_t(seq: int) -> str:
    """the ladder for a cache length: the validated table, else quarter-cache steps of at least 4 K that stay inside
    the cache ("" when none fits -- a 2 K cache has no sparse region: the DSA path starts past 2,051 rows; the TP
    prefill then compiles no sparse bucket, as before)"""
    seq = int(seq)
    if seq in SPARSE_T:
        return SPARSE_T[seq]
    step = max(4096, (seq // 4) // 128 * 128)  # the pooled-key page grid (prefill_tp: t % 128 == 0)
    return ",".join(str(step * k) for k in range(1, 5) if step * k <= seq)


def apply(mesh_shape, seq: int, isl: int | None = None) -> dict:
    """-> {knob: value} of what this call set (nothing on other meshes, nothing already in the environment)"""
    if tuple(int(v) for v in mesh_shape) != GALAXY:
        return {}
    want = dict(KNOBS)
    want["GLM53_MODEL_TRACE_MB"] = str(trace_mb(int(seq)))
    ladder = sparse_t(int(seq))
    if ladder:
        want["GLM53_PF_SPARSE_T"] = ladder
    # the 2 K / 8 K prefill buckets that fit the cache (a 127-token prompt keeps the 256 / 512 / 2048 defaults; a
    # 4 K cache gets 2048 alone)
    if (isl is None or int(isl) >= 2048) and int(seq) >= 2048:
        want["GLM53_PREFILL_BUCKETS"] = ",".join(str(b) for b in (2048, 8192) if b <= int(seq))
    out = {}
    for k, v in want.items():
        if os.environ.get(k) is None:
            os.environ[k] = v
            out[k] = v
    return out
