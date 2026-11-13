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
    # the shard's AG / RS1 tail (doc/perf/shard_8x4_design.md section 10; each knob is its own define, any subset
    # composes). GLM53_DSA_SHARD_AGK: the key-packed all-gather -- AG1 ships each token's candidate keys in one page (2
    # pages per token instead of 6: a third of both AG rounds' bytes); 4 K x 16 42.6 -> 41.0 ms, 32 K x 4 ids identical
    "GLM53_DSA_SHARD_AGK": "1",
    # GLM53_DSA_SHARD_AGB (on AGK): one y_ag chunk per block -- 16 + 32 staging batches per launch instead of 144 at t 16;
    # GLM53_DSA_SHARD_RS1_EARLY: pass 0's RS1 slices stream while pass 1 computes (two-field gates, MLA_REV 67). 4 K x 16
    # distinct, 3 in-process runs each, outputs bit-identical to both off (ids + top-4 values): 36.8 -> 35.7 ms (AGB),
    # 35.5 ms (RS1_EARLY), 34.3 ms (both); 32 K x 4 ids identical with both
    "GLM53_DSA_SHARD_AGB": "1",
    "GLM53_DSA_SHARD_RS1_EARLY": "1",
    # GLM53_DSA_SHARD_RSB: RS1 / RS2 in one 32-row chunk per destination (one completion signal per destination instead of
    # one per token: 112 -> 7 and 48 -> 3 per source at t 16; same bytes, page ids, order): 4 K x 16 33.3 -> 32.7 ms,
    # bit-identical over 3 runs; 32 K x 4 ids identical
    "GLM53_DSA_SHARD_RSB": "1",
    # GLM53_DSA_SHARD_R8: the local select and the merge in 8-bit radix digits (4 rounds + tie instead of 8 + tie; the
    # same T / T_g, quotas and tie order): 1 M x 16 40.5 -> 38.2 ms (user 0 ids identical), 4 K x 16 bit-identical over
    # 3 runs (merge -31 us per DSA layer), 32 K x 4 ids identical
    "GLM53_DSA_SHARD_R8": "1",
    # GLM53_DSA_SHARD_RSNP (on RSB): the scatter lanes send only the slices' data pages (9 of 10 / 3 of 4: the pad page
    # only made the page pairs even), paired across slices to their RSB pages: 4 K x 16 32.7 -> 32.2 ms, bit-identical
    # over 3 runs (digest e2f3c86090b7); probe B 16 layers 0-3 2925.7 -> 2876.2 us (RS1 wait 287.5 -> 246.9 us, RS2
    # landed 45.2 -> 35.2 us, the root-local marks flat)
    "GLM53_DSA_SHARD_RSNP": "1",
    # GLM53_DSA_SHARD_LOCAL_ALL (rev 72): a sparse token whose chip holds <= 512 closed local pools (every context < ~64 K
    # on 32 chips) keeps every local pool without the radix rounds -- the same records, offsets and header (host proof
    # test_local_all_matches_the_rounds): local select 174 -> 0.7 us per DSA layer at B 1; e2e 4 K x 1 18.6 -> 17.3 ms
    # (53.7 -> 57.7 tok/s/u), 32 K x 4 21.0 -> 20.6 ms, every user's ids identical (4 K x 1, 32 K x 4, 4 K x 16)
    "GLM53_DSA_SHARD_LOCAL_ALL": "1",
    # GLM53_DSA_SHARD_SEL2 (rev 73; needs R8): past 512 local pools the local select's rounds keep the same integers at less
    # coordinator work (the cores report their above counts in the tie round, packed report / pull sums, per-owner
    # quota chunks; host proof test_sel2_matches_the_rounds): coordinator round 27.0 -> 9.1 us at B 1; e2e 128 K x 1
    # 18.6 -> 18.1 ms, 1 M x 1 18.7 -> 18.2 ms; every user's ids identical (128 K x 1, 128 K x 4, 1 M x 1)
    "GLM53_DSA_SHARD_SEL2": "1",
    # the routed-expert phase reads x one tile face at a time instead of one 32-B row (kernels/mkx/xa.cpp; the same bytes
    # at the same addresses): 4 K x 16 40.8 -> 38.0 ms, 32 K x 4 22.8 -> 22.0 ms with ids and text identical
    "GLM53_MKX_XROWS_COAL": "1",
    # the TP prefill mixes the mHC streams with one program per sublayer on the flat residual (prefill_tp._layer_fused_mhc;
    # exact fp32 like the composite chain, only the summation order differs): TTFT 1 M x 1 344.7 -> 332.1 s, 32 K x 4
    # 35.7 -> 34.1 s, ISL 128 x 16 7.56 -> 7.01 s, decode unchanged, 16 identical prompts still agree; the ids move at a
    # near-tie after 7 (32 K x 4) / 35 (1 M x 1) tokens. The packed and batched prefill (GLM53_PF_PACKED / _BATCH) need it
    "GLM53_PF_MHC_POST": "fused",
    # the TP prefill's routed-MoE kernel options at full precision (prefill_tp: 808 KB expert CBs, the fused SwiGLU reads
    # the dispatch rows itself, the dispatch bounded by the plan's rows, no DRAM read for skipped rm-combine rows, the
    # bucket's padding rows out of the routed MoE, only valid rows into the shared-expert FFN, its units dealt by cost):
    # TTFT 1 M x 1 332.1 -> 281.4 s (ids identical), 32 K x 4 34.1 -> 27.4 s, ISL 128 x 16 7.01 -> 5.06 s, decode
    # unchanged; the ids move at near-ties (32 K x 4 after 7 tokens: back on the ops-post sequence; 128 x 16 user 0 at
    # token 6, "the text" -> "a text"). LoFi (GLM53_PF_MOE_FID) stays off: -0.8 % for a precision cut
    "GLM53_PF_MOE_KB": "808",
    "GLM53_PF_MOE_XGATHER": "1",
    "GLM53_PF_MOE_V2": "1",
    "GLM53_PF_MOE_RM_SKIP": "1",
    "GLM53_PF_MOE_PADMASK": "1",
    "GLM53_PF_MOE_XFFN_VALID": "1",
    "GLM53_PF_MOE_XFFN_SNAKE": "1",
    # the sparse_dsa pool + its s1 child borrow the KDA stacks' pool tensors of the same key (the programs never run at
    # once; persistent state lives in the pools' children, never lent): ~44 KB per bank at the full width, what lets the
    # B 16 x 1 M build take narrow widths; det3 4 K x 16 digest a18df85fa319 knob off = on, 32.2 ms per step both
    "GLM53_DSA_POOL_BORROW": "1",
    # the mHC deferral (stack_model.set_knobs GLM53_STACK_DEFER: sparse_kda deferred 1/rms(h) in DRAM + hn first, sparse_dsa
    # hn first; GLM53_MK_MOE_DEFER_INV the sparse_dsa MoE's deferral) and the one-tile round 1 at t <= 2
    # (GLM53_MHC_MS1, capped by GLM53_MHC_MS1_TMAX): chain_mhc_1 2026-11-13, e2e 32 K x 4 47.5 -> 48.9, 4 K x 1 53.6 ->
    # 55.9 tok/s/u; sparse_kda attn site 33.1 -> 21.1 us at B 1; the ids move only along existing near-tie sequences
    "GLM53_STACK_DEFER": "1",
    "GLM53_MK_MOE_DEFER_INV": "1",
    "GLM53_MHC_MS1": "1",
    # opt-in, not a default: GLM53_DSA_SHARD_COMB (the fold combine on the lane roots: correct, but its fp32 order moves
    # the ids at a near-tie -- fails the strict ids gate)
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
    if os.environ.get("GLM53_DSA_SHARD_AGK", want["GLM53_DSA_SHARD_AGK"]) != "1":
        want.pop("GLM53_DSA_SHARD_AGB")  # AGB rides on AGK: an explicit AGK=0 drops the AGB default with it
    if os.environ.get("GLM53_DSA_SHARD_RSB", want["GLM53_DSA_SHARD_RSB"]) != "1":
        want.pop("GLM53_DSA_SHARD_RSNP")  # RSNP rides on RSB: an explicit RSB=0 drops the RSNP default with it
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
