# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM levers for B=16 / 1M with all stack5 optimizations on (default off, pure python: no ttnn import, CPU testable).

DSV41_COPY_GUARD=1: do not allocate the duplicate weight copies of a fused decode piece when no verify row count of the adaptive spec set can use it
(the copies are bank-sharded / blaze-layout duplicates of the stock tensors, which prefill keeps using, so they cannot replace them):
  * attention-front blaze copies of wqkv + wq_b (~73 MiB/bank over 40 layers): used at verify rows T in AF_T = (8, 12) only;
  * skinny shared-expert copy (~23 MiB/bank): used at T <= 8 only;
  * skinny wo_a / wo_b copies (~48 MiB/bank): used at T <= 16.
T = U * (1 + k) for k in DSV41_VLLM_SPEC_KS (default 3,5): B=8 (U=2) -> 8 / 12 (all three pieces live), B=16 (U=4) -> 16 / 24 (attention-front and skinny shared never run;
skinny wo only at k=3). The runtime already falls back when a copy is absent (``hasattr(self, "af_w1")`` / ``self._skin is None`` / ``skin_sh is None``).
The sets are env-overridable so that the T=16 / 24 extensions can declare their rows without touching code:
DSV41_COPY_GUARD_AF_T="8,12,16", DSV41_COPY_GUARD_WOAB_MAX=24, DSV41_COPY_GUARD_SHARED_MAX=16.

``budget`` is the arithmetic of dsv4-logs/unified_stack/DRAM_LEDGER.md (MiB per bank) used by tests/test_dram_levers.py and by tools/memreport_check.py.
"""
import os

AF_T_DEFAULT = (8, 12)
WOAB_MAX_DEFAULT = 16
SHARED_MAX_DEFAULT = 8


def _env(env):
    return os.environ if env is None else env


def guard_on(env=None):
    return _env(env).get("DSV41_COPY_GUARD", "0") == "1"


def spec_ks(env=None):
    s = _env(env).get("DSV41_VLLM_SPEC_KS") or "3,5"
    return sorted({int(x) for x in str(s).split(",") if x.strip()})


def verify_rows(U, env=None):
    """verify rows per mesh row of the adaptive spec set (T = U * (1 + k))."""
    return sorted(U * (1 + k) for k in spec_ks(env))


def _ints(s, default):
    return tuple(int(x) for x in s.split(",") if x.strip()) if s else default


def want_attn_front_copy(U, env=None):
    if not guard_on(env):
        return True
    default = AF_T_DEFAULT + (
        (16,) if _env(env).get("DSV41_ATTN_FRONT_T16", "0") == "1" else ()
    )  # the T=16 gate (B=16 k=3) declares its row count
    af_t = _ints(_env(env).get("DSV41_COPY_GUARD_AF_T"), default)
    return any(t in af_t for t in verify_rows(U, env))


def want_skinny_wo_copy(U, env=None):
    if not guard_on(env):
        return True
    mx = int(_env(env).get("DSV41_COPY_GUARD_WOAB_MAX", WOAB_MAX_DEFAULT))
    return any(t <= mx for t in verify_rows(U, env))


def want_skinny_shared_copy(U, env=None):
    if not guard_on(env):
        return True
    mx = int(_env(env).get("DSV41_COPY_GUARD_SHARED_MAX", SHARED_MAX_DEFAULT))
    return any(t <= mx for t in verify_rows(U, env))


# ---------------------------------------------------------------------------------------------------------------------------------------------
# DRAM arithmetic (MiB per bank, 1M context, fp8 pool). Measured: B=8 stack5 end of build 3744.2 allocated / 187.2 free; the B=8 -> B=16 delta 337
# (cand image, 3516.6 -> 3853.6). See DRAM_LEDGER.md.
USABLE_MIB = 3930.0  # 4073 bank - 143 trace region (1.2e9 B)
B8_STACK5_ALLOC = 3744.2
B16_DELTA = 337.0
COPIES = {"attn_front": 73.0, "skinny_wo": 47.7, "skinny_shared": 23.0, "apc_paged": 66.0}
LEVERS = {  # MiB/bank freed (embed_shard: measured, embedshard/results.txt; trace_1e9: 1.2e9 -> 1.0e9 B region; bfp4_experts: 40 x (59.3 - 31.4) by tile-size ratio)
    "embed_shard": 138.1,
    "trace_1e9": 23.8,
    "bfp4_experts": 1116.0,
    "shared_bfp4": 85.0,
}


def budget(B, copies_on=("attn_front", "skinny_wo", "skinny_shared", "apc_paged"), levers=()):
    """predicted (allocated, free) MiB per bank at the end of the build."""
    alloc = B8_STACK5_ALLOC + (B16_DELTA if B == 16 else 0.0)
    alloc -= sum(v for k, v in COPIES.items() if k not in copies_on)
    alloc -= sum(LEVERS[k] for k in levers)
    return alloc, USABLE_MIB - alloc
