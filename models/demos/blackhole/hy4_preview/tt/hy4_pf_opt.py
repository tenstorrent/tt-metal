# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HY4_PF_OPT: prefill-chunk (512 rows per mesh row) levers found by program task prefill-roof, all flag-gated, default OFF.

Tokens (comma list, ``all`` = every token): 
  mm2d   explicit MatmulMultiCoreReuseMultiCast (2D) program configs for the M=512 prefill linears (attention q_a/kv_a/q_b/gate/o_proj, indexer wq_b/wk, dense MLP, shared expert)
  hifi2  HiFi2 (fp32 accumulation kept) for the prefill attention / indexer-projection / MLP linears (decode keeps HiFi4)
  a2a2   heads<->sequence all_to_all_async_generic with num_links=2 (bit-identical, 331 -> 174 us)
  kc64   sparse_sdpa k_chunk_size 64 (default 128)
  rope1  indexer q/k rope as ONE rotary_embedding_llama on the 128-wide head (identity cos/sin on the non-rope span) instead of slice/matmul/mul/mul/add/slice/concat
  mask1  indexer in-chunk causal fix as one add with a full-width mask (instead of slice + add + concat)
Never applied to decode (row count < 256 keeps the stock path).
"""
import os
from dataclasses import dataclass

ALL = ("mm2d", "hifi2", "a2a2", "kc64", "rope1", "mask1")


@dataclass(frozen=True)
class PfOpt:
    mm2d: bool = False
    hifi2: bool = False
    a2a2: bool = False
    kc64: bool = False
    rope1: bool = False
    mask1: bool = False

    @classmethod
    def parse(cls, s):
        toks = {t.strip() for t in (s or "").split(",") if t.strip()}
        if "all" in toks:
            toks = set(ALL)
        bad = toks - set(ALL)
        assert not bad, f"unknown HY4_PF_OPT tokens {bad}"
        return cls(**{t: True for t in toks})

    @classmethod
    def from_env(cls):
        return cls.parse(os.environ.get("HY4_PF_OPT", ""))

    @property
    def any(self):
        return any(getattr(self, t) for t in ALL)


def mcast_params(M, K, N, max_gx=12):
    """(gx, gy, per_core_M, per_core_N, in0_block_w, out_subblock_w) of the 2D-mcast config for x[M,K] @ w[K,N] (all multiples of 32); None if the shape does not tile."""
    if M % 256 or K % 32 or N % 32:
        return None
    mt, kt, nt = M // 32, K // 32, N // 32
    gy = 8 if mt % 8 == 0 else 4
    gx = max(d for d in range(1, max_gx + 1) if nt % d == 0)
    pn = nt // gx
    bw = max(d for d in (16, 8, 4, 2, 1) if kt % d == 0)
    sw = max(d for d in (4, 2, 1) if pn % d == 0)
    return gx, gy, mt // gy, pn, bw, sw


def make_pc(M, K, N, sigmoid=False):
    import ttnn

    p = mcast_params(M, K, N)
    if p is None:
        return None
    gx, gy, pm, pn, bw, sw = p
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=bw,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID) if sigmoid else None,
        fuse_batch=True,
    )


def plinear(opt, x, w, ckc, act=None, **kw):
    """ttnn.linear for a prefill-shaped x [1,1,M,K] (M >= 256): 2D program config when opt.mm2d, else the stock call. Activation ``sigmoid`` is fused either way."""
    import ttnn

    M, K, N = x.shape[-2], x.shape[-1], w.shape[-1]
    if opt is not None and opt.mm2d and len(x.shape) == 4 and x.shape[1] == 1 and len(w.shape) == 4 and M >= 256:
        pc = make_pc(M, K, N, sigmoid=(act == "sigmoid"))
        if pc is not None:
            return ttnn.linear(x, w, compute_kernel_config=ckc, program_config=pc, **kw)
    return ttnn.linear(x, w, compute_kernel_config=ckc, activation=act, **kw)
