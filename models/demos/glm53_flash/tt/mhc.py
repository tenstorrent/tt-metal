# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mHC hyper-connections for GLM-5.3-Flash, on top of the DeepSeek-V4 implementation.

GLM-5.3-Flash's ``Glm5NextTextHyperConnection`` is the DeepSeek-V4 parametrization exactly:
unweighted RMSNorm over the 4 packed streams, one fused projection ``fn [24, 4*4096]``, then
``pre = sigmoid(a_pre*raw + b) + eps``, ``post = 2*sigmoid(a_post*raw + b)``,
``comb = sinkhorn(softmax(a_res*raw + b) + eps)`` with one column normalisation followed by
``iters-1`` (row, column) pairs. ``TtMHCWrap`` runs that in fp32 with the fused
``mhc_split_sinkhorn`` kernel, so this module only adapts the configuration and the tensor
convention.

Tensor convention (from ``tt_mhc``): residual streams are packed on the last dim,
``[1, 1, T, 4*4096]`` fp32 with stream ``i`` at columns ``[i*4096, (i+1)*4096)`` and the
``T = B*S`` tokens down the rows. HF keeps ``[B, S, 4, 4096]``; the conversion is a reshape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import (  # noqa: F401
    TtMHCWrap,
    W,
    _compute_kernel_config,
    _upload,
    build_consts,
    mhc_expand,
)
from models.demos.glm53_flash.tt.config import GLM53FlashConfig

TILE = 32
# K-split of the decode projection: the [T,16384] x [16384,32] contraction has a single output tile,
# so the only parallelism is over K; 32 batches of 512 put it on 32 cores
KS_DECODE = 32
# "fast" (TtMHCFast, this module) | "wrap" (the DeepSeek composite TtMHCWrap, the reference path)
MHC_IMPL = os.environ.get("GLM53_MHC_IMPL", "fast")


@dataclass(frozen=True)
class MHCKernelConfig:
    """The fields ``TtMHCWrap`` reads (``MHCConfig`` in the DeepSeek-V4 reference)."""

    dim: int
    n: int
    sinkhorn_iters: int
    eps: float
    norm_eps: float

    @property
    def mix_hc(self) -> int:
        return (2 + self.n) * self.n


def mhc_kernel_config(cfg: GLM53FlashConfig) -> MHCKernelConfig:
    # HF: input_norm = UnweightedRMSNorm(eps=rms_norm_eps); hc_eps is the stochasticity eps.
    return MHCKernelConfig(
        dim=cfg.hidden_size,
        n=cfg.hc_mult,
        sinkhorn_iters=cfg.hc_sinkhorn_iters,
        eps=cfg.hc_eps,
        norm_eps=cfg.rms_norm_eps,
    )


def build_mhc(
    device, cfg: GLM53FlashConfig, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, impl: str | None = None
):
    """One hyper-connection (attention or FFN) from the raw checkpoint tensors
    ``hc_*_fn [24, 16384]`` (bf16), ``hc_*_base [24]`` and ``hc_*_scale [3]`` (fp32).
    ``impl``: "fast" (``TtMHCFast``, default) or "wrap" (``TtMHCWrap``); default from ``GLM53_MHC_IMPL``."""
    kcfg = mhc_kernel_config(cfg)
    assert tuple(fn.shape) == (kcfg.mix_hc, cfg.hc_mult * cfg.hidden_size), fn.shape
    assert base.numel() == kcfg.mix_hc and scale.numel() == 3
    impl = impl or MHC_IMPL
    cls = {"fast": TtMHCFast, "wrap": TtMHCWrap}[impl]
    return cls(device, kcfg, fn.float(), base.float(), scale.float(), dtype=ttnn.float32)


def streams_hf_to_packed(x: torch.Tensor) -> torch.Tensor:
    """HF ``[B, S, n, C]`` -> packed ``[1, 1, B*S, n*C]`` (host side, tests only)."""
    B, S, n, C = x.shape
    return x.reshape(1, 1, B * S, n * C)


def streams_packed_to_hf(x: torch.Tensor, batch: int, n: int) -> torch.Tensor:
    """Packed ``[1, 1, B*S, n*C]`` -> HF ``[B, S, n, C]`` (host side, tests only)."""
    T, nC = x.shape[-2], x.shape[-1]
    return x.reshape(batch, T // batch, n, nC // n)


# ----------------------------------------------------------------------------------------------
# TtMHCFast: the TtMHCWrap arithmetic without the data-movement composites
# ----------------------------------------------------------------------------------------------
def _rows_up(T: int) -> int:
    return -(-T // TILE) * TILE


def _view_rows(t: ttnn.Tensor, rows: int) -> ttnn.Tensor:
    """Re-declare the logical row count of a ``[1,1,T,Wd]`` TILE tensor inside its padded tile rows
    (zero-copy; ``rows`` and ``T`` round up to the same padded height). Widening (``T`` -> 32)
    makes the pad rows ordinary rows so every stage runs whole tiles with no padding bookkeeping;
    narrowing restores the contract shape. Both go through ``ttnn.reshape``'s explicit-padded-shape
    view (no device op)."""
    T, Wd = t.shape[-2], t.shape[-1]
    if rows == T and len(t.shape) == 4:
        return t
    Tp, Wp = t.padded_shape[-2], t.padded_shape[-1]
    assert _rows_up(rows) == Tp, (rows, Tp)
    return ttnn.reshape(t, ttnn.Shape([1, 1, rows, Wd]), ttnn.Shape([1, 1, Tp, Wp]))


def _view_tile_width(t: ttnn.Tensor) -> ttnn.Tensor:
    """A ``[Tp,k]`` kernel output with ``k <= 32`` as ``[1,1,Tp,32]``: the whole tile incl. its (finite)
    pad columns, so the elementwise ops below see no padding (view)."""
    Tp = t.shape[-2]
    assert t.padded_shape[-1] == W and Tp % TILE == 0, (t.shape, t.padded_shape)
    return ttnn.reshape(t, ttnn.Shape([1, 1, Tp, W]), ttnn.Shape([1, 1, Tp, W]))


class TtMHCFast(LightweightModule):
    """mHC-wrapped sublayer with the ``TtMHCWrap`` semantics (fp32 tensors, HiFi4, the fused Sinkhorn
    kernel) and the composite stages rebuilt so that a decode step issues only compute ops (matmul,
    broadcast elementwise, reduce) on zero-copy views: no slice / untilize / tilize / concat / repeat /
    fill_pad. The one layout fact: a packed ``[1,1,T,n*C]`` TILE tensor whose rows fit one tile row
    (``T <= 32``) has exactly the tile order of the stream-major ``[1,n,T,C]`` tensor (stream ``i`` =
    tiles ``[i*C/32, (i+1)*C/32)``), of its ``[(i,t), c]`` matrix form ``[n*T, C]`` and of the K-split
    ``[1,KS,T,n*C/KS]`` tensor, so all three are ``ttnn.experimental.view`` reinterpretations.

    Decode step (``T <= 32`` logical rows, widened to the 32-row tile by ``_view_rows``), per wrap:

    pre-block (13 ops incl. the norm):
      1 ``part = xk @ fn_k``: K-split (KS=32 batches of 512) batched matmul, one core per batch;
      2 ``mixes_un = fast_reduce_nc(part)``;  3 ``xx = xk * xk``;  4 ``ssp = sum(xx, W)`` (32 cores);
      5 ``ss = fast_reduce_nc(ssp)``;  6 ``rs = rsqrt(ss + eps*nC)`` (one add with an rsqrt activation;
      ``sqrt(nC)`` is folded into ``fn_k``);  7 ``mixes = mixes_un * rs``;  8 ``mhc_split_sinkhorn``;
      9 ``v = pre_tile @ S_pre``, 10 ``P = v * I_pre``: the block-diagonal ``P[t, (i,t')] = delta(t,t')
      pre[t,i]`` ``[T, n*T]``;  11 ``h = P @ x_sm[(i,t), c]`` written bf16 width-sharded over 32 cores;
      12 ``rms_norm`` (sharded multicore program config, the sublayer's input norm, bf16 in/out like the
      composite path's typecast + norm);  13 ``sharded_to_interleaved`` (the sublayers take interleaved).
    post-block (8 ops; +1 ``where`` guard at ``T < 32``):
      1 ``y32 = typecast(y)``;  2-3 ``comb_c[j,i,t] = comb[t,i,j]`` and 4-5 ``post_c[j,t]`` as ``[.., T, 1]``
      columns by one mask multiply + one ``sum`` over W each (exact);  6 ``A = I_tile * comb_c``: the
      block-diagonal ``A[(j,t), (i,t')] = delta(t,t') comb[t,i,j]`` ``[n*T, n*T]`` (view of the
      ``[n,n,T,T]`` product);  7 ``mix = A @ x_sm[(i,t), c]``;  8 ``out = mix + post_c * y`` (one
      broadcast ``addcmul`` on the ``[1,n,T,C]`` view, which IS the packed output).

    Numerics (``_logs/optmhc_probe{2,3,4,5}.log``, vs float64): fp32 elementwise ops and ``ttnn.sum`` are
    exact (3e-8); the HiFi4 fp32 matmul rounds its operands to bf16 (9e-4 rel rms on the projection at
    K=n*C, the same op the composite path uses), so the block-diagonal mixes ``h`` and ``mix`` carry
    ~4e-4..1.2e-3 rel rms (``h`` is bf16 for the sublayer anyway; ``mix`` lands on the fp32 residual,
    where the composite path's elementwise chain is exact); ``fast_reduce_nc`` rounds to bf16 (4e-4 on the
    partial sums). The block-diagonal contraction runs over the pad rows too (``0 * NaN = NaN``), hence
    the contract: the input streams' pad rows are finite (host uploads zero-pad, the embedding gathers
    pad-token rows) and ``y``'s pad rows are zeroed by the ``where`` guard; the output's pad rows are
    finite garbage, not zeros.

    Prefill (``T > 32``): the stream-major form costs one tile-block transpose in and one out per wrap
    (``[T/32, n, 32, C] <-> [n, T/32, 32, C]``), the projection splits over the ``n`` streams, ``h`` is a
    broadcast multiply + ``fast_reduce_nc``, the residual update the exact fp32 broadcast ``addcmul``
    chain over the packed column slices (the block-diagonal matrices do not scale with T), and the norm
    the interleaved multicore one (``T/32`` tile rows).
    """

    fuses_norm = True  # forward(norm=(weight, eps)) applies the sublayer's input RMSNorm itself
    supports_collapse = True  # forward(collapse=True) emits the LM head's mean over the new streams

    def __init__(self, device, cfg, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, dtype=ttnn.float32):
        assert dtype == ttnn.float32, f"TtMHCFast is fp32-only (fused op requires FLOAT32), got {dtype}"
        self.device = device
        self.cfg = cfg
        self.n = int(cfg.n)
        self.C = int(cfg.dim)
        self.nC = self.n * self.C
        self.iters = int(cfg.sinkhorn_iters)
        self.eps = float(cfg.eps)
        self.norm_eps = float(cfg.norm_eps)
        self.ckc = _compute_kernel_config()
        n, C, nC = self.n, self.C, self.nC
        mix_hc = (2 + n) * n
        assert mix_hc <= W and tuple(fn.shape) == (mix_hc, nC), fn.shape
        assert C % TILE == 0 and nC % KS_DECODE == 0 and (nC // KS_DECODE) % TILE == 0, (C, nC)
        assert C % (32 * TILE) == 0, f"the sharded norm splits C={C} over 32 cores in whole tiles"
        self.ks = KS_DECODE
        # projection weight, transposed, zero-padded to a full tile width (the kernel reads the first
        # mix_hc columns of its 32-wide input) and scaled by sqrt(nC): mixes = (x @ fn_T) * sqrt(nC) *
        # rsqrt(sum x^2 + eps*nC) = RMSNorm(x) @ fn_T. K-split views for decode (KS batches) and prefill
        # (n batches = the streams of the stream-major tensor)
        fn_t = torch.zeros(nC, W, dtype=torch.float32)
        fn_t[:, :mix_hc] = fn.float().t() * (nC**0.5)
        self.fn_k = _upload(device, fn_t, (1, self.ks, nC // self.ks, W), dtype)
        self.fn_s = ttnn.reshape(self.fn_k, [1, n, C, W])
        self.eps_nc = self.norm_eps * nC
        self.consts = _upload(device, build_consts(cfg, scale, base), (8, W, W), dtype)
        # the K-split matmul's single output tile per batch: one core per batch (32 batches)
        self.pc_ksplit = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            in0_block_w=16,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
        )
        # coefficient masks: kernel output tile [T, 32] (k real columns) times mask[.., 1, 32], summed
        # over W -> one [.., T, 1] column per batch entry. pre -> [1, n(i)]; post -> [1, n(j)];
        # comb -> [n(j), n(i)] (comb[i, j] is column i*n + j)
        m_pre = torch.zeros(1, n, 1, W)
        m_post = torch.zeros(1, n, 1, W)
        m_comb = torch.zeros(n, n, 1, W)
        for i in range(n):
            m_pre[0, i, 0, i] = 1.0
            m_post[0, i, 0, i] = 1.0
            for j in range(n):
                m_comb[j, i, 0, i * n + j] = 1.0
        self.m_pre = _upload(device, m_pre, (1, n, 1, W), dtype)
        self.m_post = _upload(device, m_post, (1, n, 1, W), dtype)
        self.m_comb = _upload(device, m_comb, (n, n, 1, W), dtype)
        # collapse mode (the LM head's unweighted mean over the n new streams, folded into the
        # residual update): cbar_i = mean_j comb[i,j] -> [1, n(i)], pbar = mean_j post_j -> [1, 1]
        m_comb_mean = torch.zeros(1, n, 1, W)
        m_post_mean = torch.zeros(1, 1, 1, W)
        for i in range(n):
            m_post_mean[0, 0, 0, i] = 1.0 / n
            for j in range(n):
                m_comb_mean[0, i, 0, i * n + j] = 1.0 / n
        self.m_comb_mean = _upload(device, m_comb_mean, (1, n, 1, W), dtype)
        self.m_post_mean = _upload(device, m_post_mean, (1, 1, 1, W), dtype)
        # decode block-diagonal forms (T == 32 rows): P[t, (i,t')] = delta(t,t') pre[t,i] =
        # (pre_tile @ S_pre) * I_pre with S_pre[k, (i,t')] = delta(k,i); A[(j,t), (i,t')] =
        # delta(t,t') comb[t,i,j] = I_tile * comb_c[j,i,t] (broadcast over t', viewed as [n*T, n*T])
        Tt = TILE
        s_pre = torch.zeros(W, n * Tt)
        for i in range(n):
            s_pre[i, i * Tt : (i + 1) * Tt] = 1.0
        self.s_pre = _upload(device, s_pre, (1, 1, W, n * Tt), dtype)
        self.i_pre = _upload(device, torch.eye(Tt).repeat(1, n), (1, 1, Tt, n * Tt), dtype)
        self.i_tile = _upload(device, torch.eye(Tt), (1, 1, Tt, Tt), dtype)
        # pad-row guard for T < 32: y's pad rows -> 0 (the contraction above would spread a NaN)
        self._rowmask: dict[int, ttnn.Tensor] = {}
        # h: [T, n*T] @ [n*T, C] with the output bf16 width-sharded over 32 cores (4 tiles each) for the
        # sharded multicore RMSNorm; the sublayers take interleaved, hence the sharded_to_interleaved
        self.h_grid = ttnn.CoreGrid(y=4, x=8)
        self.h_mem = ttnn.create_sharded_memory_config(
            [Tt, C // 32],
            self.h_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.pc_h = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=n,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=C // (32 * TILE),
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self.pc_norm = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            subblock_w=4,
            block_h=1,
            block_w=C // (32 * TILE),
            inplace=False,
        )
        # mix: [n*T, n*T] @ [n*T, C]
        self.pc_mix = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=n,
            out_subblock_h=2,
            out_subblock_w=2,
            per_core_M=n,
            per_core_N=2,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        # prefill (T > 32): the n-split projection has T/32 x n output tiles
        self.pc_proj_prefill = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            in0_block_w=16,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
        )

    # ------------------------------------------------------------------ layout helpers
    def _stream_major(self, x32: ttnn.Tensor) -> tuple[ttnn.Tensor, bool]:
        """``[1,1,Tp,nC]`` -> ``[1,n,Tp,C]``; returns (tensor, owned): a view for one tile row,
        a transposed copy otherwise."""
        Tp = x32.shape[-2]
        if Tp == TILE:
            return ttnn.experimental.view(x32, [1, self.n, Tp, self.C]), False
        blocks = ttnn.experimental.view(x32, [Tp // TILE, self.n, TILE, self.C])
        t = ttnn.transpose(blocks, 0, 1)  # [n, Tp/32, 32, C]: tile-block copy
        return ttnn.reshape(t, [1, self.n, Tp, self.C]), True

    def _packed(self, sm: ttnn.Tensor) -> ttnn.Tensor:
        """``[1,n,Tp,C]`` -> ``[1,1,Tp,nC]`` (the inverse of ``_stream_major``)."""
        Tp = sm.shape[-2]
        if Tp == TILE:
            return ttnn.experimental.view(sm, [1, 1, Tp, self.nC])
        blocks = ttnn.reshape(sm, [self.n, Tp // TILE, TILE, self.C])
        t = ttnn.transpose(blocks, 0, 1)  # [Tp/32, n, 32, C]
        ttnn.deallocate(sm)
        return ttnn.experimental.view(t, [1, 1, Tp, self.nC])

    def _coef(self, t: ttnn.Tensor, mask: ttnn.Tensor) -> ttnn.Tensor:
        """Kernel output ``[Tp,k]`` -> ``[R0, R1, Tp, 1]`` coefficient columns selected by ``mask``
        ``[R0, R1, 1, 32]`` (2 exact ops: broadcast multiply, sum over W)."""
        prod = ttnn.multiply(_view_tile_width(t), mask)  # [R0,R1,Tp,32]
        c = ttnn.sum(prod, dim=-1, keepdim=True, compute_kernel_config=self.ckc)
        ttnn.deallocate(prod)
        return c

    def _reduce_batches(self, t: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.experimental.fast_reduce_nc(t, dims=[1], compute_kernel_config=self.ckc)

    def _rowmask_for(self, T: int) -> ttnn.Tensor:
        """``[1,1,32,1]`` fp32: 1 on the ``T`` real rows, 0 on the pad rows (built once per T)."""
        if T not in self._rowmask:
            m = torch.zeros(1, 1, TILE, 1)
            m[0, 0, :T, 0] = 1.0
            self._rowmask[T] = _upload(self.device, m, (1, 1, TILE, 1), ttnn.float32)
        return self._rowmask[T]

    # ------------------------------------------------------------------ stages
    def _project(self, x32: ttnn.Tensor, x_sm: ttnn.Tensor) -> ttnn.Tensor:
        """``[1,1,Tp,nC]`` -> mixes ``[1,1,Tp,32]`` = RMSNorm(x) @ fn_T (zero columns past mix_hc)."""
        Tp = x32.shape[-2]
        if Tp == TILE:
            xk, fn_k, pc = ttnn.experimental.view(x32, [1, self.ks, Tp, self.nC // self.ks]), self.fn_k, self.pc_ksplit
        else:
            xk, fn_k, pc = x_sm, self.fn_s, self.pc_proj_prefill
        part = ttnn.matmul(xk, fn_k, compute_kernel_config=self.ckc, program_config=pc)  # [1,K,Tp,32]
        xx = ttnn.multiply(xk, xk)
        ssp = ttnn.sum(xx, dim=-1, keepdim=True, compute_kernel_config=self.ckc)  # [1,K,Tp,1]
        ttnn.deallocate(xx)
        mixes_un = self._reduce_batches(part)  # [1,1,Tp,32]
        # fast_reduce_nc keeps the padded width: re-declare the [Tp,1] column (view) so the scale
        # below is a column broadcast, not an elementwise product with rsqrt(eps) pad columns
        ss = ttnn.reshape(self._reduce_batches(ssp), ttnn.Shape([1, 1, Tp, 1]), ttnn.Shape([1, 1, Tp, W]))
        ttnn.deallocate(part)
        ttnn.deallocate(ssp)
        rs = ttnn.add(ss, self.eps_nc, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RSQRT)])
        ttnn.deallocate(ss)
        mixes = ttnn.multiply(mixes_un, rs)
        ttnn.deallocate(mixes_un)
        ttnn.deallocate(rs)
        return mixes

    def _collapse(self, x32: ttnn.Tensor, x_sm: ttnn.Tensor, pre: ttnn.Tensor, norm):
        """h ``[1,1,Tp,C]`` = sum_i pre_i x_i (fp32 interleaved), or with ``norm = (weight, eps)`` the
        sublayer's input ``RMSNorm(h)`` in bf16 interleaved."""
        Tp = x32.shape[-2]
        if Tp == TILE:
            v = ttnn.matmul(_view_tile_width(pre), self.s_pre, compute_kernel_config=self.ckc)  # [1,1,Tp,nTp]
            P = ttnn.multiply(v, self.i_pre)
            ttnn.deallocate(v)
            res_v = ttnn.reshape(x_sm, [1, 1, self.n * Tp, self.C])  # [(i,t), c]
            if norm is None:
                h = ttnn.matmul(P, res_v, compute_kernel_config=self.ckc)
                ttnn.deallocate(P)
                return h
            weight, eps = norm
            h_sh = ttnn.matmul(
                P,
                res_v,
                compute_kernel_config=self.ckc,
                program_config=self.pc_h,
                memory_config=self.h_mem,
                dtype=ttnn.bfloat16,
            )
            ttnn.deallocate(P)
            hn = ttnn.rms_norm(
                h_sh,
                epsilon=eps,
                weight=weight,
                program_config=self.pc_norm,
                memory_config=self.h_mem,
                compute_kernel_config=self.ckc,
            )
            ttnn.deallocate(h_sh)
            h = ttnn.sharded_to_interleaved(hn, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(hn)
            return h
        pre_c = self._coef(pre, self.m_pre)  # [1,n,Tp,1]
        prod = ttnn.multiply(x_sm, pre_c)  # [1,n,Tp,C]
        ttnn.deallocate(pre_c)
        h = ttnn.experimental.fast_reduce_nc(prod, dims=[1], compute_kernel_config=self.ckc)
        ttnn.deallocate(prod)
        if norm is None:
            return h
        weight, eps = norm
        hb = ttnn.typecast(h, ttnn.bfloat16)
        ttnn.deallocate(h)
        hn = ttnn.rms_norm(hb, epsilon=eps, weight=weight, compute_kernel_config=self.ckc)
        ttnn.deallocate(hb)
        return hn

    def _pre(self, x: ttnn.Tensor, norm=None):
        """x ``[1,1,T,nC]`` -> (h ``[1,1,T,C]``, post ``[Tp,n]``, comb ``[Tp,n*n]``, x32, x_sm, owned)."""
        T = x.shape[-2]
        x32 = _view_rows(x, _rows_up(T))
        x_sm, owned = self._stream_major(x32)
        mixes = self._project(x32, x_sm)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, self.n, self.iters, self.eps
        )
        ttnn.deallocate(mixes)
        h32 = self._collapse(x32, x_sm, pre, norm)
        ttnn.deallocate(pre)
        return _view_rows(h32, T), post, comb, x32, x_sm, owned

    def _post(
        self, y: ttnn.Tensor, x32: ttnn.Tensor, x_sm: ttnn.Tensor, post: ttnn.Tensor, comb: ttnn.Tensor, collapse=False
    ):
        """y ``[1,1,T,C]`` fp32 (F's output), x32 / x_sm the residual streams -> ``[1,1,T,nC]``:
        new_j = post_j * y + sum_i comb[i,j] * x_i. ``collapse`` (decode shapes): return the LM
        head's unweighted mean over the n new streams ``[1,1,T,C]`` instead (same ops: the mean
        masks fold ``1/n sum_j`` into the coefficient columns, so ``mean_j new_j = sum_i cbar_i x_i
        + pbar y`` is one ``[T, n*T]`` block-diagonal matmul plus the ``addcmul``)."""
        T = y.shape[-2]
        Tp = x32.shape[-2]
        n = self.n
        y32 = _view_rows(y, Tp)
        if Tp == TILE:
            if T != Tp:
                y32 = ttnn.where(self._rowmask_for(T), y32, 0.0)
            res_v = ttnn.reshape(x_sm, [1, 1, n * Tp, self.C])
            if collapse:
                cbar = self._coef(comb, self.m_comb_mean)  # [1, n(i), Tp, 1]
                pbar = self._coef(post, self.m_post_mean)  # [1, 1, Tp, 1]
                Pb = ttnn.multiply(self.i_tile, cbar)  # [1,n,Tp,Tp]: delta(t,t') cbar[t,i]
                ttnn.deallocate(cbar)
                mix = ttnn.matmul(
                    ttnn.experimental.view(Pb, [1, 1, Tp, n * Tp]), res_v, compute_kernel_config=self.ckc
                )  # [1,1,Tp,C]
                ttnn.deallocate(Pb)
                out = ttnn.addcmul(mix, y32, pbar, output_tensor=mix)
                ttnn.deallocate(pbar)
                if T != Tp:
                    ttnn.deallocate(y32)
                return _view_rows(out, T)
            comb_c = self._coef(comb, self.m_comb)  # [n(j), n(i), Tp, 1]
            post_c = self._coef(post, self.m_post)  # [1, n(j), Tp, 1]
            A = ttnn.multiply(self.i_tile, comb_c)  # [n,n,Tp,Tp]: delta(t,t') comb[t,i,j]
            ttnn.deallocate(comb_c)
            mix = ttnn.matmul(
                ttnn.experimental.view(A, [1, 1, n * Tp, n * Tp]),
                res_v,
                compute_kernel_config=self.ckc,
                program_config=self.pc_mix,
            )  # [(j,t), c]
            ttnn.deallocate(A)
            mix_sm = ttnn.reshape(mix, [1, n, Tp, self.C])
            out_sm = ttnn.addcmul(mix_sm, y32, post_c, output_tensor=mix_sm)  # + post_j * y
            ttnn.deallocate(post_c)
            if T != Tp:
                ttnn.deallocate(y32)
        else:
            assert not collapse, "collapse mode is for decode shapes (T <= 32)"
            comb_c = self._coef(comb, self.m_comb)  # [n(j), n(i), Tp, 1]
            post_c = self._coef(post, self.m_post)  # [1, n(j), Tp, 1]
            out_sm = ttnn.multiply(y32, post_c)  # [1,n,Tp,C]: post_j * y
            ttnn.deallocate(post_c)
            for i in range(n):
                x_i = ttnn.slice(x32, [0, 0, 0, i * self.C], [1, 1, Tp, (i + 1) * self.C])  # [1,1,Tp,C]
                c_i = ttnn.slice(comb_c, [0, i, 0, 0], [n, i + 1, Tp, 1])  # [n(j),1,Tp,1]
                c_i = ttnn.experimental.view(c_i, [1, n, Tp, 1])
                out_sm = ttnn.addcmul(out_sm, x_i, c_i, output_tensor=out_sm)
                ttnn.deallocate(x_i)
                ttnn.deallocate(c_i)
            ttnn.deallocate(comb_c)
        return _view_rows(self._packed(out_sm), T)

    # ------------------------------------------------------------------ TtMHCWrap-compatible API
    def project(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``[1,1,T,nC]`` -> mixes ``[1,1,T,32]`` (columns ``[mix_hc, 32)`` are zero)."""
        T = x.shape[-2]
        x32 = _view_rows(x, _rows_up(T))
        x_sm, owned = self._stream_major(x32)
        mixes = self._project(x32, x_sm)
        if owned:
            ttnn.deallocate(x_sm)
        return _view_rows(mixes, T)

    def hc_pre(self, x: ttnn.Tensor):
        """``[1,1,T,nC]`` -> (h ``[1,1,T,C]``, post ``[1,1,T,n]``, comb ``[1,1,T,n*n]``)."""
        T = x.shape[-2]
        h, post, comb, _, x_sm, owned = self._pre(x)
        if owned:
            ttnn.deallocate(x_sm)
        r4 = lambda t: _view_rows(ttnn.reshape(t, [1, 1, t.shape[-2], t.shape[-1]]), T)
        return h, r4(post), r4(comb)

    def hc_post(
        self, x: ttnn.Tensor, residual: ttnn.Tensor, post: ttnn.Tensor, comb: ttnn.Tensor, collapse: bool = False
    ) -> ttnn.Tensor:
        """x: F's output ``[1,1,T,C]`` fp32; residual ``[1,1,T,nC]``; post / comb as ``hc_pre`` returns them;
        ``collapse``: the mean over the n new streams ``[1,1,T,C]`` (see ``_post``)."""
        T = residual.shape[-2]
        res32 = _view_rows(residual, _rows_up(T))
        res_sm, owned = self._stream_major(res32)
        Tp = res_sm.shape[-2]
        out = self._post(x, res32, res_sm, _view_rows(post, Tp), _view_rows(comb, Tp), collapse=collapse)
        if owned:
            ttnn.deallocate(res_sm)
        return out

    def forward(self, x: ttnn.Tensor, sublayer, sublayer_dtype=None, norm=None, collapse: bool = False) -> ttnn.Tensor:
        """x: ``[1,1,T,nC]``; sublayer: ``[1,1,T,C]`` -> ``[1,1,T,C]``. Returns ``[1,1,T,nC]``.
        ``sublayer_dtype`` casts the collapsed hidden going in and back to ``x``'s dtype coming out;
        ``norm = (weight [1,1,1,C] bf16, eps)`` applies the sublayer's input RMSNorm to the collapsed
        hidden inside the pre-block (bf16 out; ``sublayer_dtype`` must then be bf16); ``collapse``
        returns the LM head's mean over the n new streams ``[1,1,T,C]`` fp32 instead of the streams
        (the last layer's FFN wrap feeding the head; decode shapes only)."""
        if norm is not None:
            assert sublayer_dtype == ttnn.bfloat16, "the fused norm emits bf16"
        h, post, comb, x32, x_sm, owned = self._pre(x, norm)
        if sublayer_dtype is None or norm is not None:
            cast = h
        else:
            cast = ttnn.typecast(h, sublayer_dtype)
            ttnn.deallocate(h)
        out = sublayer(cast)
        if sublayer_dtype is None:
            y = out
        else:
            y = ttnn.typecast(out, x.dtype)
            ttnn.deallocate(out)
            ttnn.deallocate(cast)
        new = self._post(y, x32, x_sm, post, comb, collapse=collapse)
        ttnn.deallocate(y)
        ttnn.deallocate(post)
        ttnn.deallocate(comb)
        if owned:
            ttnn.deallocate(x_sm)
        return new
