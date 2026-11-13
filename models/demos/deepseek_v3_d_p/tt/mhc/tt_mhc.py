# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Manifold-Constrained Hyper-Connections (mHC) on device:

    X' = H_res @ X + H_post.T @ F(H_pre @ X)

Mirrors the pure-torch ground truth in ../../reference/mhc/mhc_reference.py class for class
(MHCWrap -> TtMHCWrap, MHCHead -> TtMHCHead, mhc_expand -> mhc_expand) so the PCC tests can
compare method against method.

The implementation splits along the design boundary:
  - Parametrization (X -> the H matrices, including the Sinkhorn projection of H_res onto the
    doubly-stochastic manifold) is the fused kernel
    ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn, fed by `project`.
  - Computation (apply the H matrices around the sublayer F) stays composite ttnn.

Tensor convention: the n residual streams are packed along the last dim, x [1, 1, T, n*C] with
stream i at columns [i*C, (i+1)*C) and the T = B*S tokens down the tile rows. Both tiled dims
are then tile-aligned, which is the point: giving n a dim of its own pads it 4 -> 32, so every
stage tensor costs 8x its logical size (587 MB against 73.4 MB at T=640, C=7168, fp32) and the
single stream handed to F pads 1 -> 32 for 32x. These stages are bandwidth-bound, so that
padding is the cost. F is any callable [1,1,T,C] -> [1,1,T,C] (attention, MLP, ...) and owns
its own pre-norm, exactly like DeepSeek's Block -- and [1,1,T,C] is already what every other
ttnn sublayer takes, so F needs no reshape to contract over the sequence.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3_d_p.utils.weight_cache import from_torch_cached

W = 32  # one tile wide; the packed Sinkhorn state requires n*n <= W


def _compute_kernel_config():
    """HiFi4 + fp32 accumulation -- the Sinkhorn is iterative and fp32-sensitive."""
    kwargs = dict(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    return (
        ttnn.types.BlackholeComputeKernelConfig(**kwargs)
        if is_blackhole()
        else ttnn.types.WormholeComputeKernelConfig(**kwargs)
    )


def _selection_row_col(n):
    """Row-sum (RB) and col-sum (CB) broadcast matrices for the matmul-form Sinkhorn.

    Sinkhorn alternates row and column normalisations of a tiny n x n matrix per token --
    awkward on tile hardware. Flattening each token's matrix to a row of length n*n (entry
    (i,j) at column i*n+j) and packing the T tokens as M[T, W] turns

        row-sum broadcast == M @ RB        col-sum broadcast == M @ CB

    so every normalisation becomes `M / (M @ K)`: a same-shape elementwise divide, with no
    sub-tile broadcast or reduction. The pad columns [n*n, W) carry an identity block so their
    divisor is never zero (the softmax step has no eps in its denominator).
    """
    RB = torch.zeros(W, W, dtype=torch.float32)
    CB = torch.zeros(W, W, dtype=torch.float32)
    for p in range(n * n):
        pi, pj = divmod(p, n)
        for q in range(n * n):
            qi, qj = divmod(q, n)
            if qi == pi:
                RB[q, p] = 1.0  # q shares p's row -> contributes to p's row sum
            if qj == pj:
                CB[q, p] = 1.0  # q shares p's column -> contributes to p's column sum
    for p in range(n * n, W):
        RB[p, p] = 1.0
        CB[p, p] = 1.0
    return RB, CB


def build_consts(cfg, scale, base) -> torch.Tensor:
    """Bake a, b and the Sinkhorn selection matrices into the kernel's constant tensor.

    -> [8, 32, 32] fp32. The kernel is pure tile ops, so the host owns this layout; tile order
    is the host <-> kernel contract:
        0 SEL_pre   1 SEL_post   2 SEL_comb   3 base_pre   4 base_post   5 base_comb   6 RB   7 CB

    SEL_* extract and left-align each group out of the mix_hc-wide mixes and fold in the scalar
    a, so `mixes @ SEL_g` yields `a_g * raw_g` left-aligned. RB/CB are the Sinkhorn row/col-sum
    matrices (see _selection_row_col).
    """
    n = cfg.n
    a_pre, a_post, a_res = (float(scale[0]), float(scale[1]), float(scale[2]))
    base = base.float()

    sel_pre = torch.zeros(W, W)
    sel_post = torch.zeros(W, W)
    sel_comb = torch.zeros(W, W)
    for p in range(n):
        sel_pre[p, p] = a_pre
        sel_post[n + p, p] = a_post
    for p in range(n * n):
        sel_comb[2 * n + p, p] = a_res

    # per-column bias replicated down every token row (the kernel's add is tile + tile)
    base_pre = torch.zeros(W, W)
    base_post = torch.zeros(W, W)
    base_comb = torch.zeros(W, W)
    base_pre[:, :n] = base[0:n]
    base_post[:, :n] = base[n : 2 * n]
    base_comb[:, : n * n] = base[2 * n :]

    RB, CB = _selection_row_col(n)
    return torch.stack([sel_pre, sel_post, sel_comb, base_pre, base_post, base_comb, RB, CB], dim=0)


def _upload(device, t, shape, dtype):
    return ttnn.from_torch(t.contiguous().reshape(shape), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)


#: Cache names of the mHC weights (``<prefix>.<name>``); one list for the upload, the build and the check.
WRAP_CACHE_NAMES = ("fn_T", "consts")
HEAD_CACHE_NAMES = ("fn_T", "base", "scale")


def mhc_cache_complete(cache_path, cache_name_prefix: str, names) -> bool:
    """True iff every named mHC tensorbin exists. Needs ``init_checker(cache_path)`` first."""
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

    return all(pattern_exists(f"{cache_name_prefix}.{name}_dtype*.tensorbin", "V4mHC") for name in names)


def _cached_upload(device, t, shape, dtype, cache_file, *, mesh_mapper=None, on_device=True):
    """``_upload`` through the weight cache: ``t=None`` reads it, a name-less call is the plain upload."""
    if cache_file is None:
        return _upload(device if on_device else None, t, shape, dtype)
    return from_torch_cached(
        None if t is None else t.contiguous().reshape(shape),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device if on_device else None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
        cache_file=cache_file,
    )


def _upload_fn_T(device, cfg, fn, dtype, tp_factor, cache_file, on_device):
    """The transposed projection, TP-sharded in the 3D (stream, hidden, out) form so fn's rows land in the order x
    is split across chips: each chip's own hidden slice of every stream, not one whole stream. ``fn=None`` reads
    the cache."""
    n = cfg.n
    dev = device if on_device else None
    if tp_factor > 1:
        assert (
            cfg.dim % tp_factor == 0
        ), f"the mHC projection splits hidden {cfg.dim} across the TP axis, which the factor {tp_factor} does not divide"
        if fn is not None:
            assert fn.shape[1] == n * cfg.dim, f"fn is {tuple(fn.shape)}, expected [*, {n * cfg.dim}]"
        src = None if fn is None else fn.t().contiguous().reshape(1, n, cfg.dim, fn.shape[0])
        mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=tuple(device.shape), dims=(None, 2))
        if cache_file is None:
            fn_T = ttnn.from_torch(src, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dtype, mesh_mapper=mapper)
        else:
            fn_T = from_torch_cached(
                src,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
                cache_file=cache_file,
            )
        if on_device:
            shape = fn_T.shape
            fn_T = ttnn.reshape(fn_T, [1, 1, shape[1] * shape[2], shape[3]])
        return fn_T
    t = None if fn is None else fn.t()
    shape = None if fn is None else (1, 1, fn.shape[1], fn.shape[0])
    return _cached_upload(device, t, shape, dtype, cache_file, on_device=on_device)


def _streams(x, n):
    """The n per-stream slices of a packed [1,1,T,n*C] tensor."""
    T, C = x.shape[-2], x.shape[-1] // n
    return [ttnn.slice(x, [0, 0, 0, i * C], [1, 1, T, (i + 1) * C]) for i in range(n)]


def _cols(t, k):
    """The k single-column slices of [1,1,T,k], each broadcastable across a stream's C."""
    T = t.shape[-2]
    return [ttnn.slice(t, [0, 0, 0, j], [1, 1, T, j + 1]) for j in range(k)]


def _mix(streams, coeffs):
    """sum_i coeffs[i] * streams[i], with each [1,1,T,1] coefficient broadcast down its stream.

    addcmul rather than mac: mac lowers to a separate multiply and add, so it moves the
    accumulator through DRAM twice per term and measures 1.63x slower here.
    """
    y = ttnn.multiply(coeffs[0], streams[0])
    for s, c in zip(streams[1:], coeffs[1:]):
        y = ttnn.addcmul(y, s, c)
    return y


def _project(x, fn_T, norm_eps, ckc, tp_sum=None, tp_factor=1):
    """[1,1,T,n*C] -> [1,1,T,K] = RMSNorm(x) @ fn_T.

    RMSNorm carries no learned weight, so its rsqrt commutes with the linear and is applied
    after it -- one [T,1] broadcast instead of a full [T,n*C] scale.

    ``tp_sum`` is required once the streams are TP-sharded: the matmul is then a partial inner
    product and the squares a partial sum, both over this chip's slice of n*C. Every TP chip needs
    the same result -- each scales its own slice of every stream by the same scalar -- which is why
    the reduction has to end replicated rather than scattered.
    """
    mixes_un = ttnn.matmul(x, fn_T, compute_kernel_config=ckc)
    ss = ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True)
    width = x.shape[-1] * tp_factor
    if tp_sum is not None:
        mixes_un, ss = tp_sum(mixes_un), tp_sum(ss)
    ms = ttnn.multiply(ss, 1.0 / width)
    return ttnn.multiply(mixes_un, ttnn.rsqrt(ttnn.add(ms, norm_eps)))


def mhc_expand(h, n: int):
    """Embedding [1,1,T,C] -> n identical residual streams [1,1,T,n*C]."""
    return ttnn.repeat(h, [1, 1, 1, n])


def _cache_files(weight_cache_path, cache_name_prefix):
    """``name -> cache file stem`` for a module caching under ``<path>/<prefix>.<name>``, or None (no cache)."""
    if weight_cache_path is None:
        return None
    from pathlib import Path

    return lambda name: str(Path(weight_cache_path) / f"{cache_name_prefix}.{name}")


def _wrap_weights(device, cfg, fn, base, scale, dtype, tp_factor, cache_files, on_device=True):
    """``(fn_T, consts)`` of one TtMHCWrap, from the reference tensors or (all None) from the cache."""
    fn_T = _upload_fn_T(device, cfg, fn, dtype, tp_factor, cache_files and cache_files("fn_T"), on_device)
    consts = _cached_upload(
        device,
        None if scale is None else build_consts(cfg, scale, base),
        (8, W, W),
        dtype,
        cache_files and cache_files("consts"),
        on_device=on_device,
    )
    return fn_T, consts


def _head_weights(device, cfg, fn, base, scale, dtype, tp_factor, cache_files, on_device=True):
    """``(fn_T, base, a)`` of one TtMHCHead, from the reference tensors or (all None) from the cache. ``a`` is the
    host scalar ``scale[0]``, kept in the cache as a one-element tensor."""
    fn_T = _upload_fn_T(device, cfg, fn, dtype, tp_factor, cache_files and cache_files("fn_T"), on_device)
    base_t = _cached_upload(
        device,
        base,
        None if base is None else (1, 1, 1, cfg.n),
        dtype,
        cache_files and cache_files("base"),
        on_device=on_device,
    )
    if cache_files is None:
        return fn_T, base_t, float(scale[0])
    scale_t = from_torch_cached(
        None if scale is None else scale.detach().float().reshape(-1),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=None,
        memory_config=None,
        mesh_mapper=None,
        cache_file=cache_files("scale"),
    )
    return fn_T, base_t, float(ttnn.to_torch(scale_t).reshape(-1)[0])


class TtMHCWrap(LightweightModule):
    """One mHC-wrapped sublayer. DeepSeek-V4 has two per transformer block -- one around
    attention, one around the MoE FFN -- each with independent parameters.

    Parameters (constant at inference), uploaded once:
        fn:    [mix_hc, n*dim]  fused projection P
        base:  [mix_hc]         bias b
        scale: [3]              scalars a = (a_pre, a_post, a_res)
    The H matrices are not stored: they are recomputed from X every forward pass.
    """

    def __init__(
        self,
        device,
        cfg,
        fn: torch.Tensor | None,
        base: torch.Tensor | None,
        scale: torch.Tensor | None,
        dtype=ttnn.float32,
        tp_axis: int | None = None,
        num_links: int | None = None,
        topology=ttnn.Topology.Linear,
        weight_cache_path=None,
        cache_name_prefix: str | None = None,
    ):
        """``fn``, ``base`` and ``scale`` are the reference module's, or all ``None`` to read the weights from the
        cache at ``<weight_cache_path>/<cache_name_prefix>.*`` (``from_cache``)."""
        # the fused parametrization op is fp32-only; reject here rather than fail deep in hc_pre
        assert dtype == ttnn.float32, f"TtMHCWrap is fp32-only (fused op requires FLOAT32), got {dtype}"
        self.device = device
        self.cfg = cfg
        self.n = cfg.n
        self.iters = int(cfg.sinkhorn_iters)
        self.eps = float(cfg.eps)
        self.norm_eps = float(cfg.norm_eps)
        self.ckc = _compute_kernel_config()
        # tp_axis is None on a single device, where nothing is sharded and no reduction is needed.
        # fn_T is sharded along mesh axis 1 below, so that is the only TP axis supported.
        assert tp_axis in (None, 1), f"tp_axis must be None or 1 (fn_T is sharded along mesh axis 1), got {tp_axis}"
        self.tp_axis = tp_axis
        self.num_links = num_links if num_links is not None else (2 if is_blackhole() else 1)
        self.topology = topology
        self.tp_factor = device.shape[tp_axis] if tp_axis is not None else 1
        self.tt_ccl = get_tt_ccl(device) if self.tp_factor > 1 else None
        # transposed so mixes = xnorm @ fn_T, matching the reference F.linear(xf, fn).
        self.fn_T, self.consts = _wrap_weights(
            device, cfg, fn, base, scale, dtype, self.tp_factor, _cache_files(weight_cache_path, cache_name_prefix)
        )

    @classmethod
    def from_cache(cls, device, cfg, **kwargs):
        """Build from the weight cache alone (``weight_cache_path`` and ``cache_name_prefix`` in ``kwargs``): no
        torch tensors, and a missing file raises."""
        return cls(device, cfg, None, None, None, **kwargs)

    @staticmethod
    def build_ttnn_cache(
        fn, base, scale, cfg, mesh_device, cache_path, cache_name_prefix, dtype=ttnn.float32, tp_axis=None
    ):
        """Write the weights to the cache without putting any of them on the device."""
        tp_factor = mesh_device.shape[tp_axis] if tp_axis is not None else 1
        _wrap_weights(
            mesh_device,
            cfg,
            fn,
            base,
            scale,
            dtype,
            tp_factor,
            _cache_files(cache_path, cache_name_prefix),
            on_device=False,
        )

    @staticmethod
    def check_cache_complete(cache_path, cache_name_prefix: str) -> bool:
        """True iff the weights are all cached. Needs ``init_checker(cache_path)`` first."""
        return mhc_cache_complete(cache_path, cache_name_prefix, WRAP_CACHE_NAMES)

    def _tp_sum(self, t):
        """Sum ``t`` across the TP axis, leaving the result on every chip."""
        return ttnn.experimental.all_reduce_async(
            t,
            cluster_axis=self.tp_axis,
            mesh_device=self.device,
            barrier_semaphores=self.tt_ccl.barrier_semaphore_handles[self.tp_axis],
            rs_global_semaphores=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
            ag_global_semaphores=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
            num_links=self.num_links,
            math_op=ttnn.ReduceType.Sum,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
        )

    def project(self, x):
        """[1,1,T,n*C] -> mixes [1,1,T,mix_hc], the fused kernel's input."""
        tp_sum = self._tp_sum if self.tp_factor > 1 else None
        return _project(x, self.fn_T, self.norm_eps, self.ckc, tp_sum, self.tp_factor)

    def hc_pre(self, x):
        """[1,1,T,n*C] -> (y [1,1,T,C], post [1,1,T,n], comb [1,1,T,n*n]).

        y = sum_i pre_i * x_i is the single stream handed to F. The reduction uses the raw
        stream values; only the projection input is normalised.
        """
        T = x.shape[-2]
        mixes = self.project(x)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, self.n, self.iters, self.eps
        )
        # the kernel emits rank 2, [T,k]; prepending unit dims is a view, not data movement
        r4 = lambda t: ttnn.reshape(t, [1, 1, T, t.shape[-1]])
        return _mix(_streams(x, self.n), _cols(r4(pre), self.n)), r4(post), r4(comb)

    def hc_post(self, x, residual, post, comb):
        """x: F's output [1,1,T,C]; residual: [1,1,T,n*C] -> [1,1,T,n*C].

        Per output stream j:  new_j = post_j * x + sum_i comb[i,j] * residual_i,
        i.e. the residual mixing applies comb^T (still doubly-stochastic).
        """
        n = self.n
        res, cmb = _streams(residual, n), _cols(comb, n * n)
        out = [ttnn.multiply(p, x) for p in _cols(post, n)]
        for i in range(n):
            for j in range(n):
                out[j] = ttnn.addcmul(out[j], res[i], cmb[i * n + j])
        return ttnn.concat(out, dim=-1)

    def forward(self, x, sublayer, sublayer_dtype=None):
        """x: [1,1,T,n*C]; sublayer: [1,1,T,C] -> [1,1,T,C]. Returns [1,1,T,n*C].

        The mHC arithmetic needs fp32 and the sublayer has its own dtype: ``sublayer_dtype`` casts
        the collapsed hidden to it going in, and back to ``x``'s dtype coming out.
        """
        residual = x
        h, post, comb = self.hc_pre(x)
        if sublayer_dtype is None:
            h = sublayer(h)
        else:
            cast = ttnn.typecast(h, sublayer_dtype)
            ttnn.deallocate(h)
            out = sublayer(cast)
            h = ttnn.typecast(out, x.dtype)
            ttnn.deallocate(out)
            ttnn.deallocate(cast)
        return self.hc_post(h, residual, post, comb)


class TtMHCHead(LightweightModule):
    """Collapse the n streams back to 1 at the model output.

    Parameters: fn [n, n*dim], base [n], scale [1]. Pre-only -- there is no H_res here, so no
    Sinkhorn and no fused kernel: this runs once per model, not once per layer.
    """

    def __init__(
        self,
        device,
        cfg,
        fn: torch.Tensor | None,
        base: torch.Tensor | None,
        scale: torch.Tensor | None,
        dtype=ttnn.float32,
        tp_axis: int | None = None,
        num_links: int | None = None,
        topology=ttnn.Topology.Linear,
        weight_cache_path=None,
        cache_name_prefix: str | None = None,
    ):
        """``fn``, ``base`` and ``scale`` are the reference module's, or all ``None`` to read the weights from the
        cache at ``<weight_cache_path>/<cache_name_prefix>.*`` (``from_cache``)."""
        self.device = device
        self.cfg = cfg
        self.n = cfg.n
        self.eps = float(cfg.eps)
        self.norm_eps = float(cfg.norm_eps)
        self.ckc = _compute_kernel_config()
        # Same TP contract as TtMHCWrap: fn_T is sharded along mesh axis 1 in the order x is split.
        assert tp_axis in (None, 1), f"tp_axis must be None or 1 (fn_T is sharded along mesh axis 1), got {tp_axis}"
        self.tp_axis = tp_axis
        self.num_links = num_links if num_links is not None else (2 if is_blackhole() else 1)
        self.topology = topology
        self.tp_factor = device.shape[tp_axis] if tp_axis is not None else 1
        self.tt_ccl = get_tt_ccl(device) if self.tp_factor > 1 else None
        self.fn_T, self.base, self.a = _head_weights(
            device, cfg, fn, base, scale, dtype, self.tp_factor, _cache_files(weight_cache_path, cache_name_prefix)
        )

    @classmethod
    def from_cache(cls, device, cfg, **kwargs):
        """Build from the weight cache alone (``weight_cache_path`` and ``cache_name_prefix`` in ``kwargs``): no
        torch tensors, and a missing file raises."""
        return cls(device, cfg, None, None, None, **kwargs)

    @staticmethod
    def build_ttnn_cache(
        fn, base, scale, cfg, mesh_device, cache_path, cache_name_prefix, dtype=ttnn.float32, tp_axis=None
    ):
        """Write the weights to the cache without putting any of them on the device."""
        tp_factor = mesh_device.shape[tp_axis] if tp_axis is not None else 1
        _head_weights(
            mesh_device,
            cfg,
            fn,
            base,
            scale,
            dtype,
            tp_factor,
            _cache_files(cache_path, cache_name_prefix),
            on_device=False,
        )

    @staticmethod
    def check_cache_complete(cache_path, cache_name_prefix: str) -> bool:
        """True iff the weights are all cached. Needs ``init_checker(cache_path)`` first."""
        return mhc_cache_complete(cache_path, cache_name_prefix, HEAD_CACHE_NAMES)

    def _tp_sum(self, t):
        """Sum ``t`` across the TP axis, leaving the result on every chip."""
        return ttnn.experimental.all_reduce_async(
            t,
            cluster_axis=self.tp_axis,
            mesh_device=self.device,
            barrier_semaphores=self.tt_ccl.barrier_semaphore_handles[self.tp_axis],
            rs_global_semaphores=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
            ag_global_semaphores=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
            num_links=self.num_links,
            math_op=ttnn.ReduceType.Sum,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
        )

    def forward(self, x):
        """[1,1,T,n*C] -> [1,1,T,C]."""
        tp_sum = self._tp_sum if self.tp_factor > 1 else None
        mixes = _project(x, self.fn_T, self.norm_eps, self.ckc, tp_sum, self.tp_factor)
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.mul(mixes, self.a), self.base)), self.eps)
        return _mix(_streams(x, self.n), _cols(pre, self.n))
