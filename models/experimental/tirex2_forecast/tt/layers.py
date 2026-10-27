# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TiRex-2 layers in stock TTNN ops (round 0: op-level sLSTM loop, chunkwise mLSTM from matmuls).

Shapes: ``B`` device rows (multiple of 32), ``T`` padded sequence (320), ``T_real`` = 285 tokens.
"""

from __future__ import annotations

import os

import torch

from models.experimental.tirex2_forecast.reference import dataflow as df
from models.experimental.tirex2_forecast.tt.common import TILE, Precision, bmm, exact_config, upload
from models.experimental.tirex2_forecast.tt.weights import TtFFN, TtMLSTM, TtResidualBlock, TtSLSTM, TtVariate

MASK_NEG = -1e9

# Device constants (zeros, ones, masks) are owned by a ``DeviceConsts`` object per model and device. They are
# created on first use, which the warm-up forward does before any trace capture, and ``free`` never frees them.
_CONST_IDS: set = set()


class DeviceConsts:
    """Per-device constants for one model: the chunkwise mLSTM causal mask and cached zero / one tensors.

    Owned by ``TtTiRex2`` and released with it (``release``), so nothing outlives its device."""

    def __init__(self, device, chunk: int):
        import ttnn

        self.device = device
        self.chunk = chunk
        self._cache: dict = {}
        m = torch.zeros(chunk, chunk)
        m[~torch.ones(chunk, chunk, dtype=torch.bool).tril()] = MASK_NEG
        self.mask = self._register(upload(device, m.reshape(1, 1, chunk, chunk), dtype=ttnn.float32))

    def _register(self, t):
        _CONST_IDS.add(id(t))
        return t

    def full(self, value: float, shape, dtype=None):
        import ttnn

        dtype = ttnn.float32 if dtype is None else dtype
        key = (float(value), tuple(shape), str(dtype))
        if key not in self._cache:
            self._cache[key] = self._register(upload(self.device, torch.full(tuple(shape), float(value)), dtype=dtype))
        return self._cache[key]

    def tensor(self, key, host: torch.Tensor, dtype=None):
        """A cached device copy of ``host`` under ``key`` (e.g. the pad states), freed with the others."""
        import ttnn

        key = ("tensor", *key)
        if key not in self._cache:
            self._cache[key] = self._register(upload(self.device, host, dtype=dtype or ttnn.float32))
        return self._cache[key]

    def release(self):
        import ttnn

        for t in (self.mask, *self._cache.values()):
            _CONST_IDS.discard(id(t))
            ttnn.deallocate(t)
        self._cache.clear()


def free(*tensors):
    import ttnn

    for t in tensors:
        if t is not None and id(t) not in _CONST_IDS:
            ttnn.deallocate(t)


def _flat(x):
    """``[B, 1, T, K]`` -> ``[1, 1, B*T, K]`` (a view): a batched linear does not fold B into M (3-15 % of peak)."""
    import ttnn

    shape = tuple(x.shape)
    if shape[0] * shape[1] == 1:
        return x, None
    return ttnn.reshape(x, (1, 1, shape[0] * shape[1] * shape[2], shape[3])), shape


def _unflat(y, shape):
    import ttnn

    return y if shape is None else ttnn.reshape(y, (*shape[:3], y.shape[-1]))


# minimal_matmul (M, K, N) blocks and subblock per (K, N), from tests/perf/sweep_matmul.py (R1-2f, M = 131072,
# HiFi4; doc/optimize/iter_01/r1_2f/matmul_sweep.md): 2.0-2.5x faster than ttnn.linear's default config
MM_BLOCKS = {
    (512, 1536): (8, 8, 8, 2, 2),
    (512, 2048): (8, 8, 8, 4, 1),
    (512, 512): (4, 8, 8, 4, 1),
    (2048, 512): (8, 8, 4, 4, 1),
    (1408, 512): (16, 11, 4, 2, 2),  # FFN w_down, R1-2f' (iter_01/r1_2fp/matmul_sweep.md: 4.00 -> 1.82 ms)
    (64, 512): (4, 2, 8, 2, 2),
    (512, 288): (8, 8, 4, 1, 4),
}
MM_MIN_M_TILES = 64  # below this the default linear
# variance table (D5 c): roles listed here (comma separated) go back to ttnn.linear; measurement only
MM_REVERT = frozenset(r for r in os.environ.get("TIREX2_MM_REVERT", "").split(",") if r)


def _mm_config(xf, w, role=None):
    import ttnn

    if role in MM_REVERT:
        return None
    m_tiles = xf.padded_shape[-2] // TILE
    K, N = w.padded_shape[-2], w.padded_shape[-1]
    if m_tiles < MM_MIN_M_TILES or (K, N) not in MM_BLOCKS or w.dtype == ttnn.float32:
        return None  # only the swept roles (fp32-weight gate projections keep the default linear)
    k_tiles = K // TILE
    mb, kb, nb, sh, sw = MM_BLOCKS[(K, N)]
    kb = max(d for d in range(1, min(kb, k_tiles) + 1) if k_tiles % d == 0)
    grid = xf.device().compute_with_storage_grid_size()
    return ttnn.MinimalMatmulConfig(
        M_block_size=mb,
        K_block_size=kb,
        N_block_size=nb,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
    )


def linear(x, w, precision: Precision, bias=None, activation=None, dtype=None, config=None, role=None):
    import ttnn

    xf, shape = _flat(x)
    mm = _mm_config(xf, w, role)
    if mm is not None:
        acts = {"silu": ttnn.UnaryOpType.SILU}
        y = ttnn.experimental.minimal_matmul(
            xf,
            w,
            bias_tensor=bias,
            fused_activation=None if activation is None else ttnn.UnaryWithParam(acts[activation]),
            config=mm,
            dtype=ttnn.bfloat16 if dtype is None else dtype,
            compute_kernel_config=precision.matmul_config(role) if config is None else config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return _unflat(y, shape)
    y = ttnn.linear(
        xf,
        w,
        bias=bias,
        activation=activation,
        dtype=ttnn.bfloat16 if dtype is None else dtype,
        compute_kernel_config=precision.matmul_config(role) if config is None else config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return _unflat(y, shape)


def swiglu(x, w, precision: Precision):
    """``silu(x W_gate) * (x W_up)`` in one ``minimal_matmul(fuse_swiglu=True)`` on a ``swiglu_pairs`` weight."""
    import ttnn

    xf, shape = _flat(x)
    M, K = xf.padded_shape[-2], w.padded_shape[-2]
    grid = xf.device().compute_with_storage_grid_size()
    m_tiles, k_tiles = M // TILE, K // TILE
    m_block = min(8, m_tiles)
    config = ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=max(d for d in range(1, min(k_tiles, 8) + 1) if k_tiles % d == 0),
        N_block_size=2,
        subblock_h=max(d for d in range(1, min(m_block, 2) + 1) if m_block % d == 0),
        subblock_w=2,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
    )
    y = ttnn.experimental.minimal_matmul(
        xf,
        w,
        config=config,
        fuse_swiglu=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.matmul_config("ffn"),
    )
    return _unflat(y, shape)


def rms_norm(x, eps: float):
    """Gain-free RMSNorm (gains are folded into the next linear)."""
    import ttnn

    return ttnn.rms_norm(x, epsilon=eps, compute_kernel_config=exact_config(), memory_config=ttnn.DRAM_MEMORY_CONFIG)


def residual_block(x, w: TtResidualBlock, precision: Precision, role: str):
    """``W_o silu(W_h x + b_h) + W_r x + (b_o + b_r)``; ``role`` is ``embed`` or ``head``."""
    import ttnn

    h = linear(x, w.w_hidden, precision, bias=w.b_hidden, activation="silu", role=role)
    out = linear(h, w.w_out, precision, bias=w.b_out, role=role)
    free(h)
    res = linear(x, w.w_res, precision, role=role)
    y = ttnn.add(out, res)
    free(out, res)
    return y


def ffn_residual(x, w: TtFFN, precision: Precision):
    """``x + W_down(silu(W_gate xn) * W_up xn)`` with ``xn`` the gain-free RMSNorm of ``x``."""
    import ttnn

    xn = rms_norm(x, df.NORM_EPS)
    h = swiglu(xn, w.w_swiglu, precision)
    free(xn)
    y = linear(h, w.w_down, precision, role="ffn")
    free(h)
    out = ttnn.add(x, y)
    free(y)
    return out


# ------------------------------------------------------------------------------------------------- mLSTM


def mlstm_chunkwise(device, q, k, v, g, consts: DeviceConsts, eps: float = df.MLSTM_EPS, init=None):
    """Stabilized chunkwise mLSTM (see ``reference/dataflow.mlstm_chunkwise``).

    ``q [B, H, T, 64]`` (pre-scaled by 1/8), ``k [B, H, T, 64]``, ``v [B, H, T, 128]`` bf16;
    ``g [B, H, T, 32]`` fp32 with col 0 = capped input gate, col 1 = capped forget gate. -> ``h [B, H, T, 128]``
    fp32. The chunk-local terms are batched over (row, head, chunk); only the state carry loops over chunks.
    ``init``: the starting state ``(C [B*H, 1, 64, 128], n [B*H, 1, 64, 1], m [B*H, 1, 1, 1])`` (default zeros).
    """
    import ttnn

    B, H, T, DK = tuple(q.shape)
    DV = v.shape[-1]
    L = consts.chunk
    NC = T // L
    BH, N = B * H, B * H * T // L
    f32 = ttnn.float32
    qc = ttnn.reshape(q, (N, 1, L, DK))
    kc = ttnn.reshape(k, (N, 1, L, DK))
    vc = ttnn.reshape(v, (N, 1, L, DV))
    gc = ttnn.reshape(g, (N, 1, L, g.shape[-1]))
    i_col = ttnn.slice(gc, (0, 0, 0, 0), (N, 1, L, 1))
    f_col = ttnn.slice(gc, (0, 0, 0, 1), (N, 1, L, 2))
    sig = ttnn.sigmoid(f_col)
    lf = ttnn.log(sig)  # f is soft-capped to [-15, 15]: log(sigmoid) is exact enough in fp32
    free(sig, f_col)
    b_col = ttnn.cumsum(lf, dim=2)  # inclusive cumsum of log forget gates inside each chunk
    free(lf)
    b_row = ttnn.transpose(b_col, -2, -1)
    i_row = ttnn.transpose(i_col, -2, -1)
    free(i_col)
    u_row = ttnn.subtract(i_row, b_row)  # i_s - b_s
    free(i_row, b_row)
    D = ttnn.add(b_col, u_row)  # b_t - b_s + i_s
    D_masked = ttnn.add(D, consts.mask)
    free(D)
    rmax = ttnn.max(D_masked, dim=-1, keepdim=True)  # [N, 1, L, 1]
    g_last = ttnn.slice(b_col, (0, 0, L - 1, 0), (N, 1, L, 1))  # total log forget of each chunk
    a_row = ttnn.add(u_row, g_last)  # g - b_s + i_s: log weight of token s at the chunk end
    free(u_row)
    amax = ttnn.max(a_row, dim=-1, keepdim=True)  # [N, 1, 1, 1]
    _act = _Acts(ttnn)
    wa = ttnn.subtract(a_row, amax, activations=[_act.EXP])
    free(a_row)
    kT = ttnn.transpose(kc, -2, -1)  # [N, 1, DK, L]
    kw = ttnn.multiply(kT, wa, dtype=f32)
    free(kT, wa)
    dC = bmm(device, kw, vc)  # [N, 1, DK, DV]: chunk-local state, stabilized by amax
    dn = ttnn.sum(kw, dim=-1, keepdim=True)  # [N, 1, DK, 1]
    free(kw)

    # carry the states across chunks: C_{c+1} = exp(g + m - m') C_c + exp(amax - m') dC_c
    g4 = ttnn.reshape(g_last, (BH, NC, 1, 1))
    a4 = ttnn.reshape(amax, (BH, NC, 1, 1))
    dC4 = ttnn.reshape(dC, (BH, NC, DK, DV))
    dn4 = ttnn.reshape(dn, (BH, NC, DK, 1))
    if init is None:
        m = consts.full(0.0, (BH, 1, 1, 1))
        C = consts.full(0.0, (BH, 1, DK, DV))
        n = consts.full(0.0, (BH, 1, DK, 1))
    else:
        C, n, m = init
    Cs, ns, ms = [C], [n], [m]
    for c in range(NC - 1):  # the state after the last chunk is never read
        sl = lambda t, d1, d2: ttnn.slice(t, (0, c, 0, 0), (BH, c + 1, d1, d2))
        gk, ak = sl(g4, 1, 1), sl(a4, 1, 1)
        gm = ttnn.add(gk, m)
        m_next = ttnn.maximum(gm, ak)
        decay = ttnn.subtract(gm, m_next, activations=[_act.EXP])
        scale = ttnn.subtract(ak, m_next, activations=[_act.EXP])
        dCk, dnk = sl(dC4, DK, DV), sl(dn4, DK, 1)
        C = ttnn.add(ttnn.multiply(C, decay), ttnn.multiply(dCk, scale))
        n = ttnn.add(ttnn.multiply(n, decay), ttnn.multiply(dnk, scale))
        free(gk, ak, gm, decay, scale, dCk, dnk)
        m = m_next
        Cs.append(C)
        ns.append(n)
        ms.append(m)
    free(dC, dn, g_last, amax)
    C_all = ttnn.reshape(ttnn.concat(Cs, dim=1), (N, 1, DK, DV))
    n_all = ttnn.reshape(ttnn.concat(ns, dim=1), (N, 1, DK, 1))
    m_all = ttnn.reshape(ttnn.concat(ms, dim=1), (N, 1, 1, 1))
    free(*Cs, *ns, *ms)

    # outputs: intra-chunk weights exp(D - m_t) * (q . k) plus the decayed inter-chunk state
    S = bmm(device, qc, kc, transpose_b=True)  # [N, 1, L, L]
    mb = ttnn.add(b_col, m_all)  # b_t + m_prev
    m_t = ttnn.maximum(mb, rmax)
    free(b_col, rmax)
    Dm = ttnn.subtract(D_masked, m_t)
    free(D_masked)
    W = ttnn.multiply(Dm, S, input_tensor_a_activations=[_act.EXP])
    free(Dm, S)
    inter = ttnn.subtract(mb, m_t, activations=[_act.EXP])  # [N, 1, L, 1]
    free(mb)
    qC = bmm(device, qc, C_all)
    WV = bmm(device, W, vc)
    num = ttnn.add(ttnn.multiply(qC, inter), WV)
    free(qC, WV, C_all)
    qn = bmm(device, qc, n_all)  # [N, 1, L, 1]
    Wsum = ttnn.sum(W, dim=-1, keepdim=True)
    free(W, n_all, m_all)
    den = ttnn.add(ttnn.multiply(qn, inter), Wsum)
    free(qn, Wsum, inter)
    floor = ttnn.exp(ttnn.neg(m_t))
    den = ttnn.add(ttnn.maximum(ttnn.abs(den), floor), eps)
    free(floor, m_t)
    h = ttnn.divide(num, den)
    free(num, den)
    return ttnn.reshape(h, (B, H, T, DV))


def _keep(debug, name, t):
    if debug is not None:
        import ttnn

        debug[name] = ttnn.to_torch(t).float()


def mlstm_layer(
    device, x, w: TtMLSTM, precision: Precision, consts: DeviceConsts, debug: dict | None = None, init=None
):
    """``x + mLSTM(RMSNorm(x))`` (time mixer only; the FFN follows). ``debug`` (tests) collects host copies.
    ``init``: the chunkwise starting state (see ``mlstm_chunkwise``)."""
    import ttnn

    xn = rms_norm(x, df.NORM_EPS)
    _keep(debug, "xn", xn)
    qkv = linear(xn, w.w_qkv, precision, role="mlstm")  # [B, 1, T, 1536]
    q128, k128, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=4, num_kv_heads=4, transpose_k_heads=False)
    free(qkv)
    B, H, T, _ = tuple(q128.shape)
    q = ttnn.slice(q128, (0, 0, 0, 0), (B, H, T, df.QK_DIM))
    k = ttnn.slice(k128, (0, 0, 0, 0), (B, H, T, df.QK_DIM))
    free(q128, k128)
    for nm, t in (("q", q), ("k", k), ("v", v)):
        _keep(debug, nm, t)
    gs = linear(xn, w.w_gates, precision, bias=w.b_gates, dtype=ttnn.float32, config=exact_config())
    o = linear(xn, w.w_o, precision, role="mlstm")
    free(xn)
    gates = [ttnn.slice(gs, (0, 0, 0, 32 * h), (B, 1, T, 32 * (h + 1))) for h in range(df.HEADS)]
    free(gs)
    g = ttnn.concat(gates, dim=1)  # [B, 4, T, 32]
    free(*gates)
    gt = ttnn.tanh(g)
    free(g)
    g = ttnn.multiply(gt, df.GATE_CAP)  # 15 tanh(x / 15) (the 1/15 is in the weights)
    free(gt)
    _keep(debug, "g", g)
    h = mlstm_chunkwise(device, q, k, v, g, consts, init=init)
    free(q, k, v, g)
    _keep(debug, "h", h)
    hn = ttnn.layer_norm(h, epsilon=df.NORM_EPS, compute_kernel_config=exact_config())  # per head (128)
    free(h)
    _keep(debug, "hn", hn)
    hm = ttnn.experimental.nlp_concat_heads(hn)  # [B, 1, T, 512]
    free(hn)
    so = ttnn.sigmoid(o)
    free(o)
    gated = ttnn.multiply(hm, so, dtype=ttnn.bfloat16)
    free(hm, so)
    y = linear(gated, w.w_out, precision, role="mlstm")
    free(gated)
    _keep(debug, "y", y)
    out = ttnn.add(x, y)
    free(y)
    return out


# ------------------------------------------------------------------------------------------------- sLSTM


def log_sigmoid(x):
    """``log(sigmoid(max(x, -80)))``: ``ttnn.log_sigmoid`` is off by up to 2e-2 in fp32; this is ~5e-7."""
    import ttnn

    xc = ttnn.maximum(x, -80.0)
    s = ttnn.sigmoid(xc)
    free(xc)
    out = ttnn.log(s)
    free(s)
    return out


class _Acts:
    """Unary activations to fuse into binary ops."""

    def __init__(self, ttnn):
        for k in ("SIGMOID", "LOG", "EXP", "TANH"):
            setattr(self, k, ttnn.UnaryWithParam(getattr(ttnn.UnaryOpType, k)))
        self.N_FLOOR = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 1.0)


def slstm_recurrence(
    device,
    WxT,
    r,
    steps: int,
    consts: DeviceConsts,
    record: list | None = None,
    step_hook=None,
    y_fp32: bool = False,
    init=None,
):
    """sLSTM time loop. ``WxT [T, 1, B, 2048]`` fp32 (bias folded), columns (gate i f z o, head, d).

    Returns the list of per-step ``y [1, 1, B, 512]`` fp32 and the last ``(c, n, m)``. ``record`` (tests
    only) collects host copies of ``(y, c, n, m)`` after every step; ``step_hook(t)`` (profiling) runs after
    every step, e.g. to drain the device profiler buffers. ``y_fp32`` feeds the fp32 ``y`` (not its bf16 cast)
    into the recurrent matmul. ``init``: the state ``(y, c, n, m)`` before the first step, each ``[1, 1, B, 512]``
    fp32 (device constants; default: none, so step 0 uses ``m = pre_i``).
    """
    import ttnn

    T, _, B, G = tuple(WxT.shape)
    D = G // 4
    _act = _Acts(ttnn)
    f32 = ttnn.float32
    ys = []
    y_bf = c = n = m = None
    if init is not None:
        y0, c, n, m = init
        y_bf = y0 if y_fp32 else ttnn.typecast(y0, ttnn.bfloat16)
    for t in range(steps):
        wx = ttnn.slice(WxT, (t, 0, 0, 0), (t + 1, 1, B, G))
        if t == 0 and init is None:
            pre = wx
        else:
            ry = ttnn.matmul(y_bf, r, dtype=f32, compute_kernel_config=exact_config())
            pre = ttnn.add(ry, wx)
            free(ry, wx)
        pi, pf, pz, po = (ttnn.slice(pre, (0, 0, 0, j * D), (1, 1, B, (j + 1) * D)) for j in range(4))
        free(pre)
        if t == 0 and init is None:
            # n == 0 everywhere before the first step: m = pre_i, i = 1, c = tanh(z), n = 1
            free(pf)
            m, c = pi, ttnn.tanh(pz)
            free(pz)
            n = consts.full(1.0, (1, 1, B, D))
            y = ttnn.multiply(c, po, input_tensor_b_activations=[_act.SIGMOID])
            free(po)
        else:
            # activations fused into the binary ops (not into the scalar maximum: it drops them)
            xc = ttnn.maximum(pf, -80.0)
            sf = ttnn.sigmoid(xc)
            free(pf, xc)
            lfm = ttnn.add(sf, m, input_tensor_a_activations=[_act.LOG])  # log_sigmoid(f) + m
            free(sf)
            m_new = ttnn.maximum(pi, lfm)
            ig = ttnn.subtract(pi, m_new, activations=[_act.EXP])
            fg = ttnn.subtract(lfm, m_new, activations=[_act.EXP])
            free(pi, lfm, m)
            iz = ttnn.multiply(ig, pz, input_tensor_b_activations=[_act.TANH])
            free(pz)
            fc = ttnn.multiply(fg, c)
            free(c)
            c = ttnn.add(fc, iz)
            free(fc, iz)
            fn = ttnn.multiply(fg, n)
            free(n, fg)
            n = ttnn.add(fn, ig, activations=[_act.N_FLOOR])  # max(fg n + ig, 1)
            free(fn, ig)
            oc = ttnn.multiply(c, po, input_tensor_b_activations=[_act.SIGMOID])
            free(po)
            y = ttnn.divide(oc, n)
            free(oc)
            m = m_new
        if record is not None:
            record.append(tuple(ttnn.to_torch(t).float()[0, 0] for t in (y, c, n, m)))
        if y_bf is not None and not y_fp32:
            free(y_bf)
        y_bf = y if y_fp32 else ttnn.typecast(y, ttnn.bfloat16)
        ys.append(y)
        if step_hook is not None:
            step_hook(t)
    if not y_fp32:
        free(y_bf)
    return ys, (c, n, m)


def head_layer_norm(device, y, w_gnorm, head_mean, eps: float = df.NORM_EPS):
    """Per-head (128) LayerNorm of ``y [..., 512]`` fp32 on a head-split view, times ``w_gnorm`` (``head_mean`` unused)."""
    import ttnn

    shape = tuple(y.shape)
    yh = ttnn.reshape(y, (*shape[:-2], shape[-2] * (shape[-1] // 128), 128))  # each row as 4 head rows, no tile padding
    zh = ttnn.layer_norm(yh, epsilon=eps, compute_kernel_config=exact_config())
    z = ttnn.reshape(zh, shape)
    free(zh)
    out = ttnn.multiply(z, w_gnorm, dtype=ttnn.bfloat16)
    free(z)
    return out


def slstm_prepare(x, w: TtSLSTM, precision: Precision, time_major: bool = False):
    """``Wx`` time-major ``[T, 1, B, 2048]`` fp32 (gate bias folded) from the residual stream ``x``
    (``[B, 1, T, 512]``, or ``[T, 1, B, 512]`` when ``time_major``)."""
    import ttnn

    xn = rms_norm(x, df.NORM_EPS)
    if time_major:
        xT = xn
    else:
        xT = ttnn.transpose(xn, 0, 2)  # [T, 1, B, 512]
        free(xn)
    WxT = linear(xT, w.w_in, precision, bias=w.bias, dtype=ttnn.float32, config=precision.gate_config(), role="w_in")
    free(xT)
    return WxT


def slstm_finish(
    device, x, ys, w: TtSLSTM, head_mean, consts: DeviceConsts, debug: dict | None = None, time_major: bool = False
):
    """``x + GroupNorm(y)`` from the per-step outputs ``ys`` (padded with zeros to ``T``), in ``x``'s layout."""
    import ttnn

    T, _, B, _ = tuple(x.shape) if time_major else (x.shape[2], 1, x.shape[0], 1)
    steps = sum(y.shape[0] for y in ys)
    if steps < T:
        ys = [*ys, consts.full(0.0, (T - steps, 1, B, ys[0].shape[-1]))]
    if len(ys) == 1:
        Y = ys[0]
    else:
        Y = ttnn.concat(ys, dim=0)  # [T, 1, B, 512] fp32
        free(*ys)
    yn = head_layer_norm(device, Y, w.w_gnorm, head_mean)
    free(Y)
    if time_major:
        y = yn
    else:
        y = ttnn.transpose(yn, 0, 2)  # [B, 1, T, 512]
        free(yn)
    _keep(debug, "y", y)
    out = ttnn.add(x, y)
    free(y)
    return out


# fused sLSTM step kernel (ops/slstm_step) where the loop starts from zero state; TIREX2_SLSTM_KERNEL=0 turns it off
SLSTM_KERNEL = os.environ.get("TIREX2_SLSTM_KERNEL", "1") != "0"


def slstm_layer(
    device,
    x,
    w: TtSLSTM,
    precision: Precision,
    head_mean,
    steps: int,
    consts: DeviceConsts,
    debug: dict | None = None,
    step_hook=None,
    init=None,
    time_major: bool = False,
):
    """``x + GroupNorm(sLSTM(RMSNorm(x)))``; the recurrence runs time-major. ``debug`` (tests) collects host copies.
    ``init``: the state before the first step (see ``slstm_recurrence``). ``time_major``: ``x`` (and the result)
    is ``[T, 1, B, 512]``."""
    WxT = slstm_prepare(x, w, precision, time_major)
    if SLSTM_KERNEL and init is None and not precision.y_fp32:
        from models.experimental.tirex2_forecast.ops.slstm_step.op import slstm_steps

        Y, c, n, m = slstm_steps(WxT, w.r, steps)  # fused step kernel, one launch for every step
        ys = [Y]
    else:
        ys, (c, n, m) = slstm_recurrence(
            device, WxT, w.r, steps, consts, step_hook=step_hook, y_fp32=precision.y_fp32, init=init
        )
    free(WxT, c, n, m)
    return slstm_finish(device, x, ys, w, head_mean, consts, debug=debug, time_major=time_major)


# ---------------------------------------------------------------------------------------- variate mixer


def variate_block(
    device,
    x,
    w: TtVariate,
    precision: Precision,
    vmask,
    keep,
    debug: dict | None = None,
    tm_in: bool = False,
    tm_out: bool = False,
):
    """Attention across rows (variates) at every token, inside 32-row blocks that hold whole groups.
    ``tm_in`` / ``tm_out``: ``x`` / the result are time-major ``[T, 1, B, 512]`` (else ``[B, 1, T, 512]``).

    ``vmask [1, nb * 4, 32, 32]`` (0 / -1e9: same group, covariate queries never read target keys),
    ``keep [1, nb, 32, 1]`` (1 for rows of groups with more than one row; others pass through unchanged).
    """
    import ttnn

    if tm_in:
        T, _, B, Dm = tuple(x.shape)
        xT = x
    else:
        B, _, T, Dm = tuple(x.shape)
        xT = ttnn.transpose(x, 0, 2)  # [T, 1, B, 512]
    nb = B // 32
    xb = ttnn.reshape(xT, (T * nb, 1, 32, Dm))
    xn = rms_norm(xb, df.NORM_EPS)
    qkv = linear(xn, w.w_qkv, precision, role="variate")
    free(xn)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=4, num_kv_heads=4, transpose_k_heads=False)
    free(qkv)
    s = bmm(device, q, k, transpose_b=True)  # [T * nb, 4, 32, 32]
    free(q, k)
    sm = ttnn.add(ttnn.reshape(s, (T, nb * 4, 32, 32)), vmask)
    free(s)
    p = ttnn.softmax(sm, dim=-1)
    free(sm)
    a = bmm(device, ttnn.reshape(p, (T * nb, 4, 32, 32)), v, dtype=ttnn.bfloat16)
    free(p, v)
    am = ttnn.experimental.nlp_concat_heads(a)  # [T * nb, 1, 32, 512]
    free(a)
    o = linear(am, w.w_o, precision, role="variate")
    free(am)
    y = ttnn.add(xb, o)
    if debug is not None:
        _keep(debug, "attn", ttnn.reshape(o, (T, 1, B, Dm)))
    free(o)
    yn = rms_norm(y, df.NORM_EPS)
    hdn = linear(yn, w.w_wi, precision, activation="silu", role="variate")
    free(yn)
    f = linear(hdn, w.w_wo, precision, role="variate")
    free(hdn)
    y2 = ttnn.add(y, f)
    free(y, f)
    delta = ttnn.subtract(y2, xb)
    free(y2)
    kept = ttnn.multiply(ttnn.reshape(delta, (T, nb, 32, Dm)), keep)
    free(delta)
    out_t = ttnn.add(ttnn.reshape(kept, (T, 1, B, Dm)), xT)
    free(kept)
    if not tm_in:
        free(xT)
    if tm_out:
        return out_t
    out = ttnn.transpose(out_t, 0, 2)
    free(out_t)
    return out
