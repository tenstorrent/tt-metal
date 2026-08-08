# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared bidirectional-LSTM scan primitive for the Kokoro TTNN stubs.

Kokoro has three cell-by-cell unrolled bidirectional LSTMs (`l_s_t_m` on the
prosody frame axis, and the token-axis LSTMs inside `duration_encoder` and
`text_encoder`). They share one recurrence:

    gate = xᵀWih + hᵀWhh + b ;  i/f/o = sigmoid, g = tanh
    c = f·c + i·g ;  h = o·tanh(c)

with forward and (optionally) reverse directions concatenated on the feature
axis. This module is the single source of truth for that recurrence so the
trace-enabling masked fixed-capacity variant lives in ONE place.

Two paths, selected per call by `_trace_capacity(T)`:
  * DYNAMIC (default, capacity 0): scan exactly the true length T — byte-identical
    to the original per-stub loops, so every PCC gate is unchanged.
  * MASKED FIXED-CAPACITY (opt-in via env): scan a fixed capacity C >= T and gate
    each h/c update with a per-timestep validity mask m_t = (t < T_valid); padded
    frames are a state no-op, so a reverse pass keeps the initial zero state until
    the true last frame T_valid-1 and reproduces the dynamic result exactly — but
    with a capacity-fixed op sequence that CAN be trace-captured.

Env knobs (all default OFF -> dynamic path):
  KOKORO_LSTM_TRACE_CAP=<int>   pad every scan to this absolute capacity
  KOKORO_LSTM_TRACE_PAD=<int>   pad to T + this many frames (test: forces the
                                reverse pass to traverse padded tail frames first)
  KOKORO_LSTM_TRACE_DEVMASK=1   build the mask as a resident DEVICE tensor
                                (ttnn.lt) instead of a host scalar — the trace-safe
                                form with no per-step host branch
  KOKORO_LSTM_TRACE_BIASPAD=1   fill padded gates with the linear bias (emulates
                                padding x BEFORE the projection: the realistic
                                bias-leak case that pollutes an UNMASKED bi-LSTM)
  KOKORO_LSTM_TRACE_NOMASK=1    debug: disable the mask to demonstrate the collapse
"""

from __future__ import annotations

import os

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def trace_capacity(T: int) -> int:
    """Fixed trace capacity for the unrolled scan, or 0 to keep the dynamic path."""
    cap = int(os.environ.get("KOKORO_LSTM_TRACE_CAP", "0") or "0")
    pad = int(os.environ.get("KOKORO_LSTM_TRACE_PAD", "0") or "0")
    if pad > 0:
        return T + pad
    if cap > T:
        return cap
    return 0


# --------------------------------------------------------------------------- #
# Trace-capture mode
#
# The env paths above are for VALIDATION: they pad internally and slice the
# output back to T_valid, so the result length equals the dynamic length (handy
# for PCC-vs-dynamic checks) but the shape is still T-dependent. Actual trace
# capture needs a FIXED shape and a host-op-free captured region, so:
#   * the caller pre-pads the stage inputs to capacity C (the LSTM input arrives
#     already length C) and the scan returns a full length-C output (no re-slice);
#   * the validity mask is built ONCE, OUTSIDE the trace, into resident device
#     tensors (build_trace_ctx) and only sliced (pure ttnn) inside the loop — so
#     the captured region contains no from_torch / host branch.
# A single active TraceCtx (set via push/pop) is read by run_bilstm.
# --------------------------------------------------------------------------- #
class TraceCtx:
    """Resident, capacity-C trace state shared by every LSTM in one captured stage."""

    def __init__(self, C, T_valid, mask_full, omm_full):
        self.C = int(C)
        self.T_valid = int(T_valid)
        self.mask_full = mask_full  # [1, C, 1] device: 1.0 where t < T_valid
        self.omm_full = omm_full  # [1, C, 1] device: complement


_TRACE_CTX = None


def build_trace_ctx(device, C, T_valid):
    """Build the resident capacity-C validity mask OUTSIDE the trace (host work here
    is fine: this runs in *_trace_setup, not inside begin/end_trace_capture)."""
    import torch

    iota = ttnn.from_torch(
        torch.arange(C, dtype=torch.float32).reshape(1, C, 1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_DRAM,
    )
    tvv = ttnn.from_torch(
        torch.full((1, C, 1), float(T_valid), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_DRAM,
    )
    mask_full = ttnn.lt(iota, tvv)
    ones = ttnn.ones((1, C, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    return TraceCtx(C, T_valid, mask_full, ttnn.subtract(ones, mask_full))


def push_trace_ctx(ctx):
    global _TRACE_CTX
    _TRACE_CTX = ctx


def pop_trace_ctx():
    global _TRACE_CTX
    _TRACE_CTX = None


def _cell(cc, H, xp_t, h, c, whh_t):
    gate = ttnn.add(xp_t, ttnn.matmul(h, whh_t, compute_kernel_config=cc))
    i = ttnn.sigmoid(ttnn.slice(gate, [0, 0, 0], [1, 1, H]))
    f = ttnn.sigmoid(ttnn.slice(gate, [0, 0, H], [1, 1, 2 * H]))
    g = ttnn.tanh(ttnn.slice(gate, [0, 0, 2 * H], [1, 1, 3 * H]))
    o = ttnn.sigmoid(ttnn.slice(gate, [0, 0, 3 * H], [1, 1, 4 * H]))
    c_new = ttnn.add(ttnn.multiply(f, c), ttnn.multiply(i, g))
    h_new = ttnn.multiply(o, ttnn.tanh(c_new))
    return h_new, c_new


def _device_mask(device, C, T_valid):
    """Resident per-timestep validity mask as DEVICE tensors (trace-safe): builds
    mask[1,C,1] = (iota < T_valid) and its complement via pure ttnn.lt, so the
    per-step gate has NO host control flow. The trace version keeps `iota` resident
    (built once) and writes T_valid into a resident buffer OUTSIDE the trace, leaving
    this ttnn.lt as the only per-capture op. Returns (mask_full, one_minus_full)."""
    import torch  # local: keep the stub free of a global torch import

    iota = ttnn.from_torch(
        torch.arange(C, dtype=torch.float32).reshape(1, C, 1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_DRAM,
    )
    tvv = ttnn.from_torch(
        torch.full((1, C, 1), float(T_valid), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_DRAM,
    )
    mask_full = ttnn.lt(iota, tvv)  # 1.0 where t < T_valid else 0.0
    ones = ttnn.ones((1, C, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    return mask_full, ttnn.subtract(ones, mask_full)


def _run_dir(device, cc, H, xproj, whh_t, T, reverse):
    h = ttnn.zeros((1, 1, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    c = ttnn.zeros((1, 1, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    order = range(T - 1, -1, -1) if reverse else range(T)
    outs = {}
    for t in order:
        xp_t = ttnn.slice(xproj, [0, t, 0], [1, t + 1, 4 * H])
        h, c = _cell(cc, H, xp_t, h, c, whh_t)
        outs[t] = h
    return ttnn.concat([outs[t] for t in range(T)], dim=1, memory_config=_DRAM)


def _run_dir_masked(device, cc, H, xproj, whh_t, C, T_valid, reverse):
    h = ttnn.zeros((1, 1, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    c = ttnn.zeros((1, 1, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    order = range(C - 1, -1, -1) if reverse else range(C)
    devmask = os.environ.get("KOKORO_LSTM_TRACE_DEVMASK") == "1"
    nomask = os.environ.get("KOKORO_LSTM_TRACE_NOMASK") == "1"
    mask_full = omm_full = None
    if devmask:
        mask_full, omm_full = _device_mask(device, C, T_valid)
    outs = {}
    for t in order:
        xp_t = ttnn.slice(xproj, [0, t, 0], [1, t + 1, 4 * H])
        h_cand, c_cand = _cell(cc, H, xp_t, h, c, whh_t)
        if devmask:
            m = ttnn.slice(mask_full, [0, t, 0], [1, t + 1, 1])  # [1,1,1] device scalar
            om = ttnn.slice(omm_full, [0, t, 0], [1, t + 1, 1])
            c = ttnn.add(ttnn.multiply(c_cand, m), ttnn.multiply(c, om))
            h = ttnn.add(ttnn.multiply(h_cand, m), ttnn.multiply(h, om))
        else:
            mh = 1.0 if (nomask or t < T_valid) else 0.0  # host scalar (validation / debug)
            c = ttnn.add(ttnn.multiply(c_cand, mh), ttnn.multiply(c, 1.0 - mh))
            h = ttnn.add(ttnn.multiply(h_cand, mh), ttnn.multiply(h, 1.0 - mh))
        if t < T_valid:
            outs[t] = h
    return ttnn.concat([outs[t] for t in range(T_valid)], dim=1, memory_config=_DRAM)


def _pad_seq(device, H, xproj, T, C, bias=None):
    if bias is not None and os.environ.get("KOKORO_LSTM_TRACE_BIASPAD") == "1":
        ones = ttnn.ones((1, C - T, 4 * H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        pad = ttnn.multiply(ones, bias)
    else:
        pad = ttnn.zeros((1, C - T, 4 * H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.concat([xproj, pad], dim=1, memory_config=_DRAM)


def _scan(device, cc, H, xproj, whh_t, T, C, reverse, bias):
    if C:
        xproj = _pad_seq(device, H, xproj, T, C, bias)
        return _run_dir_masked(device, cc, H, xproj, whh_t, C, T, reverse)
    return _run_dir(device, cc, H, xproj, whh_t, T, reverse)


_ZERO_STATE = {}


def _zero_state(device, H):
    """Resident zero initial-state [1,1,H], created ONCE and cached. Tensor creation is a host
    write (illegal inside a trace), so the trace scan must reuse a pre-existing buffer; the eager
    warm-up call that always precedes trace capture populates this cache outside the trace. The
    state is only read as the t=0 seed and never mutated in place, so sharing one buffer is safe."""
    key = (id(device), int(H))
    z = _ZERO_STATE.get(key)
    if z is None:
        z = ttnn.zeros((1, 1, H), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        _ZERO_STATE[key] = z
    return z


def active_trace_ctx():
    """The TraceCtx pushed for the current captured stage, or None (dynamic path)."""
    return _TRACE_CTX


_FRAME_MASK = {}


def _frame_mask(device, L, valid_L):
    """Resident [1,1,L] frame-validity mask (1.0 for the first valid_L frames), built ONCE and
    cached per (device,L,valid_L). Mirrors _zero_state: the warm-up forward that precedes trace
    capture creates it OUTSIDE the trace, and the captured forward reuses the resident buffer, so no
    from_torch fires inside the capture. Read-only, so sharing one buffer across call sites is safe."""
    key = (id(device), int(L), int(valid_L))
    m = _FRAME_MASK.get(key)
    if m is None:
        import torch

        t = torch.zeros(1, 1, int(L), dtype=torch.float32)
        t[0, 0, : int(valid_L)] = 1.0
        m = ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_DRAM)
        _FRAME_MASK[key] = m
    return m


def reset_frame_masks():
    _FRAME_MASK.clear()


def zero_pad_frames(device, x):
    """Zero the padded frames (>= valid_L) of x[B,C,L] when a fixed-capacity TraceCtx is active, so
    downstream convs see the SAME zero boundary the dynamic (variable-length) path sees. Convs re-add
    their bias into the padded tail every layer, and it leaks into the valid region through the conv
    receptive field — re-masking at each block output keeps that leak from accumulating. No-op (returns
    x unchanged) on the dynamic path."""
    ctx = _TRACE_CTX
    if ctx is None:
        return x
    L = int(x.shape[-1])
    valid_L = max(1, min(L, round(ctx.T_valid * L / ctx.C)))
    return ttnn.multiply(x, _frame_mask(device, L, valid_L))


def masked_moments(device, x):
    """Frame-axis instance-norm mean/var over the VALID frames only, when a fixed-capacity TraceCtx
    is active. x is [B, C, L] (L = frame axis at this layer's resolution); the valid length scales
    with the resolution: valid_L = round(T_valid * L / C). Returns (mean, var) [B,C,1] or None (no
    active ctx -> caller keeps its dynamic full-axis reduction)."""
    ctx = _TRACE_CTX
    if ctx is None:
        return None
    L = int(x.shape[-1])
    valid_L = max(1, min(L, round(ctx.T_valid * L / ctx.C)))
    mask = _frame_mask(device, L, valid_L)  # [1,1,L]
    inv = 1.0 / float(valid_L)
    xm = ttnn.multiply(x, mask)
    mean = ttnn.multiply(ttnn.sum(xm, dim=2, keepdim=True), inv)  # [B,C,1]
    mean_x2 = ttnn.multiply(ttnn.sum(ttnn.multiply(xm, x), dim=2, keepdim=True), inv)  # sum(mask*x^2)/valid
    var = ttnn.subtract(mean_x2, ttnn.multiply(mean, mean))
    return mean, var


def _run_dir_trace(device, cc, H, xproj_C, whh_t, ctx, reverse):
    """Fixed-capacity trace scan: xproj_C is ALREADY length C; gate with the resident
    ctx mask and return a full length-C output. No host ops (no tensor creation) in this region."""
    C = ctx.C
    h = _zero_state(device, H)
    c = _zero_state(device, H)
    order = range(C - 1, -1, -1) if reverse else range(C)
    outs = {}
    for t in order:
        xp_t = ttnn.slice(xproj_C, [0, t, 0], [1, t + 1, 4 * H])
        h_cand, c_cand = _cell(cc, H, xp_t, h, c, whh_t)
        m = ttnn.slice(ctx.mask_full, [0, t, 0], [1, t + 1, 1])
        om = ttnn.slice(ctx.omm_full, [0, t, 0], [1, t + 1, 1])
        c = ttnn.add(ttnn.multiply(c_cand, m), ttnn.multiply(c, om))
        h = ttnn.add(ttnn.multiply(h_cand, m), ttnn.multiply(h, om))
        outs[t] = h
    return ttnn.concat([outs[t] for t in range(C)], dim=1, memory_config=_DRAM)


def run_bilstm(device, cc, x, fwd, rev, H):
    """Run one (bi)directional LSTM over x[1,T,in] and return [1,T,(2)H].

    fwd/rev are per-direction dicts {"wih_t":[in,4H], "whh_t":[H,4H], "bias":[1,1,4H]};
    pass rev=None for a unidirectional LSTM.

    Path selection:
      * a TraceCtx is active (push_trace_ctx) -> TRACE mode: x is already length C,
        gate with the resident mask, return a full length-C output (fixed shape,
        host-op-free) so the call can be trace-captured;
      * else trace_capacity(T) picks the env VALIDATION masked path or the DYNAMIC path."""
    ctx = _TRACE_CTX
    if ctx is not None:
        xproj_f = ttnn.linear(x, fwd["wih_t"], bias=fwd["bias"], compute_kernel_config=cc)
        hf = _run_dir_trace(device, cc, H, xproj_f, fwd["whh_t"], ctx, reverse=False)
        if rev is None:
            return hf
        xproj_r = ttnn.linear(x, rev["wih_t"], bias=rev["bias"], compute_kernel_config=cc)
        hr = _run_dir_trace(device, cc, H, xproj_r, rev["whh_t"], ctx, reverse=True)
        return ttnn.concat([hf, hr], dim=-1, memory_config=_DRAM)

    T = int(x.shape[1])
    C = trace_capacity(T)
    xproj_f = ttnn.linear(x, fwd["wih_t"], bias=fwd["bias"], compute_kernel_config=cc)
    hf = _scan(device, cc, H, xproj_f, fwd["whh_t"], T, C, reverse=False, bias=fwd["bias"])
    if rev is None:
        return hf
    xproj_r = ttnn.linear(x, rev["wih_t"], bias=rev["bias"], compute_kernel_config=cc)
    hr = _scan(device, cc, H, xproj_r, rev["whh_t"], T, C, reverse=True, bias=rev["bias"])
    return ttnn.concat([hf, hr], dim=-1, memory_config=_DRAM)
