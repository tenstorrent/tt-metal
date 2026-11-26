# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN linear-branch operations on device: frame statistics, the vdn_solve delta rule, the bidirectional scan
over frames and the boundary-state gather.

Reference: OpenVDN/vdn-minimax-h3 @e262cb57, `src/models/linear_attention/{scan,delta_rule}.py` (mirrored in
`reference_openvdn.py`). Correct first, not fast.

Precision. The reference keeps `A`, the inverse and the scan in fp32. An fp32 multiply through the FPU keeps only
~11 significand bits (see `MiniMaxH3.md`, "Audio decode precision"), which is not enough to invert `I + A` with a
condition number up to ~1000. Every fp32 product here is therefore split: an fp32 operand becomes `hi = bf16(x)`
plus bf16 residuals, and the partial products are summed in fp32 (`split_matmul`). Measured on Blackhole, a
bf16 x bf16 -> fp32 matmul is itself accurate only to ~3e-4 relative to |a||b| (not exact), which is the floor of
everything here; see `spd_inverse`.

Inverse. ttnn has no Cholesky, triangular solve or inverse. `I + A` is SPD with every eigenvalue in
`[1, 1 + trace(A)]`, so a Newton-Schulz iteration from `X0 = c I` converges for any such matrix and needs only
batched matmuls (`spd_inverse`).
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence

import torch
from loguru import logger

import ttnn

from ....utils import tensor

_last_mark: list[float] = []


def mark(device: ttnn.MeshDevice, name: str) -> None:
    """With OPENVDN_SYNC_LOG=1: wait for the device and log the section's wall time since the previous mark (to place
    a hang or time a section). With OPENVDN_PROFILE_FLUSH=1 also read the device profiler here, so no section
    overflows its op buffer. Off by default: no sync."""
    if os.environ.get("OPENVDN_SYNC_LOG") != "1":
        return
    ttnn.synchronize_device(device)
    now = time.perf_counter()
    elapsed = (now - _last_mark[0]) * 1e3 if _last_mark else 0.0
    _last_mark[:] = [now]
    logger.info(f"hybrid attention: {name} done ({elapsed:.1f} ms)")
    if os.environ.get("OPENVDN_PROFILE_FLUSH") == "1":
        ttnn.ReadDeviceProfiler(device)
        _last_mark[:] = [time.perf_counter()]


def compute_config(device: ttnn.MeshDevice) -> ttnn.DeviceComputeKernelConfig:
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def split_bf16(x: ttnn.Tensor, parts: int) -> list[ttnn.Tensor]:
    """`x ~= sum(parts)`, each part bf16: `hi = bf16(x)`, then bf16 of each remaining residual. A bf16 input is
    returned as is."""
    if x.dtype == ttnn.bfloat16:
        return [x]
    out = []
    residual = x
    for i in range(parts):
        part = ttnn.typecast(residual, ttnn.bfloat16)
        out.append(part)
        if i < parts - 1:
            residual = ttnn.subtract(residual, ttnn.typecast(part, ttnn.float32))
    return out


def split_matmul(
    a: ttnn.Tensor | list[ttnn.Tensor],
    b: ttnn.Tensor | list[ttnn.Tensor],
    *,
    parts: int = 2,
    config: ttnn.DeviceComputeKernelConfig,
) -> ttnn.Tensor:
    """fp32-grade `a @ b`: both operands split into bf16 parts, cross terms below the leading residual dropped."""
    a_parts = a if isinstance(a, list) else split_bf16(a, parts)
    b_parts = b if isinstance(b, list) else split_bf16(b, parts)
    keep = max(len(a_parts), len(b_parts))
    out = None
    for i, ai in enumerate(a_parts):
        for j, bj in enumerate(b_parts):
            if i + j >= keep:
                continue
            term = ttnn.matmul(ai, bj, dtype=ttnn.float32, compute_kernel_config=config)
            out = term if out is None else ttnn.add(out, term)
    return out


def identity(device: ttnn.MeshDevice, n: int) -> ttnn.Tensor:
    return tensor.from_torch(torch.eye(n).reshape(1, 1, n, n), device=device, dtype=ttnn.float32)


def _transpose(x: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.transpose(x, -2, -1)


def rows_to_heads(rows: ttnn.Tensor, num_frames: int, num_heads: int, head_dim: int) -> ttnn.Tensor:
    """`[1, 1, F, H*d]` -> `[F, H, 1, d]`: the per-frame, per-key-channel vectors (alpha, bridges) as row
    broadcasts over a `[F, H, d_v, d_k]` bank."""
    rows = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
    rows = ttnn.reshape(rows, [num_frames, num_heads, 1, head_dim])
    return ttnn.to_layout(rows, ttnn.TILE_LAYOUT)


def frame_statistics(
    key: ttnn.Tensor, value: ttnn.Tensor, beta: ttnn.Tensor, *, config: ttnn.DeviceComputeKernelConfig
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Per frame and head: `A = K^T diag(beta) K` (fp32, symmetrised) and `B = V^T diag(beta) K` (bf16 operands,
    fp32 result), as in the reference's `_frame_statistics_pass` with `a_fp32=True`.

    key, value: `[F, H, S, d]` bf16. beta: `[F, H, S, 1]` bf16. Returns `[F, H, d, d]` fp32 twice.

    `k * beta` of two bf16 values has at most 16 significant bits, so its two bf16 parts hold it exactly and `A` is
    exact up to fp32 accumulation.
    """
    scaled = ttnn.multiply(ttnn.typecast(key, ttnn.float32), ttnn.typecast(beta, ttnn.float32))
    a = split_matmul([_transpose(p) for p in split_bf16(scaled, 2)], key, config=config)
    a = ttnn.multiply(ttnn.add(a, _transpose(a)), 0.5)

    value_beta = ttnn.multiply(value, beta)  # bf16, as the reference's `vf * beta`
    b = ttnn.matmul(_transpose(value_beta), key, dtype=ttnn.float32, compute_kernel_config=config)
    return a, b


def newton_schulz_iterations(max_eig: float, tol: float = 1e-7, extra: int = 3) -> int:
    """Iterations for `spd_inverse` to reach `tol` from the worst start, for eigenvalues in `[1, max_eig]`, plus
    `extra` to settle at the rounding floor."""
    r0 = 1.0 - 2.0 / (1.0 + max_eig)
    if r0 <= 0.0:
        return extra
    return max(1, math.ceil(math.log2(math.log(tol) / math.log(r0)))) + extra


def spd_inverse(
    m: ttnn.Tensor, *, max_eig: float, config: ttnn.DeviceComputeKernelConfig, iterations: int | None = None
) -> ttnn.Tensor:
    """Batched inverse of SPD `m` `[..., n, n]` fp32 whose eigenvalues lie in `[1, max_eig]` (here `I + A`, `A`
    PSD, `trace(A) <= max_eig - 1`).

    Newton-Schulz in residual form: `E = I - M X`, `X <- X + X E`, from `X0 = c I` with
    `c = 2 / (1 + lambda_max_bound)` per matrix, `lambda_max_bound = trace(M) - (n - 1)` (all other eigenvalues are
    at least 1). Then `||I - M X0|| < 1` and convergence is quadratic. The count comes from the worst case
    `max_eig`, so no host sync is needed.
    """
    n = m.shape[-1]
    device = m.device()
    eye = identity(device, n)
    iterations = iterations or newton_schulz_iterations(max_eig)

    trace = ttnn.sum(ttnn.sum(ttnn.multiply(m, eye), dim=-1, keepdim=True), dim=-2, keepdim=True)  # [..., 1, 1]
    lam_max = ttnn.subtract(trace, float(n - 1))
    scale = ttnn.multiply(ttnn.reciprocal(ttnn.add(lam_max, 1.0)), 2.0)
    x = ttnn.multiply(ttnn.add(ttnn.multiply(m, 0.0), eye), scale)

    # Floor: the residual `I - M X` is only as accurate as a device matmul, ~3e-4 relative to |M||X| even with exact
    # bf16 operands (measured; a three-part split gives identical results), so the forward error is about that
    # times the condition number: 2e-3 RMSE at cond ~430, 1.5e-2 at cond ~1000.
    m_parts = split_bf16(m, 2)
    for _ in range(iterations):
        x_parts = split_bf16(x, 2)
        e = ttnn.subtract(eye, split_matmul(m_parts, x_parts, config=config))
        x = ttnn.add(x, split_matmul(x_parts, e, config=config))

    return ttnn.multiply(ttnn.add(x, _transpose(x)), 0.5)


def vdn_factor_apply(
    alpha: ttnn.Tensor | None,
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    *,
    max_eig: float,
    config: ttnn.DeviceComputeKernelConfig,
) -> tuple[ttnn.Tensor | None, ttnn.Tensor]:
    """`VdnDelta.factor_apply`: `inv = (I + A)^-1`, `transition = Diag(alpha) inv`, `injection = B inv`.

    alpha: `[F, H, 1, d]` fp32 (see `rows_to_heads`), or None for the text chunk (transition not needed).
    a, b: `[F, H, d, d]` fp32. `inv` is symmetric, so `Diag(alpha) inv = (inv * alpha_row)^T`.
    """
    eye = identity(a.device(), a.shape[-1])
    inv = spd_inverse(ttnn.add(a, eye), max_eig=max_eig, config=config)
    injection = split_matmul(b, inv, config=config)
    transition = None if alpha is None else _transpose(ttnn.multiply(inv, alpha))
    return transition, injection


def bidirectional_scan(
    transition: ttnn.Tensor,
    injection: ttnn.Tensor,
    *,
    text_state: ttnn.Tensor | None = None,
    config: ttnn.DeviceComputeKernelConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """`_run_scans`: `state <- state @ transition[f] + injection[f]` forward over frames and in reverse, from
    `text_state` `[1, H, d_v, d_k]` (or zero). Returns the prefix and suffix banks `[F, H, d_v, d_k]` fp32."""
    num_frames, heads, dv, dk = transition.shape
    t_parts = split_bf16(transition, 2)

    def frame(bank: ttnn.Tensor, f: int) -> ttnn.Tensor:
        return ttnn.slice(bank, [f, 0, 0, 0], [f + 1, heads, bank.shape[-2], bank.shape[-1]])

    def run(order: Sequence[int]) -> list[ttnn.Tensor]:
        state = text_state
        states = {}
        for f in order:
            inj = frame(injection, f)
            if state is None:
                state = inj
            else:
                step = split_matmul(state, [frame(p, f) for p in t_parts], config=config)
                state = ttnn.add(step, inj)
            states[f] = state
        return [states[f] for f in range(num_frames)]

    prefix = ttnn.concat(run(range(num_frames)), dim=0)
    suffix = ttnn.concat(run(range(num_frames - 1, -1, -1)), dim=0)
    return prefix, suffix


def parallel_bidirectional_scan(
    transition: ttnn.Tensor,
    injection: ttnn.Tensor,
    *,
    text_state: ttnn.Tensor | None = None,
    config: ttnn.DeviceComputeKernelConfig,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """`bidirectional_scan` in log2(F) batched levels instead of F sequential steps (Hillis-Steele). Each frame is
    the affine map `s -> s T_f + I_f`; composing `a` then `b` gives `(T_a T_b, I_a T_b + I_b)`, so after the levels
    frame f holds the composition of frames 0..f (forward) or F-1..f (reverse), applied once to the start state.
    Same products as the sequential scan in another association order (fp32-grade split matmuls throughout)."""
    num_frames = transition.shape[0]

    def rows(t: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
        return ttnn.slice(t, [start, 0, 0, 0], [end, t.shape[1], t.shape[2], t.shape[3]])

    def run(reverse: bool) -> ttnn.Tensor:
        t, i = transition, injection
        d = 1
        while d < num_frames:
            n = num_frames - d
            # forward: frame f <- (frame f-d) then (frame f); reverse: frame f <- (frame f+d) then (frame f)
            head, tail = (0, n), (d, num_frames)
            earlier, later = (tail, head) if reverse else (head, tail)
            earlier_t, earlier_i = rows(t, *earlier), rows(i, *earlier)
            later_t, later_i = rows(t, *later), rows(i, *later)
            later_parts = split_bf16(later_t, 2)
            new_t = split_matmul(earlier_t, later_parts, config=config)
            new_i = ttnn.add(split_matmul(earlier_i, later_parts, config=config), later_i)
            if reverse:
                t = ttnn.concat([new_t, rows(t, n, num_frames)], dim=0)
                i = ttnn.concat([new_i, rows(i, n, num_frames)], dim=0)
            else:
                t = ttnn.concat([rows(t, 0, d), new_t], dim=0)
                i = ttnn.concat([rows(i, 0, d), new_i], dim=0)
            d *= 2
        if text_state is None:
            return i
        start = ttnn.repeat(text_state, (num_frames, 1, 1, 1))
        return ttnn.add(split_matmul(start, t, config=config), i)

    return run(False), run(True)


def gather_indices(bounds: Sequence[tuple[int, int]], num_frames: int) -> dict[str, list]:
    """Host copy of the reference's `_gather_indices`."""
    last_before = [lo - 1 for lo, _ in bounds]
    first_after = [hi + 1 for _, hi in bounds]
    return dict(
        before_idx=[max(i, 0) for i in last_before],
        after_idx=[min(i, num_frames - 1) for i in first_after],
        has_before=[i >= 0 for i in last_before],
        has_after=[i < num_frames for i in first_after],
        bridge_before=[max(i + 1, 0) for i in last_before],
        bridge_after=[min(i, num_frames) for i in first_after],
    )


def bridge_matrices(bounds: Sequence[tuple[int, int]], num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """0/1 matrices `[F, F]` whose product with `log(alpha)` gives the reference's bridge exponents:
    before: `sum_{u = bridge_before[t]}^{t}`, after: `sum_{u = t}^{bridge_after[t] - 1}` (the reference's differences
    of the exclusive log-prefix sum)."""
    idx = gather_indices(bounds, num_frames)
    before = torch.zeros(num_frames, num_frames)
    after = torch.zeros(num_frames, num_frames)
    for t in range(num_frames):
        before[t, idx["bridge_before"][t] : t + 1] = 1.0
        after[t, t : idx["bridge_after"][t]] = 1.0
    return before, after


def gather_linear_state(
    prefix: ttnn.Tensor,
    suffix: ttnn.Tensor,
    alpha_rows: ttnn.Tensor,
    bounds: Sequence[tuple[int, int]],
    *,
    text_state: ttnn.Tensor | None = None,
    config: ttnn.DeviceComputeKernelConfig,
) -> ttnn.Tensor:
    """`gather_linear_state` with `bridge="alpha"`: per query frame, the states just outside its window, decayed
    in to it. prefix, suffix: `[F, H, d_v, d_k]` fp32. alpha_rows: `[1, 1, F, H*d]` fp32. Returns
    `[F, H, d_v, d_k]` fp32.

    The window bounds are static per shape, so the gather is slices, and the bridge exponents are a 0/1 matrix
    times `log(alpha)` (split three ways, so the sums are fp32-exact).
    """
    num_frames, heads, dv, dk = prefix.shape
    device = prefix.device()
    idx = gather_indices(bounds, num_frames)
    zeros = None if text_state is not None else ttnn.zeros_like(ttnn.slice(prefix, [0, 0, 0, 0], [1, heads, dv, dk]))
    outside = text_state if text_state is not None else zeros

    def pick(bank: ttnn.Tensor, rows: list[int], valid: list[bool]) -> ttnn.Tensor:
        cache = {}
        picked = []
        for r, ok in zip(rows, valid):
            if not ok:
                picked.append(outside)
                continue
            if r not in cache:
                cache[r] = ttnn.slice(bank, [r, 0, 0, 0], [r + 1, heads, dv, dk])
            picked.append(cache[r])
        return ttnn.concat(picked, dim=0)

    before = pick(prefix, idx["before_idx"], idx["has_before"])
    after = pick(suffix, idx["after_idx"], idx["has_after"])

    log_alpha = ttnn.log(ttnn.clamp(alpha_rows, min=1e-12))
    log_parts = split_bf16(log_alpha, 3)
    sel_before, sel_after = bridge_matrices(bounds, num_frames)
    exps = []
    for sel in (sel_before, sel_after):
        sel_tt = tensor.from_torch(sel.reshape(1, 1, num_frames, num_frames), device=device, dtype=ttnn.bfloat16)
        exps.append(split_matmul(sel_tt, log_parts, config=config))
    bridge_before, bridge_after = (rows_to_heads(ttnn.exp(e), num_frames, heads, dk) for e in exps)

    return ttnn.add(ttnn.multiply(before, bridge_before), ttnn.multiply(after, bridge_after))
