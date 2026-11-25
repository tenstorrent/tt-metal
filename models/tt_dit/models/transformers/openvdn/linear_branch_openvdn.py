# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN linear-attention branch on device: everything the softmax window cannot see, summarised per video token.

Reference: OpenVDN/vdn-minimax-h3 @e262cb57, `BidirectionalLinearBranch` (mirrored in `reference_hybrid_openvdn.py`),
in the released checkpoints' configuration: short conv on K and V, `vdn_solve`, `bridge="alpha"`, text state.

    features   K, V: depthwise 5x5 spatial conv per frame, then 5-tap temporal conv; Q, K, V: SiLU; Q, K: L2 norm
    statistics A = K^T diag(beta) K, B = V^T diag(beta) K per frame and head      (`linear_ops_openvdn`)
    decay      alpha = exp(-exp(A_log) * softplus(up(down(frame mean)) + dt_bias)), fp32
    text state half the prompt written into a zero state as one chunk, the start of both scans
    scans      forward and reverse over frames, then the boundary gather            (`linear_ops_openvdn`)
    readout    q . state, RMSNorm over head_dim, times a low-rank sigmoid gate

Single device for now (no tensor or sequence parallel). Correct first, not fast: the spatial conv is 25 exact shift
matmuls, and the layout changes go through row-major reshapes and permutes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import ttnn

from ....layers.linear import Linear
from ....layers.module import Module, Parameter
from ....utils import tensor
from . import linear_ops_openvdn as ops

Range = tuple[int, int]


def to_heads(x: ttnn.Tensor, num_frames: int, num_heads: int) -> ttnn.Tensor:
    """`[1, 1, F*S, H*d]` -> `[F, H, S, d]`."""
    rows, channels = x.shape[-2], x.shape[-1]
    per_frame, head_dim = rows // num_frames, channels // num_heads
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, [num_frames, per_frame, num_heads, head_dim])
    x = ttnn.permute(x, (0, 2, 1, 3))
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def from_heads(x: ttnn.Tensor) -> ttnn.Tensor:
    """`[F, H, S, d]` -> `[1, 1, F*S, H*d]`."""
    num_frames, num_heads, per_frame, head_dim = x.shape
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.permute(x, (0, 2, 1, 3))
    x = ttnn.reshape(x, [1, 1, num_frames * per_frame, num_heads * head_dim])
    return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


def heads_column(x: ttnn.Tensor, num_frames: int) -> ttnn.Tensor:
    """Per-token, per-head scalars `[1, 1, F*S, H]` -> `[F, H, S, 1]` (beta as `frame_statistics` takes it)."""
    rows, num_heads = x.shape[-2], x.shape[-1]
    per_frame = rows // num_frames
    x = ttnn.transpose(x, -2, -1)  # [1, 1, H, F*S]
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, [num_heads, num_frames, 1, per_frame])
    x = ttnn.permute(x, (1, 0, 2, 3))
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    return ttnn.transpose(x, -2, -1)


def rows_slice(x: ttnn.Tensor, start: int, end: int) -> ttnn.Tensor:
    """Rows `[start, end)` of a `[1, 1, N, C]` tensor."""
    return ttnn.slice(x, [0, 0, start, 0], [1, 1, end, x.shape[-1]])


def l2_normalize(x: ttnn.Tensor, eps: float = 1e-6) -> ttnn.Tensor:
    """`F.normalize(x, dim=-1, eps)`: `x / max(||x||, eps)`."""
    norm = ttnn.sqrt(ttnn.sum(ttnn.multiply(x, x), dim=-1, keepdim=True))
    return ttnn.multiply(x, ttnn.reciprocal(ttnn.clamp(norm, min=eps)))


def softplus(x: ttnn.Tensor) -> ttnn.Tensor:
    """`F.softplus` (beta 1), written stably: `relu(x) + log1p(exp(-|x|))`."""
    return ttnn.add(ttnn.relu(x), ttnn.log1p(ttnn.exp(ttnn.neg(ttnn.abs(x)))))


def spatial_shift_matrices(grid_h: int, grid_w: int, kernel: int) -> torch.Tensor:
    """`[k*k, S, S]` 0/1 matrices, tap `(i, j)` at index `i*k + j`: `(x^T @ P)[c, s'] = x[s + (i-p, j-p), c]` with zero
    padding, so that `sum_t w_t (x^T @ P_t)` is the reference's depthwise `conv2d` (cross-correlation, padding p)."""
    pad = kernel // 2
    size = grid_h * grid_w
    mats = torch.zeros(kernel * kernel, size, size)
    for i in range(kernel):
        for j in range(kernel):
            for y in range(grid_h):
                for x in range(grid_w):
                    ys, xs = y + i - pad, x + j - pad
                    if 0 <= ys < grid_h and 0 <= xs < grid_w:
                        mats[i * kernel + j, ys * grid_w + xs, y * grid_w + x] = 1.0
    return mats


class OpenVDNLinearBranch(Module):
    TEXT_STATE_SCALE = 0.5
    KERNEL = 5
    NORM_EPS = 1e-6

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mesh_device: ttnn.MeshDevice,
        short_conv: Sequence[str] = ("k", "v"),
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.mesh_device = mesh_device
        self.short_conv = tuple(short_conv)
        self.config = ops.compute_config(mesh_device)
        taps = self.KERNEL * self.KERNEL
        channels = self.inner_dim

        def fp32(shape):
            return Parameter(total_shape=shape, device=mesh_device, dtype=ttnn.float32)

        # Spatial taps as columns [C, 1] (they scale the transposed [C, S] frames), temporal taps as rows [1, C].
        for proj in self.short_conv:
            setattr(self, f"{proj}_sp", fp32([taps, 1, channels, 1]))
            setattr(self, f"{proj}_tm", fp32([self.KERNEL, 1, 1, channels]))

        linear = dict(mesh_device=mesh_device, compute_kernel_config=self.config)
        self.beta_proj = Linear(hidden_size, num_heads, bias=False, **linear)
        # alpha's two projections run fp32-grade (split operands), on bf16 weights.
        self.alpha_down = Parameter(total_shape=[hidden_size, head_dim], device=mesh_device)
        self.alpha_up = Parameter(total_shape=[head_dim, channels], device=mesh_device)
        self.alpha_dt_bias = fp32([1, 1, 1, channels])
        self.alpha_scale = fp32([1, 1, 1, channels])  # exp(A_log), repeated over each head's channels
        self.gate_down = Linear(hidden_size, head_dim, bias=False, **linear)
        self.gate_up = Linear(head_dim, channels, bias=True, **linear)
        self.norm_weight = fp32([1, 1, 1, head_dim])

        self._shifts: dict[tuple[int, int], list[ttnn.Tensor]] = {}

    # ------------------------------------------------------------------ weights

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        taps = self.KERNEL * self.KERNEL
        for proj in self.short_conv:
            sp = state.pop(f"short_conv.{proj}_sp.weight", None)  # [C, 1, k, k]
            if sp is not None:
                state[f"{proj}_sp"] = sp.reshape(-1, taps).T.reshape(taps, 1, -1, 1)
            tm = state.pop(f"short_conv.{proj}_tm.weight", None)  # [C, 1, k]
            if tm is not None:
                state[f"{proj}_tm"] = tm.reshape(-1, self.KERNEL).T.reshape(self.KERNEL, 1, 1, -1)

        for name, key in (("alpha_down", "alpha.down.weight"), ("alpha_up", "alpha.up.weight")):
            if key in state:
                state[name] = state.pop(key).T
        if "alpha.dt_bias" in state:
            state["alpha_dt_bias"] = state.pop("alpha.dt_bias").float().reshape(1, 1, 1, -1)
        if "alpha.A_log" in state:
            scale = torch.exp(state.pop("alpha.A_log").float())
            state["alpha_scale"] = scale.repeat_interleave(self.head_dim).reshape(1, 1, 1, -1)
        for src, dst in (("output_gate.down", "gate_down"), ("output_gate.up", "gate_up")):
            for k in [k for k in state if k.startswith(src + ".")]:
                state[dst + k.removeprefix(src)] = state.pop(k)
        if "norm.weight" in state:
            state["norm_weight"] = state.pop("norm.weight").float().reshape(1, 1, 1, -1)

    # ------------------------------------------------------------------ pieces

    def _shift_matrices(self, frame_size: tuple[int, int]) -> list[ttnn.Tensor]:
        if frame_size not in self._shifts:
            mats = spatial_shift_matrices(*frame_size, self.KERNEL)
            self._shifts[frame_size] = [
                tensor.from_torch(m.reshape(1, 1, *m.shape), device=self.mesh_device, dtype=ttnn.bfloat16) for m in mats
            ]
        return self._shifts[frame_size]

    def conv(self, tokens: ttnn.Tensor, proj: str, num_frames: int, frame_size: tuple[int, int]) -> ttnn.Tensor:
        """`LinearAttentionSepConv`: `[1, 1, F*S, C]` bf16 -> fp32. The spatial taps are exact 0/1 shift matmuls
        (one nonzero per output), scaled per channel and summed in fp32; the temporal taps are frame slices."""
        channels = tokens.shape[-1]
        per_frame = frame_size[0] * frame_size[1]
        x = ttnn.to_layout(tokens, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.to_layout(ttnn.reshape(x, [1, num_frames, per_frame, channels]), ttnn.TILE_LAYOUT)
        x_t = ttnn.transpose(x, -2, -1)  # [1, F, C, S]

        weight = getattr(self, f"{proj}_sp").data
        acc = None
        for t, shift in enumerate(self._shift_matrices(frame_size)):
            moved = ttnn.matmul(x_t, shift, dtype=ttnn.float32, compute_kernel_config=self.config)
            tap = ttnn.slice(weight, [t, 0, 0, 0], [t + 1, 1, channels, 1])
            moved = ttnn.multiply(moved, tap)
            acc = moved if acc is None else ttnn.add(acc, moved)
        x = ttnn.transpose(acc, -2, -1)  # [1, F, S, C] fp32

        pad = self.KERNEL // 2
        zeros = ttnn.zeros_like(ttnn.slice(x, [0, 0, 0, 0], [1, 1, per_frame, channels]))
        padded = ttnn.concat([zeros] * pad + [x] + [zeros] * pad, dim=1)
        weight = getattr(self, f"{proj}_tm").data
        acc = None
        for dt in range(self.KERNEL):
            part = ttnn.slice(padded, [0, dt, 0, 0], [1, dt + num_frames, per_frame, channels])
            part = ttnn.multiply(part, ttnn.slice(weight, [dt, 0, 0, 0], [dt + 1, 1, 1, channels]))
            acc = part if acc is None else ttnn.add(acc, part)
        acc = ttnn.to_layout(acc, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(ttnn.reshape(acc, [1, 1, num_frames * per_frame, channels]), ttnn.TILE_LAYOUT)

    def features(
        self, tokens: ttnn.Tensor, proj: str, num_frames: int, frame_size: tuple[int, int] | None
    ) -> ttnn.Tensor:
        """`prepare_linear_features`: `[1, 1, F*S, C]` raw bf16 -> `[F, H, S, d]` bf16. `frame_size=None` skips the
        conv (the text chunk)."""
        if frame_size is not None and proj in self.short_conv:
            x = self.conv(tokens, proj, num_frames, frame_size)
        else:
            x = ttnn.typecast(tokens, ttnn.float32)
        x = to_heads(ttnn.silu(x), num_frames, self.num_heads)
        if proj != "v":
            x = l2_normalize(x)
        return ttnn.typecast(x, ttnn.bfloat16)

    def beta(self, x: ttnn.Tensor, num_frames: int) -> ttnn.Tensor:
        """`sigmoid(beta_proj(x))` as `[F, H, S, 1]` bf16."""
        return heads_column(ttnn.sigmoid(self.beta_proj(x)), num_frames)

    def alpha(self, xv: ttnn.Tensor, num_frames: int) -> ttnn.Tensor:
        """`FrameKDAAlpha` of the per-frame mean of `xv` `[1, 1, F*S, hidden]`: `[1, 1, F, H*d]` fp32."""
        hidden = xv.shape[-1]
        per_frame = xv.shape[-2] // num_frames
        x = ttnn.to_layout(ttnn.typecast(xv, ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.to_layout(ttnn.reshape(x, [1, num_frames, per_frame, hidden]), ttnn.TILE_LAYOUT)
        mean = ttnn.multiply(ttnn.sum(x, dim=2, keepdim=True), 1.0 / per_frame)  # [1, F, 1, hidden]
        mean = ttnn.reshape(ttnn.to_layout(mean, ttnn.ROW_MAJOR_LAYOUT), [1, 1, num_frames, hidden])
        mean = ttnn.to_layout(mean, ttnn.TILE_LAYOUT)

        delta = ops.split_matmul(mean, [self.alpha_down.data], config=self.config)
        delta = ops.split_matmul(delta, [self.alpha_up.data], config=self.config)
        delta = ttnn.add(delta, self.alpha_dt_bias.data)
        return ttnn.exp(ttnn.neg(ttnn.multiply(softplus(delta), self.alpha_scale.data)))

    def text_state(self, text_x: ttnn.Tensor, text_k_raw: ttnn.Tensor, text_v_raw: ttnn.Tensor) -> ttnn.Tensor:
        """`_text_state`: `TEXT_STATE_SCALE * (B (I + A)^-1)` over all prompt rows as one chunk, `[1, H, d_v, d_k]`
        fp32."""
        length = text_x.shape[-2]
        key = self.features(text_k_raw, "k", 1, None)
        value = self.features(text_v_raw, "v", 1, None)
        a, b = ops.frame_statistics(key, value, self.beta(text_x, 1), config=self.config)
        _, injection = ops.vdn_factor_apply(None, a, b, max_eig=1.0 + length, config=self.config)
        return ttnn.multiply(injection, self.TEXT_STATE_SCALE)

    def readout(
        self,
        xv: ttnn.Tensor,
        q_raw: ttnn.Tensor,
        k_raw: ttnn.Tensor,
        v_raw: ttnn.Tensor,
        *,
        num_frames: int,
        frame_size: tuple[int, int],
        bounds: Sequence[Range],
        text_state: ttnn.Tensor | None,
    ) -> ttnn.Tensor:
        """`_readout` over exactly the frames given. Inputs `[1, 1, F*S, *]` bf16; returns `[1, 1, F*S, H*d]` bf16."""
        per_frame = frame_size[0] * frame_size[1]
        query = self.features(q_raw, "q", num_frames, frame_size)
        key = self.features(k_raw, "k", num_frames, frame_size)
        value = self.features(v_raw, "v", num_frames, frame_size)

        a, b = ops.frame_statistics(key, value, self.beta(xv, num_frames), config=self.config)
        alpha_rows = self.alpha(xv, num_frames)
        alpha = ops.rows_to_heads(alpha_rows, num_frames, self.num_heads, self.head_dim)
        transition, injection = ops.vdn_factor_apply(alpha, a, b, max_eig=1.0 + per_frame, config=self.config)
        prefix, suffix = ops.bidirectional_scan(transition, injection, text_state=text_state, config=self.config)
        state = ops.gather_linear_state(prefix, suffix, alpha_rows, bounds, text_state=text_state, config=self.config)

        out = ops.split_matmul(query, ttnn.transpose(state, -2, -1), config=self.config)  # [F, H, S, d_v] fp32
        mean_square = ttnn.multiply(ttnn.sum(ttnn.multiply(out, out), dim=-1, keepdim=True), 1.0 / self.head_dim)
        out = ttnn.multiply(out, ttnn.rsqrt(ttnn.add(mean_square, self.NORM_EPS)))
        out = from_heads(ttnn.multiply(out, self.norm_weight.data))

        gate = ttnn.sigmoid(self.gate_up(self.gate_down(xv)))
        return ttnn.typecast(ttnn.multiply(out, ttnn.typecast(gate, ttnn.float32)), ttnn.bfloat16)

    def forward(
        self,
        xv: ttnn.Tensor,
        q_raw: ttnn.Tensor,
        k_raw: ttnn.Tensor,
        v_raw: ttnn.Tensor,
        *,
        num_frames: int,
        frame_size: tuple[int, int],
        bounds: Sequence[Range],
        skip_ends: bool = True,
        text: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> ttnn.Tensor:
        """`BidirectionalLinearBranch.forward`. xv: `[1, 1, F*S, hidden]`, q/k/v_raw: `[1, 1, F*S, H*d]` (pre-norm,
        pre-RoPE projections), text: `(text_x, text_k_raw, text_v_raw)` prompt rows or None. Returns the gated, normed
        readout `[1, 1, F*S, H*d]` bf16 (before `to_out_linear`), zero on frames 0 and F-1 when `skip_ends`."""
        per_frame = frame_size[0] * frame_size[1]
        text_state = None if text is None else self.text_state(*text)
        if not skip_ends:
            return self.readout(
                xv, q_raw, k_raw, v_raw, num_frames=num_frames, frame_size=frame_size, bounds=bounds, text_state=text_state
            )
        zeros = ttnn.zeros(
            [1, 1, per_frame, self.inner_dim], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
        )
        if num_frames <= 2:
            return ttnn.concat([zeros] * num_frames, dim=2)
        start, end = per_frame, (num_frames - 1) * per_frame
        inner = self.readout(
            *(rows_slice(t, start, end) for t in (xv, q_raw, k_raw, v_raw)),
            num_frames=num_frames - 2,
            frame_size=frame_size,
            bounds=[(lo - 1, hi - 1) for lo, hi in bounds[1 : num_frames - 1]],
            text_state=text_state,
        )
        return ttnn.concat([zeros, inner, zeros], dim=2)
