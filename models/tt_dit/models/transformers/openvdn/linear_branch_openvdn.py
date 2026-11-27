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

Tensor parallel by head (`tp_mesh_axis`): every per-channel and per-head parameter is sharded on that axis, so each
device runs the branch for its own heads, the same canonical heads H3's fused QKV hands it. No sequence parallel here:
the caller passes the whole sequence. Correct first, not fast: the spatial conv is 25 exact shift matmuls, and the
layout changes go through row-major reshapes and permutes.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence

import torch

import ttnn

from ....layers.linear import ColParallelLinear, Linear
from ....layers.module import Module, Parameter
from ....utils import tensor
from . import linear_ops_openvdn as ops

Range = tuple[int, int]


@dataclasses.dataclass
class RowInputs:
    """The branch's per-row inputs, computed by the caller on its own rows instead of from the whole `xv` (stage 4):
    `beta` `[1, 1, F*S, H]` = `sigmoid(beta_proj(xv))`, `frame_mean` `[1, 1, F, hidden]` fp32 = the per-frame mean of
    `xv`, `text_beta` `[1, 1, T, H]` the same as `beta` for the prompt rows (or None). With these the branch returns
    the readout ungated, fp32: the caller applies `output_gate` on its own rows."""

    beta: ttnn.Tensor
    frame_mean: ttnn.Tensor
    text_beta: ttnn.Tensor | None = None


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


def _pad_dim(x: ttnn.Tensor, dim: int, size: int) -> ttnn.Tensor:
    """Zero-pad dim `dim` of a tensor to `size` (tile-aligned for the inner dims), by concatenating zeros."""
    if x.shape[dim] == size:
        return x
    shape = list(x.shape)
    shape[dim] = size - shape[dim]
    return ttnn.concat([x, ops.zeros(x.device(), shape, x.dtype, x.layout)], dim=dim)


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
        tp_mesh_axis: int | None = None,
    ) -> None:
        """`num_heads` is the global head count; with `tp_mesh_axis` each device holds and runs
        `num_heads / mesh_device.shape[tp_mesh_axis]` of them (`num_local_heads`)."""
        super().__init__()
        tp_factor = 1 if tp_mesh_axis is None else tuple(mesh_device.shape)[tp_mesh_axis]
        assert num_heads % tp_factor == 0, f"{num_heads} heads must divide across TP={tp_factor}"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_local_heads = num_heads // tp_factor
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.local_inner_dim = self.num_local_heads * head_dim
        self.mesh_device = mesh_device
        self.short_conv = tuple(short_conv)
        self.config = ops.compute_config(mesh_device)
        taps = self.KERNEL * self.KERNEL
        channels = self.inner_dim
        tp = tp_mesh_axis

        def fp32(shape, mesh_axes=None):
            return Parameter(total_shape=shape, device=mesh_device, dtype=ttnn.float32, mesh_axes=mesh_axes)

        # Spatial taps as columns [C, 1] (they scale the transposed [C, S] frames), temporal taps as rows [1, C].
        for proj in self.short_conv:
            setattr(self, f"{proj}_sp", fp32([taps, 1, channels, 1], [None, None, tp, None]))
            setattr(self, f"{proj}_tm", fp32([self.KERNEL, 1, 1, channels], [None, None, None, tp]))

        linear = dict(mesh_device=mesh_device, compute_kernel_config=self.config)
        col = dict(mesh_device=mesh_device, compute_kernel_config=self.config, mesh_axis=tp)
        self.beta_proj = ColParallelLinear(hidden_size, num_heads, bias=False, **col)
        # alpha's two projections run fp32-grade (split operands), on bf16 weights.
        self.alpha_down = Parameter(total_shape=[hidden_size, head_dim], device=mesh_device)
        self.alpha_up = Parameter(total_shape=[head_dim, channels], device=mesh_device, mesh_axes=[None, tp])
        self.alpha_dt_bias = fp32([1, 1, 1, channels], [None, None, None, tp])
        # exp(A_log), repeated over each head's channels
        self.alpha_scale = fp32([1, 1, 1, channels], [None, None, None, tp])
        self.gate_down = Linear(hidden_size, head_dim, bias=False, **linear)
        self.gate_up = ColParallelLinear(head_dim, channels, bias=True, **col)
        self.norm_weight = fp32([1, 1, 1, head_dim])

        self._shifts: dict[tuple[int, int], list[ttnn.Tensor]] = {}
        #: `(mesh axis, factor, all_gather(t, dim))` to split the spatial conv's frames over devices that hold the
        #: same (replicated) tokens, or None. Set by the caller. With `branch_split` the other per-frame stages are
        #: split too; `parallel_scan` takes the log-depth scan.
        self.frame_split: tuple[int, int, object] | None = None
        self.branch_split = ""  # "", "frames" or "heads"
        self._local_weights: dict[str, tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        self.parallel_scan = False

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

    def _local_conv_weights(self, proj: str) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """`proj`'s spatial and temporal taps for this device's channels under the head split (cached)."""
        key = f"{proj}_local"
        if key not in self._local_weights:
            axis, factor, _ = self.frame_split
            width = (self.num_local_heads + (-self.num_local_heads % factor)) * self.head_dim
            sp, tm = getattr(self, f"{proj}_sp").data, getattr(self, f"{proj}_tm").data
            self._local_weights[key] = (
                ttnn.mesh_partition(_pad_dim(sp, 2, width), dim=2, cluster_axis=axis),
                ttnn.mesh_partition(_pad_dim(tm, 3, width), dim=3, cluster_axis=axis),
            )
        return self._local_weights[key]

    def conv(
        self,
        tokens: ttnn.Tensor,
        proj: str,
        num_frames: int,
        frame_size: tuple[int, int],
        *,
        channel_split: bool = False,
    ) -> ttnn.Tensor:
        """`LinearAttentionSepConv`: `[1, 1, F*S, C]` bf16 -> fp32. The spatial taps are exact 0/1 shift matmuls
        (one nonzero per output), scaled per channel and summed in fp32; the temporal taps are frame slices.
        `channel_split`: `tokens` hold this device's channels of the head split (`_head_split`); the taps follow."""
        channels = tokens.shape[-1]
        per_frame = frame_size[0] * frame_size[1]
        x = ttnn.to_layout(tokens, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.to_layout(ttnn.reshape(x, [1, num_frames, per_frame, channels]), ttnn.TILE_LAYOUT)
        x_t = ttnn.transpose(x, -2, -1)  # [1, F, C, S]

        # The spatial conv is per frame: with `frame_split` (axis, factor, gather) the frames are split over those
        # devices (padded to a multiple of the factor), each convolves its share and the all-gather reassembles them.
        split = None if channel_split else self.frame_split
        if split is not None:
            axis, factor, gather = split
            x_t = _pad_dim(x_t, 1, num_frames + (-num_frames % factor))
            x_t = ttnn.mesh_partition(x_t, dim=1, cluster_axis=axis)

        sp_weight, tm_weight = (
            self._local_conv_weights(proj)
            if channel_split
            else (getattr(self, f"{proj}_sp").data, getattr(self, f"{proj}_tm").data)
        )
        weight = sp_weight
        # The frames folded into the rows make each shift one 2D matmul `[F*C, S] x [S, S]` (stage 4: 33 ms faster
        # per block than the batched form; OPENVDN_CONV_FLAT=0 keeps the batch). Same products, each output one exact
        # copy. bf16 outputs and lower fidelity were slower or not exact.
        flat = os.environ.get("OPENVDN_CONV_FLAT", "1") == "1" and channels % ttnn.TILE_SIZE == 0
        frames_here = x_t.shape[1]
        lhs = ttnn.reshape(x_t, [1, 1, frames_here * channels, per_frame]) if flat else x_t
        acc = None
        for t, shift in enumerate(self._shift_matrices(frame_size)):
            moved = ttnn.matmul(lhs, shift, dtype=ttnn.float32, compute_kernel_config=self.config)
            if flat:
                moved = ttnn.reshape(moved, [1, frames_here, channels, per_frame])
            tap = ttnn.slice(weight, [t, 0, 0, 0], [t + 1, 1, channels, 1])
            moved = ttnn.multiply(moved, tap)
            acc = moved if acc is None else ttnn.add(acc, moved)
        if split is not None:
            acc = gather(acc, 1)
            if acc.shape[1] != num_frames:
                acc = ttnn.slice(acc, [0, 0, 0, 0], [1, num_frames, channels, per_frame])
        x = ttnn.transpose(acc, -2, -1)  # [1, F, S, C] fp32

        pad = self.KERNEL // 2
        zeros = ops.zeros(self.mesh_device, [1, 1, per_frame, channels], x.dtype)
        padded = ttnn.concat([zeros] * pad + [x] + [zeros] * pad, dim=1)
        weight = tm_weight
        acc = None
        for dt in range(self.KERNEL):
            part = ttnn.slice(padded, [0, dt, 0, 0], [1, dt + num_frames, per_frame, channels])
            part = ttnn.multiply(part, ttnn.slice(weight, [dt, 0, 0, 0], [dt + 1, 1, 1, channels]))
            acc = part if acc is None else ttnn.add(acc, part)
        acc = ttnn.to_layout(acc, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(ttnn.reshape(acc, [1, 1, num_frames * per_frame, channels]), ttnn.TILE_LAYOUT)

    def features(
        self,
        tokens: ttnn.Tensor,
        proj: str,
        num_frames: int,
        frame_size: tuple[int, int] | None,
        partition=None,
        channel_split: bool = False,
    ) -> ttnn.Tensor:
        """`prepare_linear_features`: `[1, 1, F*S, C]` raw bf16 -> `[F, H, S, d]` bf16. `frame_size=None` skips the
        conv (the text chunk). `partition` (frames on dim 0 -> this device's share) is applied in the tiled head
        layout (a row-major split measured ~70 ms slower), giving `[f, H, S, d]` for the local frames.
        `channel_split`: `tokens` are this device's channels of the head split, `H = C / d` of them."""
        if frame_size is not None and proj in self.short_conv:
            x = self.conv(tokens, proj, num_frames, frame_size, channel_split=channel_split)
        else:
            x = ttnn.typecast(tokens, ttnn.float32)
        x = to_heads(ttnn.silu(x), num_frames, tokens.shape[-1] // self.head_dim)
        if partition is not None:
            x = partition(x)
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
        return self.alpha_from_mean(ttnn.to_layout(mean, ttnn.TILE_LAYOUT))

    def alpha_from_mean(self, mean: ttnn.Tensor) -> ttnn.Tensor:
        """`alpha` from the per-frame means `[1, 1, F, hidden]` fp32."""
        delta = ops.split_matmul(mean, [self.alpha_down.data], config=self.config)
        delta = ops.split_matmul(delta, [self.alpha_up.data], config=self.config)
        delta = ttnn.add(delta, self.alpha_dt_bias.data)
        return ttnn.exp(ttnn.neg(ttnn.multiply(softplus(delta), self.alpha_scale.data)))

    def output_gate(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """`sigmoid(output_gate(x))` `[1, 1, N, H*d]` for the readout."""
        return ttnn.sigmoid(self.gate_up(self.gate_down(x)))

    def text_state(
        self,
        text_x: ttnn.Tensor | None,
        text_k_raw: ttnn.Tensor,
        text_v_raw: ttnn.Tensor,
        head_split: bool = False,
        text_beta: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """`_text_state`: `TEXT_STATE_SCALE * (B (I + A)^-1)` over all prompt rows as one chunk, `[1, H, d_v, d_k]`
        fp32 (`head_split`: this device's heads only, see `_head_split`). `text_beta` (`RowInputs`) replaces
        `text_x`."""
        length = text_k_raw.shape[-2]
        local = local_channels = lambda t: t  # noqa: E731
        if head_split:
            local, local_channels, _ = self._head_split()
        key = self.features(local_channels(text_k_raw), "k", 1, None)
        value = self.features(local_channels(text_v_raw), "v", 1, None)
        beta = self.beta(text_x, 1) if text_beta is None else heads_column(text_beta, 1)
        a, b = ops.frame_statistics(key, value, local(beta), config=self.config)
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
        rows: RowInputs | None = None,
    ) -> ttnn.Tensor:
        """`_readout` over exactly the frames given. Inputs `[1, 1, F*S, *]` bf16; returns `[1, 1, F*S, H*d]` bf16
        (with `rows`: ungated, fp32)."""
        per_frame = frame_size[0] * frame_size[1]
        mark = lambda name: ops.mark(self.mesh_device, f"linear branch: {name}")  # noqa: E731

        # With `branch_split` (and `frame_split`), every per-frame stage (features, statistics, alpha, inverse,
        # readout) runs on this device's share of the frames; only the scans see all frames, between two gathers.
        if self.branch_split == "heads" and self.frame_split is not None:
            return self._readout_head_split(
                xv,
                q_raw,
                k_raw,
                v_raw,
                num_frames=num_frames,
                frame_size=frame_size,
                bounds=bounds,
                text_state=text_state,
                rows=rows,
            )
        assert rows is None, "RowInputs need the head split"
        partition = gather = None
        if self.branch_split and self.frame_split is not None:
            axis, factor, all_gather = self.frame_split
            padded_frames = num_frames + (-num_frames % factor)

            def partition(t: ttnn.Tensor) -> ttnn.Tensor:  # frames on dim 0, replicated -> this device's share
                return ttnn.mesh_partition(_pad_dim(t, 0, padded_frames), dim=0, cluster_axis=axis)

            def gather(t: ttnn.Tensor) -> ttnn.Tensor:  # local frames on dim 0 -> the F real ones, everywhere
                t = all_gather(t, 0)
                return t if t.shape[0] == num_frames else ttnn.slice(t, [0] * 4, [num_frames, *list(t.shape)[1:]])

        local = partition if partition is not None else (lambda t: t)
        query = self.features(q_raw, "q", num_frames, frame_size, partition=partition)
        mark("q features")
        key = self.features(k_raw, "k", num_frames, frame_size, partition=partition)
        mark("k features (conv)")
        value = self.features(v_raw, "v", num_frames, frame_size, partition=partition)
        mark("v features (conv)")

        a, b = ops.frame_statistics(key, value, local(self.beta(xv, num_frames)), config=self.config)
        alpha_rows = self.alpha(xv, num_frames)
        alpha = local(ops.rows_to_heads(alpha_rows, num_frames, self.num_local_heads, self.head_dim))
        mark("frame statistics and alpha")
        transition, injection = ops.vdn_factor_apply(alpha, a, b, max_eig=1.0 + per_frame, config=self.config)
        if gather is not None:
            transition, injection = gather(transition), gather(injection)
        mark("inverse")
        scan = ops.parallel_bidirectional_scan if self.parallel_scan else ops.bidirectional_scan
        prefix, suffix = scan(transition, injection, text_state=text_state, config=self.config)
        mark("scans")
        state = ops.gather_linear_state(prefix, suffix, alpha_rows, bounds, text_state=text_state, config=self.config)
        mark("state gather")

        out = ops.split_matmul(query, ttnn.transpose(local(state), -2, -1), config=self.config)  # [F, H, S, d_v]
        mean_square = ttnn.multiply(ttnn.sum(ttnn.multiply(out, out), dim=-1, keepdim=True), 1.0 / self.head_dim)
        out = ttnn.multiply(out, ttnn.rsqrt(ttnn.add(mean_square, self.NORM_EPS)))
        out = ttnn.multiply(out, self.norm_weight.data)
        if gather is not None:
            out = gather(out)
        out = from_heads(out)

        gate = ttnn.sigmoid(self.gate_up(self.gate_down(xv)))
        out = ttnn.typecast(ttnn.multiply(out, ttnn.typecast(gate, ttnn.float32)), ttnn.bfloat16)
        mark("readout")
        return out

    def _head_split(self):
        """`(local, local_channels, gather)` splitting this device's heads over the `frame_split` devices (padded to
        a multiple of the factor): `local` on a head dim 1, `local_channels` on `[.., H*d]` channel rows, `gather`
        back to all real heads on dim 1."""
        axis, factor, all_gather = self.frame_split
        heads, d = self.num_local_heads, self.head_dim
        padded = heads + (-heads % factor)

        def local(t: ttnn.Tensor) -> ttnn.Tensor:
            return ttnn.mesh_partition(_pad_dim(t, 1, padded), dim=1, cluster_axis=axis)

        def local_channels(t: ttnn.Tensor) -> ttnn.Tensor:
            return ttnn.mesh_partition(_pad_dim(t, 3, padded * d), dim=3, cluster_axis=axis)

        def gather(t: ttnn.Tensor) -> ttnn.Tensor:
            t = all_gather(t, 1)
            return t if t.shape[1] == heads else ttnn.slice(t, [0] * 4, [t.shape[0], heads, t.shape[2], t.shape[3]])

        return local, local_channels, gather

    def _readout_head_split(
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
        rows: RowInputs | None = None,
    ) -> ttnn.Tensor:
        """`readout` with the delta-rule math split by head over the SP devices: statistics, inverse, scans, state
        gather and q . state run on this device's heads (the text state comes in already split, see `forward`),
        from the conv and features on: everything but the small beta/alpha/gate projections; one gather at the end.
        With `rows` those projections come from the caller and the readout is returned ungated (fp32)."""
        per_frame = frame_size[0] * frame_size[1]
        mark = lambda name: ops.mark(self.mesh_device, f"linear branch: {name}")  # noqa: E731
        local, local_channels, gather = self._head_split()
        factor = self.frame_split[1]
        local_heads = -(-self.num_local_heads // factor)

        # The conv is depthwise and the features per head, so each device takes its channels from the raw rows.
        query, key, value = (
            self.features(local_channels(raw), proj, num_frames, frame_size, channel_split=True)
            for raw, proj in ((q_raw, "q"), (k_raw, "k"), (v_raw, "v"))
        )
        mark("q, k, v features (conv)")

        beta = self.beta(xv, num_frames) if rows is None else heads_column(rows.beta, num_frames)
        a, b = ops.frame_statistics(key, value, local(beta), config=self.config)
        alpha_rows = self.alpha(xv, num_frames) if rows is None else self.alpha_from_mean(rows.frame_mean)
        alpha_rows = local_channels(alpha_rows)
        alpha = ops.rows_to_heads(alpha_rows, num_frames, local_heads, self.head_dim)
        mark("frame statistics and alpha")
        transition, injection = ops.vdn_factor_apply(alpha, a, b, max_eig=1.0 + per_frame, config=self.config)
        mark("inverse")
        scan = ops.parallel_bidirectional_scan if self.parallel_scan else ops.bidirectional_scan
        prefix, suffix = scan(transition, injection, text_state=text_state, config=self.config)
        mark("scans")
        state = ops.gather_linear_state(prefix, suffix, alpha_rows, bounds, text_state=text_state, config=self.config)
        mark("state gather")

        out = ops.split_matmul(query, ttnn.transpose(state, -2, -1), config=self.config)  # [F, h, S, d_v]
        mean_square = ttnn.multiply(ttnn.sum(ttnn.multiply(out, out), dim=-1, keepdim=True), 1.0 / self.head_dim)
        out = ttnn.multiply(out, ttnn.rsqrt(ttnn.add(mean_square, self.NORM_EPS)))
        out = from_heads(gather(ttnn.multiply(out, self.norm_weight.data)))
        if rows is not None:
            mark("readout")
            return out

        gate = self.output_gate(xv)
        out = ttnn.typecast(ttnn.multiply(out, ttnn.typecast(gate, ttnn.float32)), ttnn.bfloat16)
        mark("readout")
        return out

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
        text: tuple[ttnn.Tensor | None, ttnn.Tensor, ttnn.Tensor] | None = None,
        rows: RowInputs | None = None,
    ) -> ttnn.Tensor:
        """`BidirectionalLinearBranch.forward`. xv: `[1, 1, F*S, hidden]`, q/k/v_raw: `[1, 1, F*S, H*d]` (pre-norm,
        pre-RoPE projections), text: `(text_x, text_k_raw, text_v_raw)` prompt rows or None. Returns the gated, normed
        readout `[1, 1, F*S, H*d]` bf16 (before `to_out_linear`), zero on frames 0 and F-1 when `skip_ends`.
        With `rows` (`RowInputs`, all F frames) `xv` and `text_x` are not used (None) and the readout comes back
        ungated, fp32."""
        per_frame = frame_size[0] * frame_size[1]
        head_split = self.branch_split == "heads" and self.frame_split is not None
        text_state = None
        if text is not None:
            text_beta = None if rows is None else rows.text_beta
            text_state = self.text_state(*text, head_split=head_split, text_beta=text_beta)
        ops.mark(self.mesh_device, "linear branch: text state")
        if not skip_ends:
            return self.readout(
                xv,
                q_raw,
                k_raw,
                v_raw,
                num_frames=num_frames,
                frame_size=frame_size,
                bounds=bounds,
                text_state=text_state,
                rows=rows,
            )
        zeros = ops.zeros(
            self.mesh_device, [1, 1, per_frame, self.local_inner_dim], ttnn.bfloat16 if rows is None else ttnn.float32
        )
        if num_frames <= 2:
            return ttnn.concat([zeros] * num_frames, dim=2)
        start, end = per_frame, (num_frames - 1) * per_frame
        if rows is not None:
            rows = RowInputs(
                beta=rows_slice(rows.beta, start, end),
                frame_mean=rows_slice(rows.frame_mean, 1, num_frames - 1),
                text_beta=rows.text_beta,
            )
        inner = self.readout(
            None if xv is None else rows_slice(xv, start, end),
            *(rows_slice(t, start, end) for t in (q_raw, k_raw, v_raw)),
            num_frames=num_frames - 2,
            frame_size=frame_size,
            bounds=[(lo - 1, hi - 1) for lo, hi in bounds[1 : num_frames - 1]],
            text_state=text_state,
            rows=rows,
        )
        return ttnn.concat([zeros, inner, zeros], dim=2)
