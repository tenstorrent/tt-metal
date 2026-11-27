# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN hybrid attention on device: the MiniMax-H3 attention with its dense softmax replaced by a frame-window
softmax plus a linear-attention branch for everything outside the window.

Reference: OpenVDN/vdn-minimax-h3 @e262cb57, `HybridAttention` (mirrored in `reference_hybrid_openvdn.py`):

    softmax_output = to_out( softmax_gate(x) * window_softmax(q, k, v) )
    linear_readout = output_gate(x) * RMSNorm( linear_attention(q_raw, k_raw, v_raw) )
    output         = softmax_output + to_out_linear(linear_readout)       # video rows only

A subclass of `MiniMaxH3Attention`, so the shared H3 code is reused unchanged: the fused QKV projection, the QK norms
with RoPE, `to_out` and the weight relayout. The checkpoint's `orig.*` keys are the H3 attention's own.

The H3 relayout permutes Q and K channels within each head (`rope_channel_permutation`), so the raw q/k the linear
branch reads come out of `to_qkv` in that order. The branch is equivariant to it as long as every per-key-channel
parameter is permuted the same way: the K conv taps and alpha's `up` rows and `dt_bias` (`_prepare_torch_state`).
q . state and the readout are unchanged, and V is not permuted.

Tensor parallel by head, as H3: each device projects, normalises and attends with its own heads (the softmax gate,
the linear branch and its gates are sharded the same way), then the outputs are gathered on TP for the column-parallel
`to_out` and `to_out_linear`. Sequence parallel by gathering: the SP-fractured rows are all-gathered so every device
holds the whole (logical, unpadded) sequence for its TP heads, and `mesh_partition` hands each device its rows back.
The heavy work is then split over the SP devices (stage 4): each window group's queries by row (`WindowSoftmax`), and
the linear branch by head, from its raw projections to the readout (`OpenVDNLinearBranch`, `_head_split`), each
reassembled with one all-gather; the per-row tail (softmax gate, `to_out`, `to_out_linear`) runs on each device's own
rows. Still repeated on every SP device: the gathers of the projections and the small beta/alpha/gate projections.
"""

from __future__ import annotations

import dataclasses
import os

import torch

import ttnn

from ....layers.linear import ColParallelLinear
from ....utils import tensor
from ..minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, rope_channel_permutation
from . import linear_ops_openvdn as ops
from .linear_branch_openvdn import OpenVDNLinearBranch, RowInputs, rows_slice
from .reference_openvdn import window_bounds
from .window_softmax_openvdn import WindowLayout, WindowSoftmax


@dataclasses.dataclass(frozen=True)
class HybridLayout:
    """The packed sequence as the hybrid attention sees it (`SequenceLayout` in the model repo): video rows
    contiguous and frame-major from `video_start`, the prompt rows at `[text_start, text_start + text_len)`.
    `seq_len` is the logical length; rows past it (the SP pad tail) take no part."""

    seq_len: int
    video_start: int
    num_frames: int
    frame_height: int
    frame_width: int
    text_start: int = 0
    text_len: int = 0

    @property
    def tokens_per_frame(self) -> int:
        return self.frame_height * self.frame_width

    @property
    def video_end(self) -> int:
        return self.video_start + self.num_frames * self.tokens_per_frame


class OpenVDNHybridAttention(MiniMaxH3Attention):
    """Set `.layout` (a `HybridLayout`) before each forward. Same call signature and output as `MiniMaxH3Attention`."""

    def __init__(
        self,
        *,
        radius: int = 1,
        chunk: int = 5,
        anchor_frames: str = "both",
        enable_text_state: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if self.vsa_config is not None or not self.is_sequence_parallel:
            raise NotImplementedError("OpenVDN hybrid attention: no VSA, and the sequence must be SP-fractured")
        if anchor_frames not in ("none", "both"):
            raise NotImplementedError(f"anchor_frames={anchor_frames!r}; the released checkpoints use 'both'")
        self.radius = radius
        self.chunk = chunk
        self.anchor_frames = anchor_frames
        self.enable_text_state = enable_text_state
        self.layout: HybridLayout | None = None

        mesh_device = self.mesh_device
        config = ops.compute_config(mesh_device)
        self.tp_factor = self.parallel_config.tensor_parallel.factor
        self.sp_factor = self.parallel_config.sequence_parallel.factor
        tp_axis = self.tp_mesh_axis if self.tp_factor > 1 else None
        self.linear_attention = OpenVDNLinearBranch(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            mesh_device=mesh_device,
            tp_mesh_axis=tp_axis,
        )
        # The linear branch's work is split over the SP devices, which all hold the whole sequence (stage 4). Env
        # switches, for A/B runs: OPENVDN_CONV_SP=0 turns the split off; OPENVDN_BRANCH_SP=heads (default) splits the
        # whole branch by head from the raw projections on, =1 splits the conv and the per-frame stages by frame
        # (slower), =0 only the spatial conv by frame; OPENVDN_PARALLEL_SCAN=0 takes the sequential scan.
        if os.environ.get("OPENVDN_CONV_SP", "1") == "1" and self.sp_factor > 1:
            self.linear_attention.frame_split = (
                self.sp_mesh_axis,
                self.sp_factor,
                lambda t, dim: self._gather(t, dim=dim, mesh_axis=self.sp_mesh_axis, factor=self.sp_factor),
            )
            self.linear_attention.branch_split = {"1": "frames", "heads": "heads"}.get(
                os.environ.get("OPENVDN_BRANCH_SP", "heads"), ""
            )
        self.linear_attention.parallel_scan = os.environ.get("OPENVDN_PARALLEL_SCAN", "1") == "1"
        col = dict(mesh_device=mesh_device, mesh_axis=tp_axis, compute_kernel_config=config)
        # Column-parallel like `to_out`: TP-gathered readout in, TP-fractured hidden out.
        self.to_out_linear = ColParallelLinear(self.inner_dim, self.hidden_size, bias=False, **col)
        # Per head, broadcast over head_dim through a 0/1 expansion matmul (exact: one nonzero per output).
        self.softmax_gate = ColParallelLinear(self.hidden_size, self.num_heads, bias=True, **col)
        expand = torch.repeat_interleave(torch.eye(self.n_local_heads), self.head_dim, dim=1)
        self.gate_expand = tensor.from_torch(expand.reshape(1, 1, *expand.shape), device=mesh_device)
        self.gate_config = config
        self._windows: dict[HybridLayout, WindowSoftmax] = {}
        self._cache: dict[tuple, ttnn.Tensor] = {}

    # ------------------------------------------------------------------ weights

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for key in [k for k in state if k.startswith("orig.")]:
            state[key.removeprefix("orig.")] = state.pop(key)
        for key in [k for k in state if k.startswith("softmax_gate.up.")]:
            state["softmax_gate." + key.removeprefix("softmax_gate.up.")] = state.pop(key)

        perm = rope_channel_permutation(self.head_dim, self.rotary_dim)
        key_channel_params = [f"linear_attention.short_conv.k_{p}.weight" for p in ("sp", "tm")] + [
            "linear_attention.alpha.up.weight",
            "linear_attention.alpha.dt_bias",
        ]
        for name in key_channel_params:
            if name in state:
                t = state[name]
                state[name] = t.reshape(self.num_heads, self.head_dim, *t.shape[1:])[:, perm].reshape(t.shape)

        super()._prepare_torch_state(state)

    # ------------------------------------------------------------------ forward

    def bounds(self, layout: HybridLayout) -> list[tuple[int, int]]:
        return window_bounds(layout.num_frames, self.radius, self.chunk)

    def _window(self, layout: HybridLayout) -> WindowSoftmax:
        if layout not in self._windows:
            # Each group's queries are split over the SP devices and the groups run on the frame-padded, tile-aligned
            # layout (stage 4). Env switches, for A/B runs: OPENVDN_WINDOW_SP=0, OPENVDN_WINDOW_ALIGNED=0 turn them
            # off; OPENVDN_WINDOW_CHUNKS="q,k" sets the SDPA chunk sizes (32,32 measured best).
            split = os.environ.get("OPENVDN_WINDOW_SP", "1") == "1" and self.sp_factor > 1
            q_chunk, k_chunk = (int(c) for c in os.environ.get("OPENVDN_WINDOW_CHUNKS", "32,32").split(","))
            self._windows[layout] = WindowSoftmax(
                self.mesh_device,
                WindowLayout(layout.seq_len, layout.video_start, layout.num_frames, layout.tokens_per_frame),
                self.bounds(layout),
                scale=self.head_dim**-0.5,
                anchor_frames=self.anchor_frames,
                sp_axis=self.sp_mesh_axis if split else None,
                sp_factor=self.sp_factor if split else 1,
                gather=(lambda t: self._gather(t, dim=2, mesh_axis=self.sp_mesh_axis, factor=self.sp_factor))
                if split
                else None,
                q_chunk=q_chunk,
                k_chunk=k_chunk,
                aligned=os.environ.get("OPENVDN_WINDOW_ALIGNED", "1") == "1",
            )
        return self._windows[layout]

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        logical_n: ttnn.Tensor | None = None,
        rope_cos: ttnn.Tensor | None = None,
        rope_sin: ttnn.Tensor | None = None,
        addcmul_residual: ttnn.Tensor | None = None,
        addcmul_gate: ttnn.Tensor | None = None,
        cu_window_seqlens: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """spatial_1BND: `[1, 1, N_local, hidden / TP]`, fractured N on SP (padded to the SP alignment) and hidden on
        TP, as `MiniMaxH3Attention` takes it; rope tables fractured the same way on SP. Returns the attention output
        (or, with the addcmul pair, `residual + output * gate`) in the same distribution. `logical_n` and
        `cu_window_seqlens` are not used: `.layout` (with its logical `seq_len`) fixes the mask."""
        assert (addcmul_residual is None) == (addcmul_gate is None), "addcmul residual/gate come as a pair"
        layout = self.layout
        padded_len = spatial_1BND.shape[2] * self.sp_factor
        assert layout is not None and layout.seq_len <= padded_len, "set .layout to the packed sequence"
        self._mark("attention start")
        bounds = self.bounds(layout)
        full_cover = all(lo <= 0 and hi >= layout.num_frames - 1 for lo, hi in bounds)
        length = layout.seq_len

        # Projections, QK norm and RoPE on the local rows, exactly as H3 does them.
        x_1BND = self._gather(spatial_1BND, dim=3, mesh_axis=self.tp_mesh_axis, factor=self.tp_factor)
        q_1BNF, k_1BNF, v_1BNF = self.to_qkv(
            x_1BND, force_transpose=False, compute_kernel_config=self.mm_compute_kernel_config
        )
        norm_kwargs = dict(
            num_heads_per_device=self.n_local_heads,
            per_head_norm=True,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=self.rope_trans_mat if rope_cos is not None else None,
        )
        q_BHNE = self.norm_q(q_1BNF, **norm_kwargs)
        k_BHNE = self.norm_k(k_1BNF, **norm_kwargs)
        v_BHNE, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            v_1BNF, num_heads=self.n_local_heads, num_kv_heads=0, transpose_k_heads=False
        )

        # The whole sequence on every SP device, for this device's heads. The window and the linear branch take their
        # rows by explicit ranges, so only the dense path trims the pad tail (a trim is a full copy).
        def whole(t: ttnn.Tensor) -> ttnn.Tensor:
            t = self._gather(t, dim=2, mesh_axis=self.sp_mesh_axis, factor=self.sp_factor)
            if t.shape[2] == length or (not full_cover and os.environ.get("OPENVDN_TRIM", "0") != "1"):
                return t
            return ttnn.slice(t, [0, 0, 0, 0], [t.shape[0], t.shape[1], length, t.shape[3]])

        # The per-row tail (softmax gate, to_out, to_out_linear) runs on this device's rows only (stage 4;
        # OPENVDN_LOCAL_OUT=0 runs it on the whole sequence and partitions after, as before). With it, the linear
        # branch's per-row projections (beta, frame means, output gate) also run on the local rows, and x is not
        # gathered at all (OPENVDN_LOCAL_ROWS=0 gathers x and runs them on the whole sequence).
        local_out = os.environ.get("OPENVDN_LOCAL_OUT", "1") == "1" and self.sp_factor > 1
        local_rows = (
            local_out
            and not full_cover
            and self.linear_attention.branch_split == "heads"
            and os.environ.get("OPENVDN_LOCAL_ROWS", "1") == "1"
        )
        x_local = x_1BND
        rows = branch_gate = None
        if local_rows:
            rows, branch_gate = self._row_inputs(x_local, layout, padded_len)
            x_1BND = None
        else:
            x_1BND = whole(x_1BND)
        q_1BNF, k_1BNF, v_1BNF, q_BHNE, k_BHNE, v_BHNE = (
            whole(t) for t in (q_1BNF, k_1BNF, v_1BNF, q_BHNE, k_BHNE, v_BHNE)
        )
        self._mark("projections and SP gather")

        def to_local(t: ttnn.Tensor) -> ttnn.Tensor:  # whole logical rows -> this device's padded rows
            if t.shape[2] < padded_len:
                t = ttnn.pad(t, [(0, 0), (0, 0), (0, padded_len - t.shape[2]), (0, 0)], 0.0)
            return ttnn.mesh_partition(t, dim=2, cluster_axis=self.sp_mesh_axis)

        if full_cover:
            softmax_BHNE = ttnn.transformer.scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                is_causal=False,
                program_config=self._sdpa_program_config(q_BHNE.shape[2], ring=False),
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )
        else:
            softmax_BHNE = self._window(layout)(q_BHNE, k_BHNE, v_BHNE)
        self._mark("window softmax")
        softmax_1BND = ttnn.unsqueeze(ttnn.transformer.concatenate_heads(softmax_BHNE), 0)
        if local_out:
            softmax_1BND = to_local(softmax_1BND)

        gate = ttnn.sigmoid(self.softmax_gate(x_local if local_out else x_1BND))  # [1, 1, N, local heads]
        gate = ttnn.matmul(gate, self.gate_expand, compute_kernel_config=self.gate_config)
        softmax_1BND = ttnn.multiply(softmax_1BND, gate)

        # Each device holds canonical heads [d * n_local, (d+1) * n_local), so gathering on TP rebuilds the
        # canonical inner order the column-parallel output projections expect.
        softmax_1BND = self._gather(softmax_1BND, dim=3, mesh_axis=self.tp_mesh_axis, factor=self.tp_factor)
        out = self.to_out(softmax_1BND, force_transpose=False, compute_kernel_config=self.mm_compute_kernel_config)
        self._mark("softmax gate, TP gather and to_out")

        if not full_cover:
            vs, ve = layout.video_start, layout.video_end
            text = None
            if self.enable_text_state and layout.text_len:
                ts, te = layout.text_start, layout.text_start + layout.text_len
                text = tuple(None if t is None else rows_slice(t, ts, te) for t in (x_1BND, k_1BNF, v_1BNF))
            readout = self.linear_attention(
                *(None if t is None else rows_slice(t, vs, ve) for t in (x_1BND, q_1BNF, k_1BNF, v_1BNF)),
                num_frames=layout.num_frames,
                frame_size=(layout.frame_height, layout.frame_width),
                bounds=bounds,
                skip_ends=self.anchor_frames == "both",
                text=text,
                rows=rows,
            )
            self._mark("linear branch")
            if local_out:
                # The video rows' readout placed in the whole sequence (zero elsewhere), then this device's rows.
                width = readout.shape[-1]
                pieces = [self._zero_rows(vs, width, readout.dtype)] if vs else []
                pieces.append(readout)
                if ve < padded_len:
                    pieces.append(self._zero_rows(padded_len - ve, width, readout.dtype))
                readout = to_local(ttnn.concat(pieces, dim=2))
                if branch_gate is not None:  # ungated fp32 readout: the output gate on the local rows
                    readout = ttnn.multiply(readout, ttnn.typecast(branch_gate, ttnn.float32))
                    readout = ttnn.typecast(readout, ttnn.bfloat16)
                readout = self._gather(readout, dim=3, mesh_axis=self.tp_mesh_axis, factor=self.tp_factor)
                out = ttnn.add(out, self.to_out_linear(readout))
        if local_out:
            self._mark("output rows assembled")
            if length < padded_len:
                out = ttnn.multiply(out, self._valid_rows(length, padded_len))  # pad rows zero, as `ttnn.pad` left them
            if addcmul_residual is not None:
                out = ttnn.addcmul(addcmul_residual, out, addcmul_gate)
            self._mark("attention end")
            return out

        if not full_cover:
            readout = self._gather(readout, dim=3, mesh_axis=self.tp_mesh_axis, factor=self.tp_factor)
            video = ttnn.add(rows_slice(out, vs, ve), self.to_out_linear(readout))
            pieces = [rows_slice(out, 0, vs)] if vs else []
            pieces.append(video)
            if ve < length:
                pieces.append(rows_slice(out, ve, length))
            out = ttnn.concat(pieces, dim=2) if len(pieces) > 1 else video

        # Back to the padded length, then each SP device keeps its own rows.
        self._mark("output rows assembled")
        if length < padded_len:
            out = ttnn.pad(out, [(0, 0), (0, 0), (0, padded_len - length), (0, 0)], 0.0)
            self._mark("pad tail")
        if self.sp_factor > 1:
            out = ttnn.mesh_partition(out, dim=2, cluster_axis=self.sp_mesh_axis)
            self._mark("mesh_partition")

        if addcmul_residual is not None:
            out = ttnn.addcmul(addcmul_residual, out, addcmul_gate)
        self._mark("attention end")
        return out

    def _mark(self, name: str) -> None:
        ops.mark(self.mesh_device, name)

    def _row_inputs(
        self, x_local: ttnn.Tensor, layout: HybridLayout, padded_len: int
    ) -> tuple[RowInputs, ttnn.Tensor]:
        """The linear branch's per-row projections on this device's rows `x_local` `[1, 1, N / SP, hidden]`:
        `RowInputs` (beta gathered to the whole sequence, a few columns wide; per-frame means as a 0/1 frame-selector
        matmul on the local rows, exact bf16 products summed in fp32, partial sums added over SP) and the output gate
        for the local rows."""
        branch = self.linear_attention
        vs, ve = layout.video_start, layout.video_end
        beta = self._gather(
            ttnn.sigmoid(branch.beta_proj(x_local)), dim=2, mesh_axis=self.sp_mesh_axis, factor=self.sp_factor
        )
        sums = ttnn.matmul(
            self._frame_selector(layout, padded_len), x_local, dtype=ttnn.float32, compute_kernel_config=self.gate_config
        )
        sums = self._gather(sums, dim=1, mesh_axis=self.sp_mesh_axis, factor=self.sp_factor)
        sums = ttnn.sum(sums, dim=1, keepdim=True)
        sums = ttnn.slice(sums, [0, 0, 0, 0], [1, 1, layout.num_frames, sums.shape[-1]])
        text_beta = None
        if layout.text_len:
            text_beta = rows_slice(beta, layout.text_start, layout.text_start + layout.text_len)
        rows = RowInputs(
            beta=rows_slice(beta, vs, ve),
            frame_mean=ttnn.multiply(sums, 1.0 / layout.tokens_per_frame),
            text_beta=text_beta,
        )
        return rows, branch.output_gate(x_local)

    def _frame_selector(self, layout: HybridLayout, padded_len: int) -> ttnn.Tensor:
        """`[1, 1, F_pad, padded / SP]` bf16 per device: 1 where a local row belongs to video frame f."""
        key = ("frames", layout, padded_len)
        if key not in self._cache:
            frames = layout.num_frames + (-layout.num_frames % ttnn.TILE_SIZE)
            sel = torch.zeros(1, 1, frames, padded_len)
            for f in range(layout.num_frames):
                start = layout.video_start + f * layout.tokens_per_frame
                sel[0, 0, f, start : start + layout.tokens_per_frame] = 1.0
            self._cache[key] = tensor.from_torch(
                sel, device=self.mesh_device, dtype=ttnn.bfloat16, mesh_axes=[None, None, None, self.sp_mesh_axis]
            )
        return self._cache[key]

    def _zero_rows(self, rows: int, width: int, dtype: ttnn.DataType = ttnn.bfloat16) -> ttnn.Tensor:
        key = ("zeros", rows, width, dtype)
        if key not in self._cache:
            self._cache[key] = ttnn.zeros([1, 1, rows, width], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device)
        return self._cache[key]

    def _valid_rows(self, length: int, padded_len: int) -> ttnn.Tensor:
        """`[1, 1, padded / SP, 1]` per device: 1 on the logical rows, 0 on the pad tail."""
        key = ("valid", length, padded_len)
        if key not in self._cache:
            valid = (torch.arange(padded_len) < length).to(torch.float32).reshape(1, 1, padded_len, 1)
            self._cache[key] = tensor.from_torch(
                valid, device=self.mesh_device, dtype=ttnn.bfloat16, mesh_axes=[None, None, self.sp_mesh_axis, None]
            )
        return self._cache[key]

    def _gather(self, t: ttnn.Tensor, *, dim: int, mesh_axis: int, factor: int) -> ttnn.Tensor:
        if factor == 1:
            return t
        return self.ccl_manager.all_gather(t, dim=dim, mesh_axis=mesh_axis, use_hyperparams=False)
