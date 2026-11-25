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

Single device for now: tensor and sequence parallel are not implemented (the H3 base supports both).
"""

from __future__ import annotations

import dataclasses

import torch

import ttnn

from ....layers.linear import Linear
from ....utils import tensor
from ..minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, rope_channel_permutation
from . import linear_ops_openvdn as ops
from .linear_branch_openvdn import OpenVDNLinearBranch, rows_slice
from .reference_openvdn import window_bounds
from .window_softmax_openvdn import WindowLayout, WindowSoftmax


@dataclasses.dataclass(frozen=True)
class HybridLayout:
    """The packed sequence as the hybrid attention sees it (`SequenceLayout` in the model repo): video rows
    contiguous and frame-major from `video_start`, the prompt rows at `[text_start, text_start + text_len)`."""

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
        if self.parallel_config.tensor_parallel.factor != 1 or self.use_ring or self.vsa_config is not None:
            raise NotImplementedError("OpenVDN hybrid attention runs on one device for now (no TP, SP or VSA)")
        if anchor_frames not in ("none", "both"):
            raise NotImplementedError(f"anchor_frames={anchor_frames!r}; the released checkpoints use 'both'")
        self.radius = radius
        self.chunk = chunk
        self.anchor_frames = anchor_frames
        self.enable_text_state = enable_text_state
        self.layout: HybridLayout | None = None

        mesh_device = self.mesh_device
        config = ops.compute_config(mesh_device)
        self.linear_attention = OpenVDNLinearBranch(
            hidden_size=self.hidden_size, num_heads=self.num_heads, head_dim=self.head_dim, mesh_device=mesh_device
        )
        self.to_out_linear = Linear(
            self.inner_dim, self.hidden_size, bias=False, mesh_device=mesh_device, compute_kernel_config=config
        )
        # Per head, broadcast over head_dim through a 0/1 expansion matmul (exact: one nonzero per output).
        self.softmax_gate = Linear(
            self.hidden_size, self.num_heads, bias=True, mesh_device=mesh_device, compute_kernel_config=config
        )
        expand = torch.repeat_interleave(torch.eye(self.num_heads), self.head_dim, dim=1)
        self.gate_expand = tensor.from_torch(expand.reshape(1, 1, *expand.shape), device=mesh_device)
        self.gate_config = config
        self._windows: dict[HybridLayout, WindowSoftmax] = {}

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
            self._windows[layout] = WindowSoftmax(
                self.mesh_device,
                WindowLayout(layout.seq_len, layout.video_start, layout.num_frames, layout.tokens_per_frame),
                self.bounds(layout),
                scale=self.head_dim**-0.5,
                anchor_frames=self.anchor_frames,
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
        """spatial_1BND: `[1, 1, N, hidden]` on one device. Returns the attention output (or, with the addcmul pair,
        `residual + output * gate`). `logical_n` and `cu_window_seqlens` are not used: the layout fixes the mask."""
        assert (addcmul_residual is None) == (addcmul_gate is None), "addcmul residual/gate come as a pair"
        layout = self.layout
        assert layout is not None and layout.seq_len == spatial_1BND.shape[2], "set .layout to the packed sequence"
        bounds = self.bounds(layout)
        full_cover = all(lo <= 0 and hi >= layout.num_frames - 1 for lo, hi in bounds)

        q_1BNF, k_1BNF, v_1BNF = self.to_qkv(
            spatial_1BND,
            force_transpose=False,
            use_persistent_buffer=self.use_persistent_ccl_buffers,
            persistent_buffer_rows=self.ag_row_capacity,
            **self._agmm_kwargs(self._qkv_pins, self.hidden_size, 3 * self.inner_dim, spatial_1BND),
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
        softmax_1BND = ttnn.unsqueeze(ttnn.transformer.concatenate_heads(softmax_BHNE), 0)

        gate = ttnn.sigmoid(self.softmax_gate(spatial_1BND))  # [1, 1, N, H]
        gate = ttnn.matmul(gate, self.gate_expand, compute_kernel_config=self.gate_config)
        softmax_1BND = ttnn.multiply(softmax_1BND, gate)

        out = self.to_out(
            softmax_1BND,
            force_transpose=False,
            use_persistent_buffer=self.use_persistent_ccl_buffers,
            persistent_buffer_rows=self.ag_row_capacity,
            **self._agmm_kwargs(self._out_pins, self.inner_dim, self.hidden_size, softmax_1BND),
        )

        if not full_cover:
            vs, ve = layout.video_start, layout.video_end
            text = None
            if self.enable_text_state and layout.text_len:
                ts, te = layout.text_start, layout.text_start + layout.text_len
                text = tuple(rows_slice(t, ts, te) for t in (spatial_1BND, k_1BNF, v_1BNF))
            readout = self.linear_attention(
                *(rows_slice(t, vs, ve) for t in (spatial_1BND, q_1BNF, k_1BNF, v_1BNF)),
                num_frames=layout.num_frames,
                frame_size=(layout.frame_height, layout.frame_width),
                bounds=bounds,
                skip_ends=self.anchor_frames == "both",
                text=text,
            )
            video = ttnn.add(rows_slice(out, vs, ve), self.to_out_linear(readout))
            pieces = [rows_slice(out, 0, vs)] if vs else []
            pieces.append(video)
            if ve < layout.seq_len:
                pieces.append(rows_slice(out, ve, layout.seq_len))
            out = ttnn.concat(pieces, dim=2) if len(pieces) > 1 else video

        if addcmul_residual is not None:
            out = ttnn.addcmul(addcmul_residual, out, addcmul_gate)
        return out
