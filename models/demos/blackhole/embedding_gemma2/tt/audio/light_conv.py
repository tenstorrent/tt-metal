# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioLightConv1d (HF `Gemma4AudioLightConv1d`, modeling_gemma4.py:490-528).

    h   = glu( ClippableLinear_start( RMSNorm_pre(x) ) )          1024 -> 2048 -> a * sigmoid(b), 1024
    y_t = sum_{k=0..4} w[:, k] * h_{t-4+k}                       causal depthwise conv, h_{<0} = 0, no bias
    out = x + ClippableLinear_end( silu( RMSNorm_conv(y) ) )      (clamp +-1e10 before conv_norm: a no-op)

Causal shift on device (exact: 0/1 matrices, one nonzero per row, bf16 output). Batch 1 (E5c): four
[S, S] shift matrices per token bucket (load_causal_shifts, shared by the layers), one 2D-multicast
matmul each. Batch > 1: per 32-row block, ctx = [previous block, own block] (64 rows,
attention.block_context) and one batched matmul with a constant selection matrix Sel [4*32, 64] gives
h_{t-4}, ..., h_{t-1} for every row. The k = 4 tap is h itself. Taps are accumulated in fp32, in
the same order on both paths.

Contract
    input   x [B, 1, S, 1024] bf16 TILE DRAM, S a multiple of 32
    output  [B, 1, S, 1024] bf16 TILE DRAM. Row t depends on rows <= t only, so padding rows (at the
            end) never reach valid rows.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.audio.attention import CTX, block_context
from models.demos.blackhole.embedding_gemma2.tt.audio.clippable_linear import ClippableLinear
from models.demos.blackhole.embedding_gemma2.tt.audio.config import (
    ATTN_BLOCK,
    TOKEN_BUCKETS,
    AudioConfig,
    audio_act_memory_config,
)
from models.demos.blackhole.embedding_gemma2.tt.audio.linear import audio_linear
from models.demos.blackhole.embedding_gemma2.tt.audio.program_configs import block_bmm_config
from models.demos.blackhole.embedding_gemma2.tt.config import compute_kernel_config_hifi4
from models.demos.blackhole.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, substate


def causal_shift_matrix(seq_len: int, shift: int) -> torch.Tensor:
    """[1, 1, S, S] 0/1 matrix M with (M h)_t = h_{t - shift} (0 for t < shift)."""
    m = torch.zeros(seq_len, seq_len)
    t = torch.arange(shift, seq_len)
    m[t, t - shift] = 1.0
    return m.reshape(1, 1, seq_len, seq_len)


def load_causal_shifts(device, kernel: int, cache_dir: Path | None = None) -> dict[int, list[ttnn.Tensor]]:
    """Per token bucket S, the kernel - 1 shift matrices (shift kernel-1 .. 1, i.e. tap k = 0 .. kernel-2),
    bf16 on device. Shared by the 12 lconv layers (AudioModel builds them once): 7.5 MB in all."""
    taps = kernel - 1
    return {
        S: [
            as_weight(
                causal_shift_matrix(S, taps - k),
                device,
                cache_dir=cache_dir,
                name=f"audio.causal_shift_s{S}_{taps - k}",
            )
            for k in range(taps)
        ]
        for S in TOKEN_BUCKETS
    }


def causal_selection(kernel: int) -> torch.Tensor:
    """Sel [1, 1, (kernel-1)*32, 64]: block row k*32 + i picks context row 32 + i - (kernel-1) + k."""
    taps = kernel - 1
    sel = torch.zeros(taps * ATTN_BLOCK, 2 * ATTN_BLOCK)
    for k in range(taps):
        for i in range(ATTN_BLOCK):
            sel[k * ATTN_BLOCK + i, ATTN_BLOCK + i - taps + k] = 1.0
    return sel.reshape(1, 1, taps * ATTN_BLOCK, 2 * ATTN_BLOCK)


class AudioLightConv1d(LightweightModule):
    def __init__(
        self,
        config: AudioConfig,
        pre_norm: RMSNorm,
        linear_start: ClippableLinear,
        taps: list[ttnn.Tensor],
        selection: ttnn.Tensor,
        conv_norm: RMSNorm,
        linear_end: ClippableLinear,
        shifts: dict[int, list[ttnn.Tensor]],
    ):
        super().__init__()
        self.config = config
        self.pre_norm = pre_norm
        self.linear_start = linear_start
        self.taps = taps
        self.selection = selection
        self.shifts = shifts
        self.conv_norm = conv_norm
        self.linear_end = linear_end
        self.compute_kernel_config = compute_kernel_config_hifi4()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: AudioConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "audio.lconv1d",
        act_dtype=ttnn.bfloat16,
        shifts: dict[int, list[ttnn.Tensor]] | None = None,
    ) -> "AudioLightConv1d":
        """`state_dict` holds the HF `layers.<i>.lconv1d.` sub-state (`.linear.` folded). `act_dtype`:
        output dtype of linear_start / linear_end, see AudioModel. `shifts`: load_causal_shifts()
        output to share between layers (loaded here when None)."""
        H, K = config.hidden_size, config.conv_kernel_size
        kw = dict(cache_dir=cache_dir)
        pre = RMSNorm.from_state_dict(
            substate(state_dict, "pre_layer_norm"), config.norm(), device, name=f"{name}.pre", **kw
        )
        cnorm = RMSNorm.from_state_dict(
            substate(state_dict, "conv_norm"), config.norm(), device, name=f"{name}.conv_norm", **kw
        )
        start = ClippableLinear.from_state_dict(
            substate(state_dict, "linear_start"), H, 2 * H, device, name=f"{name}.start", out_dtype=act_dtype, **kw
        )
        end = ClippableLinear.from_state_dict(
            substate(state_dict, "linear_end"), H, H, device, name=f"{name}.end", out_dtype=act_dtype, **kw
        )
        w = state_dict["depthwise_conv1d.weight"]
        if tuple(w.shape) != (H, 1, K):
            raise ValueError(f"{name}.depthwise_conv1d: expected {(H, 1, K)}, got {tuple(w.shape)}")
        taps = [
            as_weight(
                w[:, 0, k].float().reshape(1, 1, 1, H),
                device,
                cache_dir=cache_dir,
                name=f"{name}.tap{k}",
                dtype=ttnn.float32,
            )
            for k in range(K)
        ]
        sel = as_weight(causal_selection(K), device, cache_dir=cache_dir, name=f"audio.causal_sel_k{K}")
        if shifts is None:
            shifts = load_causal_shifts(device, K, cache_dir)
        return cls(config, pre, start, taps, sel, cnorm, end, shifts)

    def _causal_depthwise_full(self, h: ttnn.Tensor) -> ttnn.Tensor:
        """Batch 1: h_{t-s} = M_s h with the [S, S] shift matrices (2D multicast matmul, exact in bf16),
        then the same fp32 tap products and sums in the same order as the block path."""
        S = h.shape[2]
        taps = len(self.taps) - 1
        mc = audio_act_memory_config(h.device(), h.shape, ttnn.float32)
        acc = ttnn.multiply(h, self.taps[taps], dtype=ttnn.float32, memory_config=mc)  # k = taps: h_t itself
        for k in range(taps):
            sk = audio_linear(
                self.shifts[S][k], h, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config
            )
            term = ttnn.multiply(sk, self.taps[k], dtype=ttnn.float32, memory_config=mc)
            ttnn.deallocate(sk)
            nxt = ttnn.add(acc, term, memory_config=mc)
            ttnn.deallocate(term)
            ttnn.deallocate(acc)
            acc = nxt
        return acc

    def _causal_depthwise(self, h: ttnn.Tensor) -> ttnn.Tensor:
        """h [B, 1, S, D] bf16 -> fp32 [B, 1, S, D] causal depthwise conv."""
        B, _, S, D = h.shape
        if B == 1 and S in self.shifts:
            return self._causal_depthwise_full(h)
        dev = h.device()
        nb = S // ATTN_BLOCK
        taps = len(self.taps) - 1
        mc = audio_act_memory_config(dev, h.shape, ttnn.float32)
        ctx = block_context(h)  # [B, nb, 64, D]
        # both operands batched, so the block matmul config spreads the blocks over the grid
        sel = ttnn.repeat(self.selection, ttnn.Shape([B, nb, 1, 1]))
        # bf16 output: every output is one bf16 h value times 1.0, so bf16 holds it exactly (E5c; was
        # fp32, same values, twice the bytes for the slices below)
        shifted = ttnn.matmul(
            sel,
            ctx,
            dtype=ttnn.bfloat16,
            program_config=block_bmm_config(dev, taps * ATTN_BLOCK, CTX, D, per_core_M=1),
            compute_kernel_config=self.compute_kernel_config,
            memory_config=audio_act_memory_config(dev, (B, nb, taps * ATTN_BLOCK, D)),
        )  # [B, nb, taps*32, D]
        ttnn.deallocate(sel)
        ttnn.deallocate(ctx)
        acc = ttnn.multiply(h, self.taps[taps], dtype=ttnn.float32, memory_config=mc)  # k = taps: h_t itself
        acc = ttnn.reshape(acc, (B, nb, ATTN_BLOCK, D))
        for k in range(taps):
            sk = ttnn.slice(
                shifted,
                (0, 0, k * ATTN_BLOCK, 0),
                (B, nb, (k + 1) * ATTN_BLOCK, D),
                memory_config=audio_act_memory_config(dev, h.shape),
            )
            term = ttnn.multiply(sk, self.taps[k], dtype=ttnn.float32, memory_config=mc)
            ttnn.deallocate(sk)
            nxt = ttnn.add(acc, term, memory_config=mc)
            ttnn.deallocate(term)
            ttnn.deallocate(acc)
            acc = nxt
        ttnn.deallocate(shifted)
        return ttnn.reshape(acc, (B, 1, S, D))

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        D = self.config.hidden_size
        B, _, S, _ = x.shape
        if S % ATTN_BLOCK != 0:
            raise ValueError(f"audio lconv needs S % {ATTN_BLOCK} == 0, got {S}")
        # E5c: intermediates in L1 when they fit (config.audio_act_memory_config)
        dev = x.device()
        mc = audio_act_memory_config(dev, x.shape, ttnn.bfloat16)
        mc32 = audio_act_memory_config(dev, x.shape, ttnn.float32)
        h = self.pre_norm(x, memory_config=mc)
        g = self.linear_start(h)
        ttnn.deallocate(h)
        a = ttnn.slice(g, (0, 0, 0, 0), (B, 1, S, D), memory_config=mc)
        b = ttnn.slice(g, (0, 0, 0, D), (B, 1, S, 2 * D), memory_config=mc)
        ttnn.deallocate(g)
        # a * sigmoid(b) with the sigmoid as the multiply's SFPU pre-activation on b (ttnn.sigmoid's
        # default: vector mode RC = 4, accurate = 0)
        sigmoid = ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 4.0, 0.0)
        glu = ttnn.multiply(a, b, input_tensor_b_activations=[sigmoid], memory_config=mc)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        y = self._causal_depthwise(glu)
        ttnn.deallocate(glu)
        n = self.conv_norm(y, memory_config=mc32)
        ttnn.deallocate(y)
        # silu and linear_end's input clamp in one SFPU pass
        s = ttnn.unary_chain(
            n, [ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), self.linear_end.input_clamp()], memory_config=mc32
        )
        ttnn.deallocate(n)
        e = self.linear_end.project(s)
        ttnn.deallocate(s)
        out = ttnn.add(x, e, memory_config=mc)
        ttnn.deallocate(e)
        return out
