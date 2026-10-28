# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioLightConv1d (HF `Gemma4AudioLightConv1d`, modeling_gemma4.py:490-528).

    h   = glu( ClippableLinear_start( RMSNorm_pre(x) ) )          1024 -> 2048 -> a * sigmoid(b), 1024
    y_t = sum_{k=0..4} w[:, k] * h_{t-4+k}                       causal depthwise conv, h_{<0} = 0, no bias
    out = x + ClippableLinear_end( silu( RMSNorm_conv(y) ) )      (clamp +-1e10 before conv_norm: a no-op)

Causal shift on device: per 32-row block, ctx = [previous block, own block] (64 rows, attention.block_context)
and one matmul with a constant 0/1 selection matrix Sel [4*32, 64] gives h_{t-4}, ..., h_{t-1} for every
row (exact: one nonzero per row, fp32 output). The k = 4 tap is h itself. Taps are accumulated in fp32.

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
from models.demos.blackhole.embedding_gemma2.tt.audio.config import ATTN_BLOCK, AudioConfig
from models.demos.blackhole.embedding_gemma2.tt.audio.program_configs import block_bmm_config
from models.demos.blackhole.embedding_gemma2.tt.config import compute_kernel_config_hifi4
from models.demos.blackhole.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, substate


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
    ):
        super().__init__()
        self.config = config
        self.pre_norm = pre_norm
        self.linear_start = linear_start
        self.taps = taps
        self.selection = selection
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
    ) -> "AudioLightConv1d":
        """`state_dict` holds the HF `layers.<i>.lconv1d.` sub-state (`.linear.` folded). `act_dtype`:
        output dtype of linear_start / linear_end, see AudioModel."""
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
        return cls(config, pre, start, taps, sel, cnorm, end)

    def _causal_depthwise(self, h: ttnn.Tensor) -> ttnn.Tensor:
        """h [B, 1, S, D] bf16 -> fp32 [B, 1, S, D] causal depthwise conv."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        B, _, S, D = h.shape
        nb = S // ATTN_BLOCK
        taps = len(self.taps) - 1
        ctx = block_context(h)  # [B, nb, 64, D]
        # both operands batched, so the block matmul config spreads the blocks over the grid
        sel = ttnn.repeat(self.selection, ttnn.Shape([B, nb, 1, 1]))
        shifted = ttnn.matmul(
            sel,
            ctx,
            dtype=ttnn.float32,
            program_config=block_bmm_config(h.device(), taps * ATTN_BLOCK, CTX, D, per_core_M=2),
            compute_kernel_config=self.compute_kernel_config,
            memory_config=dram,
        )  # [B, nb, taps*32, D]
        ttnn.deallocate(sel)
        ttnn.deallocate(ctx)
        acc = ttnn.multiply(h, self.taps[taps], dtype=ttnn.float32, memory_config=dram)  # k = taps: h_t itself
        acc = ttnn.reshape(acc, (B, nb, ATTN_BLOCK, D))
        for k in range(taps):
            sk = ttnn.slice(shifted, (0, 0, k * ATTN_BLOCK, 0), (B, nb, (k + 1) * ATTN_BLOCK, D))
            term = ttnn.multiply(sk, self.taps[k], memory_config=dram)
            ttnn.deallocate(sk)
            nxt = ttnn.add(acc, term, memory_config=dram)
            ttnn.deallocate(term)
            ttnn.deallocate(acc)
            acc = nxt
        ttnn.deallocate(shifted)
        return ttnn.reshape(acc, (B, 1, S, D))

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        dram = ttnn.DRAM_MEMORY_CONFIG
        D = self.config.hidden_size
        B, _, S, _ = x.shape
        if S % ATTN_BLOCK != 0:
            raise ValueError(f"audio lconv needs S % {ATTN_BLOCK} == 0, got {S}")
        h = self.pre_norm(x)
        g = self.linear_start(h)
        ttnn.deallocate(h)
        a = ttnn.slice(g, (0, 0, 0, 0), (B, 1, S, D))
        b = ttnn.slice(g, (0, 0, 0, D), (B, 1, S, 2 * D))
        ttnn.deallocate(g)
        sb = ttnn.sigmoid(b, memory_config=dram)
        ttnn.deallocate(b)
        glu = ttnn.multiply(a, sb, memory_config=dram)
        ttnn.deallocate(a)
        ttnn.deallocate(sb)
        y = self._causal_depthwise(glu)
        ttnn.deallocate(glu)
        n = self.conv_norm(y)
        ttnn.deallocate(y)
        s = ttnn.silu(n, memory_config=dram)
        ttnn.deallocate(n)
        e = self.linear_end(s)
        ttnn.deallocate(s)
        out = ttnn.add(x, e, memory_config=dram)
        ttnn.deallocate(e)
        return out
