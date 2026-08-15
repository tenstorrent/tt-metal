# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `codec` (`VoxtralCodecDecoder`) -- Block 3, audio codes -> 24 kHz waveform.

    codes [T, 37] -> quantizer.decode -> [1, T, 292]
      -> CausalConv1d(292->1024, k3, replicate)
      -> 4 x { Transformer(2 layers, ALiBi + causal + sliding window 2/4/8/16)
               [+ CausalConvTranspose1d(k4, s2) for the first three] }   = 8x upsample
      -> CausalConv1d(1024->240, k7, reflect) -> unpatch = [1, 1, T*1920]

Everything is kept CHANNELS-LAST ([B, L, C]) end to end, which is what the reference's own
docstring suggests: the transformer stages want [B, L, C] anyway, so the conv blocks' [B, C, L]
transposes disappear and the length axis stays on dim 1 -- where `ttnn.slice` and `ttnn.concat`
are exact.  (Measured: `ttnn.concat` along the LAST dim at a non-tile-aligned width silently
rounds to bfloat16, 4e-4 relative.  Along dim 1 it is bit-exact.  The one place the semantic and
acoustic latents would have to be concatenated on the channel axis is avoided entirely by
splitting `decoder_blocks.0.conv`'s weight into its 256- and 36-channel halves and summing the
two partial convolutions.)

The convolutions are written as shifted matmuls rather than `ttnn.conv1d`: at T=8 frames the
tensors are far below the conv kernel's efficient regime, and this form keeps the causal padding
(replicate for the input conv, reflect for the output projection) explicit and exact.

Three details of this checkpoint that a generic codec port gets wrong:
  * `norm_eps` is 1e-2 for the codec's RMSNorms -- three orders off the usual 1e-5, and load
    bearing -- while q_norm/k_norm use 1e-6.
  * q_norm / k_norm normalise the FULL 1024-wide projection BEFORE the head split, not per head.
  * each residual branch carries a learned LayerScale vector.

`ttnn.embedding` requires a bfloat16 table, which would put 4e-3 of error on the semantic latent.
The table is therefore staged as a hi/lo bfloat16 PAIR and looked up twice: `hi + lo` recovers the
fp32 value to ~1e-7 for the cost of one extra gather.
"""

from __future__ import annotations

import math

import torch

import ttnn

from models.demos.voxtral_tts_full.tt_common import (
    COMPUTE_CONFIG,
    NEG_INF,
    stage,
    stage_weight,
    tt_gqa_attention,
    tt_linear,
    tt_merge_heads,
    tt_rms_norm,
    tt_split_heads,
    tt_swiglu,
)

DIM = 1024
N_HEADS = 8
N_KV_HEADS = 8  # MHA, no grouping
HEAD_DIM = 128
NORM_EPS = 1e-2  # params.json "norm_eps": 0.01 -- deliberate, not a typo
QK_NORM_EPS = 1e-6
SEMANTIC_DIM = 256
ACOUSTIC_DIM = 36
LEVELS = 21  # FSQ levels per acoustic dim
N_AUDIO_SPECIAL = 2
PATCH_SIZE = 240
PATCH_PROJ_KERNEL = 7

TF_BLOCKS = (1, 3, 5, 7)
TF_LAYERS = 2
UP_BLOCKS = (2, 4, 6)
WINDOWS = (2, 4, 8, 16)

# Longest transformer sequence the ALiBi/window bias tables cover.  The last stage runs at
# 8 x n_frames, so 512 admits inputs up to 64 frames (5.1 s at 12.5 Hz); the captured input is 8.
MAX_STAGE_SEQ = 512


def _alibi_slopes(n_heads):
    """`get_alibi_slopes`: geometric ratio 2^(-8/n) -> [1, 1/2, 1/4, ...] for n = 8."""
    if math.log2(n_heads).is_integer():
        r = 2.0 ** (-8.0 / n_heads)
        return torch.tensor([r ** i for i in range(n_heads)], dtype=torch.float32)
    m = 2 ** math.floor(math.log2(n_heads))
    r1, r2 = 2.0 ** (-8.0 / m), 2.0 ** (-8.0 / (2 * m))
    return torch.tensor([r1 ** i for i in range(m)] + [r2 ** i for i in range(0, 2 * m, 2)][: n_heads - m])


def _attention_bias_table(window, max_seq=MAX_STAGE_SEQ):
    """ALiBi + causal + sliding window as ONE additive [1, H, S, S] pre-softmax bias.

    Every entry depends only on `rel = j - i`, so the length-S bias is the top-left corner of
    this table and one table per window serves every call (a probed forward cannot build it)."""
    pos = torch.arange(max_seq)
    rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).float()
    bias = _alibi_slopes(N_HEADS).view(N_HEADS, 1, 1) * rel.unsqueeze(0)
    outside = (rel > 0) | (rel < -window)
    return bias.masked_fill(outside.unsqueeze(0), NEG_INF).unsqueeze(0)


class TtVoxtralCodecDecoder:
    def __init__(self, device, w):
        self.device = device
        self.__dict__.update(w)

    # ------------------------------------------------------------------ build (not probed)
    @classmethod
    def build(cls, device, torch_module):
        w = torch_module._as_dict()
        w = {k: v.detach().float() for k, v in w.items()}

        # --- semantic codebook, as a hi/lo bfloat16 pair (ttnn.embedding needs bf16)
        table = w["semantic_embedding"]
        hi = table.to(torch.bfloat16)
        lo = (table - hi.float()).to(torch.bfloat16)
        emb = (
            ttnn.from_torch(hi, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
            ttnn.from_torch(lo, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        )

        # --- decoder_blocks.0.conv [1024, 292, 3], split on the input-channel axis so the
        #     semantic and acoustic latents never need a channel-axis concat.
        c0 = w["decoder_blocks.0.conv.weight"]
        conv_in = [
            (stage_weight(c0[:, :SEMANTIC_DIM, j], device), stage_weight(c0[:, SEMANTIC_DIM:, j], device))
            for j in range(c0.shape[2])
        ]

        # --- CausalConvTranspose1d weights [in, out, 4]: `F.conv_transpose1d` contracts the
        #     INPUT-channel axis, so W[:, :, j] is already [in, out] and needs no transpose.
        conv_up = [
            [stage_weight(w[f"decoder_blocks.{b}.conv.weight"][:, :, j], device, transpose=False)
             for j in range(4)]
            for b in UP_BLOCKS
        ]

        op = w["output_proj.conv.weight"]  # [240, 1024, 7]
        conv_out = [stage_weight(op[:, :, j], device) for j in range(op.shape[2])]

        layers = {}
        for b in TF_BLOCKS:
            for li in range(TF_LAYERS):
                p = f"decoder_blocks.{b}.layers.{li}."
                layers[(b, li)] = {
                    "an": stage(w[p + "attention_norm.weight"].view(1, 1, -1), device),
                    "wq": stage_weight(w[p + "attention.wq.weight"], device),
                    "wk": stage_weight(w[p + "attention.wk.weight"], device),
                    "wv": stage_weight(w[p + "attention.wv.weight"], device),
                    "wo": stage_weight(w[p + "attention.wo.weight"], device),
                    "qn": stage(w[p + "attention.q_norm.weight"].view(1, 1, -1), device),
                    "kn": stage(w[p + "attention.k_norm.weight"].view(1, 1, -1), device),
                    "ascale": stage(w[p + "attention_scale"].view(1, 1, -1), device),
                    "fn": stage(w[p + "ffn_norm.weight"].view(1, 1, -1), device),
                    "w1": stage_weight(w[p + "feed_forward.w1.weight"], device),
                    "w2": stage_weight(w[p + "feed_forward.w2.weight"], device),
                    "w3": stage_weight(w[p + "feed_forward.w3.weight"], device),
                    "fscale": stage(w[p + "ffn_scale"].view(1, 1, -1), device),
                }

        return cls(device, {
            "emb": emb,
            "conv_in": conv_in,
            "conv_up": conv_up,
            "conv_out": conv_out,
            "layers": layers,
            "bias": [stage(_attention_bias_table(win), device) for win in WINDOWS],
            "zero_row": stage(torch.zeros(1, 1, DIM), device),
        })

    # ------------------------------------------------------------------ forward (probed)
    @staticmethod
    def _pad_left(x, source_rows):
        """Left-pad the LENGTH axis by copying rows already in `x` (replicate / reflect are both
        just index maps).  `ttnn.pad` cannot front-pad a tiled tensor on device; concat can."""
        rows = [ttnn.slice(x, [0, i, 0], [1, i + 1, x.shape[2]]) for i in source_rows]
        return ttnn.concat(rows + [x], dim=1)

    def _conv_in(self, sem, ac, length):
        """CausalConv1d(292 -> 1024, k=3, s=1, replicate), summed over the two channel halves."""
        sem_p = self._pad_left(sem, [0, 0])
        ac_p = self._pad_left(ac, [0, 0])
        y = None
        for j, (w_sem, w_ac) in enumerate(self.conv_in):
            part = ttnn.add(
                tt_linear(ttnn.slice(sem_p, [0, j, 0], [1, j + length, SEMANTIC_DIM]), w_sem),
                tt_linear(ttnn.slice(ac_p, [0, j, 0], [1, j + length, ACOUSTIC_DIM]), w_ac),
            )
            y = part if y is None else ttnn.add(y, part)
        return y

    def _conv_out(self, x, length):
        """CausalConv1d(1024 -> 240, k=7, s=1, reflect): left pad i -> x[6 - i]."""
        xp = self._pad_left(x, [6, 5, 4, 3, 2, 1])
        y = None
        for j, wj in enumerate(self.conv_out):
            part = tt_linear(ttnn.slice(xp, [0, j, 0], [1, j + length, DIM]), wj)
            y = part if y is None else ttnn.add(y, part)
        return y

    def _upsample(self, x, weights, length):
        """CausalConvTranspose1d(k=4, s=2) with the (k - stride) trim taken off the RIGHT.

        For stride 2 the transposed convolution splits cleanly by output parity:
            out[2s]   = x[s] @ W0 + x[s-1] @ W2
            out[2s+1] = x[s] @ W1 + x[s-1] @ W3
        so the whole op is two shifted matmul pairs plus an interleave, and the right trim is
        simply never computed."""
        shifted = ttnn.concat([self.zero_row, ttnn.slice(x, [0, 0, 0], [1, length - 1, DIM])], dim=1)
        even = ttnn.add(tt_linear(x, weights[0]), tt_linear(shifted, weights[2]))
        odd = ttnn.add(tt_linear(x, weights[1]), tt_linear(shifted, weights[3]))
        # [1, L, 2C] -> [1, 2L, C] interleaves the two phases along the length axis.
        return ttnn.reshape(ttnn.concat([even, odd], dim=-1), (1, 2 * length, DIM))

    def _transformer(self, x, block, seq, window_idx):
        bias = ttnn.slice(self.bias[window_idx], [0, 0, 0, 0], [1, N_HEADS, seq, seq])
        for li in range(TF_LAYERS):
            w = self.layers[(block, li)]
            h = tt_rms_norm(x, w["an"], NORM_EPS)
            # QK-norm runs over the full n_heads*head_dim width, BEFORE the head split.
            q = tt_rms_norm(tt_linear(h, w["wq"]), w["qn"], QK_NORM_EPS)
            k = tt_rms_norm(tt_linear(h, w["wk"]), w["kn"], QK_NORM_EPS)
            v = tt_linear(h, w["wv"])
            attn = tt_gqa_attention(
                tt_split_heads(q, N_HEADS, HEAD_DIM),
                tt_split_heads(k, N_KV_HEADS, HEAD_DIM),
                tt_split_heads(v, N_KV_HEADS, HEAD_DIM),
                bias, N_HEADS, N_KV_HEADS, HEAD_DIM, seq,
            )
            x = ttnn.add(x, ttnn.mul(w["ascale"], tt_linear(tt_merge_heads(attn), w["wo"])))
            h = tt_rms_norm(x, w["fn"], NORM_EPS)
            x = ttnn.add(x, ttnn.mul(w["fscale"], tt_swiglu(h, w["w1"], w["w2"], w["w3"])))
        return x

    def __call__(self, codes):
        # `codes` is [T, 37] uint32 WITH the special-token offset, exactly what the reference's
        # `strip_offset_and_trim` receives.  Its END_AUDIO cut is host-side generation control
        # (where to stop emitting), not arithmetic; the offset subtraction below is the part that
        # belongs to this block.
        t = ttnn.sub(ttnn.typecast(ttnn.to_layout(codes, ttnn.TILE_LAYOUT), ttnn.float32),
                     float(N_AUDIO_SPECIAL))
        frames = t.shape[0]

        sem_idx = ttnn.to_layout(
            ttnn.typecast(ttnn.transpose(ttnn.slice(t, [0, 0], [frames, 1]), 0, 1), ttnn.uint32),
            ttnn.ROW_MAJOR_LAYOUT,
        )
        # `ttnn.embedding` returns the table's dtype, so both halves are widened to float32
        # BEFORE they are summed -- adding them in bfloat16 would round the correction straight
        # back off and leave the plain 4e-3 lookup error.
        sem = ttnn.add(
            ttnn.typecast(ttnn.embedding(sem_idx, self.emb[0], layout=ttnn.TILE_LAYOUT), ttnn.float32),
            ttnn.typecast(ttnn.embedding(sem_idx, self.emb[1], layout=ttnn.TILE_LAYOUT), ttnn.float32),
        )

        # FSQ is parameter-free: code -> code * 2/(levels-1) - 1, the exact inverse of Block 2's
        # quantisation.
        ac = ttnn.reshape(ttnn.slice(t, [0, 1], [frames, 1 + ACOUSTIC_DIM]), (1, frames, ACOUSTIC_DIM))
        ac = ttnn.sub(ttnn.mul(ac, 2.0 / (LEVELS - 1)), 1.0)

        x = self._conv_in(sem, ac, frames)
        seq = frames
        for stage_idx, block in enumerate(TF_BLOCKS):
            x = self._transformer(x, block, seq, stage_idx)
            if stage_idx < len(UP_BLOCKS):
                x = self._upsample(x, self.conv_up[stage_idx], seq)
                seq *= 2

        x = self._conv_out(x, seq)  # [1, seq, 240]
        return ttnn.reshape(x, (1, 1, seq * PATCH_SIZE))


def build(device, torch_module=None):
    return TtVoxtralCodecDecoder.build(device, torch_module)


def codec_decoder(device, torch_module=None):
    return TtVoxtralCodecDecoder.build(device, torch_module)
