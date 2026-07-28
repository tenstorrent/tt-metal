# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Voxtral Codec DECODER (Block 3): audio codes -> 24 kHz waveform.

Mirrors reference/voxtral_codec_ref.py op-for-op. All compute runs on device; the only
host-side step is the semantic codebook gather (see _quantizer_host):

    codes [1,37,T] --quantizer--> [1,292,T] --conv(k3,replicate)--> [1,1024,T]
      --4x { Transformer(2 layers, ALiBi + causal + sliding window) [+ ConvTranspose(k4,s2)] }
      --output_proj(k7,reflect)--> [1,240,T'] --unpatch--> [1,1,T'*240] @ 24 kHz

The signal stays a channels-last [1,1,L,C] device tensor across conv stages and [1,L,C] across
transformer stages, so there is no host round-trip between ops — only codes in, waveform out.

LAYOUT / OP NOTES (what the reference's torch ops map to here):
  * `ttnn.conv1d` is native but ZERO-pad only, while every conv here is a CAUSAL left-pad with
    reflect/replicate. `_pad_causal` builds those from slice+concat (the pad is only k-1, i.e. 2
    or 6 columns, so this is a handful of slices — ttnn has no flip).
  * conv_transpose1d does not exist; done via `ttnn.conv_transpose2d` with a singleton HEIGHT and
    length on the WIDTH axis, which the XTTS-v2 vocoder work showed is ~10x faster than mapping
    length to height (height slicing hits a circular-buffer/L1 clash).
  * ALiBi + causal + sliding window collapse into ONE additive pre-softmax bias, built on host
    and cached per (S, window, dtype) — it does not depend on the weights. Because it is a
    function of (j - i) only, it is constant along diagonals, which is what makes chunking work:
    one slab-sized bias serves every chunk of every utterance.
  * QK-norm is an RMSNorm over the FULL 1024-wide projection, BEFORE the head split.
  * LayerScale multiplies each residual branch by a learned [1024] vector.
  * `norm_eps` is 1e-2 here (not 1e-5) — from params.json, and load-bearing.

PRECISION: MEASURED, not inherited. fp32 accumulation (HiFi3 + fp32_dest_acc) is on throughout;
activations outside attention are always fp32. Swept on real weights with synthetic codes, which
stress the numerics harder than real speech does (the same effect the XTTS-v2 speaker encoder saw
on random input):

    weights  attention | PCC T=64 / 256 / 469        | warm T=64 / 256 / 469
    fp32     fp32      | 0.999747 0.999761 0.999425  |  46 /  85 / 163 ms
    bf16     fp32      | 0.999204 0.999505 0.998732  |  44 /  83 / 162 ms  <- FAILS the 0.999 gate
    fp32     bf16      | 0.999800 0.999865 0.999795  |  44 /  81 / 155 ms  <- DEFAULT
    bf16     bf16      | 0.999512 0.999661 0.999610  |  42 /  79 / 153 ms

bf16 ATTENTION is strictly good: best PCC of all four, slightly faster, and it halves the largest
tensor (the attention bias). Hence the default.

bf16 WEIGHTS are now strictly BAD and the knob is kept only for experiments. They cost real
accuracy -- on their own they fall below the gate at T=469 -- and they no longer buy speed: 44 vs
46 ms. An earlier sweep showed them ~20% faster, but that gap was conv weight PREPARATION cost
(which scales with weight bytes) being paid on every call. Once that was hoisted out of the
per-call path the speed argument disappeared entirely.

Also tested and made no difference: keeping the small per-channel tensors (RMSNorm weights,
QK-norm weights, LayerScale vectors) in fp32 while the matmul weights are bf16 -- PCC 0.999512
either way, so the bf16 loss is in the matmul weights, not the norms.

On REAL speech both bf16 variants are indistinguishable (PCC 0.999982, ASR WER 0.0%); the
differences above only appear on random synthetic codes. Synthetic is kept as the gate because it
is the conservative test.

Not carried over: the XTTS-v2 HiFi-GAN result that bf16 costs 0.91-0.96 PCC. That was a 34-conv
chain with bf16 ACTIVATIONS throughout; here attention output enters the residual scaled by
LayerScale (~0.01) and there are only 8 attention ops, so it does not transfer.

CURRENT PERFORMANCE (warm, N150, defaults): 43.6 ms for 5.1 s of audio (RTF 0.0085, 117x
real-time) up to 538 ms for 120 s (RTF 0.0045, 223x). For reference, upstream report RTF 0.103 for
their WHOLE pipeline on an H200, so this block is a few percent of the total budget.

MEMORY: prepared conv weights are cached and DEDUPLICATED BY CONTENT (see _prepared). There are
only 8 distinct layouts across all 5 convs and all 12 buckets, so the cache holds 98 MB rather
than the 730 MB that keying by length alone produced -- 0.8% of an N150's DRAM instead of 6.5%.
Plus 60.8 MB of host copies, kept so a new length can still be prepared.

Validate against the reference (bit-exact goldens for all 8 stages):
    TT_METAL_HOME=<repo> PYTHONPATH=<repo> python models/experimental/voxtral_tts/tt/ttnn_voxtral_codec.py
"""

import hashlib

import torch
import ttnn

from models.experimental.voxtral_tts.reference.voxtral_codec_ref import (
    alibi_slopes,
    decoder_window_sizes,
    load_codec_state,
)
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ACOUSTIC_CODEBOOK_SIZE,
    CODEC_DIM,
    CODEC_HEAD_DIM,
    CODEC_N_HEADS,
    CODEC_NORM_EPS,
    CODEC_QK_NORM_EPS,
    DEC_CONV_BLOCKS,
    DEC_CONV_KERNELS,
    DEC_CONV_STRIDES,
    DEC_TF_BLOCKS,
    DEC_TF_LENGTHS,
    DEFAULT_CKPT,
    PATCH_PROJ_KERNEL,
    PATCH_SIZE,
    SEMANTIC_DIM,
)

DTYPE = ttnn.float32
SCALE = CODEC_HEAD_DIM**-0.5
# The reference masks with -inf. On device a max-subtracting softmax would compute
# inf-inf -> NaN, so use a large finite negative instead: exp() underflows to exactly 0,
# and it is unambiguous against real ALiBi values (which reach only about -16).
MASK_NEG = -1e9
# bf16 for the attention interior + its bias: measured best-PCC AND faster, and halves the
# largest tensor. See the sweep in the module docstring.
ATTN_DTYPE = ttnn.bfloat16
# Chunked attention. `slab` must be TILE-ALIGNED: TILE_LAYOUT pads every dim to 32, so a slab of
# 272 would silently become 288 and waste a row and column of tiles. Chunking turns attention from
# O(S^2) into O(S*slab) -- the window is only 2..16, so an unchunked q@kT computes 382M scores per
# head at S=12000 of which ~17 per row survive the mask. Measured at S=12000 (before the conv
# weight-prep fix, so absolute numbers are now lower): warm 892 -> 497 ms, cold 10580 -> 1178 ms
# (all attention shapes become identical, so kernels compile once ever), mask 2304 -> 4.2 MB. Below chunk_min the full mask is already cheap and chunking loses a few
# percent to dispatch (T=64: 89 -> 97 ms), hence the threshold.
SLAB = 512
CHUNK_MIN = 512
# Length bucketing for the CONVS. Every conv's input length scales with T (T+2, T, 2T, 4T, 8T+6),
# so each distinct utterance length compiles 5 new conv programs -- measured at 1-5 s each, and
# T is whatever the model happens to generate, so a server never stops paying it (a ONE-frame
# difference cost 5.5 s vs 181 ms warm). Rounding T up to a grid caps the shape count.
# The trade is measured: bucketing costs 7-25% on WARM steady-state (padded compute) but on a
# realistic stream of 12 distinct lengths, total time went 120.93s (none) -> 25.47s (bucket 64)
# -> 1.66s (bucket 128), i.e. 73x. Warm-only regression is the wrong thing to optimise here since
# production never repeats a length. 128 gives 12 buckets for the model's ~1500-frame ceiling.
# Set None to disable (best if you genuinely decode one fixed length repeatedly).
BUCKET = 128

COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


class TtVoxtralCodecDecoder:
    """On-device codec decoder. __call__(codes [1,37,T] int64) -> waveform torch [1,1,T*1920]."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, weight_dtype=DTYPE, attn_dtype=ATTN_DTYPE,
                 slab=SLAB, chunk_min=CHUNK_MIN, bucket=BUCKET):
        """`weight_dtype` / `attn_dtype` exist so the precision question can be MEASURED rather
        than inherited. Activations outside attention are always fp32."""
        self.device = device
        self.attn_dtype = attn_dtype
        # Chunked attention: process `slab` positions at a time, keeping the last (slab - window).
        # `slab` is TILE-ALIGNED on purpose -- TILE_LAYOUT pads every dim to 32, so a slab of 272
        # would silently become 288 and waste a row and column of tiles.
        self.slab = slab
        self.chunk_min = chunk_min  # chunk only when S exceeds this; None = never chunk
        self.bucket = bucket  # round T up to this multiple before decoding; None = off
        # With PRE-PREPARED weights the op can no longer infer weights_dtype from a host tensor,
        # so it must be stated explicitly -- and the SAME config must go to prepare_* and to the
        # conv call, or the prepared layout will not match what the kernel expects.
        self.conv_cfg = ttnn.Conv1dConfig(weights_dtype=weight_dtype)
        self.convt_cfg = ttnn.Conv2dConfig(weights_dtype=weight_dtype)
        w = load_codec_state(ckpt_path)  # weight_norm already folded by the reference loader

        dev = lambda t: ttnn.from_torch(t.contiguous(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        vec = lambda t: dev(t.reshape(1, 1, -1))  # [C] -> [1,1,C] so it broadcasts over length
        lin = lambda t: dev(t.t())  # torch Linear [out,in] -> ttnn.linear wants [in,out]
        host = lambda t: ttnn.from_torch(t.contiguous(), dtype=weight_dtype)  # conv weights, see below

        self.semantic_host = w["semantic_embedding"].float()  # host gather; see quantizer_decode

        # --- convs ---
        # Weights are kept on HOST and prepared per (conv, input length) on first use, then cached.
        # WHY: ttnn.conv1d transforms and re-uploads its weights INSIDE the op, so without this it
        # redid that work for all 5 convs on EVERY call. Hoisting it out was worth 2.6x at T=128
        # (112.8 -> 43.7 ms) and dropped the host share of wall time from 88% to 24%.
        # The cache is keyed by LENGTH because prepared weights are NOT length-independent: the
        # prepared layout encodes the sharding/blocking scheme. Same shape at every length but
        # different VALUES -- a 512-prepared weight used at L=128 computes PCC 0.19 against torch,
        # so reusing one across lengths silently produces garbage.
        self.conv_host = {
            "in": host(w["decoder_blocks.0.conv.weight"].unsqueeze(2)),      # [1024,292,1,3]
            "out": host(w["output_proj.conv.weight"].unsqueeze(2)),          # [240,1024,1,7]
            **{f"up{i}": host(w[f"decoder_blocks.{i}.conv.weight"].unsqueeze(2))
               for i in DEC_CONV_BLOCKS[1:]},                               # [1024,1024,1,4]
        }
        self._prep_cache = {}   # (conv, length) -> prepared tensor (possibly SHARED)
        self._layouts = {}      # (conv, content hash) -> the one tensor of that layout

        # --- transformer layers ---
        self.layers = {}
        for bi, n_layers in zip(DEC_TF_BLOCKS, DEC_TF_LENGTHS):
            for li in range(n_layers):
                p = f"decoder_blocks.{bi}.layers.{li}."
                self.layers[(bi, li)] = {
                    "an": vec(w[p + "attention_norm.weight"]),
                    "fn": vec(w[p + "ffn_norm.weight"]),
                    "qn": vec(w[p + "attention.q_norm.weight"]),
                    "kn": vec(w[p + "attention.k_norm.weight"]),
                    "wq": lin(w[p + "attention.wq.weight"]),
                    "wk": lin(w[p + "attention.wk.weight"]),
                    "wv": lin(w[p + "attention.wv.weight"]),
                    "wo": lin(w[p + "attention.wo.weight"]),
                    "w1": lin(w[p + "feed_forward.w1.weight"]),
                    "w2": lin(w[p + "feed_forward.w2.weight"]),
                    "w3": lin(w[p + "feed_forward.w3.weight"]),
                    "as": vec(w[p + "attention_scale"]),
                    "fs": vec(w[p + "ffn_scale"]),
                }
        self._bias_cache = {}
        self._zero_cache = {}
        self._slopes = alibi_slopes(CODEC_N_HEADS)
        self.windows = decoder_window_sizes()

    # ----------------------------------------------------------------------------------
    # Conv weight preparation (hoisted out of the per-call path)
    # ----------------------------------------------------------------------------------
    def _prepared(self, name, in_c, out_c, kernel, stride, L, transpose):
        """Prepared weight for this conv AT THIS INPUT LENGTH, DEDUPLICATED BY CONTENT.

        Prepared layouts are length-specific, but they change at only ONE length threshold per
        conv -- and for `up6` and `out` they never change at all. Measured across all 12 buckets:

            conv   distinct layouts   lengths sharing one
            in            2           {128} {256..1536}
            up2           2           {128,256} {384..1536}
            up4           2           {128} {256..1536}
            up6           1           {128..1536}   all identical
            out           1           {128..1536}   all identical

        So keying only by length stored up to 12 BYTE-IDENTICAL copies: 8 distinct layouts held as
        60 tensors, 730 MB instead of 98 MB. Hashing the prepared bytes and sharing the tensor is
        pure deduplication -- the tensors are bit-identical, so there is no accuracy question.

        Cost is one host readback per newly-seen (conv, length), on top of the 5-24 ms preparation.
        Both are first-touch only, and the hot path is a plain dict hit."""
        key = (name, L)
        if key in self._prep_cache:
            return self._prep_cache[key]
        f = self._prep_conv_t if transpose else self._prep_conv
        w = f(self.conv_host[name], in_c, out_c, kernel, stride, L)
        digest = (name, hashlib.sha1(ttnn.to_torch(w).float().numpy().tobytes()).hexdigest())
        shared = self._layouts.get(digest)
        if shared is None:
            self._layouts[digest] = w
        else:
            ttnn.deallocate(w)  # a duplicate: free it rather than keep a 17 MB twin
            w = shared
        self._prep_cache[key] = w
        return w

    def prepared_weight_stats(self):
        """(entries, distinct layouts, MB held) -- for tests and for reporting the dedup win."""
        mb = 0.0
        for t in self._layouts.values():
            n = 1
            for d in t.shape:
                n *= d
            mb += n * (4 if t.get_dtype() == ttnn.float32 else 2) / 1e6
        return len(self._prep_cache), len(self._layouts), mb

    def _prep_conv(self, w_host, in_c, out_c, kernel, stride, L):
        return ttnn.prepare_conv_weights(
            weight_tensor=w_host, input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.TILE_LAYOUT, weights_format="OIHW",
            in_channels=in_c, out_channels=out_c, batch_size=1,
            input_height=1, input_width=L, kernel_size=(1, kernel),
            stride=(1, stride), padding=(0, 0), dilation=(1, 1), has_bias=False, groups=1,
            # input_dtype is the ACTIVATION dtype (always fp32 here), NOT the weight dtype --
            # passing weight_dtype prepared a layout for bf16 activations while the real
            # activations are fp32, which silently produced PCC 0.008.
            device=self.device, input_dtype=DTYPE, compute_config=COMPUTE_CONFIG,
            conv_config=self.conv_cfg,
        )

    def _prep_conv_t(self, w_host, in_c, out_c, kernel, stride, L):
        # transpose weights are IOHW (in, out, kh, kw) -- ConvTranspose1d's [in,out,k] unsqueezed
        return ttnn.prepare_conv_transpose2d_weights(
            weight_tensor=w_host, input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.TILE_LAYOUT, weights_format="IOHW",
            in_channels=in_c, out_channels=out_c, batch_size=1,
            input_height=1, input_width=L, kernel_size=(1, kernel),
            stride=(1, stride), padding=(0, 0), dilation=(1, 1), has_bias=False, groups=1,
            # input_dtype is the ACTIVATION dtype (always fp32 here), NOT the weight dtype --
            # passing weight_dtype prepared a layout for bf16 activations while the real
            # activations are fp32, which silently produced PCC 0.008.
            device=self.device, input_dtype=DTYPE, compute_config=COMPUTE_CONFIG,
            conv_config=self.convt_cfg,
        )

    # ----------------------------------------------------------------------------------
    # Attention bias: ALiBi + causal + sliding window, one additive [1,H,S,S] term
    # ----------------------------------------------------------------------------------
    def _attn_bias(self, S, window):
        key = (S, window, self.attn_dtype)
        if key not in self._bias_cache:
            pos = torch.arange(S)
            rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).float()  # rel[i,j] = j - i
            bias = self._slopes.view(-1, 1, 1) * rel.unsqueeze(0)
            bias = bias.masked_fill(rel.unsqueeze(0) > 0, MASK_NEG)  # causal
            bias = bias.masked_fill((rel < -window).unsqueeze(0), MASK_NEG)  # window
            self._bias_cache[key] = ttnn.from_torch(
                bias.unsqueeze(0).contiguous(), dtype=self.attn_dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        return self._bias_cache[key]

    # ----------------------------------------------------------------------------------
    # Causal padding (ttnn.pad is constant-only, and there is no flip)
    # ----------------------------------------------------------------------------------
    def _pad_causal(self, x, pad, mode):
        """x [1,1,L,C] -> [1,1,L+pad,C]. `mode` mirrors torch F.pad on the length axis:
        replicate repeats column 0; reflect mirrors about column 0, excluding it."""
        if pad == 0:
            return x
        L = x.shape[2]
        if mode == "replicate":
            first = ttnn.slice(x, [0, 0, 0, 0], [1, 1, 1, x.shape[3]])
            parts = [first] * pad
        elif mode == "reflect":
            assert pad < L, f"reflect pad {pad} needs length > {pad}, got {L}"
            parts = [ttnn.slice(x, [0, 0, i, 0], [1, 1, i + 1, x.shape[3]]) for i in range(pad, 0, -1)]
        else:
            raise ValueError(mode)
        return ttnn.concat(parts + [x], dim=2)

    def _conv1d(self, x, name, in_c, out_c, kernel, stride, pad_mode):
        """Causal conv1d over channels-last [1,1,L,C]. Padding is applied explicitly, so the op
        itself runs with padding=0."""
        pad_total = kernel - stride
        x = self._pad_causal(x, pad_total, pad_mode)
        L = x.shape[2]
        out = ttnn.conv1d(
            input_tensor=x, weight_tensor=self._prepared(name, in_c, out_c, kernel, stride, L, False),
            device=self.device,
            in_channels=in_c, out_channels=out_c, batch_size=1, input_length=L,
            kernel_size=kernel, stride=stride, padding=0, dilation=1, groups=1,
            compute_config=COMPUTE_CONFIG, conv_config=self.conv_cfg,
        )
        out = out[0] if isinstance(out, (tuple, list)) else out
        return ttnn.reshape(out, [1, 1, -1, out_c])

    def _conv_transpose(self, x, name, channels, kernel, stride):
        """Length on the WIDTH axis (kernel (1,k), stride (1,s)) — the XTTS-v2 lesson. Trims
        (k - stride) samples off the RIGHT, matching upstream's trim_ratio=1.0."""
        L = x.shape[2]
        out = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self._prepared(name, channels, channels, kernel, stride, L, True),
            device=self.device,
            in_channels=channels, out_channels=channels, batch_size=1,
            input_height=1, input_width=L, kernel_size=(1, kernel), stride=(1, stride),
            padding=(0, 0), output_padding=(0, 0), dilation=(1, 1), groups=1,
            compute_config=COMPUTE_CONFIG, conv_config=self.convt_cfg,
        )
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = ttnn.reshape(out, [1, 1, -1, channels])
        trim = kernel - stride
        return ttnn.slice(out, [0, 0, 0, 0], [1, 1, out.shape[2] - trim, channels])

    def _attention_slab(self, q, k, v, bias):
        """fp32 attention with an additive pre-softmax bias. [1,H,S,d] -> [1,H,S,d].

        NOT ttnn.transformer.scaled_dot_product_attention: that op is bf16/bfp8-only, and this is
        the front of a deep conv stack where the XTTS-v2 vocoder work showed accumulated error
        does not cancel. Done as matmul + softmax in fp32 instead, which also lets us keep
        `numeric_stable=True` — XTTS-v2 Block 1 found the default softmax leaves a structured,
        dominant-aligned error that downstream layers amplify."""
        if self.attn_dtype != DTYPE:
            c = lambda t: ttnn.typecast(t, self.attn_dtype)
            q, k, v = c(q), c(k), c(v)  # the BIAS is already cached in attn_dtype -- never cast the
            #                            big tensor per call, which is what made an earlier A/B slower
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=COMPUTE_CONFIG)
        scores = ttnn.add(ttnn.multiply(scores, SCALE), bias)
        attn = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_CONFIG)
        out = ttnn.matmul(attn, v, compute_kernel_config=COMPUTE_CONFIG)
        return ttnn.typecast(out, DTYPE) if self.attn_dtype != DTYPE else out

    def _zeros(self, H, pad, d):
        key = (H, pad, d)
        if key not in self._zero_cache:
            self._zero_cache[key] = ttnn.from_torch(
                torch.zeros(1, H, pad, d), dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        return self._zero_cache[key]

    def _attention(self, q, k, v, window):
        """[1,H,S,d] -> [1,H,S,d]. Chunks when S > chunk_min, else one full-S pass.

        EXACT, not an approximation: attention is causal AND windowed, so output[i] depends only
        on input[i-window .. i]. A slab starting `window` positions early therefore has all the
        context its kept rows need; the leading `window` rows are dropped because THEIR context is
        missing. Verified against full-S attention (max abs diff ~1e-7) and against the unchunked
        device path.

        EVERY CHUNK IS EXACTLY `slab` LONG, so there is ONE cached bias per window and attention
        sees one shape for the process lifetime. Two details buy that:
          * chunk 0 starts at lo=0, so it needs NO left context -- all `slab` rows are valid
            outputs (cut=0) and local index == absolute index, so the ordinary slab bias is
            already correct. No left padding, no special-case bias.
          * the final chunk is padded on the RIGHT to `slab`. Safe with the same bias because
            causal masking already forbids any real row from looking forward into the padding;
            the padding rows are computed and discarded.
        Without this, first/last chunk lengths varied (the last with S mod C), which meant a new
        bias AND a new kernel compilation for every distinct utterance length."""
        S = q.shape[2]
        if self.chunk_min is None or S <= self.chunk_min:
            return self._attention_slab(q, k, v, self._attn_bias(S, window))
        H, d = q.shape[1], q.shape[3]
        slab = self.slab
        C = slab - window
        bias = self._attn_bias(slab, window)  # the ONE tensor, reused by every chunk
        outs, a = [], 0
        while a < S:
            lo, cut = (0, 0) if a == 0 else (a - window, window)
            hi = min(S, lo + slab)
            sl = lambda t: ttnn.slice(t, [0, 0, lo, 0], [1, H, hi, d])
            qs, ks, vs = sl(q), sl(k), sl(v)
            if hi - lo < slab:  # final chunk: pad right so the shape stays slab-sized
                z = self._zeros(H, slab - (hi - lo), d)
                qs, ks, vs = (ttnn.concat([t, z], dim=2) for t in (qs, ks, vs))
            o = self._attention_slab(qs, ks, vs, bias)
            outs.append(ttnn.slice(o, [0, 0, cut, 0], [1, H, cut + (hi - a), d]))
            a = hi
        return outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=2)

    # ----------------------------------------------------------------------------------
    # Transformer block
    # ----------------------------------------------------------------------------------
    def _block(self, x, w, window):
        """x [1,L,1024] -> [1,L,1024]. Pre-norm, QK-norm on the full projection, LayerScale."""
        L = x.shape[1]
        h = ttnn.rms_norm(x, weight=w["an"], epsilon=CODEC_NORM_EPS, compute_kernel_config=COMPUTE_CONFIG)
        q = ttnn.linear(h, w["wq"], compute_kernel_config=COMPUTE_CONFIG)
        k = ttnn.linear(h, w["wk"], compute_kernel_config=COMPUTE_CONFIG)
        v = ttnn.linear(h, w["wv"], compute_kernel_config=COMPUTE_CONFIG)
        # QK-norm over the whole 1024 width, BEFORE splitting heads
        q = ttnn.rms_norm(q, weight=w["qn"], epsilon=CODEC_QK_NORM_EPS, compute_kernel_config=COMPUTE_CONFIG)
        k = ttnn.rms_norm(k, weight=w["kn"], epsilon=CODEC_QK_NORM_EPS, compute_kernel_config=COMPUTE_CONFIG)
        heads = lambda t: ttnn.permute(ttnn.reshape(t, [1, L, CODEC_N_HEADS, CODEC_HEAD_DIM]), (0, 2, 1, 3))
        attn = self._attention(heads(q), heads(k), heads(v), window)
        attn = ttnn.reshape(ttnn.permute(attn, (0, 2, 1, 3)), [1, L, CODEC_DIM])
        r = ttnn.linear(attn, w["wo"], compute_kernel_config=COMPUTE_CONFIG)
        x = ttnn.add(x, ttnn.multiply(r, w["as"]))  # LayerScale
        h = ttnn.rms_norm(x, weight=w["fn"], epsilon=CODEC_NORM_EPS, compute_kernel_config=COMPUTE_CONFIG)
        g = ttnn.silu(ttnn.linear(h, w["w1"], compute_kernel_config=COMPUTE_CONFIG))
        u = ttnn.multiply(g, ttnn.linear(h, w["w3"], compute_kernel_config=COMPUTE_CONFIG))
        r = ttnn.linear(u, w["w2"], compute_kernel_config=COMPUTE_CONFIG)
        return ttnn.add(x, ttnn.multiply(r, w["fs"]))

    # ----------------------------------------------------------------------------------
    # Quantizer (decode side)
    # ----------------------------------------------------------------------------------
    def _quantizer_host(self, codes):
        """codes torch [1,37,T] -> HOST torch [1,1,T,292] channels-last.

        Semantic is a table lookup, acoustic is pure FSQ arithmetic. Kept on host: ttnn.embedding
        needs a BFLOAT16 table and the semantic entries are large (|x| ~ 10), so a bf16 table
        would inject ~0.4% before a deep conv stack that does not cancel error. Split out from the
        upload so the upload target is explicit."""
        T = codes.shape[2]
        sem = self.semantic_host[codes[:, 0, :].reshape(-1).long()].reshape(1, T, SEMANTIC_DIM)
        ac = codes[:, 1:, :].to(torch.float32) * 2.0 / (ACOUSTIC_CODEBOOK_SIZE - 1) - 1.0
        lat = torch.cat([sem, ac.permute(0, 2, 1)], dim=2)
        return lat.reshape(1, 1, T, SEMANTIC_DIM + 36).contiguous()

    def quantizer_decode(self, codes):
        """Host quantizer + upload, as one step (the eager path and the tests use this)."""
        return ttnn.from_torch(self._quantizer_host(codes), dtype=DTYPE,
                               layout=ttnn.TILE_LAYOUT, device=self.device)

    # ----------------------------------------------------------------------------------
    # The device-only op sequence
    # ----------------------------------------------------------------------------------
    def _graph(self, x, stages=None):
        """latents [1,1,T,292] on device -> [1,1,T',240] on device."""
        x = self._conv1d(x, "in", SEMANTIC_DIM + 36, CODEC_DIM,
                         DEC_CONV_KERNELS[0], DEC_CONV_STRIDES[0], "replicate")
        if stages is not None:
            stages["after_input_conv"] = self._chw(x)
        for stage, (tf_i, n_layers) in enumerate(zip(DEC_TF_BLOCKS, DEC_TF_LENGTHS)):
            L = x.shape[2]
            seq = ttnn.reshape(x, [1, L, CODEC_DIM])
            for li in range(n_layers):
                seq = self._block(seq, self.layers[(tf_i, li)], self.windows[stage])
            x = ttnn.reshape(seq, [1, 1, L, CODEC_DIM])
            if stages is not None:
                stages[f"after_tf{tf_i}"] = self._chw(x)
            if stage < len(DEC_CONV_BLOCKS) - 1:
                ci = DEC_CONV_BLOCKS[stage + 1]
                x = self._conv_transpose(x, f"up{ci}", CODEC_DIM,
                                         DEC_CONV_KERNELS[stage + 1], DEC_CONV_STRIDES[stage + 1])
                if stages is not None:
                    stages[f"after_up{ci}"] = self._chw(x)
        return self._conv1d(x, "out", CODEC_DIM, PATCH_SIZE, PATCH_PROJ_KERNEL, 1, "reflect")

    @torch.no_grad()
    def __call__(self, codes, return_stages=False):
        # return_stages BYPASSES bucketing on purpose: it exists to bisect against the
        # reference's per-stage goldens, and a bucketed run's stages are at the padded length
        # (T, 2T, 4T, 8T of the BUCKET), so they would not correspond. Trimming each stage
        # separately would work but is error-prone for a debug-only path.
        if self.bucket and not return_stages:
            T = codes.shape[2]
            padded = -(-T // self.bucket) * self.bucket
            if padded > T:
                # repeat the LAST frame rather than zero-pad: the tail then looks like plausible
                # audio to the causal convs instead of a hard edge. It is trimmed off either way,
                # but the transposed convs overlap, so a pathological tail is worth avoiding.
                codes = torch.cat([codes, codes[:, :, -1:].repeat(1, 1, padded - T)], dim=2)
                out = self._decode(codes, return_stages)
                wav, stages = out if return_stages else (out, {})
                wav = wav[:, :, : T * PATCH_SIZE * 8]
                return (wav, stages) if return_stages else wav
        return self._decode(codes, return_stages)

    @torch.no_grad()
    def _decode(self, codes, return_stages=False):
        lat_host = self._quantizer_host(codes)
        stages = {} if return_stages else None
        xd = ttnn.from_torch(lat_host, dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=self.device)
        x = self._graph(xd, stages)
        # unpatch: channels-last [1,1,T',240] flattens (t, c) with c fastest == the reference's
        # permute(0,2,1).reshape(B,1,T'*240)
        out = ttnn.to_torch(x).float().reshape(1, 1, -1)
        return (out, stages) if return_stages else out

    @staticmethod
    def _chw(x):
        """device [1,1,L,C] -> torch [1,C,L], to compare against the reference's layout."""
        return ttnn.to_torch(x).float().reshape(1, x.shape[2], x.shape[3]).permute(0, 2, 1)


def main():
    import time

    from models.experimental.voxtral_tts.reference import voxtral_codec_ref as ref
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import pcc

    device = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        gen = TtVoxtralCodecDecoder(device)
        w = ref.load_codec_state()
        for n_frames in (8, 24):
            codes = ref.make_synthetic_codes(n_frames)
            exp_lat = ref.quantizer_decode(codes, w)
            exp_wav = ref.reference_decode(codes, w)
            got_wav, stages = gen(codes, return_stages=True)

            tag = f"[codec T={n_frames}]"
            got_lat = TtVoxtralCodecDecoder._chw(gen.quantizer_decode(codes))
            print(f"\n{tag} {'quantizer':22s} PCC {pcc(got_lat, exp_lat):.6f}")
            x = ref.causal_conv1d(exp_lat, w["decoder_blocks.0.conv.weight"], 3, 1, "replicate")
            print(f"{tag} {'after_input_conv':22s} PCC {pcc(stages['after_input_conv'], x):.6f}")
            for stage, tf_i in enumerate(ref.DEC_TF_BLOCKS):
                x = ref.codec_transformer(x.permute(0, 2, 1), w, tf_i, 2,
                                          ref.decoder_window_sizes()[stage]).permute(0, 2, 1)
                name = f"after_tf{tf_i} (win {ref.decoder_window_sizes()[stage]})"
                print(f"{tag} {name:22s} PCC {pcc(stages[f'after_tf{tf_i}'], x):.6f}")
                if stage < 3:
                    ci = ref.DEC_CONV_BLOCKS[stage + 1]
                    x = ref.causal_conv_transpose1d(x, w[f"decoder_blocks.{ci}.conv.weight"], 4, 2)
                    print(f"{tag} {f'after_up{ci}':22s} PCC {pcc(stages[f'after_up{ci}'], x):.6f}")
            print(f"{tag} {'WAVEFORM':22s} PCC {pcc(got_wav, exp_wav):.6f}  "
                  f"shapes {tuple(got_wav.shape)} vs {tuple(exp_wav.shape)}")
            # the staged run above uses return_stages=True, which BYPASSES bucketing -- so also
            # exercise the DEFAULT path (bucketed) that real callers get.
            plain = gen(codes)
            print(f"{tag} {'bucketed (default path)':22s} PCC {pcc(plain, exp_wav):.6f}  "
                  f"shape {tuple(plain.shape)}")
            t0 = time.perf_counter()
            gen(codes)
            print(f"{tag} warm {(time.perf_counter() - t0) * 1000:.1f} ms")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
