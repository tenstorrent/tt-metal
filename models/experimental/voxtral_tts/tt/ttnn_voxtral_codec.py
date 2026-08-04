# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Voxtral Codec DECODER (Block 3): audio codes -> 24 kHz waveform.

Mirrors reference/voxtral_codec_ref.py op-for-op. All compute runs on device; the only host-side
step is the semantic codebook gather (see _quantizer_host):

    codes [1,37,T] --quantizer--> [1,292,T] --conv(k3,replicate)--> [1,1024,T]
      --4x { Transformer(2 layers, ALiBi + causal + sliding window) [+ ConvTranspose(k4,s2)] }
      --output_proj(k7,reflect)--> [1,240,T'] --unpatch--> [1,1,T'*240] @ 24 kHz

Channel width is a constant 1024 through every upsample; only the final projection narrows it, to
240, and those 240 channels then BECOME time (each frame carries a 240-sample waveform patch).
The signal stays a channels-last [1,1,L,C] device tensor across conv stages and [1,L,C] across
transformer stages, so there is no host round-trip between ops -- codes in, waveform out.

=== HOW THE REFERENCE'S TORCH OPS MAP HERE ===
  * `ttnn.conv1d` is ZERO-pad only, while every conv here is a CAUSAL left-pad with
    reflect/replicate. `_pad_causal` builds those from slice+concat (the pad is k-1, i.e. 2 or 6
    columns, so it is a handful of slices -- ttnn has no flip).
  * conv_transpose1d does not exist; done via `ttnn.conv_transpose2d` with a singleton HEIGHT and
    length on the WIDTH axis, which the XTTS-v2 vocoder work showed is ~10x faster than mapping
    length to height (height slicing hits a circular-buffer/L1 clash).
  * ALiBi + causal + sliding window collapse into ONE additive pre-softmax bias, built on host and
    cached per (S, window, dtype); it does not depend on the weights. Being a function of (j - i)
    only, it is constant along diagonals -- which is exactly what lets one slab-sized bias serve
    every chunk of every utterance.
  * QK-norm is an RMSNorm over the FULL 1024-wide projection, BEFORE the head split.
  * LayerScale multiplies each residual branch by a learned [1024] vector.
  * `norm_eps` is 1e-2 here (not 1e-5) -- from params.json, and load-bearing.

=== PRECISION: MEASURED, NOT INHERITED ===
fp32 accumulation (HiFi3 + fp32_dest_acc_en) is on throughout; activations outside attention are
always fp32. The two knobs were swept on real weights, and scored on BOTH metrics -- PCC, and the
worst single sample as a % of peak, which is what the ear notices and what PCC can hide (see
"trap: PCC hides outliers" below).

    weights  attn   | synthetic worst-sample %peak  | real speech        | warm ms
                    |  T=64    T=256   T=469        |  PCC       worst%  | 64/256/469
    fp32     fp32   |  8.95    8.32    29.08        |  0.999988  0.81%   | 46/85/163
    bf16     fp32   | 13.66   14.90    49.78        |  0.999986  1.38%   | 44/83/162
    fp32     bf16   | 10.45    8.66    11.56        |  0.999984  1.16%   | 44/81/156  <- DEFAULT
    bf16     bf16   | 13.87   10.88    25.16        |  0.999983  1.93%   | 42/79/154

bf16 ATTENTION is the default and wins on both metrics: best PCC of the four (0.999800/0.999865/
0.999795 on synthetic), best synthetic worst-sample at T=469 by a wide margin, faster, and it
halves the largest tensor (the attention bias). fp32 attention is slightly better on real speech
(0.81% vs 1.16%) but is 2.5x worse on synthetic at T=469 and ~5% slower, and synthetic is
deliberately the conservative gate.

bf16 WEIGHTS are BAD and there is no longer a knob for them. On their own they fall below the
0.999 PCC gate at T=469, and they do not buy speed either (44 vs 46 ms) -- an earlier sweep showed
~20%, but that was conv weight PREPARATION cost, since hoisted out of the per-call path. And
bf16+bf16 measures 1.93% worst-sample against a 2.00% gate, i.e. 3.5% of margin, for a ~1% gain.
If you want to re-open this, re-run the real-speech fixture, not just the synthetic one.

Made no difference: keeping the small per-channel tensors (RMSNorm / QK-norm weights, LayerScale)
in fp32 while matmul weights are bf16 -- PCC 0.999512 either way, so the bf16 loss is in the matmul
weights, not the norms.

The fp32 -> bf16 conversion of q/k/v is itself worth 3.70e-03 of worst-case error, about half of
the hand-rolled path's total 5.85e-03 (internal math alone: 4.22e-03). Real but not the dominant
term, and the sweep above already prices it in.

Not carried over: the XTTS-v2 HiFi-GAN result that bf16 costs 0.91-0.96 PCC. That was a 34-conv
chain with bf16 ACTIVATIONS throughout; here attention output enters the residual scaled by
LayerScale (~0.01) and there are only 8 attention ops, so it does not transfer.

=== PERFORMANCE (warm, N150, defaults) ===
27.3 ms for 5.1 s of audio (RTF 0.0053, 188x real-time), 97.0 ms for 37.5 s (387x), 345 ms for
120 s (348x). Upstream report RTF 0.103 for their WHOLE pipeline on an H200, so this block is
~3-5% of the end-to-end budget. It is NOT where the end-to-end answer gets decided -- Block 1 is
(87% of the parameters, 12.5 sequential steps per second of audio).

Per-block profile after the head fusions (L=4096, one of 8 blocks), which is what says the block is
now arithmetic-bound rather than movement-bound:

    norms  qkv   qkn   split  attn   merge  wo    mlp   resid | TOTAL
    0.46   1.71  0.45  0.95   9.68   0.21   0.58  7.87  1.07  | 22.97 ms

attn (42%) + mlp (34%) = 76% is real matmul work. split and merge were 11.67 and 4.93 ms before
optimization #7; they are now 0.95 and 0.21. Nothing movement-shaped is left to remove.

=== OPTIMIZATIONS APPLIED, AND WHAT EACH WAS WORTH ===
  1. bf16 attention (above): best accuracy of the four configs AND faster.
  2. Chunked windowed attention (_attention), slab 512: O(S^2) -> O(S*slab). At S=12000, warm
     892 -> 497 ms, cold 10580 -> 1178 ms, mask 2304 MB -> 4.2 MB. EXACT, not approximate.
  3. Uniform slab-sized chunks: one cached bias per window, and one attention shape for the
     process lifetime. Bias cache 23 tensors/53 MB -> 6/18.1 MB, and it stops growing with
     utterance length (measured keys: (128,2) (256,4) (512,2) (512,4) (512,8) (512,16)).
  4. Conv length bucketing (BUCKET): on a stream of 12 distinct lengths, 120.9 s -> 1.66 s (73x).
  5. Hoisted conv weight preparation (_prepared): 2.6x at short lengths (112.8 -> 43.7 ms at
     T=128); host share of wall time 88% -> 24%.
  6. Content-deduplicated prepared weights (_prepared): 730 MB -> 98 MB, bit-identical.
  7. FUSED head split/merge in _block: 1.6x on the WHOLE block (155 -> 97 ms at T=469). The
     reshape+permute pair was 41% of Block 3 -- larger than attention -- and pure data movement.
     nlp_create_qkv_heads 11.57 -> 0.95 ms and concatenate_heads 4.88 -> 0.20 ms at L=4096.
     Accuracy is unchanged (real speech PCC 0.999984, worst 1.16% of peak, both identical).
     Found late: the six rejected items below all targeted attention or the convs, and the
     reshapes were never profiled. On this hardware the wins are in op count and data movement,
     not arithmetic -- every FLOP-reducing idea here lost, and this one won.

=== MEASURED AND REJECTED -- do not retry without new information ===
  * sdpa instead of the hand-rolled attention interior: 1.44-2.27x FASTER but 3.3x worse
    worst-case error, failing 11 tests. Eight levers exhausted including a tt-metal source patch.
    Full detail and the exhaustion list in _attention_slab's docstring.
  * Smaller slab, toward the compute optimum 2*window: slab=32 computes 9x FEWER scores and runs
    9x SLOWER (334 vs 36 ms per decoder pass). Per-kernel cost dominates arithmetic here.
  * Device trace capture: 1.00x at every slab size, including 3570 ops. The async command queue
    already hides host dispatch behind device execution, so there is nothing to recover. (Trap:
    "time is in the TTNN wrapper" != "time is dispatch".)
  * Batching the chunks into one matmul: the batched attention is 2.4x faster, but building the
    stacked tensor costs more than that saves (11.30 vs 9.67 ms). The chunks overlap by `window`,
    so the gather is an unavoidable copy -- ttnn has no strided view.
  * Unchunked attention with a full [S,S] mask: identical accuracy (so chunking really is exact),
    but 3x slower at S=4096 and the mask grows quadratically to 268 MB. Only S<=1024 prefers it,
    which is a `chunk_min` question, not a chunking one -- see CHUNK_MIN.
  * Fused SwiGLU MLP (one [1024, 2*4096] matmul + ttnn.swiglu, weight ordered w3|w1): 0.77-0.86x,
    i.e. SLOWER, and maxabs 3.4e-02 vs the current path at short L so it is not even equivalent.
    Note swiglu needs 4D input. The MLP is ~116 GMAC at L=4096 -- arithmetic-bound, nothing to fuse.
  * Fused QKV projection (one [1024, 3072] matmul + 3 slices instead of three [1024, 1024]
    matmuls): 0.74-0.82x. Same FLOPs, and it BUYS data movement -- the exact mirror of why
    optimization #7 won. maxabs also drifts to 2.4e-03 at long L, so not a free swap either.
  * Fused residual+norm (`rms_norm(x, residual_input_tensor=r)` instead of add-then-norm): this one
    is genuinely FASTER, 1.54-1.73x, but the base is small -- 0.17 ms per site, 2 sites x 8 blocks
    = ~2.7 ms of 97 ms (2.8%) -- and maxabs is 4.4e-03 on the RESIDUAL path, which every later layer
    inherits. Not taken: a 2.8% gain does not justify perturbing the residual stream. Revisit only
    with the real-speech worst-sample fixture as the gate.

=== TRAPS ===
  * PCC HIDES OUTLIERS. It is a correlation: it can sit at 0.9998 while individual samples are
    badly wrong, and for audio the outliers are what you hear. Every accuracy claim here is
    therefore paired with a worst-sample bound, and the real-speech fixture asserts both.
  * Prepared conv weights are NOT length-independent -- same shape, different bytes. See _prepared.
  * `prepare_conv_*`'s `input_dtype` is the ACTIVATION dtype. See _prep_weight.

=== MEMORY ===
Prepared conv weights are cached and deduplicated by content: 8 distinct layouts across all 5
convs and all 12 buckets, so 98 MB rather than the 730 MB that keying by length alone produced
(0.8% of an N150's DRAM instead of 6.5%). Plus 60.8 MB of host copies, kept so a new length can
still be prepared, and 18.1 MB of attention bias (6 tensors).

Validate against the reference (per-stage PCC bisect + the default bucketed path):
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
    NUM_CODEBOOKS,
    PATCH_PROJ_KERNEL,
    PATCH_SIZE,
    SEMANTIC_DIM,
)

# Quantizer output width: one semantic embedding plus one scalar per acoustic codebook.
LATENT_DIM = SEMANTIC_DIM + (NUM_CODEBOOKS - 1)  # 256 + 36 = 292
# Total length gain across the three stride-2 transposed convs (12.5 Hz frames -> 100 Hz).
UPSAMPLE = 1
for _s in DEC_CONV_STRIDES:
    UPSAMPLE *= _s  # 1 * 2 * 2 * 2 = 8

DTYPE = ttnn.float32
SCALE = CODEC_HEAD_DIM**-0.5
# The reference masks with -inf. On device a max-subtracting softmax would compute
# inf-inf -> NaN, so use a large finite negative instead: exp() underflows to exactly 0,
# and it is unambiguous against real ALiBi values (which reach only about -16).
MASK_NEG = -1e9
# bf16 for the attention interior + its bias: measured best-PCC AND faster, and halves the largest
# tensor. Everything OUTSIDE attention stays DTYPE. See the sweep in the module docstring.
ATTN_DTYPE = ttnn.bfloat16
# Chunked attention -- see OPTIMIZATIONS #2/#3. `slab` MUST be tile-aligned: TILE_LAYOUT pads every
# dim to 32, so a slab of 272 would silently become 288 and waste a row and column of tiles.
# 512 is measured-optimal; both smaller and larger lose (see MEASURED AND REJECTED).
SLAB = 512
# Chunk only above this length. Below it the full mask is already cheap and chunking loses a few
# percent to per-op cost (T=64: 89 -> 97 ms). Measured crossover is S ~ 2000, so 1024 would be
# slightly better (1.06x on attention, ~0.1% end-to-end) at the cost of a 16.8 MB bias.
CHUNK_MIN = 512
# Conv length bucketing -- see OPTIMIZATIONS #4. Every conv's input length scales with T, so each
# distinct utterance length compiles 5 new conv programs at 1-5 s each, and T is whatever the model
# generates -- a ONE-frame difference cost 5.5 s vs 181 ms warm. Rounding T up caps the shape count:
# costs 7-25% on warm steady-state (padded compute), which production never sees, and 128 gives 12
# buckets for the model's ~1500-frame ceiling. Set None if you truly decode one fixed length.
# NOTE: 128 is wrong for STREAMING -- a 1-second chunk then costs the same as a 10-second one.
BUCKET = 128
# Rows in the output projection's reflected prefix. It only NEEDS PATCH_PROJ_KERNEL-1 = 6, but a
# 6-row prefix makes the following concat land off a tile boundary, and the ragged version costs
# 1.815 ms against 0.281 for the aligned one. So pad the prefix to a full 32-row tile and start the
# output slices at OUT_PREFIX-6 instead of 0. The extra 26 rows are copies of x[0]; they are finite,
# they feed the matmul, and every output row they touch is sliced away again. See _graph.
OUT_PREFIX = 32

COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


class TtVoxtralCodecDecoder:
    """On-device codec decoder. __call__(codes [1,37,T] int64) -> waveform torch [1,1,T*1920]."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, slab=SLAB, chunk_min=CHUNK_MIN, bucket=BUCKET):
        """Precision is not a parameter: fp32 weights, fp32 activations outside attention, bf16
        inside it. That combination was measured against the other three (module docstring) and the
        losers were deleted rather than left switchable."""
        self.device = device
        self.slab = slab  # attention chunk width; see the SLAB constant for why 512
        self.chunk_min = chunk_min  # chunk only when S exceeds this; None = never chunk
        self.bucket = bucket  # round T up to this multiple before decoding; None = off
        # With PRE-PREPARED weights the op can no longer infer weights_dtype from a host tensor,
        # so it must be stated explicitly -- and the SAME config must go to prepare_* and to the
        # conv call, or the prepared layout will not match what the kernel expects.
        self.conv_cfg = ttnn.Conv1dConfig(weights_dtype=DTYPE)
        self.convt_cfg = ttnn.Conv2dConfig(weights_dtype=DTYPE)
        w = load_codec_state(ckpt_path)  # weight_norm already folded by the reference loader

        dev = lambda t: ttnn.from_torch(t.contiguous(), dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
        vec = lambda t: dev(t.reshape(1, 1, -1))  # [C] -> [1,1,C] so it broadcasts over length
        lin = lambda t: dev(t.t())  # torch Linear [out,in] -> ttnn.linear wants [in,out]
        host = lambda t: ttnn.from_torch(t.contiguous(), dtype=DTYPE)  # conv weights stay on host

        self.semantic_host = w["semantic_embedding"].float()  # host gather; see _quantizer_host
        # Per-tap weights for the output projection, which does NOT use ttnn.conv1d -- see _graph's
        # last lines for why. torch stores the conv as [out, in, k]; ttnn.linear wants [in, out].
        self._out_taps = [dev(w["output_proj.conv.weight"][:, :, j].t())
                          for j in range(PATCH_PROJ_KERNEL)]
        # ...and the reflected prefix those taps slide over, as a GATHER INDEX rather than six
        # single-row slices. Row OUT_PREFIX-6+m of the prefix takes x[6-m], giving x6,x5,x4,x3,x2,x1;
        # rows 0..OUT_PREFIX-7 take x[0] and are discarded. Length-independent, so it is built once.
        idx = torch.zeros(1, 1, OUT_PREFIX, CODEC_DIM, dtype=torch.int32)
        for m in range(PATCH_PROJ_KERNEL - 1):
            idx[0, 0, OUT_PREFIX - (PATCH_PROJ_KERNEL - 1) + m, :] = (PATCH_PROJ_KERNEL - 1) - m
        self._out_prefix_idx = ttnn.from_torch(
            idx.contiguous(), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)

        # --- convs ---
        # Weights stay on HOST here and are prepared on first use by `_prepared`, which also
        # deduplicates them -- see that method for the layout table and the memory numbers.
        # WHY at all: ttnn.conv1d transforms and re-uploads its weights INSIDE the op, so without
        # this it redid that work for all 5 convs on EVERY call -- see OPTIMIZATIONS #5 for what
        # hoisting it was worth. 60.8 MB of host copies, kept so a new length can still be prepared.
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
        w = self._prep_weight(self.conv_host[name], in_c, out_c, kernel, stride, L, transpose)
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

    def _prep_weight(self, w_host, in_c, out_c, kernel, stride, L, transpose):
        """One wrapper for both prepare_conv_weights and prepare_conv_transpose2d_weights: the two
        take an identical kwarg set and differ only in the weight layout they expect
        (ConvTranspose1d's [in,out,k] is IOHW; Conv1d's [out,in,k] is OIHW).

        `input_dtype` is the ACTIVATION dtype (always DTYPE here), NOT the weight dtype. Back when
        weights were switchable, passing the WEIGHT dtype here prepared a layout for bf16
        activations while the real activations were fp32, and silently produced PCC 0.008."""
        fn = ttnn.prepare_conv_transpose2d_weights if transpose else ttnn.prepare_conv_weights
        return fn(
            weight_tensor=w_host, input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.TILE_LAYOUT, weights_format="IOHW" if transpose else "OIHW",
            in_channels=in_c, out_channels=out_c, batch_size=1,
            input_height=1, input_width=L, kernel_size=(1, kernel),
            stride=(1, stride), padding=(0, 0), dilation=(1, 1), has_bias=False, groups=1,
            device=self.device, input_dtype=DTYPE, compute_config=COMPUTE_CONFIG,
            conv_config=self.convt_cfg if transpose else self.conv_cfg,
        )

    # ----------------------------------------------------------------------------------
    # Attention bias: ALiBi + causal + sliding window as ONE additive term.
    # [1,H,S,S] on the unchunked path; [1,H,slab,slab] once chunking applies, which is the
    # normal case and is why it stays small (4.2 MB) at any utterance length.
    # ----------------------------------------------------------------------------------
    def _attn_bias(self, S, window):
        key = (S, window, ATTN_DTYPE)
        if key not in self._bias_cache:
            pos = torch.arange(S)
            rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).float()  # rel[i,j] = j - i
            bias = self._slopes.view(-1, 1, 1) * rel.unsqueeze(0)
            bias = bias.masked_fill(rel.unsqueeze(0) > 0, MASK_NEG)  # causal
            bias = bias.masked_fill((rel < -window).unsqueeze(0), MASK_NEG)  # window
            self._bias_cache[key] = ttnn.from_torch(
                bias.unsqueeze(0).contiguous(), dtype=ATTN_DTYPE, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        return self._bias_cache[key]

    # ----------------------------------------------------------------------------------
    # Causal padding (ttnn.pad is constant-only, and there is no flip)
    # ----------------------------------------------------------------------------------
    def _pad_causal(self, x, pad, mode):
        """x [1,1,L,C] -> [1,1,L+pad,C]. `mode` mirrors torch F.pad on the length axis:
        replicate repeats column 0; reflect mirrors about column 0, excluding it.

        Production only reaches `replicate` now -- it needs one slice, so it is cheap. `reflect`
        needs `pad` of them and cost 3.07 ms in the output projection, which builds its prefix with
        ttnn.gather instead (see _graph). The branch stays because it is the faithful mirror of
        F.pad and test_codec_ttnn_pcc checks both modes against torch."""
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
        """Attention with an additive pre-softmax bias, in ATTN_DTYPE. [1,H,S,d] -> [1,H,S,d].

Hand-rolled rather than ttnn.transformer.scaled_dot_product_attention, which was
        measured and rejected. sdpa DOES accept an arbitrary additive mask (`attn_mask` with
        `is_causal=False`), and dropped into this chunk loop with the same slab bias it is
        1.44-2.27x FASTER (25.1 -> 17.0 ms per decoder pass) because the fused kernel never
        materialises the [slab,slab] scores in DRAM, where this path writes them four times.

        It loses on ACCURACY. Against an exact fp64 answer at S=512:

            path                 PCC vs exact   max abs err   mean abs err
            hand-rolled (this)   0.99999559     5.85e-03      3.98e-04
            sdpa                 0.99985772     1.95e-02      2.35e-03   <- 3.3x worse worst-case

        Adopting it failed 11 tests, including the real-speech fixture's worst-sample bound -- the
        gate that guards what is audible -- and per-stage PCC 0.916 after one 2-layer stage. The
        per-slab PCC was a healthy 0.9998, so this is the PCC-hides-outliers trap firing exactly.

        DO NOT RE-LITIGATE without new information. Every lever below leaves the worst-case error at
        exactly 1.951e-02, which is why the cause is believed to be the compute kernel's arithmetic:
          * chunk geometry -- 1 vs 4 k-blocks; chunked vs unchunked full mask
          * `exp_approx_mode=False`; `fp32_dest_acc_en=True`; HiFi4 (`use_high_precision_compute`)
          * fp32 q/k/v -- rejected outright, sdpa is bf16/bfp8/bfp4 only
          * patching sdpa_program_factory.cpp so im_df/stats_df are Float32, and REBUILDING.
            Confirmed live via a marker (im_df=Float32, fp32_dest_acc_en=true) -- error unchanged.
            So closed issue #13364 ("Enable FP32 Accumulate in Flash Attention") is a red herring
            here. Isolating the inputs confirms it too: with a bf16-rounded reference (input
            conversion costing zero) sdpa's internal error is 2.07e-02 against our 4.22e-03.
        Reaching it would mean editing kernels/compute/sdpa.cpp. Re-test only if that changes.

        Separately, sdpa forbids `attn_mask` together with `is_causal` or `sliding_window_size`
        (explicit TT_FATALs), so its native block-skipping is unreachable. That costs almost nothing:
        its windowed path measured only 1.22x faster than this chunking and cannot express ALiBi at
        all (PCC 0.64 without it).

        Hand-rolling also keeps `numeric_stable=True` on the softmax. Runs in ATTN_DTYPE (bf16) --
        see the PRECISION table in the module docstring."""
        c = lambda t: ttnn.typecast(t, ATTN_DTYPE)
        q, k, v = c(q), c(k), c(v)  # the BIAS is already cached in ATTN_DTYPE -- never cast the big
        #                             tensor per call, which is what made an earlier A/B slower
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=COMPUTE_CONFIG)
        scores = ttnn.add(ttnn.multiply(scores, SCALE), bias)
        attn = ttnn.softmax(scores, dim=-1, numeric_stable=True, compute_kernel_config=COMPUTE_CONFIG)
        out = ttnn.matmul(attn, v, compute_kernel_config=COMPUTE_CONFIG)
        return ttnn.typecast(out, DTYPE)

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
        # Head split/merge via the FUSED ops, not reshape+permute. Measured at L=4096: the split
        # went 11.57 -> 0.95 ms (12x) and the merge 4.88 -> 0.20 ms (24x). Reshape+permute was the
        # single largest cost in this block -- larger than attention itself -- and it is pure data
        # movement. `nlp_create_qkv_heads` wants q separate and k|v fused, and 4D inputs (the
        # reshape to [1,1,L,C] is metadata only); it takes q/k ALREADY QK-normed, so it does not
        # conflict with normalising over the full 1024 width before the split.
        kv = ttnn.concat([k, v], dim=-1)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(q, [1, 1, L, CODEC_DIM]),
            ttnn.reshape(kv, [1, 1, L, 2 * CODEC_DIM]),
            num_heads=CODEC_N_HEADS, num_kv_heads=CODEC_N_HEADS,
            transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn = self._attention(qh, kh, vh, window)
        attn = ttnn.transformer.concatenate_heads(attn)  # [1,H,L,d] -> [1,L,H*d]
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
        return lat.reshape(1, 1, T, LATENT_DIM).contiguous()

    def quantizer_decode(self, codes):
        """Host quantizer + upload, as one step (the eager path and the tests use this)."""
        return ttnn.from_torch(self._quantizer_host(codes), dtype=DTYPE,
                               layout=ttnn.TILE_LAYOUT, device=self.device)

    # ----------------------------------------------------------------------------------
    # The device-only op sequence
    # ----------------------------------------------------------------------------------
    def _graph(self, x, stages=None):
        """latents [1,1,T,292] on device -> [1,1,T',240] on device."""
        x = self._conv1d(x, "in", LATENT_DIM, CODEC_DIM,
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
        # OUTPUT PROJECTION AS MATMULS, NOT ttnn.conv1d -- and the reason is a ttnn BUG, not speed.
        #
        # `ttnn.conv1d` here was the exact op that made Block 1's w2 unusable in BFP8. Its
        # sliding_window `halo_gather` kernel issues an out-of-range NOC write (13,897,728 bytes to
        # a nonexistent core) on the SECOND execution of this shape -- a program-cache hit -- and
        # hangs the card. Full dump in ttnn_voxtral_pipeline; STATUS.md 6.12 has the investigation.
        #
        # A k=7 stride-1 conv over an ALREADY-PADDED tensor is just a sliding-window matmul:
        #     out[t] = sum_j  xpad[t+j] @ W[j]
        # so 7 slices, 7 matmuls and 6 adds compute it exactly, touching no halo kernel at all.
        #
        # SHIFT THE OUTPUT, NOT THE INPUT. Both orders compute the same sum, but the shift has to
        # be a slice, and slicing the 1024-wide INPUT costs 0.624 ms a time against 0.145 for the
        # 240-wide OUTPUT. Multiply the full padded input by each tap, THEN slice the narrow result:
        #     conv1d (broken)                              4.29 ms
        #     7 matmuls, shift the INPUT  (slice xp)        9.16 ms   +4.87
        #     7 matmuls, shift the OUTPUT (slice y)         6.26 ms   +1.98
        #     + GATHERED prefix instead of _pad_causal      3.45 ms   -0.84   <- this, BIT-IDENTICAL
        # The input slices were 4.37 of the 9.16 ms, more than the seven matmuls together (1.93).
        #
        # THE PAD WAS THE EXPENSE, NOT THE MATMULS -- 3.07 of the 6.26 ms. `_pad_causal` builds the
        # reflection with SIX single-row slices, and against a 16 MiB TILE_LAYOUT tensor one
        # single-row slice costs 0.381 ms whether it returns 1 row or 6:
        #     one single-row slice of x        0.381 ms      six of them   2.282 ms
        #     one SIX-row slice of x           0.358 ms      ragged concat 1.815 ms
        # Cost is per op, not per byte. So take the prefix in ONE aligned 32-row slice and do the
        # reversal with ttnn.gather (see _out_prefix_idx): 3 ops and 0.071 ms instead of 8 and 3.07.
        # Bit-identical output, verified max-abs-diff exactly 0 -- gather moves data, it does not
        # arithmetic. A permutation MATMUL also works and is equally fast, but loses 2.4e-04: fp32
        # matmul multiplies at bf16 precision here, and HiFi4 + fp32_dest_acc does not change it.
        #
        # FUSING THE TAPS was measured and NOT taken. Concatenating g taps into one [1024, g*256]
        # weight cuts the 7 matmuls to 7/g and reads xp once per group instead of once per tap:
        #     g=1 (this)  3.45 ms  bit-exact      g=3   2.94 ms  err 1.1e-04
        #     g=2         3.15 ms  err 3.6e-04    g=7   worse -- its 28 MiB output goes to DRAM
        # 0.51 ms more, at the price of exactness (a [1024,768] matmul decomposes differently from
        # three [1024,240] ones) on an op that is 0.01% of wall. Not a trade worth making here.
        # Note also that blocks must be 256-aligned: at pitch 240 the half-tile column offset comes
        # back SILENTLY WRONG out of L1 (rel err 5e-01, no exception raised).
        #
        # Not parallelisable, before anyone tries: the seven passes are independent, but every ttnn
        # op already uses the whole 64-core grid, so running them at once would just give each pass
        # 9 cores. Nothing is idle. And holding xp in L1 to avoid re-reading it does not fit --
        # 16.8 MB overflows the allocator.
        L = x.shape[2]
        assert L >= OUT_PREFIX, f"output projection needs >= {OUT_PREFIX} rows, got {L}"
        off = OUT_PREFIX - (PATCH_PROJ_KERNEL - 1)
        head = ttnn.slice(x, [0, 0, 0, 0], [1, 1, OUT_PREFIX, CODEC_DIM])
        xp = ttnn.concat([ttnn.gather(head, dim=2, index=self._out_prefix_idx), x], dim=2)
        acc = None
        for j in range(PATCH_PROJ_KERNEL):
            y = ttnn.linear(xp, self._out_taps[j], compute_kernel_config=COMPUTE_CONFIG,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
            sl = ttnn.slice(y, [0, 0, off + j, 0], [1, 1, off + j + L, PATCH_SIZE],
                            memory_config=ttnn.L1_MEMORY_CONFIG)
            acc = sl if acc is None else ttnn.add(acc, sl, memory_config=ttnn.L1_MEMORY_CONFIG)
        return acc

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
                # return_stages is False in this branch, so _decode returns the waveform alone.
                return self._decode(codes)[:, :, : T * PATCH_SIZE * UPSAMPLE]
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
