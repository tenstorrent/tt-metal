# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Voxtral Codec DECODER (Block 3): audio codes -> 24 kHz waveform.

Mirrors reference/voxtral_codec_ref.py op-for-op, running entirely on device:

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
  * ALiBi + causal + sliding window collapse into ONE additive pre-softmax bias, passed as
    sdpa's `attn_mask` with `is_causal=False` (the mask already encodes causality). The bias is
    built on host once per (S, window) and cached — it does not depend on the weights.
  * QK-norm is an RMSNorm over the FULL 1024-wide projection, BEFORE the head split.
  * LayerScale multiplies each residual branch by a learned [1024] vector.
  * `norm_eps` is 1e-2 here (not 1e-5) — from params.json, and load-bearing.

PRECISION: MEASURED, not inherited. fp32 accumulation (HiFi3 + fp32_dest_acc) is on throughout;
activations outside attention are always fp32. The two dtypes that matter were swept on real
weights (synthetic codes, which stress the numerics harder than real speech does -- same effect
the XTTS-v2 speaker encoder saw):

    weights  attention | mask@S=3752 | PCC T=64 / 256 / 469        | time T=64 / 256 / 469
    fp32     fp32      |      450 MB | 0.999732 0.999762 0.999428  |  91 / 137 / 220 ms
    bf16     fp32      |      450 MB | 0.999233 0.999503 0.998757  |  67 / 117 / 206 ms  <- FAILS 0.999
    fp32     bf16      |      225 MB | 0.999746 0.999864 0.999768  |  94 / 131 / 190 ms  <- DEFAULT
    bf16     bf16      |      225 MB | 0.999537 0.999665 0.999636  |  65 / 103 / 169 ms

So bf16 ATTENTION is strictly good -- best PCC of all four, ~14% faster at length, and it halves
the largest tensor -- hence the default. bf16 WEIGHTS buy a further ~20% but cost real accuracy
(and on their own drop below the 0.999 gate at T=469), so they stay opt-in via weight_dtype.

On REAL speech both bf16 variants are indistinguishable (PCC 0.999982, ASR WER 0.0%); the
differences above only appear on random synthetic codes. Synthetic is kept as the gate because it
is the conservative test.

Not carried over: the XTTS-v2 HiFi-GAN result that bf16 costs 0.91-0.96 PCC. That was a 34-conv
chain with bf16 ACTIVATIONS throughout; here attention output enters the residual scaled by
LayerScale (~0.01) and there are only 8 attention ops, so it does not transfer.

Validate against the reference (bit-exact goldens for all 8 stages):
    TT_METAL_HOME=<repo> PYTHONPATH=<repo> python models/experimental/voxtral_tts/tt/ttnn_voxtral_codec.py
"""

import math

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
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


class TtVoxtralCodecDecoder:
    """On-device codec decoder. __call__(codes [1,37,T] int64) -> waveform torch [1,1,T*1920]."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, weight_dtype=DTYPE, attn_dtype=ATTN_DTYPE):
        """`weight_dtype` / `attn_dtype` exist so the precision question can be MEASURED rather
        than inherited. Activations outside attention are always fp32."""
        self.device = device
        self.weight_dtype = weight_dtype
        self.attn_dtype = attn_dtype
        w = load_codec_state(ckpt_path)  # weight_norm already folded by the reference loader

        dev = lambda t: ttnn.from_torch(t.contiguous(), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        vec = lambda t: dev(t.reshape(1, 1, -1))  # [C] -> [1,1,C] so it broadcasts over length
        lin = lambda t: dev(t.t())  # torch Linear [out,in] -> ttnn.linear wants [in,out]
        # conv weights stay HOST in PyTorch layout; ttnn.conv1d / conv_transpose2d move them
        # themselves and cache the device copy per call signature.
        host = lambda t: ttnn.from_torch(t.contiguous(), dtype=weight_dtype)

        self.semantic_host = w["semantic_embedding"].float()  # host gather; see quantizer_decode

        # --- convs ---
        self.conv_in = host(w["decoder_blocks.0.conv.weight"].unsqueeze(2))  # [1024,292,1,3]
        self.ups = {i: host(w[f"decoder_blocks.{i}.conv.weight"].unsqueeze(2)) for i in DEC_CONV_BLOCKS[1:]}
        self.conv_out = host(w["output_proj.conv.weight"].unsqueeze(2))  # [240,1024,1,7]

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
        self._slopes = alibi_slopes(CODEC_N_HEADS)
        self.windows = decoder_window_sizes()

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

    def _conv1d(self, x, weight, in_c, out_c, kernel, stride, pad_mode):
        """Causal conv1d over channels-last [1,1,L,C]. Padding is applied explicitly, so the op
        itself runs with padding=0."""
        pad_total = kernel - stride
        x = self._pad_causal(x, pad_total, pad_mode)
        L = x.shape[2]
        out = ttnn.conv1d(
            input_tensor=x, weight_tensor=weight, device=self.device,
            in_channels=in_c, out_channels=out_c, batch_size=1, input_length=L,
            kernel_size=kernel, stride=stride, padding=0, dilation=1, groups=1,
            compute_config=COMPUTE_CONFIG,
        )
        out = out[0] if isinstance(out, (tuple, list)) else out
        return ttnn.reshape(out, [1, 1, -1, out_c])

    def _conv_transpose(self, x, weight, channels, kernel, stride):
        """Length on the WIDTH axis (kernel (1,k), stride (1,s)) — the XTTS-v2 lesson. Trims
        (k - stride) samples off the RIGHT, matching upstream's trim_ratio=1.0."""
        L = x.shape[2]
        out = ttnn.conv_transpose2d(
            input_tensor=x, weight_tensor=weight, device=self.device,
            in_channels=channels, out_channels=channels, batch_size=1,
            input_height=1, input_width=L, kernel_size=(1, kernel), stride=(1, stride),
            padding=(0, 0), output_padding=(0, 0), dilation=(1, 1), groups=1,
            compute_config=COMPUTE_CONFIG,
        )
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = ttnn.reshape(out, [1, 1, -1, channels])
        trim = kernel - stride
        return ttnn.slice(out, [0, 0, 0, 0], [1, 1, out.shape[2] - trim, channels])

    def _attention(self, q, k, v, bias):
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

    # ----------------------------------------------------------------------------------
    # Transformer block
    # ----------------------------------------------------------------------------------
    def _block(self, x, w, bias):
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
        attn = self._attention(heads(q), heads(k), heads(v), bias)
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
    def quantizer_decode(self, codes):
        """codes torch [1,37,T] (no special-token offset) -> device [1,1,T,292] channels-last.

        Semantic is a table lookup; acoustic is pure arithmetic (FSQ has no parameters).

        The semantic gather runs on HOST and is the one host-side step in this block.
        `ttnn.embedding` requires a BFLOAT16 table, and the semantic codebook entries are large
        (|x| ~ 10), so a bf16 table would inject ~0.4% relative error into the latents before a
        deep conv stack that we already know does not cancel accumulated error. An index_select
        is exact and free, and it only changes what we upload ([1,T,292] floats instead of [1,T]
        ints). A bf16 device path is available if the accuracy cost is ever measured and accepted."""
        T = codes.shape[2]
        sem = self.semantic_host[codes[:, 0, :].reshape(-1).long()].reshape(1, T, SEMANTIC_DIM)
        ac = codes[:, 1:, :].to(torch.float32) * 2.0 / (ACOUSTIC_CODEBOOK_SIZE - 1) - 1.0  # [1,36,T]
        lat = torch.cat([sem, ac.permute(0, 2, 1)], dim=2)  # [1,T,292] channels-last
        return ttnn.from_torch(
            lat.reshape(1, 1, T, SEMANTIC_DIM + 36).contiguous(),
            dtype=DTYPE, layout=ttnn.TILE_LAYOUT, device=self.device,
        )

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, codes, return_stages=False):
        stages = {}
        x = self.quantizer_decode(codes)  # [1,1,T,292]
        x = self._conv1d(x, self.conv_in, SEMANTIC_DIM + 36, CODEC_DIM,
                         DEC_CONV_KERNELS[0], DEC_CONV_STRIDES[0], "replicate")
        if return_stages:
            stages["after_input_conv"] = self._chw(x)
        for stage, (tf_i, n_layers) in enumerate(zip(DEC_TF_BLOCKS, DEC_TF_LENGTHS)):
            L = x.shape[2]
            seq = ttnn.reshape(x, [1, L, CODEC_DIM])
            bias = self._attn_bias(L, self.windows[stage])
            for li in range(n_layers):
                seq = self._block(seq, self.layers[(tf_i, li)], bias)
            x = ttnn.reshape(seq, [1, 1, L, CODEC_DIM])
            if return_stages:
                stages[f"after_tf{tf_i}"] = self._chw(x)
            if stage < len(DEC_CONV_BLOCKS) - 1:
                ci = DEC_CONV_BLOCKS[stage + 1]
                x = self._conv_transpose(x, self.ups[ci], CODEC_DIM,
                                         DEC_CONV_KERNELS[stage + 1], DEC_CONV_STRIDES[stage + 1])
                if return_stages:
                    stages[f"after_up{ci}"] = self._chw(x)
        x = self._conv1d(x, self.conv_out, CODEC_DIM, PATCH_SIZE, PATCH_PROJ_KERNEL, 1, "reflect")
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
            t0 = time.perf_counter()
            gen(codes)
            print(f"[codec T={n_frames}] warm {time.perf_counter() - t0:.3f}s")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
