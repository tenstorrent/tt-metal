# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Voxtral Codec DECODER (Block 3): audio codes -> 24 kHz waveform.

See NOTES.md [codec-01].
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
# NOTES.md [codec-02] -- The reference masks with -inf...
MASK_NEG = -1e9
# bf16 for the attention interior + its bias: measured best-PCC AND faster, and halves the largest
# tensor. Everything OUTSIDE attention stays DTYPE. Sweep in NOTES.md [codec-01].
ATTN_DTYPE = ttnn.bfloat16
# NOTES.md [codec-03] -- Chunked attention -- see OPTIMIZATIONS in [codec-01] #2/#3...
SLAB = 512
# NOTES.md [codec-04] -- Chunk only above this length...
CHUNK_MIN = 512
# NOTES.md [codec-05] -- Conv length bucketing -- see OPTIMIZATIONS in [codec-01] #4...
BUCKET = 128
# NOTES.md [codec-06] -- Rows in the output projection's reflected prefix...
OUT_PREFIX = 32

COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


class TtVoxtralCodecDecoder:
    """On-device codec decoder. __call__(codes [1,37,T] int64) -> waveform torch [1,1,T*1920]."""

    def __init__(self, device, ckpt_path=DEFAULT_CKPT, slab=SLAB, chunk_min=CHUNK_MIN, bucket=BUCKET):
        """Precision is not a parameter: fp32 weights, fp32 activations outside attention, bf16
        inside it. That combination was measured against the other three (NOTES.md [codec-01]) and the
        losers were deleted rather than left switchable."""
        self.device = device
        self.slab = slab  # attention chunk width; see the SLAB constant for why 512
        self.chunk_min = chunk_min  # chunk only when S exceeds this; None = never chunk
        self.bucket = bucket  # round T up to this multiple before decoding; None = off
        # NOTES.md [codec-07] -- With PRE-PREPARED weights the op can no longer infer...
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
        # NOTES.md [codec-08] -- ...and the reflected prefix those taps slide over, as a...
        idx = torch.zeros(1, 1, OUT_PREFIX, CODEC_DIM, dtype=torch.int32)
        for m in range(PATCH_PROJ_KERNEL - 1):
            idx[0, 0, OUT_PREFIX - (PATCH_PROJ_KERNEL - 1) + m, :] = (PATCH_PROJ_KERNEL - 1) - m
        self._out_prefix_idx = ttnn.from_torch(
            idx.contiguous(), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)

        # NOTES.md [codec-09] -- --- convs ---
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

        See NOTES.md [codec-10].
        """
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

        See NOTES.md [codec-11].
        """
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
    # NOTES.md [codec-20] -- ALiBi + causal + sliding window as ONE additive term, [1,H,slab,slab]...
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

        See NOTES.md [codec-12].
        """
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

        See NOTES.md [codec-13].
        """
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

        See NOTES.md [codec-14].
        """
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
        # NOTES.md [codec-15] -- Head split/merge via the FUSED ops, not reshape+permute...
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

        See NOTES.md [codec-16].
        """
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
        # NOTES.md [codec-17] -- OUTPUT PROJECTION AS MATMULS, NOT ttnn.conv1d -- and the...
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
        # NOTES.md [codec-18] -- return_stages BYPASSES bucketing on purpose: it exists to...
        if self.bucket and not return_stages:
            T = codes.shape[2]
            padded = -(-T // self.bucket) * self.bucket
            if padded > T:
                # NOTES.md [codec-19] -- repeat the LAST frame rather than zero-pad: the tail then...
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
