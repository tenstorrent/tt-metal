# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `voxtral_t_t_s_audio_tokenizer` (`audio_tokenizer`) -- the whole neural
codec decoder, from integer codes to a waveform.

    quantizer.decode -> conv(292->1024, k3) -> [transformer, upsample] x4 -> output_proj -> depatch

The open-source checkpoint ships the DECODER only (no `input_proj.*` / `encoder_blocks.*`), so
there is no encode path to port.

**Everything runs channels-LAST**, `[B, 1, T, C]`. The reference permutes to channels-first around
each convolution and back for each transformer; in ttnn the matmul-shaped convolution wants
channels last anyway, so the whole chain stays in one layout and the only transpose left is the one
that brings the acoustic codes' `[B, 36, T]` around.

Sliding windows DOUBLE up the decoder -- 2, 4, 8, 16 -- because each transposed convolution doubles
the frame rate. Each stage's ALiBi mask is built from that stage's own window.

Two padding modes appear, and they are different:
  * the first convolution pads `replicate` (the boundary sample repeated), and
  * `output_proj` pads `reflect` -- `[x6, x5, x4, x3, x2, x1, x0, x1, ...]`, a mirror about index 0
    that EXCLUDES index 0 itself.

**The activation path runs in float32.** With everything in bfloat16 the full chain landed at PCC
0.9873: each stage is fine on its own, but eight codec residual blocks plus five convolutions
compound. Weights stay bfloat16 -- `ttnn.linear` takes a float32 activation against a bfloat16
weight and returns float32 -- so only the activations widen. Q/K/V (and the ALiBi mask that has to
match them) are cast back down for SDPA, which rejects float32 outright
(`sdpa_device_operation.cpp:43`).

Frame arithmetic for a 64-frame input: 64 -> 64 -> 128 -> 256 -> 512 frames, then `output_proj`
emits 240 samples per frame, de-patched to 122880 samples (12.5 Hz frames, 1920 samples each after
the 8x upsampling, 24 kHz).
"""

from __future__ import annotations

import math

import torch

import ttnn

# A PERSISTENT ZERO BUFFER, NOT A PER-CALL `ttnn.zeros`.
# `ttnn.zeros` builds its tensor on the host and enqueues a WRITE to land it on the device, and a
# captured trace cannot replay a write -- capturing this stage died on `TT_FATAL: Writes are not
# supported during trace capture`. The shape is a function of the input shape, which a trace pins,
# so the buffer is created once per (device, shape, dtype) and reused. It is READ-ONLY here and is
# therefore never deallocated by a caller.
_ZEROS = {}


def _zeros_like_buf(device, shape, dtype):
    key = (id(device), tuple(int(s) for s in shape), str(dtype))
    buf = _ZEROS.get(key)
    if buf is None:
        buf = ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        _ZEROS[key] = buf
    return buf


_MASK_MAX_SEQ = 2048
_MASK_NEG = -1.0e9


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _weight(linear, device):
    return _from_torch(linear.weight.detach().transpose(0, 1).contiguous(), device)


_TILE_BYTES = {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
_L1_BUDGET = 1_100_000


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _mcast_cfg(x, w, rows, out_dtype):
    """A full-grid 2D-multicast program config for a tall `[rows, K] x [K, N]` linear, or None.

    M goes over the grid rows and N over the grid columns. Per-core M/N are searched a few tiles
    above the minimum (a slightly larger block often divides into better subblocks), and when the
    whole per-core output does not fit L1 it is split into out-blocks. Ranked by per-core work,
    then subblock area (fp32 DEST caps it at 4 tiles), then out-block area, then K-block width.
    """
    grid = x.device().compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    mt, kt, nt = rows // 32, int(w.shape[-2]) // 32, int(w.shape[-1]) // 32
    size = lambda dt: _TILE_BYTES.get(dt, 2048)
    xs, ws = size(x.dtype), size(w.dtype)
    os_ = size(out_dtype) + (0 if out_dtype == ttnn.float32 else 4096)
    best = None
    for pm in range(-(-mt // gy), -(-mt // gy) + 5):
        if -(-mt // pm) > gy:
            continue
        for pn in range(-(-nt // gx), -(-nt // gx) + 5):
            if -(-nt // pn) > gx:
                continue
            for bh in _divisors(pm):
                for bw in _divisors(pn):
                    kb = next(
                        (
                            c
                            for c in (8, 4, 2, 1)
                            if kt % c == 0 and bh * bw * os_ + 2 * c * (bh * xs + bw * ws) <= _L1_BUDGET
                        ),
                        None,
                    )
                    if kb is None:
                        continue
                    sub = max(
                        (
                            (h, s)
                            for h in range(1, 5)
                            for s in range(1, 5)
                            if h * s <= 4 and bh % h == 0 and bw % s == 0
                        ),
                        key=lambda hs: (hs[0] * hs[1], hs[1]),
                    )
                    score = (pm * pn, -sub[0] * sub[1], -bh * bw, -kb)
                    if best is None or score < best[0]:
                        best = (score, pm, pn, bh, bw, kb, sub)
    if best is None:
        return None
    _, pm, pn, bh, bw, kb, sub = best
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=kb,
        out_subblock_h=sub[0],
        out_subblock_w=sub[1],
        out_block_h=bh,
        out_block_w=bw,
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
    )


def _fold_linear(x, w, **kwargs):
    """`ttnn.linear` with the leading batch folded into M, so the weight streams ONCE.

    A `[B, 1, T, K]` activation against a 2-D weight runs as B separate matmuls that each re-read
    the whole weight; `[1, 1, B*T, K]` is one matmul. A T that is not tile-aligned makes the fold a
    real relayout each way, still far cheaper than re-reading the weight B times. Tall results get
    a hand-sized full-grid program config.
    """
    shape = [int(d) for d in x.shape]
    lead = 1
    for d in shape[:-2]:
        lead *= d
    if lead == 1:
        return ttnn.linear(x, w, **kwargs)
    rows = lead * shape[-2]
    if rows >= 256 and rows % 32 == 0 and "program_config" not in kwargs:
        cfg = _mcast_cfg(x, w, rows, kwargs.get("dtype") or x.dtype)
        if cfg is not None:
            kwargs["program_config"] = cfg
    y = ttnn.linear(ttnn.reshape(x, [1, 1, rows, shape[-1]]), w, **kwargs)
    return ttnn.reshape(y, shape[:-1] + [int(y.shape[-1])])


_COMPUTE = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _row(x4, index):
    """One sequence row of a `[B, 1, T, C]` tensor, as `[B, 1, 1, C]`."""
    batch, channels = int(x4.shape[0]), int(x4.shape[-1])
    return ttnn.slice(x4, [0, 0, index, 0], [batch, 1, index + 1, channels])


def _alibi_window_mask(slopes, window, seq):
    """`[1, H, seq, seq]`: ALiBi bias `slope[h] * (j - i)`, blocked where `j > i` or `j < i - window`.

    Depends only on `j - i`, so the top-left `[S, S]` corner is exactly the mask for a length-`S`
    sequence -- which is what lets the forward stay free of torch calls (a per-call rebuild could
    not be captured in a trace).
    """
    pos = torch.arange(seq)
    rel = pos.unsqueeze(0) - pos.unsqueeze(1)
    bias = slopes.reshape(-1, 1, 1).float() * rel.unsqueeze(0).float()
    blocked = (rel > 0) | (rel < -window)
    return bias.masked_fill(blocked.unsqueeze(0), _MASK_NEG).unsqueeze(0)


def _compile_codec_block(device, blk, mask):
    """One `CodecTransformerBlock` as a callable on a `[B, 1, T, dim]` ttnn tensor."""
    attn = blk.attention
    ff = blk.feed_forward
    args = blk.args
    n_heads = int(attn.n_local_heads)
    n_kv_heads = int(attn.n_local_kv_heads)
    scale = 1.0 / math.sqrt(int(args.head_dim))
    qk_norm = bool(args.qk_norm)
    dim = int(blk.dim)

    attn_gamma = _norm_gamma(blk.attention_norm, device)
    attn_eps = float(blk.attention_norm.eps)
    ffn_gamma = _norm_gamma(blk.ffn_norm, device)
    ffn_eps = float(blk.ffn_norm.eps)
    wq, wk, wv, wo = (_weight(m, device) for m in (attn.wq, attn.wk, attn.wv, attn.wo))
    q_gamma = _norm_gamma(attn.q_norm, device) if qk_norm else None
    k_gamma = _norm_gamma(attn.k_norm, device) if qk_norm else None
    q_eps = float(attn.q_norm.eps) if qk_norm else 0.0
    k_eps = float(attn.k_norm.eps) if qk_norm else 0.0
    w1, w2, w3 = (_weight(m, device) for m in (ff.w1, ff.w2, ff.w3))

    attn_scale = ffn_scale = None
    if blk.layer_scale:
        attn_scale = _from_torch(blk.attention_scale.detach().reshape(1, 1, 1, dim), device, dtype=ttnn.float32)
        ffn_scale = _from_torch(blk.ffn_scale.detach().reshape(1, 1, 1, dim), device, dtype=ttnn.float32)
    if blk.post_attention_norm is not None or blk.post_ffn_norm is not None:
        raise NotImplementedError("post_attention_norm / post_ffn_norm are not ported")

    def block(h):
        seq = int(h.shape[-2])
        xn = _rms_norm(h, attn_gamma, attn_eps)
        q = _fold_linear(xn, wq, compute_kernel_config=_COMPUTE)
        k = _fold_linear(xn, wk, compute_kernel_config=_COMPUTE)
        v = _fold_linear(xn, wv, compute_kernel_config=_COMPUTE)
        if qk_norm:
            q = _rms_norm(q, q_gamma, q_eps)
            k = _rms_norm(k, k_gamma, k_eps)
        # SDPA rejects float32 outright (`sdpa_device_operation.cpp:43`) -- so this does not call
        # it. Spelling the attention out as two matmuls and a softmax keeps Q/K/V, the ALiBi mask
        # and the whole reduction in FLOAT32, which SDPA cannot do at any fidelity.
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.concat([q, k, v], dim=-1),
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            transpose_k_heads=False,
        )
        scores = _bmm(qh, kh, transpose_b=True)
        scores = ttnn.add(
            ttnn.multiply(scores, scale),
            ttnn.slice(mask, [0, 0, 0, 0], [1, n_heads, seq, seq]),
        )
        a = _bmm(_softmax(scores), vh)
        ttnn.deallocate(scores)
        r = _fold_linear(
            ttnn.experimental.nlp_concat_heads(a),
            wo,
            dtype=ttnn.float32,
            compute_kernel_config=_COMPUTE,
        )
        if attn_scale is not None:
            r = ttnn.multiply(r, attn_scale)
        h = ttnn.add(h, r)

        hn = _rms_norm(h, ffn_gamma, ffn_eps)
        r = _fold_linear(
            ttnn.multiply(
                _fold_linear(hn, w1, compute_kernel_config=_COMPUTE),
                _fold_linear(hn, w3, compute_kernel_config=_COMPUTE),
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            ),
            w2,
            compute_kernel_config=_COMPUTE,
        )
        if ffn_scale is not None:
            r = ttnn.multiply(r, ffn_scale)
        return ttnn.add(h, r)

    return block


def _compile_causal_conv1d(device, mod):
    """A `CausalConv1d` as a callable on `[B, 1, T, C_in]` -> `[B, 1, T', C_out]`."""
    conv = mod.conv
    weight = conv.weight.detach()
    out_channels, in_channels, kernel = (int(v) for v in weight.shape)
    stride = int(conv.stride[0])
    dilation = int(conv.dilation[0])
    effective_kernel = int(mod._effective_kernel_size)
    padding_total = int(mod._padding_total)
    pad_mode = str(mod.pad_mode)
    if pad_mode not in ("replicate", "reflect"):
        raise NotImplementedError(f"pad_mode {pad_mode!r} is not ported")
    reflect = pad_mode == "reflect"

    taps = [_from_torch(weight[:, :, i].transpose(0, 1).contiguous(), device) for i in range(kernel)]
    bias = None
    if conv.bias is not None:
        bias = _from_torch(conv.bias.detach().reshape(1, 1, 1, out_channels), device)

    def run(x4):
        batch, length = int(x4.shape[0]), int(x4.shape[-2])
        n_frames = (length - effective_kernel + padding_total) / stride + 1
        target = (math.ceil(n_frames) - 1) * stride + (effective_kernel - padding_total)
        extra = target - length

        pieces = []
        if padding_total > 0:
            if reflect:
                pieces.extend(_row(x4, i) for i in range(padding_total, 0, -1))
            else:
                first = _row(x4, 0)
                pieces.append(first if padding_total == 1 else ttnn.repeat(first, [1, 1, padding_total, 1]))
        pieces.append(x4)
        if extra > 0:
            if reflect:
                pieces.extend(_row(x4, length - 1 - i) for i in range(1, extra + 1))
            else:
                last = _row(x4, length - 1)
                pieces.append(last if extra == 1 else ttnn.repeat(last, [1, 1, extra, 1]))
        padded = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=2)

        padded_len = length + padding_total + extra
        out_len = (padded_len - effective_kernel) // stride + 1

        acc = None
        for i, tap in enumerate(taps):
            begin = i * dilation
            end = begin + (out_len - 1) * stride + 1
            seg = ttnn.slice(
                padded,
                [0, 0, begin, 0],
                [batch, 1, end, in_channels],
                [1, 1, stride, 1] if stride > 1 else None,
            )
            term = _fold_linear(seg, tap, compute_kernel_config=_COMPUTE)
            acc = term if acc is None else ttnn.add(acc, term)
        return acc if bias is None else ttnn.add(acc, bias)

    return run


def _compile_causal_conv_transpose1d(device, mod):
    """A `CausalConvTranspose1d` as a callable on `[B, 1, T, C]` -> `[B, 1, T * 2, C]`."""
    conv = mod.conv
    weight = conv.weight.detach()
    in_channels, out_channels, kernel = (int(v) for v in weight.shape)
    stride = int(conv.stride[0])
    if stride != 2 or kernel != 4:
        raise NotImplementedError(f"only kernel 4 / stride 2 is ported, got {kernel}/{stride}")
    right_trim = math.ceil((kernel - stride) * float(mod.trim_ratio))
    if (kernel - stride) - right_trim != 0:
        raise NotImplementedError("a non-zero left trim is not ported")

    taps = [_from_torch(weight[:, :, i].contiguous(), device) for i in range(kernel)]
    bias = None
    if conv.bias is not None:
        bias = _from_torch(conv.bias.detach().reshape(1, 1, 1, out_channels), device)

    def run(x4):
        batch, length = int(x4.shape[0]), int(x4.shape[-2])
        zero_row = _zeros_like_buf(device, [batch, 1, 1, out_channels], x4.dtype)

        def _delayed(tap):
            """`tap` applied to the PREVIOUS input step: a zero row, then steps 0..L-2."""
            head = ttnn.slice(x4, [0, 0, 0, 0], [batch, 1, length - 1, in_channels])
            return ttnn.concat([zero_row, _fold_linear(head, tap, compute_kernel_config=_COMPUTE)], dim=2)

        even = ttnn.add(_fold_linear(x4, taps[0], compute_kernel_config=_COMPUTE), _delayed(taps[2]))
        odd = ttnn.add(_fold_linear(x4, taps[1], compute_kernel_config=_COMPUTE), _delayed(taps[3]))
        out = ttnn.reshape(ttnn.concat([even, odd], dim=-1), [batch, 1, length * stride, out_channels])
        return out if bias is None else ttnn.add(out, bias)

    return run


def _norm_gamma(norm, device):
    """`[1, 1, 1, dim]` float32 TILE -- the form the spelled-out RMS norm's final multiply takes."""
    return _from_torch(norm.weight.detach().reshape(1, 1, 1, -1), device, dtype=ttnn.float32)


def _rms_norm(x, gamma, eps):
    """`x * rsqrt(mean(x^2) + eps) * gamma`, spelled out, entirely in float32.

    NOT `ttnn.rms_norm`: on this model's real inputs the stock op sits at ~9.65e-4 relative error
    where these four ops sit at 6.6e-8. A norm error is RELATIVE -- it RESCALES the whole branch
    after it -- so it shows up as a NORM RATIO rather than as a PCC drop, and the codec stacks
    eight residual blocks with two or three norms each. Measured: the codec's first transformer
    group came back at norm ratio 1.029 against torch with the stock op, and no single stage of
    the chain looked broken. `tt/vocode_stage.py` spells out the same four ops for the same
    reason, so the two bodies agree.
    """
    scale = ttnn.rsqrt(ttnn.add(ttnn.mean(ttnn.square(x), dim=-1, keepdim=True), eps))
    return ttnn.multiply(ttnn.multiply(x, scale), gamma)


def _softmax(x, dim=-1):
    """`exp(x - max) / sum(exp(x - max))`, spelled out in three ops.

    NOT `ttnn.softmax`: measured on this build against a float64 reference, the stock op's rows do
    not sum to 1 (mean 0.9943, worst 0.9611, ~1.8e-2 relative error) and no flag changes it. These
    three ops sit at 5.5e-8. A softmax that does not sum to 1 ATTENUATES the attention output it
    weights, which reads as a norm ratio below 1 at a PCC of 0.9999.
    """
    e = ttnn.subtract(x, ttnn.max(x, dim=dim, keepdim=True), activations=[ttnn.UnaryOpType.EXP])
    return ttnn.divide(e, ttnn.sum(e, dim=dim, keepdim=True))


# THE SEMANTIC CODEBOOK IS GATHERED IN TWO HALVES.
# `ttnn.embedding` requires a bfloat16 table (`embedding_device_operation.cpp:36`), and this table
# is the codec's INPUT -- everything downstream amplifies whatever it gets wrong. Measured on this
# checkpoint: a single bfloat16 table put the quantizer latent at 1.613e-3 relative, and the
# 292->1024 convolution that consumes it amplified that to 7.7e-3, which then dominated every
# later stage. So the table is split the way a compensated matmul splits a weight --
# `hi = bf16(t)`, `lo = bf16(t - hi)` -- and the two gathers are added back in float32. Two
# bfloat16 mantissas end to end is ~1e-5 on a table this size, for one extra gather and one add
# on 416 rows.
def _split_table(weight, device):
    hi = weight.to(torch.bfloat16)
    lo = (weight - hi.float()).to(torch.bfloat16)
    return (
        _from_torch(hi.contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT),
        _from_torch(lo.contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT),
    )


def _split_embedding(ids, tables, layout=None):
    hi, lo = tables
    layout = ttnn.TILE_LAYOUT if layout is None else layout
    return ttnn.add(
        ttnn.typecast(ttnn.embedding(ids, hi, layout=layout), ttnn.float32),
        ttnn.typecast(ttnn.embedding(ids, lo, layout=layout), ttnn.float32),
    )


def _bmm(a, b, transpose_b=False):
    """Head-batched attention `a @ b` (`[B, H, S, S]` scores, then `P @ V`).

    No program config: a `MatmulMultiCoreReuseProgramConfig` with per-core (batch, head, 4-tile M)
    blocks was tried here and gives WRONG results for these head-batched shapes (waveform PCC -0.005
    in the e2e test), so the stock schedule is used.
    """
    return ttnn.matmul(a, b, transpose_b=transpose_b, compute_kernel_config=_COMPUTE)


def build(device, torch_module, layers=None):
    """The codec decoder as one callable, `codes [B, 37, T]` (unshifted) -> `[B, 1, T * 1920]`.

    `layers` caps the blocks PER TRANSFORMER GROUP (None = every block). The compiled blocks are
    exposed as `.blocks`, a list of `(group, layer_id, window, block)` in forward order.
    """
    codec = torch_module
    quant = codec.quantizer
    semantic = quant.semantic_codebook
    acoustic = quant.acoustic_codebook

    n_semantic = int(semantic.num_codebooks)
    n_acoustic = int(acoustic.num_codebooks)
    n_levels = int(acoustic.n_levels)
    shift = (n_levels - 1) / 2.0
    scale = 2.0 / (n_levels - 1)
    latent_dim = int(codec.latent_dim)
    patch_size = int(codec.patch_size)

    table = _split_table(semantic.embedding.detach(), device)

    stages = []
    codec_blocks = []
    group = 0
    for blk in codec.decoder_blocks:
        name = type(blk).__name__
        if name == "CausalConv1d":
            stages.append(_compile_causal_conv1d(device, blk))
        elif name == "CausalConvTranspose1d":
            stages.append(_compile_causal_conv_transpose1d(device, blk))
        elif name == "CodecTransformer":
            mask = _from_torch(
                _alibi_window_mask(
                    blk.layers["0"].attention.alibi_slopes.detach(),
                    int(blk.args.attn_sliding_window_size),
                    _MASK_MAX_SEQ,
                ),
                device,
                dtype=ttnn.float32,
            )
            window = int(blk.args.attn_sliding_window_size)
            layer_ids = list(blk.layers_ids) if layers is None else list(blk.layers_ids)[: int(layers)]
            blocks = [_compile_codec_block(device, blk.layers[str(i)], mask) for i in layer_ids]
            codec_blocks.extend((group, int(i), window, b) for i, b in zip(layer_ids, blocks))
            group += 1

            def _stack(x4, _blocks=blocks):
                for b in _blocks:
                    x4 = b(x4)
                return x4

            stages.append(_stack)
        else:
            raise NotImplementedError(f"decoder block {name} is not ported")
    output_proj = _compile_causal_conv1d(device, codec.output_proj)

    def voxtral_t_t_s_audio_tokenizer(codes, **kwargs):
        batch, rows, frames = (int(v) for v in codes.shape)
        if frames * 8 > _MASK_MAX_SEQ:
            raise NotImplementedError(
                f"{frames} frames upsample to {frames * 8}, past the prebuilt ALiBi mask "
                f"({_MASK_MAX_SEQ}); raise _MASK_MAX_SEQ -- the mask cannot be rebuilt inside the "
                f"forward"
            )

        sem_codes = ttnn.reshape(ttnn.slice(codes, [0, 0, 0], [batch, n_semantic, frames]), [batch, frames])
        # `ttnn.embedding` requires a bfloat16 table (`embedding_device_operation.cpp:36`);
        # widen once here so every residual add downstream happens in float32.
        sem = ttnn.typecast(_split_embedding(sem_codes, table, layout=ttnn.TILE_LAYOUT), ttnn.float32)
        aco_codes = ttnn.typecast(
            ttnn.to_layout(
                ttnn.slice(codes, [0, n_semantic, 0], [batch, n_semantic + n_acoustic, frames]),
                ttnn.TILE_LAYOUT,
            ),
            ttnn.float32,
        )
        aco = ttnn.transpose(ttnn.multiply(ttnn.subtract(aco_codes, shift), scale), -2, -1)

        # float32 on BOTH sides: the semantic half is a two-gather split now and comes back
        # float32, and `ttnn.concat` requires a single dtype.
        h = ttnn.reshape(
            ttnn.concat([sem, ttnn.typecast(aco, ttnn.float32)], dim=-1),
            [batch, 1, frames, latent_dim],
        )
        for stage in stages:
            h = stage(h)
        h = output_proj(h)

        # "b (c h) t -> b c (t h)" with h = patch_size: channels-last, that is just a flatten.
        out_frames = int(h.shape[-2])
        return ttnn.reshape(h, [batch, 1, out_frames * patch_size])

    voxtral_t_t_s_audio_tokenizer.blocks = codec_blocks
    return voxtral_t_t_s_audio_tokenizer
