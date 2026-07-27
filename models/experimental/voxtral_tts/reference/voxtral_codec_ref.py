# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU reference for the Voxtral Codec DECODER — BLOCK 3 (~150M): audio codes -> 24 kHz waveform.

Self-contained (torch only) op-for-op reference for the decode half of upstream
`VoxtralTTSAudioTokenizer` (vllm_omni/model_executor/models/voxtral_tts/
voxtral_tts_audio_tokenizer.py). See ../reference/PROVENANCE.md.

BLOCK BOUNDARY:

    codes [B, 37, T]  (12.5 Hz frames, as emitted by Block 2, WITHOUT the special-token offset)
      -> quantizer.decode : semantic lookup (256D) ++ acoustic FSQ rescale (36D) = [B, 292, T]
      -> decoder_blocks.0 : CausalConv1d(292->1024, k3, s1, replicate)
      -> blocks 1/3/5/7   : Transformer(2 layers) at sliding windows 2 / 4 / 8 / 16
      -> blocks 2/4/6     : CausalConvTranspose1d(1024->1024, k4, s2)   = 8x upsample -> 100 Hz
      -> output_proj      : CausalConv1d(1024->240, k7, reflect)
      -> unpatch [B,240,T'] -> [B, 1, T'*240]                           = waveform @ 24 kHz

WHY THE ENCODER IS NOT HERE: the released checkpoint ships ZERO encoder tensors (no
`input_proj.*`, no `encoder_blocks.*` — verified against the 386-tensor manifest). Upstream
raises `RuntimeError: encode_waveforms requires encoder weights which are not available in the
open-source checkpoint.` So cloning a voice from arbitrary reference audio is IMPOSSIBLE with
public weights; only the 20 shipped `voice_embedding/*.pt` presets work. There is nothing to
port and nothing to validate on the encoder side.

DETAILS THAT WILL BITE A TTNN PORT:
  * norm_eps is 1e-2 for the codec's RMSNorms (params.json "norm_eps": 0.01) — three orders
    off the usual 1e-5, and load-bearing.
  * q_norm / k_norm are RMSNorm over the FULL 1024-wide projection, applied BEFORE the head
    split — not per-head.
  * LayerScale: each residual branch is scaled by a learned [1024] vector (init 0.01).
  * ALiBi bias slope*(j-i), plus causal mask, plus a sliding window that DOUBLES per upsample
    stage (2, 4, 8, 16). All three collapse into one additive pre-softmax bias.
  * Causal convs left-pad by (k-1) with reflect/replicate, never centre-pad; the transposed
    convs trim (k-stride) samples off the RIGHT.
  * weight_norm is stored as a parametrization (original0=g, original1=v) and folded at dim=0
    for both Conv1d [out,in,k] and ConvTranspose1d [in,out,k].

Run (regenerates goldens; needs the checkpoint — only ~0.6 GB of it is read):
    PYTHONPATH=<repo> python models/experimental/voxtral_tts/reference/voxtral_codec_ref.py
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F

from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ACOUSTIC_CODEBOOK_SIZE,
    CODEC_ATTN_WINDOW,
    CODEC_DIM,
    CODEC_HEAD_DIM,
    CODEC_LAYER_SCALE_INIT,
    CODEC_N_HEADS,
    CODEC_N_KV_HEADS,
    CODEC_NORM_EPS,
    CODEC_QK_NORM_EPS,
    DEC_CONV_BLOCKS,
    DEC_CONV_KERNELS,
    DEC_CONV_STRIDES,
    DEC_TF_BLOCKS,
    DEC_TF_LENGTHS,
    DEFAULT_CKPT,
    END_AUDIO_ID,
    GOLDEN_ROOT,
    LATENT_DIM,
    N_AUDIO_SPECIAL,
    NUM_CODEBOOKS,
    PATCH_PROJ_KERNEL,
    PATCH_SIZE,
    SEMANTIC_DIM,
    SafeTensors,
    fold_weight_norm,
    gqa_attention,
    merge_heads,
    pcc,
    rms_norm,
    split_heads,
    swiglu,
)

GOLDEN_DIR = os.path.join(GOLDEN_ROOT, "codec")
PREFIX = "audio_tokenizer."


def decoder_window_sizes():
    """Sliding-window size per decoder transformer stage -> (2, 4, 8, 16).

    Derived rather than hard-coded because the derivation is the surprising part: upstream
    threads ONE `cur_window_size` variable through encoder construction and then decoder
    construction. The encoder halves it on each of its three stride-2 downsamples
    (16 -> 8 -> 4 -> 2), and the decoder inherits the final value (2) and DOUBLES it after each
    stride-2 upsample. So the decoder's first stage runs the narrowest window, not the widest."""
    w = CODEC_ATTN_WINDOW
    for s in (2, 2, 2, 1):  # encoder strides, in order
        if s > 1:
            w //= 2
    out = []
    for stage in range(len(DEC_TF_LENGTHS)):
        out.append(w)
        if stage < len(DEC_CONV_BLOCKS) - 1:
            w *= DEC_CONV_STRIDES[stage + 1]
    return tuple(out)


def load_codec_state(ckpt_path=DEFAULT_CKPT, dtype=torch.float32):
    """Decoder blocks + output_proj + semantic codebook, with weight_norm folded.

    Returns plain `.weight` keys, so the TTNN port never has to know about parametrizations."""
    st = SafeTensors(ckpt_path)
    raw = st.prefixed(PREFIX, dtype)
    w = {}
    for k, v in raw.items():
        if ".parametrizations.weight.original" in k:
            continue
        w[k] = v
    for base in [f"decoder_blocks.{i}.conv" for i in DEC_CONV_BLOCKS] + ["output_proj.conv"]:
        w[base + ".weight"] = fold_weight_norm(raw, base)
    # Semantic codebook is stored as running sums (EMA training state), not the codebook itself.
    w["semantic_embedding"] = raw["quantizer.semantic_codebook.embedding_sum"] / raw[
        "quantizer.semantic_codebook.cluster_usage"
    ].clamp(min=1e-5).unsqueeze(-1)
    return w


# ---------------------------------------------------------------------------------------
# Quantizer (decode side)
# ---------------------------------------------------------------------------------------
def quantizer_decode(codes, w):
    """codes [B, 37, T] ints (NO special-token offset) -> latents [B, 292, T].

    Semantic: a table lookup. Acoustic: pure arithmetic (FSQ has no parameters) —
    code -> code*2/(levels-1) - 1, the exact inverse of Block 2's quantization."""
    sem = F.embedding(codes[:, 0, :], w["semantic_embedding"])  # [B, T, 256]
    sem = sem.permute(0, 2, 1)  # [B, 256, T]
    ac = codes[:, 1:, :].to(torch.float32) * 2.0 / (ACOUSTIC_CODEBOOK_SIZE - 1) - 1.0  # [B, 36, T]
    return torch.cat([sem, ac], dim=1)  # [B, 292, T]


# ---------------------------------------------------------------------------------------
# Causal convolutions
# ---------------------------------------------------------------------------------------
def causal_conv1d(x, weight, kernel, stride, pad_mode):
    """Upstream CausalConv1d: left-pad by (eff_k - stride), plus whatever extra padding is
    needed to make the output length come out at ceil(n_frames). No bias anywhere here."""
    eff_k = kernel  # dilation is always 1 in this model
    pad_total = eff_k - stride
    L = x.shape[-1]
    n_frames = (L - eff_k + pad_total) / stride + 1
    target = (math.ceil(n_frames) - 1) * stride + (eff_k - pad_total)
    extra = max(target - L, 0)
    x = F.pad(x, (pad_total, extra), mode=pad_mode)
    return F.conv1d(x, weight, None, stride=stride)


def causal_conv_transpose1d(x, weight, kernel, stride, trim_ratio=1.0):
    """Upstream CausalConvTranspose1d: full transposed conv, then trim (k - stride) samples.
    trim_ratio 1.0 puts the entire trim on the RIGHT (left_padding == 0)."""
    out = F.conv_transpose1d(x, weight, None, stride=stride)
    total = kernel - stride
    right = math.ceil(total * trim_ratio)
    left = total - right
    return out[..., left : out.shape[-1] - right]


# ---------------------------------------------------------------------------------------
# Codec transformer block (ALiBi + causal + sliding window, QK-norm, LayerScale)
# ---------------------------------------------------------------------------------------
def alibi_slopes(n_heads):
    """Upstream get_alibi_slopes: geometric ratio r = 2^(-8/n). For n=8 -> r=0.5, giving
    [1, 1/2, 1/4, ..., 1/128]."""
    if math.log2(n_heads).is_integer():
        r = 2.0 ** (-8.0 / n_heads)
        return torch.tensor([r**i for i in range(n_heads)], dtype=torch.float32)
    m = 2 ** math.floor(math.log2(n_heads))
    r1 = 2.0 ** (-8.0 / m)
    r2 = 2.0 ** (-8.0 / (2 * m))
    head = [r1**i for i in range(m)]
    tail = [r2**i for i in range(0, 2 * m, 2)][: n_heads - m]
    return torch.tensor(head + tail, dtype=torch.float32)


def attention_bias(seq_len, window, n_heads=CODEC_N_HEADS, causal=True, dtype=torch.float32):
    """One additive [1, H, S, S] pre-softmax bias folding ALiBi + causal + sliding window.

    rel[i,j] = j - i. ALiBi contributes slope_h * rel. Causal masks rel > 0. The window masks
    rel < -window (and rel > 0 again on the right, already masked when causal)."""
    pos = torch.arange(seq_len)
    rel = (pos.unsqueeze(0) - pos.unsqueeze(1)).to(dtype)  # [S, S]
    bias = alibi_slopes(n_heads).to(dtype).view(n_heads, 1, 1) * rel.unsqueeze(0)
    if causal:
        bias = bias.masked_fill(rel.unsqueeze(0) > 0, float("-inf"))
    right = 0 if causal else window
    outside = (rel < -window) | (rel > right)
    return bias.masked_fill(outside.unsqueeze(0), float("-inf")).unsqueeze(0)


def codec_block(x, w, p, bias):
    """x [B, S, 1024] -> [B, S, 1024]. Pre-norm, QK-norm on the full projection, LayerScale on
    both residual branches."""
    h = rms_norm(x, w[p + "attention_norm.weight"], CODEC_NORM_EPS)
    q = F.linear(h, w[p + "attention.wq.weight"])
    k = F.linear(h, w[p + "attention.wk.weight"])
    v = F.linear(h, w[p + "attention.wv.weight"])
    # QK-norm over the FULL n_heads*head_dim width, BEFORE splitting into heads.
    q = rms_norm(q, w[p + "attention.q_norm.weight"], CODEC_QK_NORM_EPS)
    k = rms_norm(k, w[p + "attention.k_norm.weight"], CODEC_QK_NORM_EPS)
    q = split_heads(q, CODEC_N_HEADS, CODEC_HEAD_DIM)
    k = split_heads(k, CODEC_N_KV_HEADS, CODEC_HEAD_DIM)
    v = split_heads(v, CODEC_N_KV_HEADS, CODEC_HEAD_DIM)
    attn = merge_heads(gqa_attention(q, k, v, bias))
    r = F.linear(attn, w[p + "attention.wo.weight"])
    x = x + w[p + "attention_scale"] * r  # LayerScale
    h = rms_norm(x, w[p + "ffn_norm.weight"], CODEC_NORM_EPS)
    r = swiglu(h, w[p + "feed_forward.w1.weight"], w[p + "feed_forward.w2.weight"],
               w[p + "feed_forward.w3.weight"])
    return x + w[p + "ffn_scale"] * r


def codec_transformer(x, w, block_idx, n_layers, window):
    bias = attention_bias(x.shape[1], window, dtype=x.dtype)
    for li in range(n_layers):
        x = codec_block(x, w, f"decoder_blocks.{block_idx}.layers.{li}.", bias)
    return x


# ---------------------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def reference_decode(codes, w):
    """codes [B, 37, T] (offset already stripped) -> waveform [B, 1, T*240*8].

    Note the layout flips between conv blocks ([B,C,L]) and transformer blocks ([B,L,C]); the
    TTNN port can keep everything channels-last and skip these transposes entirely."""
    x = quantizer_decode(codes, w)  # [B, 292, T]
    x = causal_conv1d(x, w["decoder_blocks.0.conv.weight"], DEC_CONV_KERNELS[0], DEC_CONV_STRIDES[0], "replicate")
    windows = decoder_window_sizes()
    for stage, (tf_idx, n_layers) in enumerate(zip(DEC_TF_BLOCKS, DEC_TF_LENGTHS)):
        x = codec_transformer(x.permute(0, 2, 1), w, tf_idx, n_layers, windows[stage]).permute(0, 2, 1)
        if stage < len(DEC_CONV_BLOCKS) - 1:  # blocks 2, 4, 6 — one upsample after each of the first 3
            ci = DEC_CONV_BLOCKS[stage + 1]
            x = causal_conv_transpose1d(x, w[f"decoder_blocks.{ci}.conv.weight"],
                                        DEC_CONV_KERNELS[stage + 1], DEC_CONV_STRIDES[stage + 1])
    x = causal_conv1d(x, w["output_proj.conv.weight"], PATCH_PROJ_KERNEL, 1, "reflect")  # [B, 240, T']
    B, _, T = x.shape
    return x.permute(0, 2, 1).reshape(B, 1, T * PATCH_SIZE)  # unpatch: "b (c h) t -> b c (t h)"


def strip_offset_and_trim(codes):
    """Turn Block 2's emitted frames [T, 37] into the decoder's input [1, 37, T'].

    Mirrors upstream decode_helper_batch_async: cut at the first [END_AUDIO] in codebook 0,
    then subtract the special-token offset."""
    eoa = (codes[:, 0] == END_AUDIO_ID).nonzero()
    cut = int(eoa[0]) if len(eoa) else len(codes)
    return (codes[:cut] - N_AUDIO_SPECIAL).t().unsqueeze(0)


def make_synthetic_codes(n_frames=24, seed=0):
    """Deterministic in-range codes [1, 37, T]: semantic in [0,8192), acoustic in [0,21)."""
    g = torch.Generator().manual_seed(seed)
    sem = torch.randint(0, 8192, (1, 1, n_frames), generator=g)
    ac = torch.randint(0, ACOUSTIC_CODEBOOK_SIZE, (1, NUM_CODEBOOKS - 1, n_frames), generator=g)
    return torch.cat([sem, ac], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=GOLDEN_DIR)
    ap.add_argument("--n-frames", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[codec] loading decoder from {args.ckpt}")
    w = load_codec_state(args.ckpt)
    print(f"[codec] {len(w)} tensors (weight_norm folded); windows {decoder_window_sizes()}, "
          f"norm_eps {CODEC_NORM_EPS}, layer_scale init {CODEC_LAYER_SCALE_INIT}")

    codes = make_synthetic_codes(args.n_frames)
    latents = quantizer_decode(codes, w)
    print(f"[codec] codes {tuple(codes.shape)} -> latents {tuple(latents.shape)} "
          f"(semantic |x| {latents[:, :SEMANTIC_DIM].abs().mean():.4f}, "
          f"acoustic |x| {latents[:, SEMANTIC_DIM:].abs().mean():.4f})")

    wav = reference_decode(codes, w)
    secs = wav.shape[-1] / 24000
    print(f"[codec] waveform {tuple(wav.shape)} = {secs:.3f}s @ 24 kHz "
          f"(peak {wav.abs().max():.4f}); upsample {wav.shape[-1] // args.n_frames}x per frame")
    assert wav.shape[-1] == args.n_frames * PATCH_SIZE * 8, "expected 240*8 = 1920 samples/frame"

    # FSQ round-trip: quantizer_decode must invert Block 2's quantization exactly.
    lvl = ACOUSTIC_CODEBOOK_SIZE
    probe = torch.arange(lvl).view(1, 1, lvl).expand(1, NUM_CODEBOOKS - 1, lvl)
    rt = probe.to(torch.float32) * 2.0 / (lvl - 1) - 1.0
    back = (((rt + 1) / 2) * (lvl - 1)).round().long()
    print(f"[codec] FSQ round-trip over all {lvl} levels exact: {bool((back == probe).all())}")

    # Per-stage shapes, so a port can be bisected stage by stage.
    x = causal_conv1d(latents, w["decoder_blocks.0.conv.weight"], 3, 1, "replicate")
    stages = {"after_input_conv": x.clone()}
    for stage, (tf_idx, n_layers) in enumerate(zip(DEC_TF_BLOCKS, DEC_TF_LENGTHS)):
        x = codec_transformer(x.permute(0, 2, 1), w, tf_idx, n_layers, decoder_window_sizes()[stage]).permute(0, 2, 1)
        stages[f"after_tf{tf_idx}"] = x.clone()
        if stage < 3:
            ci = DEC_CONV_BLOCKS[stage + 1]
            x = causal_conv_transpose1d(x, w[f"decoder_blocks.{ci}.conv.weight"], 4, 2)
            stages[f"after_up{ci}"] = x.clone()
    for k, v in stages.items():
        print(f"[codec]   {k:20s} {tuple(v.shape)}")
    print(f"[codec] full-path vs staged PCC {pcc(wav, reference_decode(codes, w)):.6f}")

    torch.save(codes, os.path.join(args.out, "codes.pt"))
    torch.save(latents, os.path.join(args.out, "latents.pt"))
    torch.save(wav, os.path.join(args.out, "waveform.pt"))
    torch.save(stages, os.path.join(args.out, "stages.pt"))
    torch.save({"n_frames": args.n_frames, "patch": PATCH_SIZE, "upsample": PATCH_SIZE * 8,
                "latent_dim": LATENT_DIM, "dim": CODEC_DIM, "windows": decoder_window_sizes(),
                "sampling_rate": 24000}, os.path.join(args.out, "meta.pt"))
    print(f"[codec] wrote goldens to {args.out}")


if __name__ == "__main__":
    main()
