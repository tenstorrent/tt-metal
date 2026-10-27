# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared pieces for the Voxtral-TTS CPU reference (all three blocks), torch-only.

A seek-per-tensor safetensors reader, the params.json config, and the primitives (RMSNorm,
SwiGLU, GQA attention, RoPE, weight-norm folding, codebook offsets, PCC). Config values are from the
released checkpoint's params.json; N_DECODING_STEPS and FM_NORM_EPS, which it does not set, are
upstream fallbacks.

Run to check the config against a real checkpoint's tensor manifest (no weights needed):
    PYTHONPATH=<repo> python models/experimental/voxtral_tts/reference/voxtral_common_ref.py
"""

import json
import os
import struct

import torch
import torch.nn.functional as F

from models.experimental.voxtral_tts.reference.voxtral_paths import CKPT_NAME, DOWNLOAD_HINT, MODEL_DIR

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(MODEL_DIR, CKPT_NAME)
DEFAULT_PARAMS = os.path.join(MODEL_DIR, "params.json")
MANIFEST = os.path.join(_HERE, "CKPT_MANIFEST.json")  # name -> {dtype, shape}; lets tests run weight-free
GOLDEN_ROOT = os.path.join(_HERE, "..", "golden")

# ---------------------------------------------------------------------------------------
# Config — mistralai/Voxtral-4B-TTS-2603 params.json
# ---------------------------------------------------------------------------------------
# The backbone: AR backbone (Ministral-derived). NOTE n_heads*head_dim (4096) != dim (3072):
# the attention output projection is [3072, 4096], i.e. wider inside than the residual stream.
DIM = 3072
N_LAYERS = 26
N_HEADS = 32
N_KV_HEADS = 8  # GQA, 4 query heads per KV head
HEAD_DIM = 128
HIDDEN_DIM = 9216  # SwiGLU inner
NORM_EPS = 1e-5
ROPE_THETA = 1_000_000.0
VOCAB_SIZE = 131072
TIED_EMBEDDINGS = True
ATTN_DIM = N_HEADS * HEAD_DIM  # 4096
KV_DIM = N_KV_HEADS * HEAD_DIM  # 1024

# Audio tokenization (shared by the backbone, flow model and codec)
SAMPLING_RATE = 24000
FRAME_RATE = 12.5
PATCH_SIZE = 240  # pretransform_patch_size; 24000 / 240 = 100 Hz pre-downsampling
SEMANTIC_CODEBOOK_SIZE = 8192
N_ACOUSTIC_CODEBOOK = 36
ACOUSTIC_CODEBOOK_SIZE = 21  # FSQ levels per acoustic dim
NUM_CODEBOOKS = 1 + N_ACOUSTIC_CODEBOOK  # 37 tokens per 12.5 Hz frame
SEMANTIC_DIM = 256
ACOUSTIC_DIM = 36
LATENT_DIM = SEMANTIC_DIM + ACOUSTIC_DIM  # 292

# Audio special tokens (upstream AudioSpecialTokens; ids are enum ORDER, not config)
EMPTY_AUDIO_ID = 0
END_AUDIO_ID = 1
N_AUDIO_SPECIAL = 2  # every code emitted by the quantizer is offset by this

# The flow model: flow-matching acoustic transformer
FM_DIM = 3072
FM_N_LAYERS = 3
FM_N_HEADS = 32
FM_N_KV_HEADS = 8
FM_HEAD_DIM = 128
FM_HIDDEN_DIM = 9216
FM_INPUT_DIM = 3072  # llm_projection input == backbone dim
FM_TIME_THETA = 10000.0  # TimeEmbedding theta (NOT rope_theta, which is unused: no RoPE here)
FM_NORM_EPS = 1e-5  # dataclass default; absent from params.json
N_DECODING_STEPS = 7  # vLLM-Omni's default; params.json does not set it
CFG_ALPHA = 1.2  # vLLM-Omni's _DEFAULT_CFG_ALPHA; params.json does not set it
FM_SEMANTIC_OUT = 8320  # pad_to_multiple(8192 + 2, 128)

# Codec decoder
CODEC_DIM = 1024
CODEC_HIDDEN_DIM = 4096
CODEC_N_HEADS = 8
CODEC_N_KV_HEADS = 8  # MHA (no grouping)
CODEC_HEAD_DIM = 128
CODEC_NORM_EPS = 1e-2  # params.json "norm_eps": 0.01 — deliberate, not a typo
CODEC_QK_NORM_EPS = 1e-6
CODEC_LAYER_SCALE_INIT = 0.01
CODEC_ATTN_WINDOW = 16  # base; halved per encoder downsample, so the DECODER sees 2,4,8,16
PATCH_PROJ_KERNEL = 7
# decoder_transformer_lengths "2,2,2,2" / convs_kernels "3,4,4,4" / convs_strides "1,2,2,2"
DEC_TF_LENGTHS = (2, 2, 2, 2)
DEC_CONV_KERNELS = (3, 4, 4, 4)
DEC_CONV_STRIDES = (1, 2, 2, 2)
# decoder_blocks.N: 0 = CausalConv1d(292->1024, k3), 2/4/6 = CausalConvTranspose1d(k4, s2);
# odd N = 2-layer transformer, window 2/4/8/16.
DEC_CONV_BLOCKS = (0, 2, 4, 6)
DEC_TF_BLOCKS = (1, 3, 5, 7)
DEC_WINDOWS = (2, 4, 8, 16)


# ---------------------------------------------------------------------------------------
# Minimal safetensors reader (replaces the `safetensors` package)
# ---------------------------------------------------------------------------------------
# Format: u64 little-endian header length | JSON header | raw tensor bytes. Each header entry
# is {"dtype", "shape", "data_offsets": [start, end]} with offsets relative to the data start.
_ST_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


class SafeTensors:
    """Lazy, seek-per-tensor reader: a block reads only its own slice of the single-file
    checkpoint."""

    def __init__(self, path=DEFAULT_CKPT):
        if not os.path.exists(path):
            raise FileNotFoundError(f"checkpoint not found: {path}\n{DOWNLOAD_HINT}")
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(n))
            self._data_start = 8 + n
        self.header.pop("__metadata__", None)

    def __contains__(self, name):
        return name in self.header

    def keys(self):
        return self.header.keys()

    def shape(self, name):
        return tuple(self.header[name]["shape"])

    def get(self, name, dtype=torch.float32):
        """Read one tensor and cast (reference math is fp32; the checkpoint is bf16)."""
        e = self.header[name]
        a, b = e["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self._data_start + a)
            buf = bytearray(f.read(b - a))  # bytearray: writable, so frombuffer stays quiet
        t = torch.frombuffer(buf, dtype=_ST_DTYPES[e["dtype"]]).reshape(e["shape"])
        return t.to(dtype)

    def prefixed(self, prefix, dtype=torch.float32, strip=True):
        """All tensors under `prefix` as a dict (key relative to the prefix if strip)."""
        return {(k[len(prefix) :] if strip else k): self.get(k, dtype) for k in self.header if k.startswith(prefix)}


def load_params(path=DEFAULT_PARAMS):
    """Mistral params.json -> dict. Plain JSON; no vllm config machinery needed."""
    with open(path) as f:
        return json.load(f)


def load_manifest(path=MANIFEST):
    """name -> {dtype, shape} for the released checkpoint. Vendored metadata (no weights) so the
    structural tests run without the checkpoint download."""
    with open(path) as f:
        return json.load(f)


def random_state_from_manifest(prefix="", seed=0, scale=0.02, keys=None, dtype=torch.float32):
    """Random weights at the real checkpoint shapes, for wiring/shape tests with no download.
    `keys` restricts the set (e.g. two backbone layers instead of 26)."""
    man = load_manifest()
    g = torch.Generator().manual_seed(seed)
    names = [k for k in man if k.startswith(prefix)] if keys is None else list(keys)
    out = {}
    for k in names:
        shape = tuple(man[k]["shape"])
        # Norm vectors at 1.0 and LayerScale at its init, or the residual stream explodes.
        if k.endswith(("_norm.weight", "norm.weight")) or k.endswith("cluster_usage"):
            t = torch.ones(shape, dtype=dtype)
        elif k.endswith(("attention_scale", "ffn_scale")):
            t = torch.full(shape, CODEC_LAYER_SCALE_INIT, dtype=dtype)
        else:
            t = torch.randn(shape, generator=g, dtype=dtype) * scale
        out[k[len(prefix) :] if prefix else k] = t
    return out


# ---------------------------------------------------------------------------------------
# Primitives (replace apex FusedRMSNorm / flash-attn / einops)
# ---------------------------------------------------------------------------------------
def rms_norm(x, weight, eps):
    """torch.nn.RMSNorm: x * rsqrt(mean(x^2) + eps) * weight, reduction over the last dim."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def swiglu(x, w1, w2, w3):
    """Upstream FeedForward: w2(silu(w1 x) * w3 x). All three are bias-free here."""
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)


def split_heads(x, n_heads, head_dim):  # [B, S, n*d] -> [B, n, S, d]
    B, S, _ = x.shape
    return x.view(B, S, n_heads, head_dim).permute(0, 2, 1, 3)


def merge_heads(x):  # [B, n, S, d] -> [B, S, n*d]
    B, n, S, d = x.shape
    return x.permute(0, 2, 1, 3).reshape(B, S, n * d)


def repeat_kv(x, repeats):  # [B, n_kv, S, d] -> [B, n_kv*repeats, S, d], interleaved
    if repeats == 1:
        return x
    B, n, S, d = x.shape
    return x.unsqueeze(2).expand(B, n, repeats, S, d).reshape(B, n * repeats, S, d)


def gqa_attention(q, k, v, bias=None):
    """[B,n,S,d] x [B,n_kv,S,d] -> [B,n,S,d]. `bias` is added to the scores pre-softmax
    (causal mask / sliding window / ALiBi all arrive that way). Scale is 1/sqrt(head_dim)."""
    k = repeat_kv(k, q.shape[1] // k.shape[1])
    v = repeat_kv(v, q.shape[1] // v.shape[1])
    scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    if bias is not None:
        scores = scores + bias
    return torch.softmax(scores, dim=-1) @ v


def rope_cis(seq_len, head_dim, theta, offset=0):
    """Mistral-native RoPE table as a complex tensor [S, head_dim/2], rotating INTERLEAVED pairs
    (dims (0,1), (2,3), ...), not HF's half-split, as the Mistral-format checkpoint expects."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(offset, offset + seq_len).float()
    return torch.polar(torch.ones(seq_len, freqs.shape[0]), torch.outer(t, freqs))


def apply_rope(x, cis):  # x [B, n, S, d] (d even), cis [S, d/2]
    B, n, S, d = x.shape
    xc = torch.view_as_complex(x.float().reshape(B, n, S, d // 2, 2))
    out = torch.view_as_real(xc * cis.view(1, 1, S, d // 2)).reshape(B, n, S, d)
    return out.to(x.dtype)


def causal_bias(seq_len, dtype=torch.float32):
    m = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    return torch.triu(m, diagonal=1).view(1, 1, seq_len, seq_len)


def fold_weight_norm(state, key):
    """torch weight_norm parametrization (original0=g magnitude, original1=v direction) ->
    a plain conv weight. dim=0 for both Conv1d [out,in,k] and ConvTranspose1d [in,out,k],
    matching how the checkpoint stores g as [N,1,1]."""
    g = state[key + ".parametrizations.weight.original0"]
    v = state[key + ".parametrizations.weight.original1"]
    return torch._weight_norm(v, g, 0)


def pcc(a, b):
    """Pearson correlation of two tensors, flattened — the accuracy gate used across blocks."""
    a, b = a.detach().flatten().float(), b.detach().flatten().float()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


# ---------------------------------------------------------------------------------------
# Codebook offsets for the 37-way audio token embedding (upstream MultiVocabEmbeddings)
# ---------------------------------------------------------------------------------------
def codebook_sizes(include_special=True):
    """[semantic, acoustic x 36], each +N_AUDIO_SPECIAL when include_special."""
    extra = N_AUDIO_SPECIAL if include_special else 0
    return [SEMANTIC_CODEBOOK_SIZE + extra] + [ACOUSTIC_CODEBOOK_SIZE + extra] * N_ACOUSTIC_CODEBOOK


def codebook_offsets():
    """Per-codebook base offset into the single flat embedding table (cumsum of sizes).
    -> [0, 8194, 8217, 8240, ...]; total 9022, padded to 9088 in the checkpoint."""
    sizes = codebook_sizes()
    out, acc = [], 0
    for s in sizes:
        out.append(acc)
        acc += s
    return torch.tensor(out, dtype=torch.long)


def main():
    """Config self-check against the vendored manifest — runs with no weights present."""
    man = load_manifest()
    print(f"[common] manifest: {len(man)} tensors")

    def chk(name, shape):
        if name not in man:
            return f"MISSING {name}"
        got = tuple(man[name]["shape"])
        return None if got == tuple(shape) else f"SHAPE {name}: ckpt {got} != cfg {tuple(shape)}"

    checks = [
        ("mm_audio_embeddings.tok_embeddings.weight", (VOCAB_SIZE, DIM)),
        ("mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight", (9088, DIM)),
        ("norm.weight", (DIM,)),
        ("layers.0.attention.wq.weight", (ATTN_DIM, DIM)),
        ("layers.0.attention.wk.weight", (KV_DIM, DIM)),
        ("layers.0.attention.wo.weight", (DIM, ATTN_DIM)),
        ("layers.0.feed_forward.w1.weight", (HIDDEN_DIM, DIM)),
        (f"layers.{N_LAYERS - 1}.ffn_norm.weight", (DIM,)),
        ("acoustic_transformer.input_projection.weight", (FM_DIM, N_ACOUSTIC_CODEBOOK)),
        ("acoustic_transformer.llm_projection.weight", (FM_DIM, FM_INPUT_DIM)),
        ("acoustic_transformer.semantic_codebook_output.weight", (FM_SEMANTIC_OUT, FM_DIM)),
        ("acoustic_transformer.acoustic_codebook_output.weight", (N_ACOUSTIC_CODEBOOK, FM_DIM)),
        (f"acoustic_transformer.layers.{FM_N_LAYERS - 1}.ffn_norm.weight", (FM_DIM,)),
        ("audio_tokenizer.quantizer.semantic_codebook.embedding_sum", (SEMANTIC_CODEBOOK_SIZE, SEMANTIC_DIM)),
        ("audio_tokenizer.decoder_blocks.0.conv.parametrizations.weight.original1", (CODEC_DIM, LATENT_DIM, 3)),
        ("audio_tokenizer.decoder_blocks.6.conv.parametrizations.weight.original1", (CODEC_DIM, CODEC_DIM, 4)),
        (
            "audio_tokenizer.output_proj.conv.parametrizations.weight.original1",
            (PATCH_SIZE, CODEC_DIM, PATCH_PROJ_KERNEL),
        ),
        ("audio_tokenizer.decoder_blocks.7.layers.1.attention.q_norm.weight", (CODEC_DIM,)),
        ("audio_tokenizer.decoder_blocks.7.layers.1.ffn_scale", (CODEC_DIM,)),
    ]
    bad = [m for m in (chk(n, s) for n, s in checks) if m]
    for m in bad:
        print("  " + m)
    print(f"[common] config vs checkpoint: {len(checks) - len(bad)}/{len(checks)} OK")

    enc = [k for k in man if k.startswith(("audio_tokenizer.input_proj", "audio_tokenizer.encoder_blocks"))]
    print(
        f"[common] codec ENCODER tensors in checkpoint: {len(enc)} "
        f"({'present' if enc else 'ABSENT -> reference-audio voice cloning impossible; preset voices only'})"
    )
    n_layers = len({k.split(".")[1] for k in man if k.startswith("layers.")})
    print(f"[common] backbone layers in checkpoint: {n_layers} (cfg {N_LAYERS})")
    print(
        f"[common] frame rate {FRAME_RATE} Hz x {NUM_CODEBOOKS} tokens/frame; "
        f"{N_DECODING_STEPS} Euler steps, cfg_alpha {CFG_ALPHA}"
    )
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
