# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU reference for the Voxtral-TTS autoregressive backbone — BLOCK 1 (3.4B, Ministral-derived).

Self-contained (torch only) op-for-op reference for what vLLM-Omni delegates to its registered
`MistralForCausalLM` plus the audio-token embedding it borrows from the codec module. Written
from vllm_omni/model_executor/models/voxtral_tts/{voxtral_tts_audio_generation.py,
voxtral_tts_audio_tokenizer.py} (see ../reference/PROVENANCE.md).

BLOCK BOUNDARY (the tensor in -> out a TTNN port must match):

    inputs_embeds [1, S, 3072]
        -> 26 x { RMSNorm -> GQA causal attention (RoPE) -> RMSNorm -> SwiGLU }   layers.*
        -> norm            (final RMSNorm)                                        norm.weight
        = hidden_states [1, S, 3072]

`hidden_states[:, -1]` is the *h* that Block 2 (flow matching) consumes per frame. Everything
that BUILDS inputs_embeds lives on the input side of the block and is reproduced here because
the decode loop needs it every frame:

    text token      -> tok_embeddings[id]                                (a plain lookup)
    audio frame     -> sum over 37 codebooks of embeddings[code_c + offset_c]   (embed_frame)

The text LM head (tied to tok_embeddings) is only used for the text/EOS path; the semantic
code head lives in Block 2, so it is deliberately NOT part of this block.

ARCHITECTURE NOTES worth carrying into the port:
  * n_heads*head_dim = 4096 != dim = 3072. wq is [4096, 3072] and wo is [3072, 4096]: the
    attention interior is WIDER than the residual stream. Do not assume square.
  * GQA 32/8 (4 query heads per KV head), interleaved repeat_kv.
  * RoPE is Mistral-native INTERLEAVED-pair rotation, not HF half-split. See rope_cis().
  * No biases anywhere; RMSNorm (pre-norm) throughout; SwiGLU MLP.
  * There is no positional table and no sliding window in params.json — plain causal attention
    over max_seq_len 65536, so context length costs KV-cache, nothing else.

Run (regenerates goldens; needs the checkpoint):
    PYTHONPATH=<repo> python models/experimental/voxtral_tts/reference/voxtral_backbone_ref.py
"""

import argparse
import os

import torch
import torch.nn.functional as F

from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ATTN_DIM,
    DEFAULT_CKPT,
    DIM,
    GOLDEN_ROOT,
    HEAD_DIM,
    HIDDEN_DIM,
    KV_DIM,
    N_AUDIO_SPECIAL,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    NORM_EPS,
    NUM_CODEBOOKS,
    ROPE_THETA,
    SafeTensors,
    apply_rope,
    causal_bias,
    codebook_offsets,
    gqa_attention,
    merge_heads,
    pcc,
    rms_norm,
    rope_cis,
    split_heads,
    swiglu,
)

GOLDEN_DIR = os.path.join(GOLDEN_ROOT, "backbone")


def load_backbone_state(ckpt_path=DEFAULT_CKPT, dtype=torch.float32):
    """The 26 transformer layers + final norm + both embedding tables.

    ~6.9 GB in fp32 (3.4B params), so this is the one block whose loader is genuinely heavy;
    `SafeTensors` still seeks per tensor rather than slurping the 8 GB file."""
    st = SafeTensors(ckpt_path)
    w = {"norm": st.get("norm.weight", dtype)}
    for i in range(N_LAYERS):
        p = f"layers.{i}."
        for k in ("attention.wq", "attention.wk", "attention.wv", "attention.wo",
                  "attention_norm", "ffn_norm", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"):
            w[p + k] = st.get(p + k + ".weight", dtype)
    w["tok_embeddings"] = st.get("mm_audio_embeddings.tok_embeddings.weight", dtype)
    w["audio_embeddings"] = st.get("mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight", dtype)
    return w


# ---------------------------------------------------------------------------------------
# Input side: building inputs_embeds
# ---------------------------------------------------------------------------------------
def embed_text(w, token_ids):
    """text token ids [S] (or [1,S]) -> [1, S, 3072]."""
    ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1)
    return w["tok_embeddings"][ids].unsqueeze(0)


def embed_frame(w, codes):
    """One audio frame's 37 codes -> [1, 1, 3072].

    Upstream MultiVocabEmbeddings: each codebook occupies its own slice of ONE flat table, so
    the code is shifted by that codebook's offset before lookup, and the 37 vectors are SUMMED
    (`input_embedding_concat_type: sum`). `codes` are already offset by N_AUDIO_SPECIAL, i.e.
    exactly what Block 2 emits."""
    c = torch.as_tensor(codes, dtype=torch.long).reshape(-1)
    assert c.numel() == NUM_CODEBOOKS, f"expected {NUM_CODEBOOKS} codes, got {c.numel()}"
    return w["audio_embeddings"][c + codebook_offsets()].sum(0).view(1, 1, DIM)


def embed_frames(w, codes):
    """[T, 37] -> [1, T, 3072] (the batched form, used to prefill a reference-voice prompt)."""
    c = torch.as_tensor(codes, dtype=torch.long)
    return w["audio_embeddings"][c + codebook_offsets().view(1, -1)].sum(1).unsqueeze(0)


# ---------------------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------------------
def _layer(x, w, p, cis, bias, cache=None):
    """One pre-norm GQA + SwiGLU block. `cache` (dict) makes it incremental: k/v are appended
    and the whole cache is attended, mirroring the paged KV-cache a TTNN decode path would use."""
    h = rms_norm(x, w[p + "attention_norm"], NORM_EPS)
    q = split_heads(F.linear(h, w[p + "attention.wq"]), N_HEADS, HEAD_DIM)
    k = split_heads(F.linear(h, w[p + "attention.wk"]), N_KV_HEADS, HEAD_DIM)
    v = split_heads(F.linear(h, w[p + "attention.wv"]), N_KV_HEADS, HEAD_DIM)
    q, k = apply_rope(q, cis), apply_rope(k, cis)
    if cache is not None:
        if p in cache:
            k = torch.cat([cache[p][0], k], dim=2)
            v = torch.cat([cache[p][1], v], dim=2)
        cache[p] = (k, v)
    attn = merge_heads(gqa_attention(q, k, v, bias))  # [1, S, 4096]
    x = x + F.linear(attn, w[p + "attention.wo"])  # 4096 -> 3072
    h = rms_norm(x, w[p + "ffn_norm"], NORM_EPS)
    return x + swiglu(h, w[p + "feed_forward.w1"], w[p + "feed_forward.w2"], w[p + "feed_forward.w3"])


@torch.no_grad()
def reference_forward(inputs_embeds, w, n_layers=N_LAYERS):
    """Block 1 prefill: [1, S, 3072] -> hidden_states [1, S, 3072]. Causal, no cache.

    `n_layers` is only for tests: a shortened stack lets the wiring be checked at real widths
    without holding all 26 layers of fp32 weights in RAM."""
    S = inputs_embeds.shape[1]
    cis = rope_cis(S, HEAD_DIM, ROPE_THETA)
    bias = causal_bias(S, inputs_embeds.dtype)
    x = inputs_embeds
    for i in range(n_layers):
        x = _layer(x, w, f"layers.{i}.", cis, bias)
    return rms_norm(x, w["norm"], NORM_EPS)


@torch.no_grad()
def reference_prefill_then_step(inputs_embeds, w, step_embeds, n_layers=N_LAYERS):
    """Prefill [1,P,3072], then feed `step_embeds` [1,T,3072] one position at a time through a
    KV-cache. Returns (prefill_hidden [1,P,3072], step_hidden [1,T,3072]).

    This is the shape the real decode loop has — Block 2 only ever sees the LAST position's
    hidden state — so it is also the golden a traced TTNN decode step should be checked against."""
    cache = {}
    P = inputs_embeds.shape[1]
    cis = rope_cis(P, HEAD_DIM, ROPE_THETA)
    x = inputs_embeds
    for i in range(n_layers):
        x = _layer(x, w, f"layers.{i}.", cis, causal_bias(P, x.dtype), cache)
    prefill_hidden = rms_norm(x, w["norm"], NORM_EPS)

    outs = []
    for t in range(step_embeds.shape[1]):
        x = step_embeds[:, t : t + 1]
        cis_t = rope_cis(1, HEAD_DIM, ROPE_THETA, offset=P + t)
        for i in range(n_layers):
            x = _layer(x, w, f"layers.{i}.", cis_t, None, cache)  # no mask: all cached pos are past
        outs.append(rms_norm(x, w["norm"], NORM_EPS))
    return prefill_hidden, torch.cat(outs, dim=1)


class IncrementalBackbone:
    """Stateful prefill + single-step decode over a KV-cache.

    `reference_prefill_then_step` runs a fixed list of steps; the real generation loop has to
    interleave Block 2 between steps, so it needs this. Deliberately shaped like a TTNN traced
    decoder (build once -> prefill -> step per token) so the port is a drop-in at this seam."""

    def __init__(self, w, n_layers=N_LAYERS):
        self.w, self.n_layers = w, n_layers
        self.cache, self.pos = {}, 0

    def reset(self):
        self.cache, self.pos = {}, 0

    @torch.no_grad()
    def prefill(self, inputs_embeds):
        """[1, P, 3072] -> hidden of the LAST position only [1, 1, 3072] (all Block 2 ever sees)."""
        P = inputs_embeds.shape[1]
        cis = rope_cis(P, HEAD_DIM, ROPE_THETA, offset=self.pos)
        bias = causal_bias(P, inputs_embeds.dtype)
        x = inputs_embeds
        for i in range(self.n_layers):
            x = _layer(x, self.w, f"layers.{i}.", cis, bias, self.cache)
        self.pos += P
        return rms_norm(x[:, -1:], self.w["norm"], NORM_EPS)

    @torch.no_grad()
    def step(self, emb):
        """[1, 1, 3072] -> hidden [1, 1, 3072]. No mask: every cached position is in the past."""
        cis = rope_cis(1, HEAD_DIM, ROPE_THETA, offset=self.pos)
        x = emb
        for i in range(self.n_layers):
            x = _layer(x, self.w, f"layers.{i}.", cis, None, self.cache)
        self.pos += 1
        return rms_norm(x, self.w["norm"], NORM_EPS)


@torch.no_grad()
def text_logits(hidden, w):
    """Tied text head, for the EOS/text path only (Block 2 owns the semantic code head)."""
    return hidden @ w["tok_embeddings"].t()


def make_synthetic_inputs(n_text=6, n_frames=4, seed=0):
    """Deterministic inputs_embeds-shaped input without needing the tokenizer: a few text
    token ids followed by a few audio frames, embedded exactly as the real prompt would be."""
    g = torch.Generator().manual_seed(seed)
    text_ids = torch.randint(0, 32000, (n_text,), generator=g)
    sem = torch.randint(0, 8192, (n_frames, 1), generator=g)
    ac = torch.randint(0, 21, (n_frames, NUM_CODEBOOKS - 1), generator=g)
    frames = torch.cat([sem, ac], dim=1) + N_AUDIO_SPECIAL
    return text_ids, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=GOLDEN_DIR)
    ap.add_argument("--n-text", type=int, default=6)
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=3, help="decode steps to golden after prefill")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[backbone] loading {N_LAYERS} layers from {args.ckpt} (fp32, ~6.9 GB)")
    w = load_backbone_state(args.ckpt)

    text_ids, frames = make_synthetic_inputs(args.n_text, args.n_frames)
    embeds = torch.cat([embed_text(w, text_ids), embed_frames(w, frames)], dim=1)
    print(f"[backbone] inputs_embeds {tuple(embeds.shape)} ({args.n_text} text + {args.n_frames} frames)")

    hidden = reference_forward(embeds, w)
    print(f"[backbone] prefill hidden {tuple(hidden.shape)} "
          f"(mean {hidden.mean():+.4f}, std {hidden.std():.4f})")

    # Incremental path must reproduce prefill exactly, and gives the per-step goldens.
    P = embeds.shape[1] - args.n_steps
    pre, steps = reference_prefill_then_step(embeds[:, :P], w, embeds[:, P:])
    print(f"[backbone] cache path: prefill PCC {pcc(pre, hidden[:, :P]):.6f}, "
          f"steps PCC {pcc(steps, hidden[:, P:]):.6f} ({args.n_steps} steps)")

    # Single-frame embedding must equal the batched one (the decode loop uses the single form).
    one = embed_frame(w, frames[0])
    print(f"[backbone] embed_frame vs embed_frames: max abs diff "
          f"{(one - embed_frames(w, frames[:1])).abs().max():.3e}")

    torch.save(embeds, os.path.join(args.out, "inputs_embeds.pt"))
    torch.save(hidden, os.path.join(args.out, "hidden_states.pt"))
    torch.save(steps, os.path.join(args.out, "step_hidden.pt"))
    torch.save(text_ids, os.path.join(args.out, "text_ids.pt"))
    torch.save(frames, os.path.join(args.out, "frames.pt"))
    torch.save({"n_text": args.n_text, "n_frames": args.n_frames, "prefill_len": P,
                "n_steps": args.n_steps, "dim": DIM, "n_layers": N_LAYERS,
                "attn_dim": ATTN_DIM, "kv_dim": KV_DIM, "hidden_dim": HIDDEN_DIM},
               os.path.join(args.out, "meta.pt"))
    print(f"[backbone] wrote goldens to {args.out}")


if __name__ == "__main__":
    main()
