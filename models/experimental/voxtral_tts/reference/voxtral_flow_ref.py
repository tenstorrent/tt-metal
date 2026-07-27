# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU reference for the Voxtral-TTS flow-matching acoustic transformer — BLOCK 2 (390M).

Self-contained (torch only) op-for-op reference for upstream
`FlowMatchingAudioTransformer` (vllm_omni/model_executor/models/voxtral_tts/
voxtral_tts_audio_generation.py). See ../reference/PROVENANCE.md.

BLOCK BOUNDARY (per generated frame):

    h = backbone hidden state [B, 3072]
      -> semantic_codebook_output(h) -> mask -> argmax        = semantic code   [B, 1]
      -> 7 x Euler step of a 3-layer bidirectional transformer over a 3-TOKEN sequence
         with classifier-free guidance                        = acoustic floats [B, 36]
      -> clamp(-1,1) -> scale to 21 FSQ levels -> round       = acoustic codes  [B, 36]
      = audio_codes [B, 37]   (+N_AUDIO_SPECIAL offset, ready for Block 1's embed_frame)

WHY THIS BLOCK IS THE INTERESTING ONE FOR TTNN:

  * The transformer sees a sequence of exactly THREE tokens —
    [input_projection(x_t), time_projection(t_emb), llm_projection(h)] — and reads the
    velocity off position 0. Tiny per call, but called 7 x per frame, and every call is
    THE SAME SHAPE. Upstream wraps the whole ODE solver in one CUDA graph and reports 47%
    lower latency / 2.5x RTF; the direct analogue is capturing all 7 steps in ONE device trace.
  * Attention is BIDIRECTIONAL and unmasked (no RoPE, no causal mask) despite the GQA 32/8
    layout — `rope_theta` in params.json is inert for this module. Do not add RoPE.
  * CFG is done by batching cond+uncond to 2B in a single forward (uncond = zeroed h), so the
    step is a batch-2 graph, not two graphs.
  * x_0 is Gaussian noise, so this block is NOT deterministic unless the generator is seeded.
    `decode_frame(..., x_0=...)` takes an explicit x_0 so PCC tests stay deterministic.

Run (regenerates goldens; needs the checkpoint — only ~1.6 GB of it is read):
    PYTHONPATH=<repo> python models/experimental/voxtral_tts/reference/voxtral_flow_ref.py
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F

from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    ACOUSTIC_CODEBOOK_SIZE,
    CFG_ALPHA,
    DEFAULT_CKPT,
    EMPTY_AUDIO_ID,
    END_AUDIO_ID,
    FM_DIM,
    FM_HEAD_DIM,
    FM_HIDDEN_DIM,
    FM_INPUT_DIM,
    FM_N_HEADS,
    FM_N_KV_HEADS,
    FM_N_LAYERS,
    FM_NORM_EPS,
    FM_TIME_THETA,
    GOLDEN_ROOT,
    N_ACOUSTIC_CODEBOOK,
    N_AUDIO_SPECIAL,
    N_DECODING_STEPS,
    SEMANTIC_CODEBOOK_SIZE,
    SafeTensors,
    gqa_attention,
    merge_heads,
    pcc,
    rms_norm,
    split_heads,
    swiglu,
)

GOLDEN_DIR = os.path.join(GOLDEN_ROOT, "flow")
PREFIX = "acoustic_transformer."


def load_flow_state(ckpt_path=DEFAULT_CKPT, dtype=torch.float32):
    """The 33 acoustic_transformer tensors, keyed relative to the module."""
    st = SafeTensors(ckpt_path)
    w = st.prefixed(PREFIX, dtype)
    # time_embedding.inv_freq is registered persistent=True upstream but is ABSENT from the
    # released checkpoint, so it must be recomputed (deterministic — see time_embedding()).
    w["time_embedding.inv_freq"] = _inv_freq(FM_DIM, FM_TIME_THETA)
    return w


def _inv_freq(dim, theta):
    half = dim // 2
    return torch.exp(-math.log(theta) * torch.arange(half).float() / half)


def time_embedding(t, inv_freq):
    """Sinusoidal time embedding. t [B, 1] -> [B, dim] as cat(cos, sin) (that ORDER; upstream
    is `torch.cat((emb.cos(), emb.sin()), dim=-1)`)."""
    emb = t.float() @ inv_freq.unsqueeze(0)  # einsum("bi,j->bj") for i == 1
    return torch.cat((emb.cos(), emb.sin()), dim=-1)


def _block(x, w, p):
    """Bidirectional pre-norm block: NO RoPE, NO mask. GQA 32/8 over a 3-token sequence."""
    h = rms_norm(x, w[p + "attention_norm.weight"], FM_NORM_EPS)
    q = split_heads(F.linear(h, w[p + "attention.wq.weight"]), FM_N_HEADS, FM_HEAD_DIM)
    k = split_heads(F.linear(h, w[p + "attention.wk.weight"]), FM_N_KV_HEADS, FM_HEAD_DIM)
    v = split_heads(F.linear(h, w[p + "attention.wv.weight"]), FM_N_KV_HEADS, FM_HEAD_DIM)
    attn = merge_heads(gqa_attention(q, k, v, bias=None))  # [B, 3, 4096]
    x = x + F.linear(attn, w[p + "attention.wo.weight"])
    h = rms_norm(x, w[p + "ffn_norm.weight"], FM_NORM_EPS)
    return x + swiglu(h, w[p + "feed_forward.w1.weight"], w[p + "feed_forward.w2.weight"],
                      w[p + "feed_forward.w3.weight"])


@torch.no_grad()
def predict_velocity(x_t, llm_output, t_emb, w):
    """[B,36], [B,3072], [B,3072] -> velocity [B,36].

    The 3-token sequence is assembled here; the velocity is read off POSITION 0 only. The other
    two positions exist purely so attention can mix time and LLM conditioning into it."""
    seq = torch.cat(
        [
            F.linear(x_t, w["input_projection.weight"]).unsqueeze(1),  # [B,1,3072]
            F.linear(t_emb, w["time_projection.weight"]).unsqueeze(1),
            F.linear(llm_output, w["llm_projection.weight"]).unsqueeze(1),
        ],
        dim=1,
    )  # [B, 3, 3072]
    for i in range(FM_N_LAYERS):
        seq = _block(seq, w, f"layers.{i}.")
    final = rms_norm(seq, w["norm.weight"], FM_NORM_EPS)
    return F.linear(final[:, 0, :], w["acoustic_codebook_output.weight"])  # [B, 36]


@torch.no_grad()
def semantic_code(llm_hidden, w):
    """h [B,3072] -> semantic code [B,1]. Greedy argmax over the masked semantic logits:
    [EMPTY_AUDIO] is forbidden ([END_AUDIO] is allowed — that is how generation stops), and
    everything past the real codebook (the pad up to 8320) is forbidden."""
    logits = F.linear(llm_hidden, w["semantic_codebook_output.weight"]).float()
    logits[:, EMPTY_AUDIO_ID] = -float("inf")
    logits[:, N_AUDIO_SPECIAL + SEMANTIC_CODEBOOK_SIZE :] = -float("inf")
    return logits.argmax(dim=-1, keepdim=True)


@torch.no_grad()
def decode_frame(sem_code, llm_hidden, w, cfg_alpha=CFG_ALPHA, n_steps=N_DECODING_STEPS,
                 x_0=None, noise_scale=1.0, return_trace=False):
    """Euler-integrate the velocity field to acoustic codes. [B,1], [B,3072] -> [B,36] ints.

    `x_0=None` draws fresh Gaussian noise (real inference); pass x_0 for a deterministic test.
    Frames whose semantic code is [END_AUDIO] are not decoded — their acoustic slots become
    [EMPTY_AUDIO] — which is why the returned codes must be read together with sem_code."""
    B = sem_code.shape[0]
    should_decode = (sem_code != END_AUDIO_ID).reshape(B)
    x = (torch.randn(B, N_ACOUSTIC_CODEBOOK) if x_0 is None else x_0.clone()) * noise_scale
    timesteps = torch.linspace(0, 1, n_steps + 1)
    zero_h = torch.zeros_like(llm_hidden)
    alpha = torch.as_tensor(cfg_alpha).reshape(-1, 1).expand(B, 1) if not torch.is_tensor(cfg_alpha) \
        or cfg_alpha.ndim == 0 else cfg_alpha.reshape(B, 1)
    trace = []
    for i in range(n_steps):
        t, dt = timesteps[i], timesteps[i + 1] - timesteps[i]
        t_emb = time_embedding(t.view(1, 1).repeat(B, 1), w["time_embedding.inv_freq"])
        # cond + uncond batched into ONE forward (2B); uncond zeroes the LLM conditioning.
        v_all = predict_velocity(
            torch.cat([x, x], dim=0),
            torch.cat([llm_hidden, zero_h], dim=0),
            torch.cat([t_emb, t_emb], dim=0),
            w,
        )
        v = alpha * v_all[:B] + (1 - alpha) * v_all[B:]
        x = x + v * dt
        if return_trace:
            trace.append(x.clone())
    codes = _fsq_quantize(x)
    codes[~should_decode] = EMPTY_AUDIO_ID
    codes = codes + N_AUDIO_SPECIAL
    return (codes, torch.stack(trace)) if return_trace else codes


def _fsq_quantize(x):
    """clamp to [-1,1], rescale onto 0..levels-1, round. Mirrors upstream exactly (note the
    manual clamp — upstream does NOT tanh here, unlike the codec's encode path)."""
    x = torch.clamp(x, -1, 1)
    return (((x + 1) / 2) * (ACOUSTIC_CODEBOOK_SIZE - 1)).round().long()


@torch.no_grad()
def reference_frame(llm_hidden, w, **kw):
    """Full Block 2: h [B,3072] -> audio_codes [B,37] (semantic ++ acoustic), offset applied."""
    sem = semantic_code(llm_hidden, w)
    return torch.cat([sem, decode_frame(sem, llm_hidden, w, **kw)], dim=1)


def make_synthetic_inputs(batch=2, seed=0):
    """Deterministic h and x_0 so the block can be exercised without Block 1. h is scaled to a
    plausible post-RMSNorm magnitude (unit-ish per channel)."""
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(batch, FM_INPUT_DIM, generator=g)
    x_0 = torch.randn(batch, N_ACOUSTIC_CODEBOOK, generator=torch.Generator().manual_seed(seed + 1))
    return h, x_0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=GOLDEN_DIR)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[flow] loading acoustic_transformer ({FM_N_LAYERS} layers) from {args.ckpt}")
    w = load_flow_state(args.ckpt)
    print(f"[flow] {len(w)} tensors; {N_DECODING_STEPS} Euler steps, cfg_alpha {CFG_ALPHA}, "
          f"3-token sequence, dim {FM_DIM}")

    h, x_0 = make_synthetic_inputs(args.batch)
    sem = semantic_code(h, w)
    codes, trace = decode_frame(sem, h, w, x_0=x_0, return_trace=True)
    frame = torch.cat([sem, codes], dim=1)
    print(f"[flow] semantic {sem.reshape(-1).tolist()} | acoustic range "
          f"[{int(codes.min())}, {int(codes.max())}] | frame {tuple(frame.shape)}")
    print(f"[flow] ODE trace {tuple(trace.shape)}: |x| step0 {trace[0].abs().mean():.4f} "
          f"-> step{N_DECODING_STEPS - 1} {trace[-1].abs().mean():.4f}")

    # cfg_alpha=1 must equal the purely conditional field (a cheap check that CFG is wired right).
    only_cond = decode_frame(sem, h, w, cfg_alpha=1.0, x_0=x_0, return_trace=True)[1][-1]
    print(f"[flow] cfg_alpha 1.2 vs 1.0 final-x PCC {pcc(trace[-1], only_cond):.6f} "
          f"(should be < 1.0 — guidance is doing something)")

    # A single velocity evaluation is the unit a TTNN trace would capture.
    t_emb = time_embedding(torch.zeros(args.batch, 1), w["time_embedding.inv_freq"])
    v0 = predict_velocity(x_0, h, t_emb, w)
    print(f"[flow] one velocity eval: {tuple(v0.shape)} (mean {v0.mean():+.4f}, std {v0.std():.4f})")

    torch.save(h, os.path.join(args.out, "llm_hidden.pt"))
    torch.save(x_0, os.path.join(args.out, "x_0.pt"))
    torch.save(t_emb, os.path.join(args.out, "t_emb.pt"))
    torch.save(v0, os.path.join(args.out, "velocity.pt"))
    torch.save(trace, os.path.join(args.out, "ode_trace.pt"))
    torch.save(frame, os.path.join(args.out, "audio_codes.pt"))
    torch.save({"batch": args.batch, "n_steps": N_DECODING_STEPS, "cfg_alpha": CFG_ALPHA,
                "dim": FM_DIM, "n_layers": FM_N_LAYERS, "seq_len": 3,
                "n_acoustic": N_ACOUSTIC_CODEBOOK, "levels": ACOUSTIC_CODEBOOK_SIZE},
               os.path.join(args.out, "meta.pt"))
    print(f"[flow] wrote goldens to {args.out}")


if __name__ == "__main__":
    main()
