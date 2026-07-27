"""PCC of our Block 1 (AR backbone) against mistral_inference — Mistral's OWN reference.

vLLM-Omni delegates the backbone to vLLM's MistralForCausalLM, which is not CPU-runnable here
(paged attention, no GPU). mistral_inference is the authoritative reference for this
architecture AND reads the same consolidated/params.json format, which makes it the right thing
to check against — in particular it settles the RoPE convention (interleaved pairs vs HF's
half-split), the one choice that fails silently.

Its Attention calls xformers memory_efficient_attention, which has no CPU kernel. We patch in
torch's F.scaled_dot_product_attention (causal). That is a THIRD implementation, independent of
both ours and xformers', so agreement validates our attention as well as our layer wiring —
it is not circular.

    PYTHONPATH=$TT_METAL_HOME ./cmp_venv/bin/python compare_backbone.py
"""

import os
import pathlib
import sys

import torch
import torch.nn.functional as F

REPO = os.environ.get("TT_METAL_HOME") or str(pathlib.Path(__file__).resolve().parents[4])
CKPT = f"{REPO}/models/experimental/voxtral_tts/reference/weights/consolidated.safetensors"
sys.path.insert(0, REPO)

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as B  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (  # noqa: E402
    DIM, HEAD_DIM, HIDDEN_DIM, NORM_EPS, N_HEADS, N_KV_HEADS, N_LAYERS, ROPE_THETA,
    SafeTensors, apply_rope, pcc, repeat_kv, rms_norm, rope_cis, swiglu,
)

RESULTS = []


def report(name, a, b, gate=0.9999):
    p = pcc(a, b)
    mx = (a.float() - b.float()).abs().max().item()
    ok = p >= gate
    RESULTS.append((name, p, mx, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name:56s} PCC {p:.8f}  maxabs {mx:.3e}")


def _sdpa_causal(q, k, v, attn_bias=None, p=0.0, scale=None):
    """xformers layout (B,S,H,D) -> torch SDPA (B,H,S,D). Always causal (see docstring)."""
    q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    return F.scaled_dot_product_attention(q_, k_, v_, is_causal=True).transpose(1, 2)


def main():
    import mistral_inference.transformer_layers as mtl
    from mistral_inference.rope import apply_rotary_emb, precompute_freqs_cis

    mtl.memory_efficient_attention = _sdpa_causal  # CPU substitute

    S = 12
    torch.manual_seed(3)

    # ---------------------------------------------------------------------------------
    # 1) RoPE table + application — the convention question
    # ---------------------------------------------------------------------------------
    print("=== BLOCK 1: RoPE convention (interleaved pairs vs half-split) ===")
    their_cis = precompute_freqs_cis(HEAD_DIM, S, ROPE_THETA)
    our_cis = rope_cis(S, HEAD_DIM, ROPE_THETA)
    report("b1 rope table (complex, real part)", our_cis.real, their_cis.real)
    report("b1 rope table (complex, imag part)", our_cis.imag, their_cis.imag)

    xq = torch.randn(S, N_HEADS, HEAD_DIM)
    xk = torch.randn(S, N_KV_HEADS, HEAD_DIM)
    tq, tk = apply_rotary_emb(xq, xk, their_cis)
    oq = apply_rope(xq.permute(1, 0, 2).unsqueeze(0), our_cis)  # (S,H,D) -> (1,H,S,D)
    ok_ = apply_rope(xk.permute(1, 0, 2).unsqueeze(0), our_cis)
    report("b1 apply_rope on Q", oq, tq.permute(1, 0, 2).unsqueeze(0))
    report("b1 apply_rope on K", ok_, tk.permute(1, 0, 2).unsqueeze(0))

    # ---------------------------------------------------------------------------------
    # 2) Primitives
    # ---------------------------------------------------------------------------------
    print("\n=== BLOCK 1: primitives ===")
    st = SafeTensors(CKPT)
    g = st.get("layers.0.attention_norm.weight", torch.float32)
    mn = mtl.RMSNorm(DIM, eps=NORM_EPS)
    with torch.no_grad():
        mn.weight.copy_(g)
    x = torch.randn(S, DIM)
    with torch.no_grad():
        report("b1 RMSNorm (eps 1e-5)", rms_norm(x, g, NORM_EPS), mn(x))

    ff = mtl.FeedForward(dim=DIM, hidden_dim=HIDDEN_DIM)
    w1 = st.get("layers.0.feed_forward.w1.weight", torch.float32)
    w2 = st.get("layers.0.feed_forward.w2.weight", torch.float32)
    w3 = st.get("layers.0.feed_forward.w3.weight", torch.float32)
    with torch.no_grad():
        ff.w1.weight.copy_(w1); ff.w2.weight.copy_(w2); ff.w3.weight.copy_(w3)
        report("b1 SwiGLU FeedForward (3072->9216->3072)", swiglu(x, w1, w2, w3), ff(x))
    del ff, w1, w2, w3

    kv = torch.randn(1, N_KV_HEADS, S, HEAD_DIM)
    their_rep = torch.repeat_interleave(kv, N_HEADS // N_KV_HEADS, dim=1)
    report("b1 repeat_kv GQA 32/8 (interleaved)", repeat_kv(kv, N_HEADS // N_KV_HEADS), their_rep)

    # ---------------------------------------------------------------------------------
    # 3) One full transformer block on real layer-0 weights
    # ---------------------------------------------------------------------------------
    print("\n=== BLOCK 1: full transformer block (real weights) ===")
    blk = mtl.TransformerBlock(dim=DIM, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS,
                               n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM, norm_eps=NORM_EPS)
    blk = blk.float().eval()

    def load_layer(i):
        p = f"layers.{i}."
        sd = {
            "attention.wq.weight": st.get(p + "attention.wq.weight", torch.float32),
            "attention.wk.weight": st.get(p + "attention.wk.weight", torch.float32),
            "attention.wv.weight": st.get(p + "attention.wv.weight", torch.float32),
            "attention.wo.weight": st.get(p + "attention.wo.weight", torch.float32),
            "attention_norm.weight": st.get(p + "attention_norm.weight", torch.float32),
            "ffn_norm.weight": st.get(p + "ffn_norm.weight", torch.float32),
            "feed_forward.w1.weight": st.get(p + "feed_forward.w1.weight", torch.float32),
            "feed_forward.w2.weight": st.get(p + "feed_forward.w2.weight", torch.float32),
            "feed_forward.w3.weight": st.get(p + "feed_forward.w3.weight", torch.float32),
        }
        missing, unexpected = blk.load_state_dict(sd, strict=False)
        assert not [m for m in missing if "lora" not in m] and not unexpected, (missing, unexpected)
        # our reference wants the same tensors keyed without the trailing ".weight"
        return {k[: -len(".weight")]: v for k, v in sd.items()}

    x1 = torch.randn(1, S, DIM) * 0.5
    from models.experimental.voxtral_tts.reference.voxtral_backbone_ref import _layer
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import causal_bias

    for i in (0, 1, 13, N_LAYERS - 1):
        w = {f"layers.{i}." + k: v for k, v in load_layer(i).items()}
        with torch.no_grad():
            theirs = blk(x1[0], their_cis)  # unbatched (S, dim)
            ours = _layer(x1, w, f"layers.{i}.", our_cis, causal_bias(S, x1.dtype))
        report(f"b1 TransformerBlock layer {i}", ours[0], theirs)

    # ---------------------------------------------------------------------------------
    # 4) Full 26-layer stack + final norm (one block object, weights swapped per layer)
    # ---------------------------------------------------------------------------------
    print("\n=== BLOCK 1: full 26-layer stack ===")
    with torch.no_grad():
        t = x1[0].clone()
        for i in range(N_LAYERS):
            load_layer(i)
            t = blk(t, their_cis)
        t = rms_norm(t, st.get("norm.weight", torch.float32), NORM_EPS)  # same final norm both sides
        w_all = B.load_backbone_state(CKPT)
        ours = B.reference_forward(x1, w_all)
    report("b1 26 layers + final norm", ours[0], t, gate=0.999)

    print("\n=== SUMMARY ===")
    n_ok = sum(1 for _, _, _, ok in RESULTS if ok)
    for name, p, mx, ok in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:56s} {p:.8f}")
    print(f"  {n_ok}/{len(RESULTS)} checks pass")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
