# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only (torch, no device) validation of SageAttention 'smooth K' for the
BGE-M3 encoder-SDPA BF8-score path.

Hypothesis: q256/k2048 SDPA (21.5 ms/call, would cross sub-1s) needs a BF8 score
CB (CB_QK). Raw BF8 score gave full-model PCC 0.31. SageAttention shows the cause
is K's shared-per-channel-bias outlier inflating the pre-softmax scores S=QK^T so
BF8's ~3-mantissa-bit precision destroys the inter-token signal. Subtracting the
per-token mean of K is MATHEMATICALLY EXACT for attention (the mean is a per-row
constant that cancels in softmax) but re-centers S so BF8 can represent it.

This probe captures REAL BGE-M3 layer activations (Q,K) and compares softmax of:
  (a) fp32 reference:          softmax(QK^T)
  (b) BF8 score, raw:          softmax(bf8(QK^T))
  (c) BF8 score, smooth-K:     softmax(bf8(Q(K-mean_tok(K))^T))
Reports PCC/cosine of the attention PROBABILITIES vs the fp32 reference. If (c)
recovers toward 1.0 while (b) is ~0.3, smooth-K reopens the sub-1s path.

Run: python models/demos/wormhole/bge_m3/tests/perf/smooth_k_bf8_probe.py
"""

from __future__ import annotations

import torch

MODEL_ID = "BAAI/bge-m3"
SEQ = 8192
HEAD_DIM = 64


def tt_bfloat8_b(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Simulate TT bfloat8_b: per-16-element block shared 8-bit exponent, each
    element 1 sign + 7 mantissa bits (bfloat8 block float). We emulate by, per
    block along the last dim, taking the block max-abs exponent and quantizing
    each element's mantissa to 7 bits relative to that shared exponent.
    """
    orig_shape = x.shape
    xf = x.reshape(-1, block).to(torch.float32)
    # shared exponent per block = floor(log2(max|x|))
    maxabs = xf.abs().amax(dim=1, keepdim=True)
    maxabs = torch.where(maxabs == 0, torch.ones_like(maxabs), maxabs)
    shared_exp = torch.floor(torch.log2(maxabs))  # per block
    # mantissa: 7 bits => 2^7 levels relative to 2^(shared_exp+1)
    scale = torch.pow(2.0, shared_exp - 6)  # 7-bit mantissa step (incl implicit)
    q = torch.round(xf / scale) * scale
    return q.reshape(orig_shape).to(x.dtype)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0
    return (torch.dot(a, b).item()) / denom


def capture_qk(layer_idx: int):
    """Capture real Q,K for one encoder layer at S8192 via forward hooks."""
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).eval()
    backbone = model.roberta if hasattr(model, "roberta") else model

    torch.manual_seed(42)
    vocab = model.config.vocab_size
    input_ids = torch.randint(1, vocab, (1, SEQ), dtype=torch.long)

    captured = {}
    layer = backbone.encoder.layer[layer_idx].attention.self

    def hook(module, inp, out):
        hs = inp[0]  # [1, S, D]
        q = module.query(hs)
        k = module.key(hs)
        B, S, D = q.shape
        H = module.num_attention_heads
        dh = D // H
        q = q.view(B, S, H, dh).permute(0, 2, 1, 3)  # [1,H,S,dh]
        k = k.view(B, S, H, dh).permute(0, 2, 1, 3)
        captured["q"] = q.detach()
        captured["k"] = k.detach()

    h = layer.register_forward_hook(hook)
    with torch.no_grad():
        backbone(input_ids=input_ids, attention_mask=None)
    h.remove()
    return captured["q"], captured["k"]


def main():
    print("Smooth-K BF8-score validation (host-only, TT bfloat8_b emulation)\n")
    for layer_idx in (0, 11, 23):
        q, k = capture_qk(layer_idx)  # [1,H,S,dh]
        H = q.shape[1]
        scale = 1.0 / (HEAD_DIM**0.5)

        # fp32 reference scores + softmax (per head)
        s_ref = torch.matmul(q, k.transpose(-1, -2)) * scale  # [1,H,S,S]
        p_ref = torch.softmax(s_ref, dim=-1)

        # (b) raw BF8 score: quantize S to bf8, then softmax
        s_raw_bf8 = tt_bfloat8_b(s_ref)
        p_raw = torch.softmax(s_raw_bf8, dim=-1)

        # (c) smooth-K: subtract per-token mean of K (mean over token dim S)
        k_smooth = k - k.mean(dim=2, keepdim=True)  # [1,H,1,dh] bias removed
        s_smooth = torch.matmul(q, k_smooth.transpose(-1, -2)) * scale
        s_smooth_bf8 = tt_bfloat8_b(s_smooth)
        p_smooth = torch.softmax(s_smooth_bf8, dim=-1)

        # sanity: smooth-K in fp32 must equal reference (exactness of the identity)
        exact = pcc(torch.softmax(s_smooth, dim=-1), p_ref)

        pcc_raw = pcc(p_raw, p_ref)
        pcc_smooth = pcc(p_smooth, p_ref)
        # also report score dynamic range before/after centering
        rng_raw = (s_ref.abs().mean().item(), s_ref.abs().amax().item())
        rng_sm = (s_smooth.abs().mean().item(), s_smooth.abs().amax().item())
        print(
            f"layer {layer_idx:>2}: softmax-PCC vs fp32  raw-bf8={pcc_raw:.4f}  "
            f"smooth-K-bf8={pcc_smooth:.4f}  (fp32 smooth-K identity check={exact:.5f})"
        )
        print(
            f"          |S| mean/max  raw={rng_raw[0]:.2f}/{rng_raw[1]:.2f}  "
            f"smoothed={rng_sm[0]:.2f}/{rng_sm[1]:.2f}"
        )


if __name__ == "__main__":
    main()
