#!/usr/bin/env python3
"""
Decode-step-0 reference for the Qwen3.6-27B text model.

Reuses the faithful CPU oracle in hf_ref.py (same RoPE positions 0..N-1, same
DeltaNet recurrence, same gating/norm hypotheses).

Decode step 0 = feeding the predicted token ' Paris' (id 11751) at position 5,
given the 5-token prompt already processed. By causality this equals a full
forward on the 6-token sequence [760, 6511, 314, 9338, 369, 11751] read at
position index 5 (the last position).

Dumps, for position 5 only:
  embed       : input embedding                       [1, 5120]
  layers      : residual-stream output after each of  [64, 1, 5120]
                the 64 decoder layers
  final_norm  : output of the final RMSNorm           [1, 5120]
  logits      : lm_head logits                         [1, 248320]
to /home/yito/work/hf_dec_ref.pt

Run inside docker image qwen36-wh-test:latest:
  docker run --rm -e HF_HUB_OFFLINE=1 \
    -v /home/yito/tt-metal:/home/yito/tt-metal -v /home/yito/work:/home/yito/work \
    -w /home/yito/tt-metal qwen36-wh-test:latest \
    /opt/venv/bin/python3 models/demos/qwen36_27b/t3k/hf_dec_ref.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hf_ref as R  # the faithful oracle

OUT_PT = "/home/yito/work/hf_dec_ref.pt"
POS = 5  # position index of ' Paris' in the 6-token sequence


def main():
    w = R.Weights(R.HF_DIR)

    # 6-token sequence: "The capital of France is" + predicted " Paris" (11751)
    input_ids = torch.tensor([[760, 6511, 314, 9338, 369, 11751]], dtype=torch.long)
    seq = input_ids.shape[1]
    assert seq == 6
    position_ids = torch.arange(seq, dtype=torch.long)  # 0..5, same as prefill
    cos, sin = R.build_rope(position_ids)

    embed = w.get("model.language_model.embed_tokens.weight")  # [VOCAB, HIDDEN]
    hidden = F.embedding(input_ids, embed)  # [1, seq, HIDDEN]
    embed_pos = hidden[:, POS, :].clone()   # [1, HIDDEN]

    layer_acts = torch.empty(R.N_LAYERS, 1, R.HIDDEN, dtype=R.DTYPE)
    for li in range(R.N_LAYERS):
        hidden = R.decoder_layer(hidden, w, li, cos, sin)
        layer_acts[li] = hidden[:, POS, :].detach()
        print(f"  layer {li:2d} ({R.LAYER_TYPES[li]:16s}) "
              f"pos{POS} mean={hidden[:, POS, :].mean().item():.5f} "
              f"std={hidden[:, POS, :].std().item():.5f}")

    norm_w = w.get("model.language_model.norm.weight")
    final_norm_full = R.rmsnorm_w_plus_1(hidden, norm_w)  # [1, seq, HIDDEN]
    final_norm_pos = final_norm_full[:, POS, :]           # [1, HIDDEN]

    lm_head = w.get("lm_head.weight")  # [VOCAB, HIDDEN]
    logits_pos = F.linear(final_norm_pos, lm_head)  # [1, VOCAB]

    # top-5 of position-5 logits (predicted token AFTER ' Paris')
    topv, topi = torch.topk(logits_pos[0], 5)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(R.HF_DIR, trust_remote_code=True)
        decoded = [tok.decode([int(i)]) for i in topi]
    except Exception as e:
        decoded = [f"<id {int(i)}>" for i in topi]
        print("tokenizer load failed:", e)

    print("\n=== TOP-5 TOKEN AFTER ' Paris' (position 5) ===")
    for v, i, d in zip(topv.tolist(), topi.tolist(), decoded):
        print(f"  id={i:7d} logit={v:9.4f}  {d!r}")

    out = {
        "embed": embed_pos,             # [1, 5120]
        "layers": layer_acts,           # [64, 1, 5120]
        "final_norm": final_norm_pos,   # [1, 5120]
        "logits": logits_pos,           # [1, 248320]
    }
    torch.save(out, OUT_PT)
    print(f"\nSaved decode-step-0 reference to {OUT_PT}")
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
