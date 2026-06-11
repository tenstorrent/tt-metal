# MiniMax-M2 — Prefill bring-up

TTNN implementation of **MiniMax-M2** prefill inference for Tenstorrent.
Target: one Blackhole Galaxy (4×8). Model: 62-layer MoE decoder, 256 experts /
top-8 sigmoid routing, GQA (48 q / 8 kv heads, head_dim 128, partial RoPE 64),
QK-norm, SiLU-SwiGLU experts. Config: [`configs/MiniMax-M2/config.json`](configs/MiniMax-M2/config.json).

## Read these first
- **[PREFILL_PROPOSAL.md](PREFILL_PROPOSAL.md)** — architecture + work breakdown (living doc;
  always reflects current status: validated vs scaffold).
- **[SKELETON.md](SKELETON.md)** — the serving-skeleton work split (who owns which seam).

## Status (single Wormhole, TP=1, random weights, seq=128)
Model math validated vs the HuggingFace reference (`MiniMaxM2*`, loaded via
loaded from a downloaded checkpoint via `HF_MODEL`):

| Component | PCC vs HF |
|---|---|
| attention block | 0.9991 |
| MoE router (decomposed) | sigmoid 0.99999 / weights 0.9993 |
| experts (SiLU SwiGLU) | 0.9990 |
| full decoder layer | 0.99993 |

Not yet validated (need the multi-card Blackhole box): TP/EP/SP collectives, paged
KV read-back, full-model logits, real weights, bfp4 accuracy, decode, long context.

## Tests
```
pip install -r requirements.txt          # transformers==4.57.1 (M2 is a trust_remote_code model)
pytest models/demos/minimax_m2/tests/unit/
```

## Layout
```
tt/              model: attention/ experts/ topk.py mlp.py layer.py model.py
tt/runners/      prefill serving skeleton (scaffold) — see SKELETON.md
configs/MiniMax-M2/   config.json (dims only; modeling code is NOT vendored —
                      HF-reference tests load it from a checkpoint via HF_MODEL)
tests/unit/      module-by-module PCC tests vs the HF reference
```
