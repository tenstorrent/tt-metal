# Qwen3.6-27B Galaxy v2 — Bringup Log

Live tracker. Append-only. Mirrors the format in
`models/demos/olmo_galaxy/BRINGUP_LOG.md`.

## Current Status

**Stage:** V2-1 (bulk copy) complete. V2-2 (qwen_model_config.py) in progress.

| stage | status |
|---|---|
| V2-1 bulk copy from llama3_70b_galaxy | DONE |
| V2-2 qwen_model_config.py — is_qwen36 + full-grid | in progress |
| V2-3 load_checkpoints.py — qwen3.6 HF key map | pending |
| V2-4 llama_attention.py — is_qwen36 branch | pending |
| V2-5 qwen36_delta_attention.py + decoder dispatch | pending |
| V2-6 llama_ccl.py — dual-dtype persistent buffer keys | pending |
| V2-7 Block-test suite (Relay Race) | pending |
| V2-8 Full-model PCC + Paris-token parity | pending |
| V2-9 Trace capture parity | pending |
| V2-10 Tracy perf sheet + PERF.md | pending |

## Model Overview

| Param | Value |
|-------|-------|
| dim | 5120 |
| n_layers | 64 (hybrid: `[lin,lin,lin,full] × 16`) |
| Full-attn heads (Q) | 24 (6:1 GQA) |
| Full-attn KV heads | 4 |
| head_dim | 256 |
| rope_dim (partial RoPE) | 64 (rotary factor 0.25) |
| mrope_section | [11, 11, 10] (text mode collapses) |
| mrope_theta | 10,000,000 |
| MLP intermediate | 13,824 |
| vocab_size | 248,320 (padded to 248,832) |
| norm_eps | 1e-6 |
| Norm | zero-centered RMSNorm (`w' = w + 1`) |
| Linear-attn | DeltaNet — 16 K-heads, 48 V-heads, head_dim 128, conv_kernel 4 |

## Sessions

### Session 1 — 2026-05-14
- V2-1: bulk-copied `llama3_70b_galaxy/` → `qwen3_6_galaxy_v2/` via
  `rsync -a --exclude '__pycache__'`. Tree intact.
- Removed inherited `README.md` and `PERF.md` (llama3_70b-specific);
  wrote v2-specific `README.md` + this `BRINGUP_LOG.md`.
- Next: V2-2 — modify `qwen_model_config.py` to introduce `is_qwen36`,
  set qwen3.6 hyperparams, `use_prefetcher=False`, and reclaim the
  prefetcher-worker cores into the compute grid.
