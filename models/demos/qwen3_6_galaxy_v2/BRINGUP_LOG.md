# Qwen3.6-27B Galaxy v2 — Bringup Log

Live tracker. Append-only. Mirrors the format in
`models/demos/olmo_galaxy/BRINGUP_LOG.md`.

## Current Status

**Stage:** V2-2 complete (commit `45b2138d759`). Next: V2-3 (load_checkpoints.py qwen3.6 key map).

| stage | status | commit |
|---|---|---|
| V2-1 bulk copy from llama3_70b_galaxy | DONE | `45b2138d759` |
| V2-2 qwen36_model_config.py — is_qwen36 + full-grid | DONE | `45b2138d759` |
| V2-3 load_checkpoints.py — qwen3.6 HF key map | pending | |
| V2-4 llama_attention.py — is_qwen36 branch | pending | |
| V2-5 qwen36_delta_attention.py + decoder dispatch | pending | |
| V2-6 llama_ccl.py — dual-dtype persistent buffer keys | pending | |
| V2-7 Block-test suite (Relay Race) | pending | |
| V2-8 Full-model PCC + Paris-token parity | pending | |
| V2-9 Trace capture parity | pending | |
| V2-10 Tracy perf sheet + PERF.md | pending | |

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
- **V2-1** (commit `45b2138d759`):
  bulk-copied `llama3_70b_galaxy/` → `qwen3_6_galaxy_v2/` via
  `rsync -a --exclude '__pycache__'`. Tree intact (16 .py files in tt/).
  Removed inherited `README.md` and `PERF.md` (llama3_70b-specific);
  wrote v2-specific `README.md` + this `BRINGUP_LOG.md`.

- **V2-2** (same commit): added
  `tt/qwen36_model_config.py` with `TtQwen36ModelArgs(TtModelArgs)`.
  Subclasses `TtModelArgs` directly (not `TtQwenModelArgs` — qwen3.6 head
  layout 24×256 diverges too far from qwen3-32B 64×80). Sets:
  - `is_qwen36 = True` flag for downstream branches in `llama_*.py`
  - `use_prefetcher = False` everywhere
  - `sub_core_grids = (1,0) → (6,9)` = 60 contiguous Tensix cores (was
    50 with the 70B col-4 carve-out). +20% compute area for matmul / CCL.
  - All qwen3.6 hyperparams documented in the docstring at the top of
    the file.
  - Hybrid layer pattern populated from HF `config.layer_types` in
    `_set_qwen36_hf_params()`.

  Smoke-test: `python -c "from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs"` imports without error.

  Per-op program configs (matmul-2d, sharded norm, etc.) are NOT yet
  re-tuned for the 60-core grid — that's intentional and will be done
  per-op as V2-4 through V2-6 wire things up. Inherited 50-core configs
  will still run on the 60-core grid; extra cores simply sit idle until
  per-op tuning.

- **Next: V2-3** — update `tt/load_checkpoints.py` for qwen3.6 HF keys
  (`model.language_model.*` prefix; new `q_norm`/`k_norm` keys; new
  `linear_attn.*` keys for DeltaNet-layers).
