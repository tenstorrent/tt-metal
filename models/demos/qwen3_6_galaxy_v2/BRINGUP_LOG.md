# Qwen3.6-27B Galaxy v2 — Bringup Log

Live tracker. Append-only. Mirrors the format in
`models/demos/olmo_galaxy/BRINGUP_LOG.md`.

## Current Status

**Stage:** V2-9 BLOCKED on V2-decode (dirty workspace). Prefill end-to-end PCC > 0.99 verified through 64L. Decode codepath never wired in v2; eager decode crashes at the embedding-output / decoder-input layout boundary. Trace machinery in `generator.py` is already wired; gated on V2-decode landing. See `tests/test_decode_trace_parity.py` module docstring for the full blocker chain.

| stage | status | commit |
|---|---|---|
| V2-1 bulk copy from llama3_70b_galaxy | DONE | `45b2138d759` |
| V2-2 qwen36_model_config.py — is_qwen36 + full-grid | DONE | `45b2138d759` |
| V2-3 load_checkpoints.py — qwen3.6 HF key map | DONE | `27e57ca0aa1` |
| V2-5a qwen36_delta_attention.py (NEW, +545 lines) | DONE | `27e57ca0aa1` |
| V2-6 llama_ccl.py — dual-dtype CCL buffer keys | DONE | `27e57ca0aa1` |
| V2-norm distributed_norm.py — zero_centered kwarg | DONE | `27e57ca0aa1` |
| V2-rope llama_rope.py — partial RoPE | DONE | `27e57ca0aa1` |
| V2-4 llama_attention.py — is_qwen36 (QKVG + QK-norm + RoPE + gate) | DONE | `41c190106dd` |
| V2-decoder llama_decoder.py — hybrid dispatch | DONE | `41c190106dd` |
| V2-embedding llama_embedding.py — bf16 force | DONE | `41c190106dd` |
| V2-model llama_model.py — per-layer + rope_setup threading | DONE | `58ee671e46e` |
| V2-config2 model_config populated (~100 keys) | DONE | `6472cdd551f` |
| V2-device-smoke setup + 7 construction bugs fixed | DONE | `16ba2ca1fcc` `9a9b2c86439` |
| V2-7 Block-test suite (DeltaNet 0.9995, full-attn 0.9997) | DONE | `eaefe1e13b8` |
| V2-7b 1L/4L hybrid + decoder gather/scatter | DONE | `bc2b24d3074` |
| V2-7c 64L hidden + logits PCC + Paris parity | DONE | `2227b2709c0` |
| V2-decode (qwen3.6 decode end-to-end) — REQUIRED for V2-9 | **BLOCKED** | dirty (V2-9 attempt) |
| V2-9 Trace capture parity (test added as skipped sentinel) | **BLOCKED** | dirty (this session) |
| V2-10 Tracy perf sheet + PERF.md | pending | |

### V2-9 attempt (2026-05-14) — findings

- Generator trace machinery (`begin_trace_capture` / `end_trace_capture` /
  `execute_trace`, `trace_ids_decode`, `_capture_trace_text`,
  `_decode_easy_trace_text`, `release_traces` in `__del__`) is already
  wired in `tt/generator.py` — no `_TRACE_SUPPORTED=False` flag exists
  in v2. (v1's flag was a manual gate; v2 inherits 70B's hot path.)
- Eager decode is broken before trace can even be attempted. Five
  separate layout/contract mismatches between v2's inherited 70B decode
  contract (batch-32 packed in T-dim, L1-WIDTH-sharded residual via
  `DECODE_RESIDUAL_MEMCFG`, `tt_sharded_distributed_rmsnorm`) and the
  qwen3.6 attention/DeltaNet blocks (written against v1's single-user
  `[B=batch, 1, T=1, H]` DRAM-interleaved contract with
  `tt_distributed_rmsnorm`).  Full chain in
  `tests/test_decode_trace_parity.py` docstring.
- Two small, safe infrastructure fixes APPLIED (no impact on the
  passing prefill tests, validated):
    - `TtLlamaAttention.prefetch`: skip `insert_tensor(self.wqkv)` /
      `insert_tensor(self.wo)` for `is_qwen36=True` (we use `wqkvg`).
    - `_NoOpPrefetcherSetup.worker_sub_device_id` attribute added +
      synced in `setup_decode`.  `TtTransformer.forward(mode='decode')`
      reads this unconditionally for the stall-group set call.
- Test `tests/test_decode_trace_parity.py` lands as a skipped sentinel
  (flag `_DECODE_ENABLED=False`).  Flip to `True` once V2-decode lands.
- **Static review on qwen3.6 forward paths found no host-write
  blockers** for trace capture: no `from_torch(device=...)` /
  `to_torch` / `copy_host_to_device_tensor` calls in the hot path.
  All persistent buffers (DeltaNet `dn_state_buffer`,
  `conv_state_buffer`, `_conv_zero_pad`) are allocated at
  `__init__` and written in-place via `ttnn.copy` (V2-5a contract).
  The v1 PERF.md residual `to_memory_config` host-write blocker is
  in the *70B branch* of `TtLlamaAttention.forward_decode`, NOT in
  `_forward_decode_qwen36`.  So once V2-decode runs eager, the trace
  capture itself should succeed without intervention.

### Recommended V2-decode plan (predecessor)

Rather than incrementally bridging the 70B↔qwen3.6 decode boundaries
one layout converter at a time, mirror v1's decode contract directly:
add a `forward_decode_qwen36` entry point on `TtTransformer` that
takes `[B=batch, 1, T=1, H]` natively, uses DRAM-interleaved residual
throughout, and calls `tt_distributed_rmsnorm` (the prefill primitive,
which is already verified at PCC > 0.99 for 64L).  The generator's
`ttnn_decode_forward` should dispatch to this for qwen3.6.  Estimated
effort: 1-2 sessions.

After V2-decode lands and eager decode is PCC-verified vs HF reference:

- Flip `_DECODE_ENABLED=True` in `test_decode_trace_parity.py`
- Static review predicts trace capture will succeed cleanly.
- Then V2-10: tracy perf sheet, target >= 17 tok/s/user 64L decode
  (olmo precedent on the same mesh).

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

- **Wave 1** (commit `27e57ca0aa1`): five parallel sub-agents landed
  V2-3 (load_checkpoints.py qwen3.6 HF key map — 9 tests), V2-5a
  (qwen36_delta_attention.py NEW +545 lines), V2-6 (llama_ccl dual-dtype
  buffer keys — 6 tests), V2-norm (distributed_norm zero_centered — 2
  tests), V2-rope (llama_rope partial-RoPE — 3 tests). All cpu_only,
  20/20 tests pass, cross-imports clean.

  Key findings folded forward:
  - HF DeltaNet uses SPLIT linear_attn weights
    (`in_proj_qkv/z/a/b`), not combined `qkvz/ba` form the plan brief
    expected.
  - q_proj fused with output gate: `[12288, 5120]` = 2 × 24 × 256 ×
    5120. `convert_hf_qkv_to_meta_format` skipped for qwen3.6.
  - QKVG persistent CCL buffer width = 56 × 256 = 14336 (per-chip 3584
    across 4 mesh cols).

- **Wave 2** (commit `41c190106dd`): three parallel sub-agents landed
  V2-4 (llama_attention.py +763 lines — QKVG split + per-head QK-norm +
  partial RoPE + sigmoid-gate; 2 tests), V2-decoder (llama_decoder.py
  +61 lines — hybrid dispatch, all 4 scenarios verified; 4 tests),
  V2-embedding (llama_embedding.py +9 lines — bf16 force; 4 tests).

  Full Wave-1 + Wave-2 suite: **30 passed, 1 skipped (placeholder), 0
  failed**. Cross-imports across all 9 v2 modules verified.

  TODO for V2-model: must thread `rope_setup` onto `TtLlamaAttention`
  instances (qwen3.6 forward path reads `self.rope_setup`; the 70B
  path constructs its own internally).

- **Next: Wave 3** — V2-model (llama_model.py per-layer instantiation,
  rope_setup creation + threading, qwen36 weight ingestion via V2-3's
  `standardize_hf_keys_qwen36`) and V2-generator (DeltaNet state
  plumbing through trace). Both CPU-buildable with mocks. Then V2-7
  block tests on BH GLX (strictly sequential — one device session at
  a time, `tt-smi -r` between failures).
