# Time Series Transformer (TTNN)

TTNN bring-up of HuggingFace's `TimeSeriesTransformerForPrediction` — a vanilla
encoder-decoder Transformer for probabilistic time-series forecasting — on
Tenstorrent hardware. Implements [Bounty #32140](https://github.com/tenstorrent/tt-metal/issues/32140).

## Status at a glance

| Stage | Requirement | Status |
|---|---|---|
| **Stage 1 — Correctness** | Full encoder-decoder architecture, all 3 distribution heads, PCC + NLL/CRPS/MAE within 5% of HF | ✅ **Complete** — 9/9 correctness tests pass |
| **Stage 1 — Performance** | <50 ms latency, ≥100 seq/s throughput, 100 samples in <1s | ⚠️ **Partial** — sample generation passes; latency and throughput do not meet target (see [Performance](#performance-current-gap-to-stage-1-targets-and-why)) |
| **Stage 2 — Basic optimizations** | Causal-mask precompute, KV-cache fused ops, sharded memory configs, op fusion | 🟡 **In progress** — causal-mask precompute and KV-cache fusion (V, BS=1) done; distribution-head fusion and L1 memory config not yet integrated |
| **Stage 3 — Deep optimization** | Flash attention, pipelining, batch scaling, stretch targets | ⬜ **Not started** — blocked on Stage 2 closing the latency gap first |

This README documents what is implemented, what is measured on real Wormhole
hardware, and exactly what remains — no perf numbers are hidden behind
`xfail`; every claim below is reproducible with the commands in
[Validation & Usage](#validation--usage).

## Platforms
- Wormhole (`n150`, `n300`)

## Architecture

- **Model**: Vanilla encoder-decoder Transformer, 2 encoder layers + 2 decoder
  layers, `d_model=26`, 2 attention heads (padded to 32-wide tiles on device)
- **Probabilistic forecasting**: distribution head outputs parameters rather
  than point estimates
  - **Student-T** (default) — used in production generation/sampling paths
  - **Normal** — routing + generation validated (`test_tst_distributions.py`)
    with a synthetic head, since the pinned checkpoint only ships Student-T
  - **Negative Binomial** — same as Normal
- **Feature support**: value embedding with lag features, past/future
  temporal encodings, static categorical + static real features, mean
  scaling, observed mask
- **HF checkpoint**: `huggingface/time-series-transformer-tourism-monthly`
  (pinned to `2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35`)
- **Dataset**: Tourism Monthly, `hf-internal-testing/tourism-monthly-batch`
  (pinned to `81c7ee3cf3317e51beb97327df55926cd5bbfadb`)
- **Generation path**: `generate_traced_cached()` in
  `tt/tst_model_cached_additions.py` — one TTNN trace per decoder layer,
  chained through real device buffers, KV-cache for autoregressive decode,
  with an optional (hardware-verified for same-step ordering, but not
  currently a reliable latency win — see [2CQ Investigation](#2cq-investigation))
  2-command-queue mode. This is the **canonical Stage 1 deliverable path**.
  `tt/tst_model.py` separately contains an earlier, KV-cache-less traced path
  (`generate_traced()`) and the original untraced reference (`generate()`);
  both remain in the codebase because other tests use them as correctness
  gates, but neither is benchmarked against the bounty's Stage 1 performance
  targets.

## Directory Layout

```text
time_series_transformer/
├── README.md
├── Changelog.md                        # Root-cause history: layer-threading bug,
│                                        # PCC threshold policy, decode-path
│                                        # retirement plan, L1/2CQ investigations
├── requirements.txt
├── probe_2cq_event_ordering.py         # Hardware probe: verifies 2CQ event
│                                        # choreography (not part of the acceptance
│                                        # suite; a one-off correctness check, kept
│                                        # for provenance — see "2CQ Investigation")
├── scripts/
│   └── save_reference_tensors.py       # Generates PCC/e2e validation artifacts
├── reference/
│   ├── config.json                     # Static model provenance (committed)
│   └── config_runtime.json             # Dynamic environment versions
├── tests/
│   ├── conftest.py                     # Adds package root to sys.path
│   ├── test_tst_pcc.py                 # Per-layer PCC validation (encoder + decoder)
│   ├── test_tst_e2e.py                 # End-to-end CRPS, exact NLL, mean-prediction
│   │                                   # MAE vs HF reference
│   ├── test_tst_perf.py                # Latency, throughput, 2CQ correctness + perf
│   ├── test_tst_stage2.py              # KV-cache fused-op correctness + allocation
│   ├── test_tst_distributions.py       # Student-T / Normal / NegativeBinomial routing
│   ├── test_tst_embedding_pcc.py       # Embedding-layer PCC vs HF reference
│   ├── test_tst_dist_head_fusion_pcc.py        # Isolated PCC gate: ttnn dist-head
│   │                                            # projections vs host torch (unwired)
│   ├── test_tst_dist_head_fusion_traced_pcc.py # Same gate on a real traced hidden state
│   ├── test_tst_e2e_traced.py          # Correctness gate for tst_model.py's
│   │                                   # generate_traced() (the superseded,
│   │                                   # KV-cache-less traced path — not the
│   │                                   # Stage 1 deliverable path)
│   └── diagnostics/                    # Development-time root-cause scripts, NOT
│                                        # part of the acceptance suite
└── tt/
    ├── tst_config.py                   # Shared constants (D_MODEL, heads, lags, ...)
    ├── tst_weights.py                  # HF state_dict -> padded TTNN weights, load_weights()
    ├── tst_io.py                       # Torch <-> TTNN input conversion helpers
    ├── tst_embedding.py                # Value projection, lag features, mean scaler,
    │                                   # static/temporal feature concat
    ├── tst_encoder_layer.py            # Encoder block (self-attention + FFN)
    ├── tst_decoder_layer.py            # Decoder block (masked self-attn + cross-attn + FFN)
    ├── tst_ffn.py                      # Shared feed-forward block
    ├── tst_distribution.py             # Distribution-head projection, sampling, NLL
    │                                   # (Student-T, Normal, NegativeBinomial) + an
    │                                   # unwired ttnn projection path (see PCC gates above)
    ├── ttnn_utils.py                   # TTNN helper functions (padded layer norm, etc.)
    ├── tst_model.py                    # run_encoder/run_decoder_step, generate(),
    │                                   # generate_traced() (KV-cache-less, correctness
    │                                   # gate only), teacher_forced_nll()
    ├── tst_model_cached_additions.py   # generate_traced_cached() — the Stage 1
    │                                   # deliverable path: per-layer trace, KV-cache,
    │                                   # optional 2CQ
    └── attention/
        ├── ops.py                      # Shared attend()/softmax primitives
        ├── self_attention.py           # Encoder (unmasked) + decoder (causal) self-attn
        ├── cross_attention.py          # Decoder-over-encoder cross-attention (+ KV precompute)
        ├── kv_cache.py                 # KV-cache allocation + single-token cached self-attn
        └── masks.py                    # Causal mask construction
```

## Setup

```bash
source python_venv/bin/activate
python_env/bin/python -m ensurepip --upgrade
python_env/bin/python -m pip install -r models/demos/time_series_transformer/requirements.txt
```

## Validation & Usage

### 1. Generate Reference Tensors

Downloads the pinned HF model and tourism batch, runs a full forward pass
(including the 24-step teacher-forced future window), and saves reference
tensors to `reference/` (gitignored except `config.json`).

```bash
python models/demos/time_series_transformer/scripts/save_reference_tensors.py
```

### 2. Run the Bounty Acceptance Suite

The bounty acceptance suite is the four files below, run explicitly rather
than `pytest tests/`, because `tests/diagnostics/` contains development-time
root-cause scripts (not acceptance tests) that would otherwise also be
collected.

```bash
cd /root/tt-metal/models/demos/time_series_transformer && \
PYTHONPATH=/root/tt-metal/ttnn:/root/tt-metal/tools:/root/tt-metal/build_Release/lib:. \
TT_METAL_HOME=/root/tt-metal \
ARCH_NAME=wormhole_b0 pytest tests/test_tst_pcc.py tests/test_tst_e2e.py \
  tests/test_tst_perf.py tests/test_tst_stage2.py -v -s
```

This must be run as a single chained command (`cd ... &&` followed
immediately by the `pytest` invocation). Test paths are relative to the demo
directory, and pytest's working directory at invocation time matters — the
`&&` chain guarantees both run in that directory. Replace `/root/tt-metal`
with your actual tt-metal repo root if different.

**Latest run** (Wormhole n300, real hardware, no `xfail` markers anywhere in
`test_tst_perf.py` — a failing perf test is a correct, honest result, not a
placeholder):

| Test | Result |
|------|--------|
| `test_encoder_pcc` | PCC 0.9999925 ✓ |
| `test_decoder_pcc` | PCC 0.9999812 ✓ |
| `test_e2e_generate` (CRPS) | HF 1325.85, TTNN 1337.04 — diff 0.84% (≤5%) ✓ |
| `test_e2e_exact_nll` | HF 7.7098, TTNN 7.7096 — diff <0.01% (≤5%) ✓ |
| `test_e2e_mean_prediction_mae` | HF 1788.94, TTNN 1776.72 — diff 0.68% (≤5%) ✓ |
| `test_sample_generation_under_1s` | 227.0 ms (target <1000 ms) ✓ |
| `test_2cq_matches_single_queue_output` | NaN/Inf-free, correctly shaped, distributionally consistent ✓ |
| `test_update_cache_v_matches_slice_write_ground_truth` | exact match ✓ |
| `test_update_cache_allocated_correctly_at_bs1` | ✓ |
| `test_single_sequence_latency` (1CQ) | **126.9 ms** — FAIL (target <50 ms) |
| `test_single_sequence_latency_2cq` (2CQ) | **91.9 ms** — FAIL (target <50 ms) |
| `test_throughput_seqs_per_second` | **24.3 seq/s** — FAIL (target ≥100 seq/s) |

**9 passed, 3 failed.** These are hard `assert` failures, not `xfail` — do
not re-add `xfail` markers to hide them.

### 3. Additional Test Files (not part of the 4-file acceptance command above)

```bash
ARCH_NAME=wormhole_b0 pytest tests/test_tst_distributions.py \
  tests/test_tst_embedding_pcc.py tests/test_tst_e2e_traced.py \
  tests/test_tst_dist_head_fusion_pcc.py \
  tests/test_tst_dist_head_fusion_traced_pcc.py -v -s
```

- `test_tst_distributions.py` — validates Student-T, Normal, and
  NegativeBinomial distribution-head routing and generation, per-distribution.
- `test_tst_embedding_pcc.py` — embedding-layer PCC checks (encoder/decoder
  embeddings, loc/scale, static categorical embedding) against HF reference.
- `test_tst_e2e_traced.py` — correctness gate for `tst_model.py`'s
  `generate_traced()`, the earlier KV-cache-less traced path. Not used for
  Stage 1 performance claims (see [Architecture](#architecture)).
- `test_tst_dist_head_fusion_pcc.py` / `test_tst_dist_head_fusion_traced_pcc.py`
  — PCC gates for the ttnn distribution-head projection functions
  (`student_t_params_ttnn`, etc.) against the host torch reference, both on
  synthetic and on real traced hidden states. These functions are correctness-
  verified but **not yet wired into any production generation path** — see
  Stage 2 status below.

## Performance: Current Gap to Stage 1 Targets, and Why

**Single-sequence latency target is <50 ms (B=1). Latest measured: 126.9 ms
(1CQ), 91.9 ms (2CQ). Throughput target is ≥100 seq/s; latest measured:
24.3 seq/s.** These are hard test failures, not `xfail` — see the table above
and `tests/test_tst_perf.py`'s own module docstring.

**Run-to-run variance is real and material**, not just noise on the margins.
Across sessions on the same code path:

| Metric | Session A | Session B (latest) |
|---|---|---|
| Latency, 1CQ | 96.1 ms | 126.9 ms |
| Latency, 2CQ | 110.2 ms | **91.9 ms** |
| Throughput | 35.3 seq/s | 24.3 seq/s |

Notably, 2CQ beat 1CQ by ~28% in the latest session despite the opposite
ranking in the prior one, and despite 2CQ's cross-step overlap correctness
being unverified (see [2CQ Investigation](#2cq-investigation)). Treat any
single run's numbers as a sample from a wide distribution, not a fixed
constant — this is itself evidence that host-dispatch overhead, not raw
device compute, dominates per-step latency, since device kernel time doesn't
vary session to session the way these totals do.

Two separate root causes contribute to the gap:

**1. Architecture: one trace per decoder layer, not one fused trace for the
stack.** `generate_traced_cached()` originally captured a single fused trace
for both decoder layers. That was a real correctness bug: layer 1's Q/K/V
were projected from the raw input embedding instead of layer 0's real output,
so layer 1 never attended over layer 0's content (PCC ~0.31–0.33 against
`generate()` when this was caught). The fix — one trace per layer, chained
through real device buffers — is correct, but costs `num_decoder_layers`
separate `execute_trace` calls per autoregressive step instead of 1. This
"works against the Stage 1 latency target and will need to be re-measured and
possibly optimized" (per `tt/tst_model_cached_additions.py`'s own module
docstring) once correctness was confirmed. That re-optimization has not
happened yet and is likely the dominant remaining latency cost — more than
any individual op.

**2. Op-level cost inside `write_prep`**, confirmed via per-step timing
(`verbose=True` on `run_traced_generation_cached`): `slice_write_kv`
~20–22 ms, `qkv_linear` ~7 ms, `to_layout_kv` ~6 ms, `qkv_split` ~5 ms per
step. Stage 2 Change 4 (KV-cache fused op) partially addresses this for V at
BS==1 — see [KV-Cache Fused Op for V](#kv-cache-fused-op-for-v) below.

### 2CQ Investigation

A 2-command-queue (writer queue + compute queue, event-gated handoff per the
tt-metal tech report "Advanced Performance Optimizations for Models") was
implemented and hardware-verified for **same-step** ordering:

- `probe_2cq_event_ordering.py` — single-decoder-layer hardware probe, exact
  (0.000000 max abs diff) agreement between 2CQ and single-queue paths across
  8 steps with deliberately distinct per-step marker values.
- `test_2cq_matches_single_queue_output` (full model) — passes: NaN/Inf-free,
  correctly shaped, distributionally consistent with the single-queue
  baseline.

**Open item, not yet closed:** the module docstring in
`tt/tst_model_cached_additions.py` flags that **cross-step** overlap (layer
0's write for step k+1 racing against the last layer's compute for step k)
has NOT been separately verified — only same-step ordering has.
`test_throughput_seqs_per_second` and `test_sample_generation_under_1s` both
run `use_2cq=True` across many sequential steps, which is exactly that
unverified scenario. `use_2cq` defaults to `False` in
`generate_traced_cached()` pending explicit CQ fencing and a multi-step
regression test (the bar an external reviewer set for enabling it by
default).

2CQ has shown a real latency benefit in the latest session (91.9 ms vs
126.9 ms, ~28% faster) but a regression in an earlier one (110.2 ms vs
96.1 ms) — see the variance table above. The autoregressive loop has a
genuine data dependency (step k+1's CPU embedding prep needs step k's output
already read back and sampled), which serializes the host loop regardless of
which command queue anything runs on; that dependency, not 2CQ itself, is the
likely reason the benefit is inconsistent.

### What Would Actually Close the Gap (Stage 2/3, not yet implemented)

- **Merge the per-layer traces back toward one fused trace per step**, or
  find a write-free formulation that allows it — the likely highest-leverage
  remaining item, but carries real regression risk given the layer-threading
  bug already found here once.
- **Wire in distribution-head fusion (Change 2)** — `student_t_params_ttnn`
  / `normal_params_ttnn` / `negative_binomial_params_ttnn` are implemented
  and PCC-verified against the host reference (both on synthetic hidden
  states and on a real traced decoder output — see
  `test_tst_dist_head_fusion_pcc.py` / `test_tst_dist_head_fusion_traced_pcc.py`)
  but not yet called from `run_traced_generation_cached()`. Wiring them in
  would remove one full-tensor `ttnn.to_torch()` readback per step.
- **L1 memory config (Change 3)** — measured, not applied. Investigation 1
  (documented in `test_tst_perf.py`) found that moving decoder hot-path
  weights to `ttnn.L1_MEMORY_CONFIG` made latency **worse**, not better
  (98.0 ms → 117.6 ms, +20%), and throughput worse too (28.2 → 25.4 seq/s),
  likely from pooled-L1 contention between weights, KV cache, trace, and mask
  buffers rather than weight footprint itself (which is only 0.3% of pooled
  L1 capacity). `HOT_PATH_MEMORY_CONFIG` is kept defined in the test file but
  intentionally unused pending a real fix for the contention mechanism, not a
  re-application of the same change.
- **Causal-mask precompute (Change 1)** — done: `_precompute_causal_masks_host()`
  builds all `T_max` masks host-side once before the decode loop; each step
  is a single `copy_host_to_device_tensor` with no per-step torch mask
  construction.
- Reducing the ~6–7 ms/step `execute_trace` floor via sharding or op fusion.

## Update: KV-Cache Fused Op for V

`ttnn.update_cache` is used directly for the V cache at BS==1 (K stays on
`slice_write` unconditionally at all batch sizes — its transposed
`[B,H,D,T_max]` layout is incompatible with `update_cache`'s fixed
seq-at-dim(-2) contract; V at BS>1 also stays on `slice_write`, since
`update_cache` hard-asserts `padded_shape()[0] == 1` and B=4/32/33/100 all
`TT_FATAL` on that check).

Correctness: exact match vs. `slice_write` ground truth over an 8-step probe
(`tests/diagnostics/probe_v_cache_update_cache*.py`), and gated in CI by
`test_update_cache_v_matches_slice_write_ground_truth` +
`test_update_cache_allocated_correctly_at_bs1`.

Measured latency effect (single-sequence, B=1, S=1, isolated steady-state
measurement, separate from the acceptance-suite run above — different
JIT/thermal state, consistent with the variance already documented in
[Performance](#performance-current-gap-to-stage-1-targets-and-why)):

| Metric | Before | After | Change |
|---|---|---|---|
| `write_prep`/step | ~47 ms | ~36 ms | −11 ms |
| `slice_write_kv`/step | ~27 ms | ~20 ms | −7 ms |
| Median single-sequence latency | 124.3 ms | 111.8 ms | −10% |

A real, correctness-verified improvement, but on its own it does not close
the gap to the Stage 1 targets. Per-step cost is spread roughly evenly across
`write_prep`, `trace_exec`, and `readback`/`sample`/`cpu_prep` — no single op
dominates, so no single further op-level change is expected to reach 50 ms.
Closing that gap needs Stage 2/3 restructuring (op fusion, sharding, or a
KV-cache scheme that avoids per-step host writes entirely).

## On the Horizon

- Wire `student_t_params_ttnn` into `run_traced_generation_cached()` now that
  it is PCC-verified against a real traced hidden state — the readback it
  removes is a full `[BS, 1, D_MODEL]` `ttnn.to_torch()` call every decode
  step.
- Investigate the readback-time bottleneck observed at B=4/S=10 (throughput
  test) independently of the per-layer-trace merge work already targeting
  single-sequence latency — the two batch shapes show different per-step
  cost balances and may need different fixes.
- Establish explicit CQ fencing on the shared K/V-cache and mask buffers, and
  add a multi-step (not single-marker-probe) regression comparing 1CQ vs 2CQ
  output, before enabling `use_2cq=True` by default.
- Re-verify Stage 1 perf criteria (<50 ms, ≥100 seq/s) once the per-layer
  trace merge and distribution-head fusion land.

## Provenance
- **Model**: pinned to revision `2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35`
- **Dataset**: pinned to revision `81c7ee3cf3317e51beb97327df55926cd5bbfadb`
- **Test hardware**: Wormhole n300
- See `reference/config.json` for full architectural parameters
- Runtime environment logged in `reference/config_runtime.json`
- See `Changelog.md` for the full root-cause history behind every claim in
  this README (layer-threading bug, PCC threshold policy, L1/2CQ
  investigations, decode-path retirement plan)
