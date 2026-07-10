# Time Series Transformer (TTNN)

## Platforms
- Wormhole (`n150`, `n300`)

## Overview
This directory contains the `ttnn` implementation and validation suite for the
`TimeSeriesTransformerForPrediction` model used in monthly tourism forecasting.

- **Architecture**: Vanilla Encoder-Decoder Transformer with Student-T probabilistic distribution head
- **HF checkpoint**: `huggingface/time-series-transformer-tourism-monthly`
- **Dataset**: Tourism Monthly (`hf-internal-testing/tourism-monthly-batch`)
- **Validation**: Per-layer PCC + end-to-end NLL/CRPS/mean-prediction-MAE within 5% of HF reference
- **Generation path**: KV-cache + single-trace-replayed-24x autoregressive decode,
  with an optional (hardware-verified-correct, but not currently latency-beneficial —
  see "Performance" below) 2-command-queue mode. This is `generate_traced_cached()`
  in `tt/tst_model_cached_additions.py` — the canonical Stage 1 deliverable path.
  `tt/tst_model.py` separately contains an earlier, KV-cache-less traced path
  (`generate_traced()`) and the original untraced reference (`generate()`); both
  remain in the codebase because other tests use them as correctness gates, but
  neither is the path benchmarked against the bounty's Stage 1 performance targets.

## Directory Layout

```text
time_series_transformer/
├── README.md
├── requirements.txt
├── probe_2cq_event_ordering.py        # Hardware probe: verifies 2CQ event
│                                       # choreography (not part of the acceptance
│                                       # suite; a one-off correctness check, kept
│                                       # for provenance — see "2CQ Investigation")
├── scripts/
│   └── save_reference_tensors.py      # Generates PCC/e2e validation artifacts
├── reference/
│   ├── config.json                    # Static model provenance (committed)
│   └── config_runtime.json            # Dynamic environment versions
├── tests/
│   ├── conftest.py                    # Adds package root to sys.path
│   ├── test_tst_pcc.py                # Per-layer PCC validation (encoder + decoder)
│   ├── test_tst_e2e.py                # End-to-end CRPS, exact NLL, mean-prediction
│   │                                  # MAE vs HF reference
│   ├── test_tst_perf.py               # Latency, throughput, 2CQ correctness + perf
│   ├── test_tst_distributions.py      # Student-T / Normal / NegativeBinomial routing
│   ├── test_tst_embedding_pcc.py      # Embedding-layer PCC vs HF reference
│   ├── test_tst_e2e_traced.py         # Correctness gate for tst_model.py's
│   │                                  # generate_traced() (the superseded,
│   │                                  # KV-cache-less traced path — not the
│   │                                  # Stage 1 deliverable path)
│   └── diagnostics/                   # Development-time root-cause scripts, NOT
│                                       # part of the acceptance suite — see
│                                       # tests/diagnostics/README.md
└── tt/
    ├── tst_model.py                   # Weight loading (load_weights), run_encoder,
    │                                  # run_decoder_step, generate(), generate_traced()
    ├── tst_model_cached_additions.py  # generate_traced_cached() — the Stage 1
    │                                  # deliverable path: KV-cache, single-trace
    │                                  # x24-replay, optional 2CQ
    ├── tst_embedding.py               # Value projection, lag features, mean scaler
    ├── tst_encoder_layer.py           # Encoder block (self-attention + FFN)
    ├── tst_decoder_layer.py           # Decoder block (masked self-attn + cross-attn + FFN)
    ├── tst_attention.py               # Attention mechanism, KV-cache allocation
    ├── tst_distribution.py            # Distribution-head parameter projection
    │                                  # (Student-T, Normal, NegativeBinomial)
    └── ttnn_utils.py                  # TTNN helper functions (layer norm, etc.)
```

## Setup

```bash
source python_env/bin/activate
pip install -r models/demos/time_series_transformer/requirements.txt
```

## Validation & Usage

### 1. Generate Reference Tensors

Downloads the pinned HF model and tourism batch, runs a forward pass, and saves
reference tensors to `reference/` (gitignored except `config.json`).

```bash
python models/demos/time_series_transformer/scripts/save_reference_tensors.py
```

### 2. Run the Bounty Acceptance Suite

The bounty acceptance suite is the three files below, run explicitly rather
than `pytest tests/`, because `tests/diagnostics/` contains development-time
root-cause scripts (not acceptance tests — see `tests/diagnostics/README.md`)
that would otherwise also be collected.

```bash
cd /root/tt-metal/models/demos/time_series_transformer && PYTHONPATH=/root/tt-metal/ttnn:/root/tt-metal/tools:/root/tt-metal/build_Release/lib:. TT_METAL_HOME=/root/tt-metal ARCH_NAME=wormhole_b0 pytest tests/test_tst_pcc.py tests/test_tst_e2e.py tests/test_tst_perf.py tests/test_tst_stage2.py -v -s
```

This must be run as a single chained command (`cd ... &&` followed immediately by
the `pytest` invocation on the same line/chain). Running `cd` as a separate
command first and `pytest tests/...` as a second separate command will fail
with `ERROR: file or directory not found: tests/test_tst_pcc.py`, because the
test paths are relative to the demo directory, and pytest's own working
directory at invocation time matters — the `&&` chain guarantees both run in
that directory; two independently-issued shell commands relying on `cd`'s
persistence are not equivalent if anything in between changes directory, and
in practice this was confirmed to fail when split apart. Replace
`/root/tt-metal` with your actual tt-metal repo root if different.

Actual run on this HEAD, no xfail markers anywhere in test_tst_perf.py
(test_tst_perf.py's own module docstring: a failing perf test right now is
a correct, honest result, not a placeholder -- xfail was deliberately
removed once the underlying S/B workload bug was fixed):

| Test | Result |
|------|--------|
| `test_encoder_pcc` | PCC 0.9999925 ✓ |
| `test_decoder_pcc` | PCC 0.9999812 ✓ |
| `test_e2e_generate` | CRPS within threshold ✓ |
| `test_e2e_exact_nll` | passes ✓ |
| `test_e2e_mean_prediction_mae` | passes ✓ |
| `test_sample_generation_under_1s` | passes ✓ |
| `test_2cq_matches_single_queue_output` | passes ✓ |
| `test_update_cache_v_matches_slice_write_ground_truth` | passes ✓ |
| `test_update_cache_allocated_correctly_at_bs1` | passes ✓ |
| `test_single_sequence_latency` | **115.0 ms** -- FAIL (target <50 ms) |
| `test_single_sequence_latency_2cq` | **104.0 ms** -- FAIL (target <50 ms) |
| `test_throughput_seqs_per_second` | **28.2 seq/s** -- FAIL (target ≥100 seq/s) |

9 passed, 3 failed. These are hard `assert` failures, not `xfail` -- do not
re-add xfail markers to hide them. Run-to-run hardware variance is real
(a separate session measured 111.8ms/98ms-range steady state on the same
code path), but the gap to the 50ms/100 seq/s targets is consistent and
substantial across every run so far, not noise.

### 3. Additional Test Files (not part of the 3-file acceptance command above)

```bash
ARCH_NAME=wormhole_b0 pytest tests/test_tst_distributions.py tests/test_tst_embedding_pcc.py tests/test_tst_e2e_traced.py -v -s
```

- `test_tst_distributions.py` — validates Student-T, Normal, and NegativeBinomial
  distribution-head routing and generation, per-distribution.
- `test_tst_embedding_pcc.py` — embedding-layer PCC checks (encoder/decoder
  embeddings, loc/scale, static categorical embedding) against HF reference.
- `test_tst_e2e_traced.py` — correctness gate for `tst_model.py`'s `generate_traced()`,
  the earlier KV-cache-less traced path. Both tests pass; this path is not used
  for Stage 1 performance claims (see Overview).

## Performance: Current Gap to Stage 1 Targets, and Why

**Single-sequence latency target is <50 ms (B=1); measured 115.0 ms (1CQ)
and 104.0 ms (2CQ) on the last full run. Throughput target is ≥100 seq/s;
measured 28.2 seq/s.** These are hard test failures, not xfail -- see the
table above and `tests/test_tst_perf.py`'s own docstring.

Two separate causes contribute to the gap:

**1. Architecture: one trace per decoder layer, not one fused trace for
the stack.** `generate_traced_cached()` originally captured a single fused
trace for both decoder layers. That was a real correctness bug: layer 1's
Q/K/V were projected from the raw input embedding instead of layer 0's
real output, so layer 1 never attended over layer 0's content (PCC ~0.31-0.33
against `generate()` when this was caught). The fix -- one trace per layer,
chained through real device buffers -- is correct, but costs
`num_decoder_layers` separate `execute_trace` calls per autoregressive step
instead of 1. Per `tt/tst_model_cached_additions.py`'s own module docstring,
this "works against the Stage 1 latency target and will need to be
re-measured and possibly optimized" once correctness was confirmed. That
re-optimization has not happened yet, and is likely the dominant remaining
latency cost -- more than any individual op.

**2. Op-level cost inside `write_prep`**, confirmed via `_print_op_times()`:
`slice_write_kv` ~20-22 ms, `qkv_linear` ~7 ms, `to_layout_kv` ~6 ms,
`qkv_split` ~5 ms per step. Stage 2 Change 4 (KV-cache fused op) partially
addresses this for V at BS==1 -- see "KV-Cache Fused Op for V" below.

### 2CQ Investigation

A 2-command-queue (writer queue + compute queue, event-gated handoff per
the tt-metal tech report "Advanced Performance Optimizations for Models")
was implemented and hardware-verified for **same-step** ordering:

- `probe_2cq_event_ordering.py` -- single-decoder-layer hardware probe,
  exact (0.000000 max abs diff) agreement between 2CQ and single-queue
  paths across 8 steps with deliberately distinct per-step marker values.
- `test_2cq_matches_single_queue_output` (full model) -- passes: NaN/Inf-free,
  correctly shaped, distributionally consistent with the single-queue
  baseline.

**Open item, not yet closed:** the module docstring in
`tt/tst_model_cached_additions.py` flags that **cross-step** overlap
(layer 0's write for step k+1 racing against the last layer's compute for
step k) has NOT been separately verified -- only same-step ordering has.
`test_throughput_seqs_per_second` and `test_sample_generation_under_1s`
both run `use_2cq=True` across many sequential steps, which is exactly
that unverified scenario. `use_2cq` defaults to `False` in
`generate_traced_cached()` pending this.

2CQ has not demonstrated a reproducible net latency benefit even where
ordering is confirmed correct: the autoregressive loop has a genuine data
dependency (step k+1's CPU embedding prep needs step k's output already
read back and sampled), which serializes the host loop regardless of which
command queue anything runs on.

### What Would Actually Close the Gap (Stage 2/3, not yet implemented)

- **Merge the per-layer traces back toward one fused trace per step**, or
  find a write-free formulation that allows it -- flagged above as the
  likely highest-leverage remaining item, but carries real regression risk
  given the layer-threading bug already found here once.
- **Distribution head fusion (Change 2)** -- not yet implemented; would
  remove one `ttnn.to_torch()` full-tensor readback per step.
- **L1 memory config (Change 3)** -- not started. `scripts/measure_weight_footprint.py`
  confirms the full weight set fits comfortably in pooled L1
  (247,234 bytes / 0.236 MB, 0.3% of the ~78.29 MB pooled ceiling on
  this 8x7 grid), so footprint is not the blocker; real contention with
  activations/KV-cache/trace buffers during decode has not yet been measured.
- Reducing the ~5.9-7ms/step `execute_trace` floor via sharding or op fusion.

## On the Horizon

- Investigate the readback-time bottleneck observed at B=4/S=10 (throughput
  test) independently of the sharded-FFN work already targeting single-
  sequence latency — the two batch shapes show different per-step cost
  balances and may need different fixes.
- Same-session A/B perf benchmark: sharded vs. forced-unsharded FFN at BS=100
  to confirm latency/throughput improvement from the HEIGHT_SHARDED FFN path
  under development.
- Re-verify Stage 1 perf criteria (<50ms, ≥100 seq/s) once sharding lands.

## Update: KV-Cache Fused Op for V Re-Enabled (This Revision)

The Stage 2 "Change 4" xfail (`tests/test_tst_stage2.py`) previously claimed
"no measured evidence the TILE<->ROW_MAJOR conversion tax nets positive" for
using `ttnn.update_cache` on the V cache. This was incorrect: at BS==1, V
cache is already allocated `TILE_LAYOUT` (see `allocate_kv_cache`), so there
is no conversion tax on this path to begin with. A prior session had already
built and correctness-verified this exact change
(`tests/diagnostics/probe_v_cache_update_cache*.py`: exact match vs.
`slice_write` ground truth, multistep, on real hardware) but it was reverted
without the xfail author consulting those probes.

Re-applied and re-verified on hardware (wormhole_b0). Correctness unaffected
(7/7 PCC + e2e tests still pass: encoder PCC 0.9999925, decoder PCC
0.9999812, CRPS/NLL/MAE all within 5% threshold). Measured latency effect,
single-sequence (B=1, S=1), steady state: (Note: the 124.3 ms/111.8 ms figures below come from a separate, more recent measurement session than the 249.8 ms figure in the acceptance-suite table above -- different JIT/thermal state, consistent with the run-to-run variance already documented elsewhere in this README, not a discrepancy between two different claims.)

| Metric | Before | After | Change |
|---|---|---|---|
| `write_prep`/step | ~47 ms | ~36 ms | -11 ms |
| `slice_write_kv`/step | ~27 ms | ~20 ms | -7 ms |
| `test_single_sequence_latency` median | 124.3 ms | 111.8 ms | -10% |
| `test_single_sequence_latency_2cq` median | 99.6 ms | 103.3 ms | within noise (small sample) |

This is a real, correctness-verified improvement but does **not** close the
gap to the Stage 1 targets (<50 ms latency, ≥100 seq/s throughput) on its
own. Per-step cost is spread roughly evenly across `write_prep`,
`trace_exec`, and `readback`/`sample`/`cpu_prep` -- no single op dominates,
so no single further op-level change is expected to reach 50 ms. Closing
that gap requires Stage 2/3 restructuring (op fusion, sharding, or a
KV-cache scheme that avoids per-step host writes entirely), consistent with
the "What Would Actually Close the Gap" section above. `test_tst_stage2.py`
now asserts the real correctness/allocation invariants this change relies
on instead of xfailing with outdated reasoning.

## Provenance
- **Model**: pinned to revision `2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35`
- **Dataset**: pinned to revision `81c7ee3cf3317e51beb97327df55926cd5bbfadb`
- See `reference/config.json` for full architectural parameters
- Runtime environment logged in `reference/config_runtime.json`
