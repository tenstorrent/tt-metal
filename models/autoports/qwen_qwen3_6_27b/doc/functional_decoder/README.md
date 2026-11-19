# Qwen3.6-27B functional decoder

All hard validation gates pass. Independent [stage review, round 2](stage_review_2.md)
returned **clean-pass**. The advertised **262,144-token context is preserved**.
Work is confined to the functional decoder for `Qwen/Qwen3.6-27B`, checkpoint
revision `8f33b98937c6031bb49e713dee3e8e185aafd26f`.

The implementation covers both text layer kinds: linear attention (represented
by real layer 0) and full attention (real layer 3). The target has hidden width
5120, MLP width 17408 and 64 layers. Tests use the actual config and checkpoint
shapes. See [context contract](../context_contract.json),
[work log](work_log.md), and [repair evidence](AUTOFIX.md).

## Loading and forward contract

```python
FunctionalDecoder.from_state_dict(
    state_dict, *, hf_config, layer_idx, mesh_device
)
layer.create_recurrent_state(batch=1)
layer.create_paged_state(page_table)
layer.prefill_forward(x, *, state, start_pos=0, rope=None)
layer.decode_forward(x, *, state, current_pos=None, rope=None)
```

Use a single 1×1 mesh. `state_dict` has layer-local HF keys, for example
`input_layernorm.weight`; `hf_config` may be the text config or its enclosing
config. Loading and state construction are explicit setup boundaries.

Inputs and outputs are BF16 tiled DRAM tensors `[batch, sequence, 5120]`.
Decode consumes one token per row. Full-attention decode supports batches
1–32 on its 8×4 grid. Validation covers batch 1 at full context, batches 1/2/32
at short lengths, and four mixed-position rows. It does not measure every
batch/context combination; available DRAM determines how many concurrent
full-context requests fit.

Prefill accepts every positive logical length that fits the remaining context,
with a nonnegative absolute `start_pos`. It returns every logical output row.
Dense work is chunked at at most 512 tokens. Linear recurrence uses 32-token
blocks with neutral padding. Full attention uses 32-key blocks for cache extents
below 512 tokens and 512-key blocks for larger extents. Larger blocks bound the
native running-state rounding that failed the full-context accuracy gate.

For full-attention continuation, exact one-token updates advance to the native
key-block boundary before aligned chunk fills. Existing prefix entries remain
intact. A list of inputs, states, starts and RoPE tuples describes requests with
different lengths and prefix positions; each request owns writable state. This
functional path processes those requests independently.

### Linear attention state

Convolution state is BF16 `[batch, 3, 10240]`; recurrent state is FP32
`[batch, 48, 128, 128]`. Both are mutated in place. Continuation/decode require
the exact state after the preceding token. The linear layer has no KV page
table or RoPE; supplied recurrent history represents its position. `start_pos`
still identifies the absolute logical range for the public context check.

### Full-attention state and positions

The setup page table is int32 `[batch, logical_pages]`, with arbitrary physical
ownership mapping and 32-token pages. K/V are BF16
`[physical_pages, 4, 32, 256]`. The caller reserves pages for every prompt and
subsequent decode token. Setup initializes all cache storage to finite zero.
For tables with at least 16 pages, setup appends uniquely owned padding pages
until the table width is divisible by 16. Use the returned `state.page_table`
shape when refreshing ownership; appended pages are part of its stable buffer.

Prefix pages must contain the request's history. `current_pos` is a device
int32 tensor `[batch]`; each row may have a different valid absolute position
in `[0, 262143]`. The caller supplies valid positions and non-conflicting page
ownership. Forward does not read positions or page contents to the host.

RoPE `(cos, sin)` are device BF16 tensors `[batch, 1, sequence, 64]`, using
absolute text positions. Text uses the same position on all three MRoPE axes;
the remaining 192 head dimensions are unrotated. Vision encoding is outside
the text decoder-layer contract.

### Numerical policy

Weights, activations and KV use BF16, except folded `(1 + weight)` RMSNorm
scales, recurrent state and decay parameters kept in FP32. Compute uses HiFi4,
accurate math and FP32 destination accumulation. Native attention retains some
BF16 intermediates; the configuration does not claim fully FP32 attention.
The acceptance threshold remains **PCC ≥ 0.995**, with no model-specific waiver.

## Trace ownership

The caller owns warmup, capture, replay and release. Prepare persistent weights,
inputs, positions, RoPE, cache and page-table buffers before capture. Warm each
eager prefill/continuation signature that will interleave with the retained
trace. Restore state after warmup/capture when it changed; refresh the contents
of stable input buffers before replay. Decode itself consists of TTNN device
operations and stable state writes.

`run_trace_reuse.py` retains one trace across changing prepared request lengths,
starts and page ownership. `run_mixed_decode.py` checks independent batched
positions. `run_decoder.py` checks changed input/position replay and repeated
bit equality, and proves trace allocation tracking is active by deliberately
creating an unsafe live buffer and requiring replay rejection. No broad
allocation acknowledgment or program-cache exemption is used.

## Validation interpretation

Short contract tests cover lengths 1, 31, 32, 33, 127, 128, 129, 511, 512, 513,
1025, then 17 on the same loaded instance. Prefix windows start at 1, 32, 33,
128, 511, 512 and 544; mixed rows use different starts and lengths. Continuation
uses per-row PCC of `output - residual_input`, and dropped-history negative
controls must fail the same 0.995 gate.

The long harness executes the complete TT decoder for every prompt token.
Its reduced HF oracle processes every K/V or recurrent-state update, then
compares selected output rows (early boundaries and the final 32 rows), avoiding
a quadratic full-output CPU attention reference. It enforces both whole-output
and selected per-row contribution PCC. Full attention additionally resumes the
last 32 rows from the real filled TT cache. Near-max decode compares eager and
traced results for bit equality. This is selected-row HF validation, not an
HF comparison of all 262,144 output rows.

[Runtime audit](runtime_audit.json) follows all 16 reachable forward methods;
[native wrapper inspection](native_runtime_audit.md) checks hidden fallback
branches. Runtime sentinels reject Python from/to-torch calls inside measured
prefill and capture. Host setup, HF comparisons, input refresh and profiler
drains remain outside measured passes. The HF reference's optional-kernel
fallback warning describes CPU reference execution, not the TT forward path.

[Watcher audit](watcher_audit.json) distinguishes clean final runs from the
retained failed tiny-fill experiment, with compressed raw logs and hashes.
Watcher and profiler run separately. All device jobs are serialized and bounded
by timeouts. No operator reset or manual process termination was used.

## Reproduction

On the recorded machine, activate `python_env` and source
`/home/ttuser/qwen36-27b/lane-env.sh`. Leave `TT_VISIBLE_DEVICES` unset; each
runner opens physical chip 0 as a 1×1 mesh. Set
`TT_METAL_TRACE_ALLOC_TRACKING=1`. Commands and exact output paths are recorded
in each result JSON and the work log. Representative module invocations:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_contract --layer 3 --output RESULT.json
python -m models.autoports.qwen_qwen3_6_27b.tests.run_long_context --layer 3 --length 262143 --output RESULT.json
python -m models.autoports.qwen_qwen3_6_27b.tests.run_trace_reuse --layer 3 --output RESULT.json
python -m models.autoports.qwen_qwen3_6_27b.tests.run_mixed_decode --layer 0 --output RESULT.json
pytest -q --noconftest models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py
```

Use the recorded bounded `timeout` wrappers for device runs. Normal pytest
uses deterministic synthetic weights derived from committed real-weight stats;
it requires no HF download. Artifact runners use the locally cached pinned
checkpoint and load only the relevant layer's shards.

Final correctness and performance artifacts follow. Historical failures remain
in the work log and AutoFix reports; they are not accepted results.

## Final validation results

Both kinds preserve the full advertised context. These are real-weight results
on decoder source `476f9d99fa90a1268f264964037dbc6837b320adc9f48d172dc1156b8b35962d`.

| Layer kind | Prefill PCC at 262144 | Traced decode PCC at 262144 | Minimum selected prefill contribution PCC |
|---|---:|---:|---:|
| linear_attention | 0.999964119 | 0.999986067 | 0.999609271 |
| full_attention | 0.999876821 | 0.999574258 | 0.997773059 |

[Acceptance index](acceptance.json) maps every gate to raw artifacts.
`gdn_contract_final` and `full_contract_final` contain the short/resume checks;
`*_trace_reuse_final` records one retained trace across six requests.
`*_final_context_262143/262144` contains all-token TT context tests.
`layer_0/3_batch32_final` and `layer_0/3_watcher_final` cover real batched
execution. `synthetic_pytest_final.log` records four passing real-shape tests.

## Warmed per-layer performance

Batch 1, prefill 512 tokens, traced decode at context 513; one Blackhole chip.
Numbers are **kernel sum / kernel plus inter-op gaps, milliseconds**. These are
decoder-layer timings, not full-model throughput.

| Layer kind | Warmed prefill ms | Warmed traced decode ms |
|---|---:|---:|
| linear_attention | 16.051 / 18.580 | 2.735 / 2.798 |
| full_attention | 6.317 / 6.479 | 2.326 / 2.371 |

Tables, filtered CSVs, compressed raw ops and provenance are in
[linear attention reports](perf/linear_attention/metrics.json) and
[full attention reports](perf/full_attention/metrics.json). Each directory has
`prefill_perf_report.txt/.csv` and `decode_perf_report.txt/.csv`.
Totals use the filtered `Device Time [μs]` and `Op-to-Op Gap [μs]` columns;
reported overlap is zero. `tt-perf-report==1.5.0 --arch blackhole` uses its P150
nominal roofline reference for this P300 host; measured durations are unchanged.
The unclassified reshape warning concerns category labeling only.

[Profiler audit](profiler_audit.json) confirms final collection has no dropped
markers. Capacity is 5000 programs, with profiler drains between complete
passes and outside signposts. Earlier overflow runs are historical evidence.
Raw `ops.csv.gz` can be decompressed to replay the exact report commands saved
in metrics.json. The raw compressed console logs preserve compiler output.

## Limitations and handoff

Full-context HF comparison samples output rows as described above; it does not
compare every long output row. Synthetic activations are seeded BF16 values
scaled to 0.1, with real checkpoint weights in acceptance runs. This is layer
validation, not text-generation or full-model accuracy evidence.

The functional implementation uses DRAM tensors and serial mixed-request
prefill. Unaligned large-cache continuation can use up to 511 exact token
updates before a block boundary. Those paths are correct but have no optimized
latency claim. Later optimization, multi-chip, full-model and serving work has
not begun. Native SDPA still contains reduced-precision running state; the
selected geometry is validated rather than advertised as exact arithmetic.

Independent review and local commit records follow in the work log.

Accepted performance was rerun with allocation tracking and tracebacks disabled,
Watcher unset, and program-cache exclusion explicitly disabled. See
[review remediation](work_log.md) and the environment in each metrics JSON.
Prior tracked profiles remain under `perf_tracked/` as historical evidence.
