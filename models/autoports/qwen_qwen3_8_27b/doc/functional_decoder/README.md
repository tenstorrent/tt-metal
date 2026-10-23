# Qwen3.8-27B functional decoder

Single Blackhole device, real Qwen/Qwen3.8-27B text configuration and checkpoint revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. This stage implements only the decoder.
All implementation, correctness, context, watcher, trace-allocation and performance gates pass. Independent rereview returned **clean-pass**; see `STAGE_REVIEW.md`. Local checkpoint SHAs are recorded in `work_log.md`.

## Model and API contract

The HF model uses Qwen3.5 text classes: hidden5120, MLP17408, 48 linear-attention
and16 full-attention layers, maximum context262144. Full attention has24 query heads,
4 KV heads, head width256 and partial RoPE width64. Linear attention has16 key heads,
48 value heads, width128 and causal convolution width4. Both paths include both RMS
norms, residuals and the dense SiLU-gated MLP. Full attention applies a sigmoid output
gate; linear attention applies SiLU. See `AUTODEBUG.md` for the independent HF audit.

```python
FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device)
decoder.allocate_state(*, batch_size, context_length=None)
decoder.snapshot(state)
decoder.prefill_forward(hidden_states, *, state, start_position=0,
                        page_table=None, cos=None, sin=None)
decoder.decode_forward(hidden_states, *, state, page_table=None,
                       current_pos=None, cos=None, sin=None)
```

Weight loading accepts full HF checkpoint keys or layer-only keys. Setup converts weights
once to BF16, using HiFi4 and FP32 destination accumulation. Linear recurrent storage and
its state-sensitive arithmetic use FP32. Runtime tensor math, conversions and cache writes
stay on device; Python plans chunks and launches TTNN operations.

Prefill accepts BF16 tiled `[B,S,5120]` for any positive logical S within context,
returns every logical output and mutates request state. Internal chunks are128 tokens.
Full attention pads only the last physical chunk to32 tokens and slices logical outputs;
causal masking prevents padded rows from affecting valid outputs. Prefix continuation
starts at a128-token boundary and uses absolute `start_position`. Call prefill on individual
chunks and snapshot when a resume point is needed; a single long call returns its final
state only. Linear state cannot be reconstructed by slicing a later state.

Decode accepts `[B,1,5120]`. Full-attention `current_pos` is a device INT32 vector[B];
`page_table` is device row-major INT32[B,pages_per_request], mapping logical pages into
the shared physical pool. Rows have a multiple of8 entries (physical capacity256-token
alignment), but logical lengths have no alignment restriction. K/V shape is
`[physical_pages,4,32,256]`, BF16. Caller-supplied cos/sin have shape`[B,1,S,64]` for
prefill, `[B,1,1,64]` for decode and represent absolute positions. Linear state tracks the
prefix implicitly and does not consume a page table, RoPE or current-position tensor.
Owners must supply the state for the exact prefix, and avoid overlapping writable pages.

## Prefix-cache and trace ownership

| Layer kind | Prefix hit needs | Resume payload per request | Release |
|---|---|---|---|
| linear_attention | Identical prefix and weights; convolution and recurrence snapshot at that boundary, plus prefix position | 3207168 logical bytes =48*128*128*4 +3*10240*2. Physical tiled storage3801088 bytes because convolution height pads to32. Position metadata is additional (e.g.8 bytes). | After final request owner and every trace referencing the backing allocations retire |
| full_attention | Identical prefix, weights, absolute RoPE positions and retained page ownership | 4096 bytes/token BF16 K/V, rounded to32-token pages, plus4 bytes/page mapping. Pool rows reserve groups of8 pages. Prefix-position metadata is additional. | Return pages after final prefix/request owner and referencing trace retire |

The snapshot helper clones the entire allocated K/V pool for test isolation. A serving owner
can retain prefix pages with reference counts instead; the stage does not implement a serving
cache manager. Resume at128 matches continuous execution with PCC1.0 for both kinds.

Prepare supported physical shapes and compile eager paths before first capture. Request states,
weights and trace input/output allocations must stay alive while the trace exists; refresh only
input contents (activations, positions, RoPE and page table). The test restores state after warmup
and capture. Three identical replay results are bitwise identical. Changed input and position,
and full-attention page remapping, pass under the same trace. Capture uses trace_region_size=0.
Allocation tracking includes program-cache allocations (`TT_METAL_TRACE_ALLOC_TRACKING=1`,
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`). This is a layer trace contract, not a claim of a
serving scheduler or a cross-request trace cache.

## Capability evidence

Default acceptance PCC>=0.995; no model-specific relaxation.

| Claim | Linear evidence | Full-attention evidence | Scope/risk |
|---|---|---|---|
|262144-token prefill|linear_262144.json: .999938965|full_262144.json: .996442795|All TT outputs computed; HF comparison is final128 outputs using full prefix state|
|Nonaligned262142 then final-position decode|linear_262142_decode_max.json: prefill .999936640, decode .999996245/.999994516|full_262142_decode_k256.json: prefill .996382296, decode .999959588/.999991536|Decode positions262142/262143; no context reduction|
|Boundary/reuse/continuation|linear_watcher_requests.json|full_watcher_requests.json|Same loaded decoder, fresh requests lengths1,31,32,33,127,128,129,385,31; changed inputs; resume at128|
|Paged addressing|Linear state is nonpaged by definition|full_watcher_requests.json: remap .999932349, written-cache .999988377|Permuted pages; changed table under same trace|
|Determinism and runtime audit|linear_watcher_requests.json|full_watcher_requests.json|Three identical replays; guards reject Torch ops and TTNN host conversions inside forward|
|Watcher clean|watcher_linear/generated/watcher/watcher.log|watcher_full/generated/watcher/watcher.log|TT_METAL_WATCHER=10; no disabled features, no fatal/sanitize/overflow errors|

Inputs are deterministic synthetic residual activations with real target weights. Stress tests
use std0.1; linear watcher also tests std0.012563 derived from sampled real embedding weights
(`input_activation_stats.json`). This layer stage does not evaluate generated language.
Synthetic CI weights use each real tensor's stored mean/std/shape/dtype (`weight_stats.json`).
No smaller HF configuration is substituted. Runtime audit covers helpers through a TorchDispatchMode
and patched conversion functions; setup and explicit output comparisons remain host boundaries.

## Numerical repairs

The original recurrence lost long-context state precision. The decoder-local generic kernel
preserves FP32 state decay/subtraction/copy and uses separate FPU-compatible matmul copies.
See `long_linear_debug/AUTODEBUG.md` and `AUTOFIX.md`; final state PCC at262142 is .999973834.

Implicit SDPA decode k32/core1 failed at maximum context. Explicit k256/core16 and exact
exponent configuration passes. k512/core16 exceeded L1 and was rejected. The combined repair
is verified; individual knob causality is not isolated. See `full_context_debug/AUTODEBUG.md`
and `AUTOFIX.md`. Neither repair lowers model capability.

## Reproduction and measurement

`work_log.md` records the source build, container environment, commands and historical failures.
Run `python -m models.autoports.qwen_qwen3_8_27b.tests.run_decoder --layer 0 --seq 385 --requests --resume --output <file.json>`
(or layer3 with `--remap`). Use `--seq 262144 --prefill-only` and `--seq 262142` for capability.
Real weights default to ignored `.cache/qwen38-weights/layer_{0,3}.safetensors`; `--synthetic`
requires only checked-in statistics. `tests/test_functional_decoder.py` supplies pytest cases.

All accepted batch measurements are in [PERFORMANCE.md](PERFORMANCE.md) and `performance.json`; independent review is recorded in `STAGE_REVIEW.md`.

## Performance basis (batch-1 completed)

The headline is exactly4096 input positions plus128 supplied-hidden-input decode steps,
batch1, one request. These are single-layer runs, not end-to-end token generation.
Decode measures the128th warmed traced step, whose context is4224.

|Layer kind|Prefill device us|Decode device us|Useful prefill FLOPs / HiFi4 peak|Compulsory decode traffic / DRAM peak|
|---|---:|---:|---:|---:|
|linear_attention|284743.541|2660.428|6.696%|56.827%|
|full_attention|108787.730|2456.269|18.051%|60.576%|

Both denominators use the complete layer's device firmware timestamp envelope, including
attention, norms, movement and all gaps, at1350MHz. They do not use matmul time, a sum
of per-op utilization, or host time. The per-run `whole_layer_timing.json`
records exact cycles; actual linear directory is `tracy/linear_b1_complete`, full is
`tracy/full_b1`. Each has copied `ops.csv`, filtered `{prefill,decode}_perf_report.csv`
and separately rendered human-readable `.txt` tables. The filtered table's Device Time
column is microseconds and sums kernel time; it is **not** the whole-layer denominator.

`tests/roofline.py` and `performance.json` record the reproducible arithmetic and assumptions.
Peak is one participating p300c ASIC:120 physical cores*4096 FLOP/cycle*1.35GHz/4 HiFi4
=165.888TFLOP/s, and512GB/s DRAM. The runtime exposes110 compute cores; peak still uses
all physical cores of the selected device. This is conservative compared with a110-core
available-grid roofline. Sources: p300 specifications at
https://docs.tenstorrent.com/aibs/blackhole/p300.html and installed tt-perf-report1.4.1
Blackhole fidelity formula (`perf_report.py:418`). Useful FLOPs include dense matrices,
causal attention or recurrent updates, and approximate scalar norms/gates/activations;
there is no sparse/MoE work. Scalar transcendental operations count as one operation.
DRAM traffic is an explicit compulsory-traffic **lower bound**, not hardware-counter
utilization: weights, cache/state and residual input/output. Intermediate spills, repeated
core reads and small metadata/norm traffic are excluded. Timing still includes their work.
No percentage is clamped.

Agentic planning workload (`../agentic_serving.json`):32 concurrent conversations,
32768 live tokens each, typical4096 context. These are planning estimates, not a measured
serving load. Additional device batch rows1,8,16,32 are collected for layer compatibility;
there is no serving scheduler or memory-pool capacity claim in this stage.

The original FP32 PCC statistic can round/clamp to1.0 on very large output arrays.
`profile_full_b32.json` compares both statistics on identical tensors: FP321.0,
FP64 .999974527, relative RMSE .007774166. The runner now uses FP64 PCC. Historical
FP32 results are retained as historical measurements; a value1.0 is not a bitwise-parity
claim. Determinism uses torch.equal independently.

Batch32 linear's first profile exceeded Tracy's32K source-location limit and device
marker capacity even though its command exited0. It is excluded from performance.
The decode-only profile option retains4096 prefill and128 decode steps while omitting
measurement of a second batch32 prefill; periodic profiler drains occur outside the final
measured decode window. No device markers or host signposts may be missing in an accepted
capture. The batch1 headline profile still includes warmed prefill and decode.

The native-context tests use batch1; separate batch rows validate larger batches at4096
context. This evidence does not claim every Cartesian combination of batch and context fits.
For example, batch32 full-attention KV alone at262144 tokens needs34359738368 bytes,
which exceeds the measured34178731008-byte allocator before weights or buffers. Request
admission and pool sizing belong to later stages; the single-request advertised context is
preserved in full.

The Transformers “fast path is not available / falling back to torch” warning refers only
to the CPU HF oracle (flash-linear-attention is not installed). It is outside every TTNN
forward and measured window. It is not a runtime fallback in FunctionalDecoder.

Batch scaling is weak in this functional layout. Full-attention batch32 decode takes
63.195ms, including60.067ms of matmuls, versus2.456ms/1.923ms at batch1. Linear batch8
spends15.679ms in matmuls versus1.978ms at batch1. The `[B,1,H]` layout gives each
request its own tiled batch plane; the TTNN broadcast matmuls process separate planes
with padded32-row tiles instead of merging requests into shared rows. This explains the
near-linear matmul-time growth and repeated weight movement. The row shapes/fidelity in
`tracy/*/decode_perf_report.*` and raw ops.csv are the evidence. It is a measured layout
cost, not a correctness or device-health failure. Changing this geometry belongs to the
explicitly excluded optimized-decoder stage. Supplemental DRAM estimates assume one
weight read per independent batch plane and still exclude extra within-plane rereads and
intermediate traffic. Batch1 headline estimates are unchanged.

The HF oracle is the installed Transformers5.12.1 `modeling_qwen3_5.py`, copied as
`hf_modeling_installed.py` (SHA256395439341ea5ba4dd14c103e830383caa75686cfaf6b693dd7efb58622224da6).
`hf_modeling_reference.py` is a separate upstream inspection snapshot, not the file imported
by tests; it must not be used to infer the installed reference version.

Linear batch16's optional Tracy GUI event stream hit its source-location limit during final
shutdown, **after** all measured signposts. Unlike the rejected batch32 capture, its four
exported signposts and all11171 prefill/286 decode device rows are present. Every measured
firmware timestamp matches its `(global_call_count, trace_id, replay_session)` C++ profiler
row; there are zero dropped device markers. `tracy/linear_b16/capture_validation.json`
records this control. The numeric measurement windows are complete and accepted; later
GUI events are not used for timing.

## Final allocation-tracked validation

`tests/validate.sh`, captured in `final_validation.log`, exports
TT_METAL_TRACE_ALLOC_TRACKING=1 and TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0.
All final real-weight cases below passed with the runtime host-fallback guard and bitwise
repeated-replay determinism. Every non-ragged row resumes at128 with PCC1.0.

|Evidence JSON|Prefill PCC|Traced decode PCC|Changed input/position PCC|
|---|---:|---:|---:|
|linear_b32_final.json|0.999962364|0.999965921|0.999932172|
|full_b32_final.json|0.999817065|0.999952644|0.999947153|
|linear_b8_final.json|0.999969095|0.999954239|0.999988505|
|linear_b16_final.json|0.999964486|0.999935549|0.999958918|
|full_b16_final.json|0.999818128|0.999946031|0.999957049|
|full_b1_fp64.json|0.999962094|0.999983369|0.999990661|

`ragged_decode.json` tests eight randomly ordered current positions
[33,31,1,385,128,32,129,127] in one batch with a permuted page pool. Same-trace
replay PCC is .999965301; refreshed per-request positions/activations give .999954488.
Full-attention B16/B32 and B1 controls also remap page ownership under the same trace.

The full B1 FP64 control (`full_b1_fp64.json`) gives prefill PCC .999962094 and
relative RMSE .008713095 on4096 outputs, resolving the earlier rounded1.0 statistic.
`source_comment_cleanup.diff/json` records only documentation changes to the kernel and
module after profiling: executable C++ tokens and Python AST are unchanged. Final tests
record the new source hashes. `weights_provenance.json` records the real partial checkpoint
SHA256 values; checkpoint payloads are not committed.

Synthetic CI: **14/14 passed in153.34s**, real config shapes at lengths1,31,32,33,
127,128,129 for each kind, deterministic weight statistics and input std0.012563.
`synthetic_pytest.xml`, `synthetic_summary.json`, `synthetic_cases/*/result.json` and
`final_validation.log` retain the evidence. These runs include trace allocation tracking,
three repeated replays, changed inputs/positions, the runtime fallback guard and129-token
resume checks.
