# Qwen3.6-27B fused decoder

Stage 2 implementation and evidence for `Qwen/Qwen3.6-27B`, revision
`8f33b98937c6031bb49e713dee3e8e185aafd26f`. Stage 2 is complete. The final combined source passed correctness, context,
Watcher and latency runs. Profiler reconciliation and the fresh independent
[stage review](stage_review_1.md) passed (`clean-pass`, round 1).

The standalone `tt/fused_decoder.py` implements both text layer kinds with
real checkpoint shapes and BF16 weights/activations, HiFi4 compute and FP32
accumulation. It imports no functional runtime implementation. Tests explicitly
select `FusedDecoder` and compare its results to HF and the functional decoder.

## Selected graph

Linear attention packs QKV/Z/B/A into one projection (narrow gates padded to
whole tiles), uses native causal convolution plus SiLU and returned history,
and uses flat-input native chunk GDN to fuse L2 normalization, query scaling,
head replication and recurrence. Head-major output continues through per-head
RMSNorm, then dedicated head concatenation precedes flat SiLU gating. Single-row
decode uses the same native phased prep/scan kernels with logical gate masking,
removing explicit gate pads; other shapes use the public native GDN path.

Full attention packs Q/K/V into one projection and uses dedicated head split,
shape-preserving partial rotate-half RoPE and head concatenation. K/V cache
updates are fused on disjoint input core sets; temporary L1 inputs are released
before SDPA. Native paged attention, key-block geometry and KV ownership remain
unchanged.

Shared-weight linear ops fold batch/sequence into their matrix row dimension
and restore the public shape afterward, avoiding repeated weight reads for
batched requests. Batch 1 takes the unchanged direct path.

SiLU/sigmoid are folded into binary multiplication. MLP gate/up stay separate:
packing, native SwiGLU and real matmul activation fusion were tested and lost.
Native RMSNorm absorbs casts in general shapes. Single-row hidden-width decode
uses an eight-core native norm with casts fused into its required transfers. All named
patterns and alternatives are assessed in [pattern assessment](pattern_assessment.md).

## Public contract

`FusedDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device)`
accepts layer-local keys. Public methods match FunctionalDecoder:
`create_recurrent_state(batch)`, `create_paged_state(page_table)`,
`prefill_forward(x, *, state, start_pos=0, rope=None)`, and
`decode_forward(x, *, state, current_pos=None, rope=None)`.

Inputs/outputs are tiled BF16 `[batch, sequence, 5120]` on a single 1x1 mesh.
Both layer kinds preserve arbitrary valid logical lengths within the unchanged
262,144-token context contract. Capacity at this limit is validated at batch1;
batch32 is validated on short requests, as in the functional contract. Dense work chunks at 512. Internal GDN time
padding is neutral (`beta=g=0`) and native convolution uses the logical end
when producing the next history. Public callers need no tile/chunk alignment.

Recurrent convolution history is now **ROW_MAJOR BF16 `[batch,3,10240]`**;
recurrent state stays tiled FP32 `[batch,48,128,128]`. Use the factory or
refresh tensors with the returned state layout. Full-attention state stays
BF16 `[physical_pages,4,32,256]` and uses the original arbitrary-ownership
32-token page table, finite padding, and per-row device int32 positions.

Setup creates persistent native GDN constants, convolution exchange scratch,
three semaphores and logical-end buffers for 1..512 rows. They live with the
layer and precede trace capture. Each request owns its state. Layer operations
are serialized on the owning queue; concurrent calls on one layer instance
are unsupported because native convolution scratch is shared. A single
retained trace can interleave prepared eager requests as validated by the
trace-reuse harness. No broad allocation-tracking exemption is used.

## Validation and reproduction

Activate `python_env`, source `/home/ttuser/qwen36-27b/lane-env.sh`, and set
`QWEN_DECODER=fused`; leave `QWEN_FUSION_CANDIDATE` unset for the delivered path.
Use physical chip 0 with a 1x1 mesh and serialize bounded jobs. Correctness uses
`TT_METAL_TRACE_ALLOC_TRACKING=1`, `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`.
Timing disables allocation tracking/tracebacks and leaves Watcher unset.
Watcher and Tracy are separate runs. Exact argv, environment, source hashes
and checkpoint provenance are in each result JSON and the work log.

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_contract --layer 0 --output result.json
python -m models.autoports.qwen_qwen3_6_27b.tests.run_long_context --layer 3 --length 262143 --output result.json
python -m models.autoports.qwen_qwen3_6_27b.tests.run_trace_reuse --layer 0 --output result.json
pytest -q --noconftest models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py
```

Commands above omit the outer `flock ... timeout ...` wrapper for readability;
the [work log](work_log.md) records the actual bounded commands and results.

Successful short tests cover lengths 1,31,32,33,127,128,129,511,512,513,1025,17;
continuation boundaries; mixed rows; shuffled pages; dropped-history negative
controls; batches 1/2/32; repeated bit equality; changed-input trace replay;
and an unsafe-allocation positive control. Long HF comparison processes all
state/cache updates and samples selected output rows while TT executes every
decoder row, matching the functional stage's acceptance method.

[Runtime audit](runtime_audit.json) traverses 13 forward methods;
[native audit](native_runtime_audit.md) checks wrapper paths. Functional/HF
reference computation and host refresh are outside measured forward.

## Evidence interpretation

All timings are **per decoder layer**, not full-model throughput or serving
TTFT. The inputs are seeded BF16 activations at real model dimensions, with
real checkpoint weights in acceptance/benchmark runs. Synthetic smoke tests
use committed checkpoint weight statistics. No text generation is claimed.

Candidate source snapshots are in `candidates/`. Exact pre-format snapshots
are archived by SHA in `source_archives/`; `measured_sources.json` maps successful
candidate results to exact archives. `candidate_results.csv` gives the timing
and PCC ledger; `candidate_sources.json` is the original snapshot index.
Formatting did not change candidate algorithms. The final source is formatted
before final gates and performance collection; final JSON hashes identify it.

The remaining dense matmuls retain the functional BF16/HiFi4 policy. Precision,
DRAM sharding, optimized-decoder, multi-chip, full-model and serving stages are
not part of this delivery. There is no advertised capability reduction.

## Final evidence

The exact reproducible serialized suite is [validate_final.sh](validate_final.sh).
It separates correctness, Watcher, timing and profiler regimes. The first
combined-source context failure and repair are documented in [AutoFix](AUTOFIX.md)
and [work log](work_log.md). Final results use the delivered_* prefix;
before_lifetime_fix_* and other candidate artifacts are historical evidence.

## Final latency and accuracy

Warmed per-layer wall time on physical chip0, batch1, prefill512 / decode context513;
decode is the median of 100 traced replays. Prefill rows are single warmed passes;
small candidate prefill differences are not statistically resolved gains. Baseline and final use the same harness,
weights, BF16/HiFi4 policy and tracking-disabled regime. Watcher is measured separately.

| Layer kind | Prefill functional → fused (ms) | Traced decode functional → fused (ms) | Decode speedup | HF PCC prefill / decode, functional → fused |
|---|---:|---:|---:|---|
| Linear attention (GDN) | 17.848 → 5.565 | 2.777 → 2.132 | 1.302× | 0.999967342 / 0.999997185 → 0.999970580 / 0.999997493 |
| Full attention | 7.087 → 5.784 | 2.383 → 2.041 | 1.167× | 0.999979640 / 0.999985743 → 0.999979522 / 0.999985391 |

The final default reproduces the selected combined candidate within0.02% GDN and
0.002% full-attention decode timing; those differences are measurement noise,
not additional claimed speedups. The selected combination reproduces the fastest correct
candidate family; GDN logical_gates has the same one-token graph. The full length matrix (128,511,512,513,4097) is in
[performance comparison](performance_comparison.json) and its CSV.

| Layer kind | Batch32 prefill functional → fused (ms) | Batch32 traced decode functional → fused (ms) |
|---|---:|---:|
| GDN | 104.895 → 26.460 | 73.506 → 7.971 |
| Full attention | 70.057 → 22.321 | 63.961 → 4.921 |

Batch32 uses33 logical prefill tokens and100 final traced replays. Baseline sources
are identified in [batch32 performance](batch32_performance.json).

## Full-context acceptance

| Layer kind | Prefill PCC at262144, functional → fused | Decode PCC at context262144, functional → fused | Minimum selected fused contribution PCC |
|---|---:|---:|---:|
| linear_attention | 0.999964119 → 0.999954320 | 0.999986067 → 0.999797103 | 0.999236844 |
| full_attention | 0.999876821 → 0.999876229 | 0.999574258 → 0.999962332 | 0.997757338 |

The GDN long-context decode delta is−0.000189 PCC, still above the unchanged0.995
gate; full attention improves. Native convolution/GDN and normalization alter
rounding and operation order. Short direct functional output, contribution and state
controls pass; the exact long-context delta was not isolated to a single fused op.
Full-context eager and traced decode are bit-equal. All output rows execute on TT;
HF checks selected output rows while processing all state/cache updates. The updated
[context contract](../context_contract.json) records exact artifacts and layout changes.

## Stop criterion and profiler interpretation

Active BF16 dense weights plus mandatory recurrent-state/KV traffic yield independent
memory floors of1.509888ms GDN and1.458184ms full attention at512GB/s per chip.
The final unprofiled traced medians reach70.81% and71.43% of this nominal roofline.
The last two kept incremental decode changes for each kind are below3%; exact
candidate names and gains are in work_log.md and AUTOFIX.md. The explicit stop rule
is met without claiming eight hours or counting idle time.

[roofline.json](roofline.json) reconciles unprofiled wall timing with same-profile
host timing, device kernel sums and gaps. Profiling overhead is kept separate.
[profiler_audit.json](profiler_audit.json) confirms BF16/HiFi4 dense rows and no
tilize/untilize or reshard ops. Native normalization uses four necessary transfers
per decode layer; full attention additionally uses two required K/V input shards.
All other host tensor setup is outside measured forward.

Final prefill/decode tables, CSV and compressed raw operation traces are in
`perf/delivered_profile_0/` and `perf/delivered_profile_3/`; matched functional reports
are in `perf/functional_profile_0/` and `perf/functional_profile_3/`. Reports include
tt-perf-report advice. Dense matmul bandwidth remains the dominant cost; broader
DRAM sharding and precision work is recorded as a next-stage hand-off.

[acceptance.json](acceptance.json) indexes all34 final-source result files,
[Watcher audit](delivered_watcher_audit.json) indexes four compressed clean raw logs,
and [runtime audit](runtime_audit.json) verifies the reachable forward path.
Independent review returned `clean-pass` in [round 1](stage_review_1.md).
Local checkpoint SHAs are recorded in the work log; no push is authorized.

Local implementation/evidence checkpoint: `e759ed7e37818386b3b880af1cd191eb72bea27c` on `main`; never pushed.
