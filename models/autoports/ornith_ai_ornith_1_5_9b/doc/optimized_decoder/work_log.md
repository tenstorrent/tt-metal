# Optimized decoder work log

Stage 3, Ornith-1.5-9B revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53.
Starting clean commit 814ba2ac. Enabled tt-autodebug/model-bringup verified in Codex inventory/config.
Installed environment.py exports active. Serial P300c chip 0; no foreign device owners at startup.
No reset performed or authorized. BF16 activations/norms, FP32 recurrent state, PCC >=0.995,
262144 context retained. Scope excludes multichip/full-model/vLLM.

## Operation-topology audit (before tuning)

Measured fused baseline: linear prefill/decode 27/48 ops, full 33/54 ops.
Source: ../fused_decoder/tracy_final/{linear,full}/{prefill,decode}_perf_report.csv.

| Current operations | Candidate replacement | Action/evidence |
|---|---|---|
| Packed QKV/Z/B/A or Q/gate/K/V | Tuned separate BFP8 projections vs packed | Packed wins; closing_queue.json |
| Packed gate/up and down; BF16 HiFi4 | BFP8 HiFi2/LoFi, BFP4 LoFi candidates | BFP8 selected; initial/diagnostic/role sweeps |
| Four sharded projection boundaries | Carry L1 residual/norm/output; phase-specific shards | Kept; movement_queue.json |
| DRAM projections, 4 readers, block 1, 64 cores | Per-role geometry/reader/block sweep | 1,200 rows; matmul_geometry.csv |
| Minimal matmul SwiGLU prefill | Explicit large-prefill program config | Minimal wins; adapted 2D measured |
| Causal conv + flat chunk GDN | Already dedicated composites; preserve inactive tails | GDN alias + direct history kept; final gates |
| Paged SDPA BF16 cache | BFP8 cache; SDPA chunk geometry | BFP8, K128 prefill / K256 decode |
| Decode RoPE packed lookup, history untilize | Small operator-required conversions | Tiled lookup/direct history kept; head composite rejected |
| No host crossings in measured path | Runtime guard | Final correctness and profile runs |
| Dense MLP, no routed experts/CCL | MoE and collective families inapplicable | HF config + source |

All decisions and measurements below are provisional until final integrated validation.

## Initial precision and geometry measurements

Exact commands, wall duration and exit codes are appended to `queue_results.jsonl`;
`initial_queue.json` and `geometry_queue.json` are the reproducible serial job manifests.
Each successful benchmark uses the same HF reference, real checkpoint weights, seven
warmed prefill samples and seven 50-replay nonblocking decode groups as stage 2.
Inputs remain the seeded BF16 hidden-state probes used by the fused baseline; no full-model
text-quality claim is made from these decoder measurements.

- First BFP8/HiFi2 linear candidate: 4.2797 ms prefill / 0.80682 ms decode,
  output PCC 0.999782/0.999666. Same-regime BF16 fused control: 5.8716/1.20753 ms.
- Full BFP8/HiFi2: 4.9921/0.74611 ms, versus fused 6.5570/1.13008 ms.
- BFP8 MLP LoFi: 3.9151/0.80679 ms linear and 4.7518/0.74562 ms full;
  candidate for improved prefill. Attention LoFi changes decode by less than 0.1%,
  with a larger PCC delta on linear attention; provisional tie goes to HiFi2.
- All-MLP BFP4/LoFi and attention BFP4/LoFi initially fail unchanged PCC assertions.
  These are diagnostic candidates, never default policy. Follow-up measurements
  separate gate/up from down and retain failed-score diagnostic timing/profiler rows.
- The geometry harness initially shadowed its integer core-count variable inside a
  closure. Corrected the local variable name; `*_closure_error.log` preserves the
  host failure. No device reset or correctness waiver was involved.
- Current checkout's opted-in multi-core DRAM pipeline supports ragged K blocks,
  multiple column passes and reader counts above three. Source:
  `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:1260` and
  `matmul_device_operation.cpp:1720`. Installed skill examples describe an older
  contract. Tests use this checkout's actual implementation and record errors.
- Single-reader block 4 requested 1,980,352 bytes of static L1 buffers against
  1,572,864 available. Multi-core reader retries make larger blocks legal;
  before/after timings are in `candidate_measurements.csv`. A first L1 error
  is not the rejection basis for the family.

## Topology and precision follow-ups

The sharded-residual candidate carries a width-sharded L1 stream through both
residual adds and both main norms. It returns DRAM only at the public layer boundary
and specialized attention/gating boundaries. At 32 residual cores it measures
0.72567 ms linear / 0.66865 ms full, versus 0.80679 / 0.74562 ms interleaved.
Separate gate/up plus binary SiLU beats packed gate/up: 0.79826 / 0.73794 ms.
Combined sharded+split linear measures 0.71762 ms. These are provisional until
integrated geometry, final path and all correctness gates are checked.

`diagnostic_*.json` deliberately sets `diagnostic_only: true`. These runs record the
unchanged PCC threshold failure and continue to collect traced timing, without claiming
VALIDATION_PASS. The delivered tests reject diagnostic-only reports. All-MLP BFP4/LoFi
linear decode is 0.61676 ms with output PCC 0.989899; attention BFP4/LoFi is
0.66691 ms with output PCC 0.977612. The narrower gate/up trial is 0.62993 ms,
PCC 0.993891; down-only is 0.70438 ms, PCC 0.995913 (contribution scores remain
in the artifact). Packed prefill MLP remains BFP8 in the narrower gate/up trial:
its explicit overrides target separate decode gate/up, not packed_mlp.
All BFP4 results are stage-8 candidates; stage 3 never selects BFP4 by default.

## Adapted layout, prefill and attention candidates

`movement_queue.json` retries the 16-core norm with a legal four-tile subblock
(the original eight-tile subblock exceeded the FP32 norm contract). It also retries
large 2D prefill using one-row output blocks and legal divisors up to eight columns,
on 8x8 and 11x10 grids. These adapted candidates run, rather than being dismissed
at the first L1 error, but lose to explicit minimal matmul on the measured 1024 row.
The minimal projection config M=4,K=16,N=8, subblock=1x2 improves both layer kinds.
Per-chunk comparisons and final selected-path validation are still pending.

A coherent 8x8 activation grid keeps separate gate/up outputs and their SiLU product
in L1, feeds the down projection directly, and carries both residual boundaries.
`movement_c64_l0.json` / `movement_c64_l3.json`: 0.70646 / 0.63910 ms traced decode.
The latter also uses BFP8 cache and SDPA K chunk 128. 16/32-core variants are
measured in the same queue, not rejected on unrelated precision evidence.

BFP8 KV prefill explicitly casts K/V for paged_fill_cache; decode updates remain
BF16. The same-context K-chunk sweep covers 32,64,128,256 under BF16 and BFP8 caches.
Initial cache scores pass; short-prefix and per-user batch-32 checks must pass
before selection. Logical batch remains one active user per state; padded tile
rows are never additional users.

## Integrated decoder and movement anomaly

The selected implementation is now standalone in `tt/optimized_decoder.py`; experimental
subclasses use an immutable test baseline `tests/optimized_reference.py`. This makes
candidate commands reproducible after production defaults change. Candidate reports
carry the frozen base hash and every experiment helper hash. Source snapshots in
`sources/` preserve earlier helper versions.

The fused GDN primitive is exposed under `ttnn._ttnn.operations.transformer`, not the
public transformer wrapper. The initial AttributeError is retained; using the actual
binding enables the state-alias candidate. Each receiver reads its full disjoint
head/V slab into its circular buffer before compute publishes the final slab. The
writer emits final state only after its compute completion. Initial/final views share
that same slab; no head reads another head's state. Scratch output is consumed within
one forward and does not escape. Decode alias + 64-core MLP carry measures 0.68923 ms;
the fused norm retry is correct but slower (0.69111 ms combined). No rejection rests
on the initial API/config error.

The integrated default reproduces 0.68920 ms linear decode. Its first smoke includes
logical lengths 1,33,1024; final length uses a different seeded input than a single-case
benchmark, so its PCC is not substituted for the matched single-case score.

Observed anomaly: matmul logs say the requested rectangular output grid is replaced
with a compact row-major grid. Source: matmul_device_operation.cpp:299. The model's
explicit consumer conversions keep the operation legal and correct, but this can
introduce reshards. `native_c{32,64}_l{0,3}.json` adapts both norm program grids and
all residual/MLP boundaries to that actual output grid. It passes PCC and is slower:
64-core native 0.69631 / 0.64491 ms versus selected 0.68923 / 0.63852 ms.
Resolution: controlled; keep measured lower whole-layer latency and document the
required boundary conversions rather than claiming they vanished.

Profiler setup failures were host-only: an overbroad filename replacement corrupted
an overlay include name, then Tracy's respawn command stripped quotes from inline JSON.
Corrected the overlay names and passed policy through an exact JSON file. Failed logs
are retained as `*_overlay_error.log` / `*_json_error.log`. Successful BFP4 profiles
show actual BF16 x BFP4 LoFi MLP kernels. These diagnostic profiles never claim PCC pass.

## Final movement and composite follow-ups

The per-role geometry sweep measured 1,200 cases across five roles, two layer kinds,
BFP8 and BFP4 MLP weights. Each role uses actual activations recorded at decoder
projection boundaries. One-reader controls and readers 2/3/4/6/8, 32/64 activation
cores and K blocks 1/2/4/8/16 are included. `roles_*.json` and provenance sidecars
retain successes and exact L1/config errors; `matmul_geometry.csv` is the combined
search table. Whole-layer integration uses readers 8 for packed input, 6 for gate/up,
2 with block 2 for down, 4 for output. Against uniform readers 4, integration changes
linear 0.68923 -> 0.68874 ms and full 0.63852 -> 0.63691 ms. These are small gains;
no microbenchmark winner is claimed as a whole-layer win without that comparison.

History conversion audit found repeated untilize/tilize around concatenation.
`history_reuse.json` replaces this chain with an explicit row-major intermediate:
0.68923 -> 0.66704 ms. Persistent row-major history (`history_rm_adapted.json`)
uses 49,152 bytes instead of tile-padded 524,288 bytes and measures 0.66721 ms
(within the observed timing spread). The first row-major test failure was its reset
helper's inplace multiply output-layout validation. Reset now computes zeros then
copies into the original backing allocation; it is outside the measured path.

Integrated non-aligned 1025-token prefill exposed mixed sharded/interleaved outputs
at the final one-token chunk. The public prefill boundary now normalizes each chunk
output to DRAM before concatenating. Failure retained in
`linear_final_tail_layout_error.log`; the full non-aligned/continuation battery
then passed. This is not a public sequence-alignment restriction.

For full attention, `nlp_create_qkv_heads_decode` was tested with the legal output
shard and real packed projections. Its producer/consumer conversions make the
whole layer slower: `heads_composite.json`, 0.65156 vs 0.63691 ms. Separate reshapes
remain. `paged_fused_update_cache` uses disjoint K and V cores and passes with
0.62665 ms. Padding the lookup indices to a tile lets embedding directly emit tiles
and avoids its extra tilize: combined `cache_lookup.json` 0.62484 ms. Padding
indices are inactive lookup rows, never logical users or cache positions.

GDN precision probing confirmed the implementation's explicit HiFi4/FP32/no-approx
contract (`chunk_gdn_compute_config.hpp:31`), so HiFi2/LoFi cannot be selected through
this op. Receivers 4/8 have no legal BH=32 geometry on 11x10; the already measured
phased dispatch is the alternative. Receiver 2 is legal and correct but slower:
`gdn_alias_r2.json`, 0.67905 ms versus ~0.667 ms. No source/kernel edits outside the
stage scope were made to bypass these op contracts.

## Closing topology comparison and stop-rule accounting

`closing_queue.json` measures separate attention projections at readers 1/2/4/8,
32/64 activation cores, block 1, using the same BFP8/HiFi2 attention and BFP8/LoFi
MLP control. Best legal separate candidates are 0.84578 ms linear (4 readers,
64 cores) and 0.77511 ms full (8 readers,64 cores). Matched packed control is
0.80679/0.74562 ms (`mlp_lofi_l*.json`). Packing remains faster even after tuning
separate projections; this decision is not based on operation count.

`history_direct*.json` combines the width slice with untilize/unpadding and writes
row-major history into its stable destination with slice's output_tensor. It
measures ~0.6437 ms linear decode. The destination-only ablation is within timing
noise; removing the separate copy is legal, and the integrated path must match
the best correct timing. The model does not add custom kernels or a host fallback.

Stop-rule target: representative decoder mix, three linear layers per full-attention
layer (HF layer_types), and independent batch-32 requests. This is a layer-stage
weighted budget, not a measured full model. On S1024 candidates, the final two
kept changes are full-attention fused cache + tiled lookup and linear direct history.
Using the immediately preceding correct pair (L=0.66721,F=0.63691 ms), the first
changes 3L+F from 2.63855 to 2.62648 ms (0.46%); direct history changes it to
2.55597 ms (2.68%). Both gains are <3%. The final measured default and conservative
roofline fractions are generated in `perf_summary.json`; correctness gates are
independent of this optimization stopping decision.

Conservative roofline uses 512 GB/s for the single P300c chip's eight DRAM banks,
BFP8 physical tile storage (1088 bytes per 1024 values), actual DRAM shard padding,
and necessary FP32 recurrent read/write or paged KV reads. It excludes all other
activation traffic. Approximate candidate fractions: linear 71.9%, full 70.4%.
Profiler's header DRAM/FLOPs fractions are not substituted for this accounting:
it ignores exponent storage and attributes pipeline math to eight storage cores.
`perf_summary.json` reconciles host, kernel sum and op gaps from the same signpost;
unprofiled amortized replay latency is separately labeled. Single-replay profiling
adds synchronization/instrumentation overhead, not a runtime torch conversion.

## Optimize checklist disposition

| Item | Decision and evidence |
|---|---|
| Real weights/shapes, traced warmed timing, logical lengths | validate_optimized, final_performance.csv; exact checkpoint revision and serial chip recorded |
| Topology audit before knobs | Opening table; phase-specific decisions and closing split controls above |
| Best correct integrated path | candidate_measurements.csv, final default S1024 rerun; BFP4 diagnostics excluded |
| Serving/full-model generation requirements | Not applicable to a decoder-only stage; no full model or server created |
| Profiler + advice | tracy_final/{linear,full}/{prefill,decode}_perf_report.{csv,txt}; BFP4 counterpart under bfp4_profile |
| Roofline/device/host reconciliation | perf_summary.json; actual raw kernel durations, not erroneous reported core utilization |
| Short/long/tail prefill | final_performance.csv, S128/1023/1024/1025/8192, plus full advertised context; no model TTFT claim |
| Dominant role geometry + BH readers | 1,200 rows matmul_geometry.csv, one-reader controls, BFP8/BFP4 crossings and per-role integration |
| Packed input projections | 16 tuned separate controls, packed wins under matched weight/fidelity policy |
| Packed gate/up | topology_split_mlp vs packed; keep separate decode, fused minimal SwiGLU prefill |
| Large prefill program config | Adapted legal 2D 8x8/11x10 configs, K/output/subblock sweeps vs minimal; movement_queue.json |
| Sharded residual and phase-specific activations | sharded_split_mlp, movement/native grid experiments; selected 64-core rectangle |
| SDPA and cache dtype | BF16/BFP8 Kchunk32/64/128/256; selected prefill128/decode256; short/batch32/long cache checks |
| Canonical precision/fidelity | Class policy BFP8 default; attention HiFi2, MLP LoFi, recurrent HiFi4/FP32; actual profiler dtypes |
| Separate attention precision | attn_lofi, all_lofi, diagnostic attention BFP4; no measurable LoFi decode gain with larger PCC delta |
| Composite ops and movement | GDN alias, conv history, head split, fused cache, tiled lookup; successful/rejected timings above |
| Runtime host crossings | device_only guards plus profiler zero host ops; setup uploads and test/reference reads are outside forward |
| Row-parallel decompositions and persistent CCL | No inter-device collective in this single-device decoder; later multichip stage is excluded |
| Logical batch vs tile padding | Independent row states, mixed lengths/starts/page tables, batch32 replay equality and latency |
| Cumulative integrated contracts | final_accuracy.csv, *_final.json, *_watcher.json, *_long.json, paired and 128-token trajectory |
| LM head/sampling | Not part of decoder-stage graph; no terminal-model performance claim |
| MoE active-expert execution | Not applicable: checkpoint uses dense gate/up/down MLPs, no expert/router modules |
| Evidence/hardware | All jobs serial/bounded; watcher separate from profiler; trace allocation tracker includes program cache; no resets |

## Not tried (stop rule)

- Exhaustive per-role prefill kernel-family matrices at 64/128/256 and a larger
  physical chunk. 1024-row 2D/minimal comparisons were measured; other final lengths
  use the selected config. A short-tail-specific choice may reduce prefill by a few
  microseconds per chunk; no measured gain is claimed. Same-request chunk-size search
  beyond the correctness control remains a later prefill hand-off.
- New fused kernels for partial RoPE + Q/K normalization or convolution state update.
  Existing composite alternatives were tested; new tt-metal operation development
  is outside this stage's permitted files. Expected gain is bounded by the respective
  non-matmul rows in the final profile, not assumed to equal their full cost.
- Dense multi-row projection batching is deferred to serving optimization; this
  stage preserves and measures the serial independent-state batch contract.
- Stage-8 precision probe and full-model text/accuracy selection. BFP4 candidates,
  including failing scores and real BFP4/LoFi perf rows, are retained without changing
  stage-3 weights from BFP8.

## Reproduction and evidence boundaries

From the checkout root, active bring-up Python environment and local HF cache:

```bash
bash models/autoports/ornith_ai_ornith_1_5_9b/tests/run_optimized_final.sh
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_sweep models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_decoder/initial_queue.json
bash models/autoports/ornith_ai_ornith_1_5_9b/tests/run_optimized_tuning.sh --skip-profiles
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.run_optimized_candidate_profiles
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_perf_reports --root models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_decoder/bfp4_profile
```

Other experiment queues use the same sweep command with their exact `*_queue.json`;
`queue_results.jsonl` records complete commands, timestamps, exit codes and durations.
`run_optimized_final.sh` runs correctness with allocation tracking (including program
cache), positive control, real-weight pytest, watcher separately, final profiles,
paired/stress, matched before/after timing, batch32 and advertised-context gates.
Every device child has a bounded timeout. Parent wait_for logs retain queue status.
Benchmarks have no watcher/profiler/tracker; those tools run in separate evidence jobs.

The only allocations acknowledged by the trace tracker are exact returned output
tensors. Each output is consumed before the next replay overwrites it. State, page
maps, positions, inputs, weights and program buffers are prepared before capture;
request reuse changes their contents, not their addresses. Linear GDN scratch is a
transient of its own forward; the recurrence allocation is persistent and disjoint
from scratch. No broad warmup/forward allocation exemption is used. The positive
control intentionally makes an unacknowledged live-trace allocation and must fail.

Logical context is unchanged. Full-attention BFP8 K/V allocation at 262144 is
570,425,344 bytes combined (previous BF16 1,073,741,824). Linear history is 49,152
bytes plus 2,097,152-byte FP32 recurrent state. Decode weight copies are
232,849,408 bytes linear / 222,822,400 full. Prefill weights and RoPE lookup copies
are additional persistent setup allocations; no capacity reduction is advertised.
Long-context tests execute every TT prefill token and compare sampled first/middle/
terminal windows against complete HF prior state/KV history, not truncated history.

`archive_optimized_evidence.py` retains raw captures locally and commits lossless
compressed logs/large CSVs with SHA256 provenance. TTNN profiler source overlay
symlinks only avoid an existing Tracy source-location ID collision; they resolve to
byte-identical checkout kernel sources. No kernel implementation was changed.

The target batch-32 stop-rule ablation is now measured directly:
`stop_batch32_before_l0.json` -> `stop_batch32_after_l0.json`: 20.57291 ->
20.40763 ms (0.80%); `stop_batch32_before_l3.json` -> `stop_batch32_after_l3.json`:
19.61170 -> 19.37679 ms (1.20%). Both final kept decode changes are <3% on the
largest supported batch. Conservative batch fractions exceed 70% for each kind
(about 72.6% linear and 71.9% full at these short mixed prefixes). These direct
measurements supplement the S1024 representative mix estimate above.

Final profiler advice prompted a 110-core minimal-matmul trial: prefill improves
from ~3.60/4.04 ms to 3.384/3.894 ms under the same candidate cache settings.
L1 MLP-input placement alone measures 3.581/4.057 ms and does not give a repeatable
win. Explicit fused-MLP configs initially omitted the required core grid and
failed validation; they are retried with an explicit legal grid before disposition.
Fixed-signature caller-owned prefill traces are also measured and correct:
`advice_prefill_trace_l*.json`, 3.533/3.998 ms. Decode is already traced; its remaining
GDN op gap persists with tracing. The prefill API remains caller-traceable without
introducing shape-specific capture state into the decoder.

The adapted explicit MLP config passes with grid 11x10. On the same candidate path,
M4/K16/N8, subblock1x2 gives 3.2094/3.7130 ms prefill; M8/K16/N8 gives
3.3014/3.9261, M4/K8/N8 gives 3.2810/3.8255. Moving the input to L1 with the winning
config loses: 3.2327/3.7930 ms. Both the larger grid and explicit fused-MLP config
are integrated. All projections now use the explicit minimal prefill config;
decode configs and BFP8 policies are unchanged. The final affected-path suite is
rerun with `run_optimized_final.sh --selected-only`. Earlier baseline timings remain
valid (unchanged FusedDecoder and timing regime); earlier optimized signoff is
preserved under `pre_advice/` with its exact source snapshot and source hashes.

The HF log's optional fast-path warning refers to the CPU reference implementation
(missing optional FLA/causal-conv extensions), not the TTNN decoder. `device_only`
raises on torch dispatch, from_torch, to_torch and host-to-device copies inside the
measured forward. Reference computation, initial weight conversion, caller input/
position refresh, state restoration and result inspection are outside timing.
Timing groups repeat a fixed input/position; advancing-state correctness is checked
separately by the 128-step trajectory and changed-input/position trace replays.

Final source integrity is checked by `tests/audit_optimized_evidence.py`: each final
correctness, watcher, profile, stress, context and default timing artifact must match
the current implementation SHA256. It also verifies actual dominant matmul dtype/
fidelity rows, context metadata, roofline thresholds and the allocation-tracker
positive control. This host-only audit supplements the device gates; it cannot turn
an old source run into a final pass.

`stage8_precision_handoff.json` distinguishes numeric per-layer passes from accepted
stage defaults. Down-only BFP4 meets 0.995 on layer 0 (minimum 0.995332) but misses
it on layer 3 (0.994838). All-MLP BFP4 misses on both. The narrower linear policy is
retained for stage 8; it is not vetoed by synthetic weights. BFP8 remains the explicit
stage-3 default required by the goal, independent of these candidate outcomes.

## Final integrated gates

`selected_acceptance.log` completed with exit 0 after the selected prefill changes.
All final correctness, watcher, pytest, tracking/positive control, profiles,
paired-state comparisons, 128-token trajectories, batch32 and long-context runs
passed on source b678803e78b74ede66d61e7609c6ca9147b2977dbba702f8431eb621ef997c4e.
Final S1024 prefill is 3.17440 / 3.66489 ms versus fused 5.93262 / 6.54346;
traced decode is 0.643936 / 0.624672 ms versus 1.207735 / 1.130236.
The default reproduces the best correct candidate within the recorded sub-microsecond
decode sampling spread, while improving prefill. The 0.643618 candidate point does
not establish a distinct faster topology over the final 0.643936 point; per-sample
ranges overlap and the selected decode graph/configs are identical.

The final host-only evidence audit passes. It checks every final source hash,
actual BFP8/HiFi2 and BFP8/LoFi matmul rows, unchanged context, PCC and >=70% roofline
at S1024 and batch32. All detailed values are generated in final_accuracy.csv,
final_performance.csv and perf_summary.json. Independent review and local commit
provenance are recorded below after closure.

## Independent review and local closure

Independent xhigh round 1 review returned `clean-pass`, with no required work or
quality-gate waiver. See [stage_review_round1.md](stage_review_round1.md). The
review checked the final implementation hash, all 24 final source-bound artifacts,
continuation per-row scores and failing negative controls, trace lifetimes,
runtime precision, profiler provenance, context coverage and stop-rule accounting.
The reviewed implementation remains unchanged after that verdict.

The following nonblocking hand-offs remain explicit:

- The role sweep covers K blocks 1/2/4/8/16, not every potentially legal
  non-power-of-two down-projection block. No global-optimum claim is made.
- Generic prefill reports label minimal matmuls `SLOW` and omit `program_config`.
  The selected calls, including fused SwiGLU, pass an explicit
  `MinimalMatmulConfig` through `config=`. Adapted configuration measurements,
  rather than that generic field, establish the selected tuning.
- The compact host audit omits continuation `min_row_contribution_pcc`; the
  independent reviewer checked those positive scores and failing controls
  directly. No additional device run was required.

Existing stop-rule hand-offs and stage-8 precision candidates remain unchanged.
Only stage-owned decoder, tests and documentation are included in local commits.
No push is performed.

Local implementation/evidence checkpoint:
`ac5d3a06df710b5e001579b50e88f9d0eeb7aa13` (`Optimize Ornith-1.5-9B TTNN decoder`).
All commit hooks passed. The source-bound host audit passed immediately before
commit with the reviewed implementation hash above. This SHA entry is committed
separately as documentation-only provenance; the implementation is unchanged.
