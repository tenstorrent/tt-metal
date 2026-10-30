# Fused decoder (stage 02)

Status: complete; independent stage-review returned `clean-pass`.

Target: text-only `google/gemma-4-31B-it`, pinned HF revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`. All reference/config/tokenizer loads
retain that revision. This stage changes only the new fused decoder, tests and docs.
Functional code and evidence remain intact. No optimized-decoder, multi-chip,
full-model, generator or vLLM implementation is introduced.

## Runtime and selected graph

`FusedDecoder.from_state_dict` and its public prefill/decode/cache interfaces match
the functional stage. The implementation is standalone and never delegates runtime
to FunctionalDecoder. Test entrypoints import FusedDecoder explicitly and record its
module/class. Construction owns weight conversion/packing and scalar extraction.
Runtime accepts only TTNN tensors and performs no torch/host conversion or fallback.

Selected transformations:

- Dedicated rotate-half RoPE; prefill flattens batch into sequence around the kernel
  to preserve each row's cos/sin without per-request kernel dispatch. Decode uses
  the per-row-position HF RoPE kernel, sharing cos/sin transfers between Q and K.
- Dedicated prefill QKV head creation, with tied KV for full attention. Sliding
  attention keeps separate projection weights and combines projected K/V for the
  head kernel. Full attention packs Q/K weights once and deletes original copies. B32 sliding
  prefill uses an additional packed QKV copy; decode retains separate projections.
- Decode head transforms are chosen by measured batch size. Dedicated splitting
  wins at larger batches; B1 retains the cheaper reshape sequence. The decode
  splitter lacks tied-KV mode, so full attention duplicates only the projected K
  activation for that op, then discards its redundant K output.
- Dedicated prefill head concatenation. Full attention uses two16-head groups;
  direct32-head concatenation at D512 exceeds per-core L1.
- Full attention reuses the raw K/V RMS reduction, applying K gamma to normalized V.
  BF16 rounding differs slightly and must pass the same0.995 PCC bar.
- GELU-tanh folds into gate/up multiplication; layer scalar folds into residual add.
- Batched prefill cache fill removes per-row K/V slices and fill calls. Device row
  identities are prepared during setup; one index slice is shared by all chunks.
  Controlled comparisons favor this rewrite at B8/B16/B32 for both layer kinds.
- Fused paged K/V decode update uses disjoint K/V core sets; update temporaries are
  released before SDPA. Sliding Q and K stay sharded through their consumers.

Required movement has concrete contracts: RMSNorm rejects height-sharded input,
so dedicated decode split outputs pass through DRAM for normalization. Full Q
must be in DRAM for its long-context SDPA configuration: retaining sharded Q still
collides with static L1 buffers after temporary-lifetime repair. These adapted
paths were made runnable and measured, rather than rejected on their first error.
For full attention at B>=34, K is relocated through DRAM before cache update to
recover the top-of-L1 placement left fragmented by RoPE temporaries. B36/S33 then
passes like the functional path; B37/S33 exceeds L1 in both. This boundary depends
on page-table geometry, not a new public batch cap. See AUTOFIX.md. There are no
tilize/untilize calls or host fallback in final measured forwards.

BF16 weights/activations/KV and HiFi4/FP32 accumulation remain unchanged. The
functional full-attention K128/core-cap16/precise-exp SDPA configuration is retained.
Full attention packing does not retain a second weight copy. Sliding attention
retains one 168 MiB QKV replica for its measured B32 prefill gain; no request-time
packing or allocation of persistent weights occurs. The selected decode graph reproduces the best correct candidates within measured
variation. Sub-microsecond historical wins are not claimed as meaningful differences;
whole-layer device results are compared with the functional baseline below.

## Context, caches and trace lifetime

The advertised and supported context remains262144. Physical prefill chunks are
at most1024 tokens, with metadata-only logical views for tails. Public prompt
lengths need not be divisible by tiles, pages or chunks. Explicit prefix continuation
starts at a page boundary; this pre-existing caller-owned-cache contract is unchanged.

K/V cache remains separate BF16 `[blocks,kv_heads,32,head_dim]` DRAM tensors.
Bytes per token remain16384 sliding and8192 full; a batch1 maximum-context cache
uses4GiB and2GiB respectively. Full attention stores one Q/K packed weight family
instead of separate Q and K; tensor element counts are unchanged. Sliding retains
a 176160768-byte setup-only packed QKV replica: measured B32 prefill was 2467 ms
versus 2525 ms for separate projections, while separate decode remained faster.
This is the explicit copy reason; maximum-context validation includes that copy. No capacity
reduction, bounded-cache eviction or precision change is used to obtain a gain.

The caller prepares inputs, RoPE and page tables before capture. Decode positions
and page rows are device tensors; trace replay consumes updated values in the same
allocations. Trace-allocation tracking includes program-cache allocations, with no
exemptions. Tests warm supported physical signatures, consume/drop eager outputs
before replay and retain cache/input ownership until teardown. Changed row order,
positions, pages, unseen logical lengths within a prepared bucket and resumed
prefill are covered on the final implementation. Layer harness host refresh is
outside measured device windows; this is not a token-out generator claim.

## Workload and measurements

Agent workload remains the stage01 estimate: concurrency32, working set32768 tokens
per agent, typical context4096. Configured decode batches are1,8,16,32. Stage02
measures every batch; B1/concurrency1 is the regression/headline row. Headline
profiles execute4096 input tokens followed by128 teacher-forced decode positions.
Each meaningful layer kind is reported separately.

`performance_<kind>_batch<B>.json` reports whole-layer device windows: first FW start
through last FW end across all operations, including internal gaps. Decode averages
128 complete traced windows; caller refresh/drain gaps between replays are excluded.
Host trace timings in candidate JSON are only for ranking, never device-time telemetry.

`tracy_final/<kind>/batch1/` and `tracy_batched_fill/<kind>/batch<B>/` retain
raw processed ops CSV, native tt-perf-report
prefill/decode tables and CSV, collection log and exact source/command manifest.
`profile_provenance.json` connects every report to the source and raw CSV hash.
Whole-layer roofline formulas use useful dense projection+attention FLOPs and
estimated physical DRAM I/O, not matmul-only duration or averaged op percentages.
The whole-ASIC peak basis is inherited from the validated functional report:
165.888TFLOP/s at HiFi4 and512GB/s for one participating p300c ASIC. Available worker
grid11x10 is recorded separately from that whole-device theoretical denominator.

DRAM estimates count every ordinary DRAM operand/output once, sparse active KV and
cache tile updates, excluding L1-only traffic. BinaryNg profiler shape metadata can
retain prefill dimensions under shape-independent program reuse; the estimator
uses actual decode geometry for those operations. Padding is counted; redundant
per-core reads and hardware cache reuse are not modeled. These are estimates,
not memory-controller counters. No suspicious percentage is clamped.

## Evidence and rejected options

See [pattern inventory](pattern_inventory.md) for every graph-fusing pattern and
repo-discovered cache/shared-norm rewrite. Candidate manifests retain commands,
PCC failures/successes and host timing rows. All first API/layout failures were
adapted or tied to exact op constraints; candidate failures are not final gates.

- `candidates/`: initial RoPE, packing, activation and head-transform comparisons.
- `candidates_adapted/`: repaired full-attention L1/shape paths and shared K/V norm.
- `candidates_combined/`: compatible combinations at B1/B32, including fused cache.
- `candidates_batches/`: batch-dependent split selection and direct-sharded norm
  rejection, with shared cos/sin transfer.
- `acceptance_manifest.json`: final source hashes, commands and exits.
- `validation_queue.json`: real-weight context/boundary/batch/watcher runs.
- `equivalence_*.json`: direct functional-versus-fused output and cache PCC.
- `pytest_final.log`: synthetic real-shape regression tests on fused runtime.
- `AUTODEBUG.md` / `AUTOFIX.md`: profiler overflow diagnosis and capacity repair.

MLP packing and matmul-fused GELU were numerically valid but slower than separate
projections plus binary GELU. Dedicated decode head concatenation was slower even
after padding adaptation. Full prefill concatenation was adapted to fit L1 rather
than discarded. Absent patterns (convolution, TopK, distributed RMSNorm, explicit
softmax/reductions) have no matching subgraph in this single-device decoder.

## Limitations and review

These are representative single layers0/5, not full-model quality or throughput.
Long HF comparisons sample selected output rows while TT executes the complete
prompt and the reference constructs all required prefix KV. No text-generation
quality or serving-readiness claim follows from this stage.

Final tables and independent review are recorded below and in [work_log.md](work_log.md).
Nothing is pushed. The optional browser-copy warning does not lose the local
.tracy file: it exists in each custom capture directory. Mixed-column warnings
are classified in AUTOFIX.md; all required timestamp and capture-ID audits pass.

The headline B1 device graph is unchanged by batched fill and the B>=34 K-placement
repair. Its retained final profile has source a18a3b4; fresh B8/B16/B32 profiles
use the final source. Exact full hashes and commands are in collection manifests.
The B1 maximum-context run likewise retains the same runtime graph; only the
440-byte setup index payload was added afterward. No capability was reduced.

The main reproducible entrypoints are `run_fused_decoder`, `test_fused_equivalence`,
`check_fused_batch_boundaries`, `check_prefill_index_boundary` and
`check_full_batch_capacity` under this autoport's tests package. Actual commands
and exits are recorded in the manifests. For a fresh profile, use
`run_fused_profile_queue --artifact-root <new-local-directory> --batches 1 8 16 32`;
existing captures are never silently reused across a changed model/profile harness.


## Final acceptance results

All correctness, context, repeated-run, watcher and profiler-completeness gates passed. Independent final review is recorded in stage_review.md. Latencies below are whole-layer device microseconds; prefill is warmed and decode is the mean of 128 warmed trace replays. Each row uses 4096 input tokens.

| Layer kind | Batch | Prefill before → after (µs) | Reduction | Decode before → after (µs) | Reduction |
|---|---:|---:|---:|---:|---:|
| sliding_attention | 1 | 97224.4 → 70874.0 | 27.10% | 3039.46 → 2996.81 | 1.40% |
| sliding_attention | 8 | 768889.8 → 566412.7 | 26.33% | 3248.81 → 3213.68 | 1.08% |
| sliding_attention | 16 | 1537798.5 → 1131258.8 | 26.44% | 3664.53 → 3628.31 | 0.99% |
| sliding_attention | 32 | 3060012.7 → 2201642.5 | 28.05% | 4518.44 → 4380.07 | 3.06% |
| full_attention | 1 | 136683.9 → 99991.4 | 26.84% | 3343.52 → 3263.95 | 2.38% |
| full_attention | 8 | 1089629.1 → 801013.6 | 26.49% | 4029.44 → 3947.91 | 2.02% |
| full_attention | 16 | 2176647.2 → 1600111.0 | 26.49% | 4869.85 → 4750.46 | 2.45% |
| full_attention | 32 | 4338213.6 → 3198766.5 | 26.27% | 6602.36 → 6361.75 | 3.64% |

Exact values and baseline paths: performance_summary.json. Final profile paths and collection/report commands: profile_provenance.json.

At context 262144, HF-reference PCC remains above 0.995:

| Layer kind | Prefill before → after | Decode before → after |
|---|---:|---:|
| sliding_attention | 0.999933120 → 0.999932825 | 0.999913867 → 0.999908895 |
| full_attention | 0.998180195 → 0.998191698 | 0.999979899 → 0.999988295 |

The small BF16 differences from shared normalization, GELU and RoPE are immaterial to the acceptance bar. correctness_summary.json also records direct output/cache comparisons at B1/B32 with non-aligned S1025. The final batched watcher runs cover S2049, logical tails, continuation, changed positions/pages, ten trace replays and six interleaved prefills. watcher_audit.json records 55 inspected files with no fatal/error/corruption/overflow/illegal matches. Boundary manifests cover head-split, fused-cache and batched-fill branch boundaries, including prefill-only B110/B111; full-attention capacity controls confirm both implementations pass B36/S33 and encounter the inherited physical L1 limit at B37/S33.

The tt-perf-report kernel breakdown identifies matmul and SDPA as the remaining dominant work. At B1 sliding prefill they consume approximately 39089 and 23370 µs; full prefill 42571 and 46773 µs. Decode matmul is approximately 2351/2617 µs for sliding/full, versus SDPA 163/162 µs. These kernel subtotals explain the gains; they are never used as the roofline denominator. No applicable fusion pattern remains unassessed: runnable alternatives were compared and rejected options are in pattern_inventory.md. Further layout, precision or custom-kernel tuning belongs to the subsequent optimization stage.

final_candidate_comparison.json records the original decode selection epoch. The later batched-fill rewrite changes prefill only for configured batches; its final host medians are recorded in batched_timing_*.json. Sliding B8/B16/B32 are 3213.91/3628.47/4389.75 µs and full are 3944.96/4751.33/6365.72 µs. These reproduce the selected fastest correct decode graph with small run variation; a strict win over every historical noisy sample is not claimed. Whole-device decode improves over the functional baseline in every row. Batched prefill was selected using controlled ABBA comparisons (compare_batched_fill.py and its output), then independently profiled on the final source.
