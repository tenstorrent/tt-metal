# Full-model profiler evidence and advice

All collections finish successfully with the effective source-path overlay,
model-directory cwd and fresh absolute output directories. Watcher is disabled.
`commands.jsonl` in the parent records collection commands; each
`*_provenance.json` records exact report commands and SHA256 of the full raw CSV
and lossless signpost-window archive. Every original column and row inside the
window is retained. Tables use advice enabled, Blackhole architecture and
merged devices. Uncompressed CSVs can be reconstructed with gzip -dc.

| Window | Device operations per rank | Scope |
|---|---:|---|
| decode | 104 | One real linear + full-attention layer, embedding, final norm/head, position advance, split greedy/feedback |
| prefill | 158 | Public B1 generation's reset/admission, 128-token prefill and first sampled token/readback |
| chunk_prefill | 439 | Same public path, 32769 logical tokens: linear32768+1, full16384+16384+1 padded internally |
| guard_full | 6 | Real B32/31-active full-pool recurrent and convolution protection, one linear layer |
| guard_selective | 5 | Same guard with inactive-row-only FP32 copy |

The alternate sampling/masked traces are prepared but not replayed in the B1
profile. Tracy's warnings that these alternate trace IDs have no replay rows
are expected. The measured decode window has104 device rows on **each** rank,
zero host operations and one replay of each selected model/sampling trace.
There are no dropped-marker, missing-device-data or source-collision failures
in the accepted collections. Failed broader B32 capture attempts remain logged.

## Decode budget

| Component | Merged kernel + gap time |
|---|---:|
| Embedding/entry | 0.034938 ms |
| Linear layer | 0.243511 ms |
| Full-attention layer | 0.262858 ms |
| Final norm, DRAM-sharded head and logits layout | 0.601917 ms |
| Device position advance | 0.001337 ms |
| Split greedy, sampler state and token feedback | 0.593567 ms |

Packed stack estimate:24×0.243511+8×0.262858 = **7.947128ms**. Add entry/terminal
once:9.178887ms versus measured full resident8.708730ms. This is not an
unexplained >10–15% gap. The estimate sums separately merged op/gap maxima and
instrumentation; its negative residual must not be labeled negative host time.
The stage5 optimized private S1024 layer latencies give
24×0.290397+8×0.275834 =9.176189ms before terminal work, but include private
entry/exit conversions and longer attention history. They are retained as the
requested layer-budget reference, not misrepresented as the packed lower bound.

Same-run reduced replay at position131: firmware spans1.726575/1.729251/1.728515/
1.720941ms on ranks0–3 (1350MHz); synchronized host1.808981ms, remainder0.079730ms.
That remainder includes two replay calls, final synchronization and profiler
instrumentation. The projection+head+valid-KV bandwidth floor is0.753066ms for
that exact reduced run. Full32-layer quiet-host resident reaches about47.9% of
its conservative stored-BF8 projection/head+KV floor, above the10% functional
bar; no70% roofline claim is made. `layer_stack_accounting.json` and
`../perf_summary.json` expose every term and scope.

The selected head kernel is556.647us, versus749.273us in the prior accepted
full-model profile. Full terminal norm/head/layout falls784.385→601.917us.
The report labels the head89.2% DRAM utilization; its simplified byte model
omits BF8 tile-header storage, so this is not a whole-model utilization claim.
`measured_matmul_policy.json` preserves all11 dominant decode matmul attributes:
BF16 inputs, BF8 weights, expected HiFi2/LoFi fidelity, explicit program configs
and readers/bank. The table's Cores=8 is not the reader count; derived FLOPs
percentages above100% are not used as utilization evidence.

Masked protection falls0.309320→0.031982ms per linear layer, predicting
6.656112ms full-stack savings. The full model measures6.546332ms saved, with
identical final tokens, exact preserved inactive state and zero host-boundary
counters. Remaining masked overhead is0.730ms versus all-active; the guard
profile predicts0.768ms. `masked_gap_accounting.json` records the comparison.
The generic copy report's INT32 output label reflects the active-mask operand
returned by generic_op; the operation bit-copies FP32 recurrence, as source and
four-rank bitwise tests prove. No conversion to INT32 occurs.

## Prefill and metadata interpretation

`prefill_chunk_kernels.csv` gives each dominant matmul role, logical/physical
chunk rows, recorded tensor metadata, measured time, dtype, fidelity, memory
and program attributes at128 and32769 logical tokens. The public before/after
TTFT sweep additionally covers513 and16385. Its physical buckets are recorded
in `../prefill_summary.json`; padding remains internal.

MinimalMatmul deliberately excludes row count from the program hash and
updates runtime M_tiles (`minimal_matmul_device_operation.cpp:101–125` and
`minimal_matmul_program_descriptor.cpp:13,545`). Tracy reuses tensor metadata
by that hash (`tools/tracy/process_ops_logs.py:465–495`). Consequently a128-row
call can retain warmup metadata32, or a16384-row call can display32768. The
annotated table's executed rows come from the deterministic validated plan
and ordered layer calls; the unedited raw table retains those stale metadata
values. Kernel durations/operation identity remain measured. Do not use its
cached M field for a FLOPs/physical-row assertion. Decode head shapes are fixed.

## Advice disposition

- **Decode gaps / enable tracing:** already two nonblocking traces, confirmed
  by replay IDs and runtime host-boundary tripwires. Generic tracing advice
  cannot remove these measured device gaps. No per-token host synchronization,
  input refresh or logits transfer remains in the resident loop.
- **Prefill gaps:** eager orchestration is retained for explicit per-request
  chunk history, continuation, scheduler mappings and variable logical lengths.
  Profiler host instrumentation inflates the gap: use the quiet full-model
  TTFT sweep, not54.7ms reduced-profile span, for ranking. A new prefill trace
  catalogue is not adopted by this change; this is a performance decision, not
  a claim that TTNN cannot trace prefill. Existing all-bucket preparation and
  no-late-compilation contract are preserved.
- **MinimalMatmul missing program_config:** it uses its own explicit
  MinimalMatmulConfig, preserved in raw attributes (large chunks K16,
  Mblock4/Nblock8/subblock1x2, grid11x10). The generic Matmul program_config
  suggestion does not apply to that API. Prior phase-specific geometry and
  adapted row/CCL families remain in the stage5 topology ledger.
- **Large prefill input in DRAM:** retained to fit legal32768/16384 operation
  rows and bounded workspace; the stage5 DRAM-prefill/phase2D comparisons and
  maximum-context gates justify it. Moving the complete activation to L1 is
  not a legal wholesale substitution at these sizes.
- **Small subblocks / higher fidelity:** the inherited short-prefill geometry
  and measured precision policy are unchanged. Stage5 independently swept
  projection roles/fidelity. The user's preservation constraint takes priority
  over generic HiFi2/HiFi4 suggestions for selected LoFi MLP. No new datatype
  frontier is inferred from advice.
- **LM-head DRAM sharding:** applied to repeated decode and measured against
  adapted split/whole controls. General prefill retains its arbitrary-height
  interleaved head; its one-time terminal cost is about0.750ms in both prefill
  windows. Reusing the decode head for shape-specialized prefill is a possible
  further small optimization, not claimed here.
- **Generic sampler bottleneck:** absent. TopkLargeIndices is266.217us; the
  complete split sampler is6.8% of full token-out time. No force-argmax,
  full-vocabulary gather or generic TopK is in the measured104-op decode path.
  Correct forced-argmax control is4.57× slower at sampling alone.

Optimization completion uses the checklist's measured/kept/rejected decisions,
including preserved source-equivalent stage5 trials. It does not invoke the
70% or time-box stop rules. Further prefill tracing/head specialization are
identified performance candidates, not hidden runtime fallbacks or failed gates.
