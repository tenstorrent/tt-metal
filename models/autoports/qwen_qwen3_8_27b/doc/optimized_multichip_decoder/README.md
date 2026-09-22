# Qwen3.8-27B optimized multichip decoder

Status: complete; independent stage review `clean-pass`. This is Stage5 of the repo-local
autoport pipeline, starting from completed multichip commit `1623f9cb595`.
No full-model or serving implementation is part of this stage.

The measured model is the real dense Qwen/Qwen3.8-27B text decoder on a
four-chip Blackhole `MeshShape(1,4)`. Attention heads and MLP channels are
tensor-parallel; the layer weights are not replicated model copies. Layer0
represents linear attention and layer3 represents full attention. MoE routing,
LM-head, sampling, and text-generation requirements do not apply to this decoder.

## Evidence and measurement contract

- The initial operation-topology audit is the first table in [work_log.md](work_log.md).
- [commands.log](commands.log) records exact commands. Each timing result has
  a JSON report, process log, commit marker, and source hashes; deduplicated
  source snapshots are in `sources/`.
- `final_installed_library.sha256` identifies the actually installed
  native extensions used after the native repair. Build and installation are
  recorded separately because compiling does not update the loaded copies.
- [candidate_summary.csv](candidate_summary.csv) and
  [projection_summary.csv](projection_summary.csv) summarize completed runs.
  Projection microtraces use recorded real decoder inputs and real weights,
  and include that projection's input/output layout operations.
- All decode timing uses warmed TTNN trace replay. Prefill uses warmed eager
  execution. Correctness compares against frozen single-chip TTNN fixtures
  anchored to the completed decoder's HF checks; single-chip performance is
  never used as this stage's before measurement.
- Every forward is guarded against Torch operations and host conversion.
  Trace checks include changed inputs, positions and page tables, output/state
  bitwise parity, and cache write ownership. The accepted output PCC floor is
  .995, including per-user checks.
- Profiler runs and watcher runs are separate. Per-device perf tables are the
  accounting source: merged four-device timestamps can create false large gaps.
  Raw profiler archive paths and hashes are in `raw_archive_manifest.json`.
  `format_originals_manifest.json` preserves byte-exact originals of text
  normalized by repository hooks; compressed raw operation CSVs remain readable
  by the accounting script.

## Default-path results

Batch1, logical prefill128, real TP4 mesh; milliseconds. Decode is warmed trace
replay, with one replay plus synchronization per timed sample. Prefill uses60
warmed samples in the final run. The queued supplement amortizes host dispatch
but is not substituted for the inherited primary timing.

| Layer kind | Before prefill | Default prefill | Before decode | Default decode | Decode reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear attention | 1.415843 | 1.248232 | 0.474841 | 0.421990 | 11.1% |
| Full attention | 1.233765 | 1.177266 | 0.351291 | 0.306304 | 12.8% |
| Linear → full stack | 2.616180 | 2.347569 | 0.791311 | 0.704825 | 10.9% |

Artifacts: `before_l0.json`, `before_l3.json`, `before_stack_retry.json`, and
`after_review_l0/l3/stack.json`. Final runs use empty model policy overrides,
the installed native libraries and the optimized launchers' default
`TT_MESH_PASS_THROUGH_THREAD_POOL=1`. Each report records the effective policy,
model/runner hashes and runtime environment. Warmed prefill improves for all
three workloads; decode improves by 10.9–12.8%. Earlier candidate or pre-review
values are retained as historical evidence and are not substituted here.
Non-aligned 33/2049 and aligned 2048 before controls also exist.

| Kind | Before prefill PCC | Default prefill PCC | Before decode PCC | Default decode PCC |
| --- | ---: | ---: | ---: | ---: |
| Linear | 0.999995410 | 0.999995410 | 0.999999285 | 0.999999225 |
| Full | 1.000000000 | 1.000000000 | 0.999999583 | 0.999999404 |

All default output, cache and recurrent-state PCCs pass the accepted.995 floor;
minimum state PCC is.999957 linear and.999928 stacked. Exact trace replay,
changed-input replay and state parity also pass. The final evidence index
checks current source, strict PCC, process exit and selected runtime setup.

## Selected policy and comparisons

- BFP4/LoFi projection weights, BF16 activations/residual/norms, FP32
  accumulation and recurrence, BFP8 paged KV. Every projection group was
  compared with BF8/LoFi, BF8/HiFi2 and reduced accumulation alternatives.
- Two DRAM readers per bank for attention/output/down; three for packed gate/up.
  Logical input grids are10/8/40/8 cores with K blocks16/6/4/17 respectively.
  Residual and AR use40 cores. Both norms keep the same L1 residual layout.
- Packed QKVZBA or QKVG attention and packed gate/up win against adapted split
  attention, separate gate/up and fused SwiGLU alternatives.
- Bounded64–256-token prefill uses1D matmul: linear K8/L1, full K20/DRAM,
  with role-specific output/down blocks. Longer prefill retains minimal/2D
  programs and uses4096-token chunks. Public logical lengths remain unrestricted.
- Decode SDPA keeps short8x2, otherwise11x10, with K128 reduced internally as
  needed to divide mapped cache capacity. Paired logical4094 K128 improves
  queued decode from~.313 to~.307ms without changing KV layout or dtype.

[collective_contracts.md](collective_contracts.md) records coherent topology,
payload, persistent-buffer and residual comparisons. The best adapted sharded
residual with fused distributed norm is~.775ms per stack; fused AGMM/MMRS
families are slower. BF8 CCL does not give a repeatable material improvement.
[inter_layer_contract.md](inter_layer_contract.md) specifies the required
boundary for later assembly: no gather, reshard or all-reduce between layers.

`matmul_geometry_search.csv` contains per-role precision-locked geometry,
real-input projection microtraces, whole-layer traces and PCC. Output subblocks
are source-derived from the native factory because that API does not expose
an override. FP32 limits subblock area to4; reader/grid sweeps change legal N
blocks and K blocks, including non-powers-of-two. Projection summaries retain
full input/output memory configs; final profiler rows provide actual op timing.
All candidate matrices, policy overrides and failed/adapted logs remain indexed
in `candidate_summary.csv`, `topology_family_results.json` and `work_log.md`.

## Native repair

The multi-reader DRAM matmul factory queried physical NoC distances through a
non-unit mesh. It now creates descriptors for each mesh coordinate and resolves
that coordinate's physical device for reader placement. This preserves true
TP execution. A second fix handles readers that own only N padding: they drain
their computation without writing a nonexistent logical output shard.

Both fixes compiled using the existing native `ttnn` build. The prescribed
Docker wrapper could not run because Docker is unavailable; no compiler or
dependency was installed. Eighteen final native mesh/cache/placement tests and eleven
existing unit-mesh controls pass. Reports and patches are in `AUTODEBUG_dram_*`,
`AUTOFIX_dram_mesh.md`, and `native_*` logs. Final watcher and model stress
also pass for the selected policy.

Multi-reader remote-coordinate execution is explicitly unsupported. This stage
uses four local physical chips. Heterogeneous harvesting is source-covered by
per-coordinate placement but has not been tested on different harvesting maps.

## Correctness, stress and watcher

`final_correctness_pytest.log`: 31 passed in 291.74 seconds, covering both kinds, non-aligned
logical lengths through4097, continuation and batches through32, plus two-kind
stacks atB1/2/3/8/16. `native_lint_final.log`: 18 passed after adopting the repo error fixture, covering all
reader counts, TP4 and offset TP2, output tails and repeated allocation/cache
generations. No tests are skipped or xfailed in these selected suites.

`final_stress_stack_b1/b32.json`:100 eager and100 queued trace iterations with
bitwise output/state parity through both kinds. Minimum PCC.999750/.999827,
minimum per-user output PCC.999993/.999989. Every forward is device-only.

`final_watcher.log`: four cases pass under watcher10 with no disabled features,
including linear logical2049, both kinds atB32/S257 and stackB3/S33.
Each case exits 0 through driver shutdown. The additional
`watcher_review_prefill_trace_stack` B1/S128 case validates both layer kinds
through the repaired short-prefill trace and closes cleanly. Full ETH instrumentation requires
O3 plus supported watcher noinline to fit the kernel buffer. The router teardown
repair clears owned-NoC packet tags after barriers; the original Os/noinline
assertion and reset recovery are preserved in the anomaly ledger. No profiler
runs overlap watcher.

## Context and memory

The updated [context contract](../context_contract.json) remains262,144 tokens
with no public sequence-alignment requirement. All five final default capacity
probes pass: exact-max prefill for both kinds, near-max prefill followed by
last-position decode for both kinds, and the mixed stack. Their strict PCC
minima are.99992996 or higher, with trace/state parity where decode is run.

`memory_capacity_plan.json` is validated against the current model source and
process exit markers. Stored projection copies total7,015,956,480B per rank;
the conservative complete-model resource plan totals30,069,112,832B versus
34,138,688,512B physical DRAM. Probes reserve12,889,243,648B plus real
full-length activations/state. These are decoder capacity probes and resource
arithmetic; they do not claim a completed full-model implementation.

## Profiler evidence and accounting

`tracy/review_profile_l0`, `review_profile_l3`, `review_profile_stack` and their
long-context variants contain per-device prefill/decode tables and CSVs with
advice enabled. `performance_accounting.json` reconciles the stored-tile read
roofline, device kernel-plus-gap span and the same signposted host measurement.
Profile instrumentation affects timings; the headline table uses unprofiled
100-replay default runs.

| Profile | Read lower bound ms | Device span ms | Same-window host ms |
| --- | ---: | ---: | ---: |
| review_profile_l0 | 0.110880 | 0.435705 | 0.506147 |
| review_profile_l3 | 0.105290 | 0.313456 | 0.379845 |
| review_profile_stack | 0.216170 | 0.743605 | 0.813322 |

The read lower bound includes projection bank/tile padding and KV reads but
excludes activation traffic, recurrent state, constants and writes. It is not
a measured bandwidth claim. The installed tool hardcodes8 DRAM matmul compute
workers and can report~101% FLOPs utilization. Runtime attributes prove16/24
workers; `final_matmul_rows.csv` adds the corrected denominator and preserves
the original rows. Its DRAM percentages also use logical bytes.

`inter_layer_profile_audit.json` proves the stack boundary on all four devices.
The two internal reductions per layer remain faster overall than the measured
sharded-residual, fused-normalization, AGMM and MMRS families. Their final
profiles retain the adapted downstream consumers and persistent buffers.


## Review repair, tracing and host dispatch

The independent review found four short-prefill BF16 identity casts per layer.
The final branch guards dtype equality; `review_cast_audit.json` compares the
before/after device operations on all four ranks. Correctness and final
performance were refreshed after the repair.

Prefill-trace advice was implemented in the caller-owned test harness with
stable input/state/page-table addresses, exact changed-input and state checks,
and a full-watcher stack control. `review_trace_prefill_*` covers both kinds at
128/2049 plus B3/S33 and B32/S257. Ordinary eager prefill remains the comparable
headline metric. The explicit trace option is available for callers that can
preserve its storage contract.

| Optional trace case | Eager prefill ms | Traced prefill ms |
| --- | ---: | ---: |
| Linear B1/S128 | 1.311211 | 0.897442 |
| Full B1/S128 | 1.110037 | 0.734681 |
| Linear B1/S2049 | 5.313371 | 5.018578 |
| Full B1/S2049 | 4.476514 | 4.129954 |
| Stack B3/S33 | 4.276233 | 3.658669 |
| Stack B32/S257 | 59.483830 | 59.070747 |

`AUTODEBUG_profile_gap.md` and `AUTOFIX_profile_gap.md` localize historical
host timing outliers to the completion wait. A targeted pool-bypass experiment
improves eager prefill across all three workloads and removes sampled large
completion tails. The selected launchers set the pass-through option before
mesh creation. The evidence supports workload-specific host-overhead reduction;
it does not claim a general thread-pool correctness defect or prove the exact
cause of every historical outlier. Final accounting uses fresh untouched runs.

The rank-adapted monolithic GDN probe passes PCC but costs 0.551348 ms decode;
serial preparation increases S2049 prefill to 7.432725 ms, and serial scan has
no repeatable advantage. The final path retains phased GDN with parallel scan
and preparation. These were measured rejections after adapting the contracts.

[advice_closure.md](advice_closure.md) maps applicable optimization requirements
to trials and decisions. [cumulative_contract.md](cumulative_contract.md) records
the selected topology, precision, logical batch, attention, cache, norm and
matmul contracts together. The final independent verdict and local checkpoint
SHAs are recorded in [work_log.md](work_log.md).
