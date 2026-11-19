# Optimized multichip decoder: Ornith-1.5-9B

Correctness, final profiles, consolidated audit and 24 separate diagnostics-off timing runs pass. Independent [stage review round 2](stage_review_round2.md) returns **clean-pass**. Local checkpoint SHAs are recorded in [work_log.md](work_log.md); nothing is pushed.

This is stage 5 only: no full-model or vLLM work. The measured implementation is TP4 on four P300c Blackhole devices 0–3, mesh `[1,4]`, FABRIC_1D, Linear topology and two links. Projection weights and attention heads are sharded across ranks; replicated residuals do not mean replicated decoder execution. HF revision: `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`.

Final production SHA256: `b1d5016dd769ab65fce82e27d483f73a33fa2e746e29e21952d521052948d2d8`.

## Final measurements

Decode rows measure a packed two-layer stack: one real linear-attention layer and one real full-attention layer. These are decoder-stack times, not full-model token latency. Before uses the frozen completed multichip decoder with the same packed-I/O harness, inputs, batch, capacity and serial regime. All after numbers come from the final default. All 45 rows now use fresh serialized processes with allocation tracking and tracebacks disabled, watcher/profiler unset and no Tracy. [final_timing_audit.json](final_timing_audit.json) verifies 24 source-bound artifacts and matching workloads, references, warmups, harnesses and diagnostic settings; the original tracked runs remain correctness evidence.

| Traced warmed decode workload | Before, ms | After, ms | Speedup |
|---|---:|---:|---:|
| B32, 4K prefix, 1024 answer tokens (primary) | 19.855421 | 0.828631 | 23.96× |
| B32, 64K prefix, 128 answer tokens | 24.221873 | 3.635508 | 6.66× |
| B32, 128K prefix, 1024 answer tokens | 28.889898 | 6.731746 | 4.29× |
| B1, short prefix | 0.578599 | 0.483845 | 1.20× |
| B32, short prefix | 18.990080 | 0.587436 | 32.33× |

| Warmed prefill per layer | Before, ms | After, ms |
|---|---:|---:|
| Linear, S1024 | 2.098791 | 1.963015 |
| Full attention, S1024 | 2.215763 | 2.073185 |
| Linear, S65536 | 141.840874 | 101.194906 |
| Full attention, S65536 | 574.769887 | 532.573878 |

All 45 matched timing rows, including short and chunk-boundary lengths, are in [final_performance.csv](final_performance.csv). Private B1/S1024 decode changes from 0.307956 to 0.290397 ms for linear attention and 0.288181 to 0.275834 ms for full attention. The B1 128-in/128-out FUNCTIONAL checks reach 40.4% and 41.5% of their conservative bandwidth floors, above the unchanged 10% minimum.

The matched table retains measured prefill regressions: 2.22% at linear S513, 4.03% at full-attention S512, and 0.24% at full-attention S513. The pass does not claim improvement at every length.

The primary B32 workload reaches 56.1% of its useful-bandwidth roofline estimate; no primary 70% claim is made. The optimization search stopped at eight hours of serialized hardware-facing work at 05:40 UTC. [stop_rule.json](stop_rule.json) freezes that point; [device_time_accounting.json](device_time_accounting.json) separately records later mandatory repair and validation. Job intervals include setup, compilation, validation, teardown and profiler host processing; they are not kernel execution hours. Untried items are handoffs in the work log.

## Correctness and final evidence

All 82 selected final device jobs pass. [final_audit.json](final_audit.json) binds them to current production Python and kernel hashes; [final_accuracy.csv](final_accuracy.csv) lists every artifact. Minimum output/state/contribution PCC is **0.99571836**, above the unchanged **0.995** threshold. Replicated-rank checks retain **0.99999**. The primary 4K/1024-step run has minimum PCC 0.99607718.

Coverage includes every batch 1–32, both layer kinds, changing-input trace replay, terminal state, refresh and allocation-tracker positive controls, long answers, reordered/subset/mixed pools, reset/isolation and interleaved traces. Eight recorded-HF checks cover B1/B32 and short/long inputs for both kinds, including residual-subtracted contributions. Retained short-prefill boundary controls are indexed in [retained_short_prefill_evidence.json](retained_short_prefill_evidence.json), with their source-equivalence review. Torch execution and host-transfer guards protect the runtime path; the source fallback audit is clean.

Watcher runs use interval 10, NOINLINE and assertions, separately from profiles. Actual production convolution/gate helpers pass 2,048 direct records with bitwise agreement for convolution, gates, history, recurrent state and output against preserved controls; standalone ordered recurrence also passes 104- and 7-core waves. The legacy convolution discrepancy is controlled by identical tensors from the production-substitution test, not a relaxed threshold: [anomaly resolution](conv_gates_current_audit/anomaly_resolution.json).

The added 128K/B32/1024-step test exposed a fixed-K64 SDPA precision regression. HiFi4 alone did not fix it, and K128 failed at step 1,013. K256 and K512 passed the complete test. The capacity-aware repair passes the original workload and populated-128K watcher test. See [AutoFix](context128k_autofix/AUTOFIX.md), its independent AutoDebug report and [control matrix](context128k_autofix/controls.json). The quality gates and context limit were unchanged.

## Selected implementation

Weights remain BFP8. Attention projections use HiFi2, MLP uses LoFi; recurrent state stays FP32 with the preserved HiFi4 arithmetic. Decode residuals and matmul outputs stay BF16. Async all-reduce uses BF8 payload and setup-owned BF8 receive buffers, with BF16 output. Packed attention input projections remain fused. Decode MLP gate/up remain separate after adapted packed families measured slower; larger prefill retains fused packed SwiGLU.

Complete ordered linear pools, including B1, use tiled history, fused convolution/gate preparation and ordered recurrence/postnorm/gating with NoC zero fills. Private B1 state remains supported. DRAM-sharded matmuls explicitly use two workers per bank for packed input/gate/up and four for output/down. The runtime attributes confirm these values; the report's Cores=8 field is not workers per bank.

Paged decode SDPA uses HiFi2, FP32 destination accumulation, BF16 running statistics/intermediates and the TTNN default exponential approximation. The decoder selects the smallest K64/128/256/512 that limits local running-statistic updates to 200 per worker over the original page-table capacity, with an explicit 16-worker-per-head cap. At B32, approximately 4K/64K/128K/max capacity selects K64/K128/K256/K512. This is an empirically validated numerical-risk bound, not a mathematical PCC guarantee. It needs no position readback or mid-request trace change. Overprovisioning cache capacity can select a larger K while the current position is still short.

Prefill defaults to 32768-token linear chunks and 16384-token full-attention chunks for replicated residuals with nonfused prefill. Other modes retain 1024; explicit overrides are honored. Above 4096 operation rows, hidden normalization stages through DRAM; projection inputs stay in DRAM above 8192. Measured 2D projection configs apply at 512–1024 operation rows. Norm output and prefill RS/AG payload use BFP8 at 512+ operation rows, with BF16 outputs/residuals. Full-attention 511 logical rows pad to 512 before normalization; linear 511 stays 511. Padding, masking and slicing are internal responsibilities, so public logical lengths need not be aligned.

The operation-topology audit precedes the candidate search at the start of [work_log.md](work_log.md). [Topology decisions](topology_decisions.md) compares coherent residual, collective-placement, fused CCL/matmul, packing, dtype and persistent-buffer families, including adaptations and measured rejections. This model is dense; MoE requirements do not apply.

## Inter-layer and trace contract

Share one `MultichipCollectives` workspace across a serial stack. Default `residual_layout="replicated"`, `native_local=True`, `movement_optimized=True` executes TP4. Decode residuals are BF16, replicated across mesh ranks and WIDTH_SHARDED locally across the origin 8×4 grid with `[32,128]` shards. Inner attention/MLP features remain rank-local. Necessary row-parallel reductions occur inside layers; **no gather, reshard or all-reduce occurs between decoder layers**.

Pack requests once into BF16 `[1,1,B,4096]`, 1≤B≤32, at stack entry. Chain `decode_batch_packed` outputs directly into the next layer and unpack after the stack. Do not restore DRAM or another shard contract at every boundary. The list API additionally pays entry/exit packing. Private `decode_forward` returns the same local L1 residual layout. Prefill crosses layers as logical BF16 DRAM-interleaved rows.

Use `allocate_batch_state` for shared pools, including pooled linear B1. Keep state objects and backing buffers alive with every trace that references them. Complete ordered linear pools use the fused path; subsets/permutations/mixed pools use device-only per-request attention with grouped dense projections. Duplicate state regions are invalid. Full attention aliases shared KV pools or packs/scatters private and mixed pools as needed.

Page tables contain local page indices. Attention tables own K/32 alignment and pad local physical page zero **before** pool offsets; cache-update tables remain unchanged. Reset linear requests through `write_linear_state`, then prefill the new prefix. Reset full attention by writing its new valid prefix from position zero and resetting its position; later entries remain masked. Warm every replay/refresh program before capture. Trace signatures include batch, cache/table capacity (thus K) and pool layout. Shape changes require preparation and capture. Allocation tracking includes program-cache allocations; only exact outputs overwritten by that same trace are acknowledged.

## Context and memory

Supported context remains **262144**. Final per-layer gates execute complete prefill at 262139/262143/262144 and terminal decode at 262143: `final_v2_max_context_{0,3}.json` and [context_contract.json](../context_contract.json). Long-context output checks sample first/middle/terminal windows while TTNN executes every logical token; short/medium tests compare every row. The populated 128K/B32 test separately validates 1,024 decode steps and final state.

Additional capacity-transition watcher tests allocate matching physical caches and exercise ordered, reordered, subset and mixed pools. They sample first/last/valid-prefix state pages to avoid downloading every unused page; ordinary pool tests retain complete-state checks. These short-position capacity tests are not claims of fully populated B32 maximum-context numerical coverage.

Private linear B1 history uses 12,288 bytes/chip/layer. Tiled pooled history reserves 393,216 bytes for B1–32; pooled B1 adds 380,928 bytes/layer, or 9,142,272 across 24 linear layers. FP32 recurrence backing remains in DRAM. Shared BF8 AR receive storage is 1,114,112 bytes/chip versus 2,097,152 before; this L1 saving is not subtracted from the conservative DRAM budget. Adaptive SDPA adds at most 1,536 table bytes/chip at B32 and enlarges temporary SDPA circular buffers, without changing KV/recurrent allocations. The analytic all-layer B1 budget is 17,579,245,568 bytes/chip. No all-layer allocation or populated B32 maximum-context execution is claimed.

## Profiles, advice and reproduction

[Final profile summary](final_profile_summary.json) reconciles the same-run device span, merged table and host latency. [Final advice](final_advice.md) links every actionable item to a trial or explicit stop-rule handoff. Tables and CSV/provenance files are under `tracy_final_v2/{0,3}` and `tracy_final_v2_target/stack32`. Raw attributes verify the selected dtypes/fidelities. Generic FLOPs percentages above 100 and the report's incomplete modeled-ops DRAM roofline are not whole-path utilization claims.

Source `/home/ttuser/ornith-9b/lane-env.sh`. Hardware commands use `/home/ttuser/ornith-9b/tt-device-stage5.lock`, bounded timeouts and serial ownership. `tests/run_stage5_final_jobs.py` records commands/environment, immutable source snapshots, hashes, timing intervals, exit codes, recovery and watcher byte ranges. Completed labels cannot be overwritten: use fresh labels/output paths when repeating job lists.

- `final_v2_all_jobs.json`: complete selected 82-job queue; `final_v2_resume_jobs.json` resumes after the invalid transition fixture was repaired.
- `final_v2_{gates,kernel,recorded,perf,prefill,profile,context128k_answer}_jobs.json`: exact categorized commands.
- `final_manifest.json`, `final_audit.json`, `final_accuracy.csv`, `final_performance.csv`: current-source acceptance and timing evidence.
- `review_timing{,_additional}_jobs.json`, `review_timing_map.json`, `final_timing_audit.json`: separate diagnostics-off measurements and acceptance.
- `timing_acceptance_host_checks.json`: negative controls for diagnostic settings and pair mismatches.
- `final_sdpa_policy_host.json`: 270,336 host-only scheduling/alignment checks.
- `final_v2_style_validation.json`: formatting, lint and source hashes; unchanged C++ formatting evidence is retained separately.
- `device_jobs.jsonl`, `recovery_jobs.jsonl`, generated evidence indexes and `source_snapshots`: experiment provenance.

After device jobs finish, run `tests.stage5_perf_reports` for collections `final_v2` and `final_v2_target` (the latter with `--stack --measurement .../final_v2_profile_target32.json`), then `tests.prepare_stage5_manifest --collection final_v2`, `tests.audit_stage5_evidence`, and `tests.summarize_stage5_final`, all under module prefix `models.autoports.ornith_ai_ornith_1_5_9b`.

Raw evidence stays local. `archive_stage5_evidence.py` retains hashes for large binary captures and writes deterministic lossless archives for large text/CSV and logs; `archive_manifest.json` maps archives to raw paths. Decompress an archived path before rerunning an audit on a checkout without local raw files. Nothing is pushed.
