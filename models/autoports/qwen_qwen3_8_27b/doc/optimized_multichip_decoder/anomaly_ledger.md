# Optimization anomaly ledger

| Observed anomaly / evidence | Investigation and control | Resolution |
| --- | --- | --- |
| Missing S128 stack reference (`before_stack.log`) | Device path completed; generated a frozen single-chip correctness fixture, then reran TP4 `before_stack_retry`. | Controlled; single-chip timing excluded. |
| Multi-reader matmul queried non-unit mesh (`native_reader{2,3}_l0.log`) | AutoFix traced physical NoC placement to missing mesh coordinate. Descriptor adapter now constructs per-coordinate native programs; full TP4 and offset TP2 cache/placement tests. | Native fix, built/installed; 18 mesh/cache/placement tests and final model validation pass. |
| Early native build not loaded | Build wrote build_Release/ttnn; Python used installed source-tree/build-lib copies. Installed both runtime components and recorded binary hashes. | Controlled; early native_reader artifacts excluded from post-fix claims. |
| Padding-only reader lacked output storage (`installed_reader2_l0.log`) | AutoFix proved N130 tiles, padded readers144, actual output capacity135. Zero-write tail workers still drain compute. | Native fix; model and focused tests pass. |
| Descriptor test helper namespace wrong | Actual factory is in `ttnn._ttnn.operations.matmul`, not top-level ttnn. Fixed helper, repeated12-case native suite. | Test fix. |
| Sharded L1 first attempt (`sharded_l1_control.log`) | Adapted residual adds, RMSNorm and collective buffers to local hidden1280. `sharded_l1_retry` and cumulative families pass without a timed restore to replicated residual. | Adapted and measured slower. |
| AGMM sender-axis assertion (`family_agmm_bf16_k8.log`) | Reduced workers/link to5 on10-core axis or4 on8-core axis for two links; output/down K blocks16/17 divide local K. | Adapted full families pass and lose. |
| Narrow split projection conversion (`family_split_attention_l0.log`) | Full-shard allocation proof refutes overrun. Loader pads N32 weights and model slices before semantics. | Adapted split controls pass and lose; broader validator change excluded. |
| Full attention BFP8 weights differ from frozen BFP4 raw cache (`precision_attention_bf8_lofi_l3.log`) | Output PCC>.9999, changed-input cache-consuming replay and ownership pass. Explicit cache-diagnostic reruns keep all output/bitwise gates. | Higher latency rejects BFP8 weights; final default uses strict checks. |
| BFP4 KV raw cache PCC~.9815 (`kv_bfloat4_b_l3_s128.log`) | Correct fill/update dtypes; diagnostic short/4097 checks preserve output/per-user/replay/ownership gates. Short decode is similar, longer decode differs by~0.003ms. | Retain accepted BFP8 cache parity; diagnostic results excluded from final gates. |
| Prefill2D L1 allocation2,070,528>1,572,864 B | Adapted output block height1 and4, several grids/K blocks at real2048. | Legal candidates pass and lose to minimal matmul. |
| Minimal K16/N16/M8 L1 allocation1,979,392>1,572,864 B | Retried M8 with N8; M4/N16 and M4/N8 controls also measured. | Legal candidates pass and lose. |
| Host timing outliers / MPI shared-memory warnings | Retained raw samples; removed only unreferenced owned shared-memory segments after mapping/fd audit. Old-policy controls reproduce baseline. Queued timing supplements primary single-replay medians. | Controlled; no outlier-based speedup claims. |
| Native-test nanobind shutdown diagnostics | Existing unit-mesh controls show the same diagnostics and exit0. | Controlled module teardown warning. |
| AutoFix linked TensorSpec probe initialized UMD unintentionally | Recorded exact02:19:46.648–02:19:47.811 interval; preceding model had closed02:18:18.703 and next started later. | No timing overlap; subsequent agent work source-only. |
| Full watcher instrumented fabric exceeds26KiB in prior O3 configuration | AutoFix found supported Os and O3+noinline settings; fresh caches and actual compiler flags prove recompilation. | Size adapted with O3+noinline, retaining every watcher check. |
| Os and O3+noinline watcher model passes then process aborts134 | Post-abort L1 and PC evidence identifies packet-tag handoff assertion on every ERISC1. Empty watcher mesh closes. Reset/list/mesh recovery passed. | Fixed owned-NoC tag cleanup; `watcher_full_eth_tags_fix_repeat` passes decoder gates and exits0 after driver close. Failed JSON remains excluded. |

| Wrapper edited while a long child ran (`watcher_full_eth_tags_fix`) | Native driver closes normally, but Bash resumes at a stale file offset and exits2. Freeze wrapper and repeat exact traffic check. | Repeat exits0; first run excluded from acceptance. |

Wrapper `*.exit_status` markers now capture process teardown as well as Python
checks. Watcher and profiler metrics are labeled separately in candidate summaries.

Profiler metadata anomaly: final packed MLP reports~101% FLOPs utilization.
The installed `tt_perf_report/perf_report.py:930–937` hardcodes8 DRAM matmul
workers, ignoring runtime `num_workers_per_dram_bank=2/3`. Native descriptor
tests and raw operation attributes prove16/24 compute workers. Original
advice tables remain intact; `final_matmul_rows.csv` adds runtime attributes,
worker counts and corrected compute utilization. No >100% claim is accepted.
DRAM percentages use logical tensor bytes and omit bank/tile padding; the
separate accounting lower bound includes stored tile and bank padding.
Resolution: controlled tool limitation with source and runtime evidence.


The first independent review found four identity BF16 casts per short-prefill
layer. The projection branch now guards dtype equality. The final
`review_cast_audit.json` compares all four devices for both kinds; the model
regressions and runtime profiles were refreshed after the repair. Resolution:
fixed, with device-operation removal checked independently of host timing.

The short full-attention trace timing discrepancy is investigated in
`AUTODEBUG_profile_gap.md` and `AUTOFIX_profile_gap.md`. Original host traces
localize the extra delay to C++ `finish_nolock` (591/869 us versus a healthy
327 us). The normal control gives 0.739 ms traced prefill and 0.305 ms decode;
the earlier 2.009 ms prefill median does not reproduce. Pass-through host
thread pools remove one event-dispatch handoff. Paired real TP4 controls improve
eager prefill by 6.5–8.6% across both kinds and their stack with effectively
unchanged decode medians. Diagnostic maximum decode synchronization time falls
from 2.211 to 0.311 ms, and the corresponding new profile has a 0.071 ms
host-minus-device gap. Thread counters show pool/scheduling sensitivity, but
snapshots extend beyond the timed intervals and do not identify the exact
cause of each historical outlier. No kernel-causation or universal latency
claim is made. Resolution: controlled transient completion-path delay; the
selected optimized launcher defaults to the measured pass-through setting.
Untouched final profiles and medians remain the acceptance evidence.

The adapted monolithic GDN and serial scan/preparation trials all pass PCC.
Monolithic decode (0.551348 ms) and serial preparation at S2049 (7.432725 ms
prefill) lose; serial scan overlaps the selected phased path. Resolution:
measured rejection after adapting the monolithic rank and normalization.
