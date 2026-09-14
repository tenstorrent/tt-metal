# Stage 7 anomaly ledger

| Observation | Evidence and affected path | Investigation / control | Resolution |
| --- | --- | --- | --- |
| Initial heartbeat and sysmem-base mismatch before inference | `AUTOFIX_startup.md`, ownership/recovery logs; initialization only | Bounded reset/list/mesh attempts; operator identified foreign telemetry owners and restored reservation | Recovered. Resumed exact TP4 mesh and all subsequent full-model gates pass. No UMD check bypass or foreign PID termination by this stage. |
| One head experiment stalled during unchanged weight upload | `triage_head_startup/`, `AUTOTRIAGE_head_startup.md`; before candidate execution | Live triage found a transfer wait, not a reader-geometry result; owned process terminated after capture, reset/list/mesh and identical retry succeeded | Controlled infrastructure recurrence; no claimed head result for the stalled run. Root cause remains unproven; raw evidence retained. |
| Reader1 local-logit PCC about.79 with inherited padding | `AUTODEBUG_head_geometry.md`, failed and compatible head logs | Source identifies20-tile physical bank stride versus19 expected by one reader. Exact reader-dependent padding restores PCC>=.99999988. Reader3 also required adapted tail widths | Fixed candidate contract; correctly padded reader1/3 measured slower and rejected. Selected reader2 retains valid padding. |
| Larger head K block exceeds L1 | `head_16384_block10_reader2.log`:101760-byte overlap | Same16384-column/two-reader geometry with block5 passes PCC and improves latency | Selected block5. No native validation bypass. |
| Lower head multiply fidelity misses local PCC | `head_16384_block5_lofi_report.json`: rank3 PCC.99892634; other ranks>.999; all local greedy equal | Same real head weights, recorded real activation, bank geometry, BF8/BF16 and FP32 accumulation; only multiply fidelity changes. Moderate numerical loss, not catastrophic layout divergence | Rejected at existing.999 local gate.942.765us versus selected HiFi2 949.427us is not an accepted performance result. No decoder precision change or broad datatype frontier. |
| Watcher Ethernet instrumentation exceeds code buffer | `watcher_contract_b3.log`:29104>26624B before model load | Existing NOINLINE1/fabricO3 instrumentation settings; no ETH disablement. Both the fit contract and final prefill watcher/tracker runs pass | Resolved instrumentation configuration; original failure preserved. |
| Generic live-trace allocation warning | Untracked full-run logs; allocator warns on potential risk | Stable buffer/capture-order design, reduced allocation tracking, final watcher tracking, full-layer retained-output and exact repeated-request tests | Tested lifetimes pass. Warning is not hidden and is not used as evidence of safety by itself. |
| Generic packet-size and L1-semaphore advice | `full_path_checklist.md`, native source refs |4352/8192 payload A/B on reduced and full model;8192 improves full decode about2%. L1 semaphore warning indicates intentional allocation region, not observed fragmentation |8192 selected. No L1 capacity error on selected full path; no L1_SMALL reservation introduced solely to silence a warning. |
| Perf tool reports eight DRAM matmul workers / excessive FLOPs utilization | Raw advice tables; inherited native program-factory evidence | Native two/three-reader programs use16/24 workers over8 banks; stored-tile bytes include physical padding | Original tables retained; invalid FLOPs denominator is excluded from utilization claims. |
| Gap-inclusive layer expansion exceeds measured full decode | `profile_analysis_terminal.json`, `interim_perf_summary.json` | Separate representative kernels, profiled gaps, isolated replay/sync latencies and full unprofiled window | Explicitly labeled extrapolation. No negative host overhead or fabricated full64-layer device interval. |
| Initial final profile used cache160/history1 | `profile_final.*`; original reducedG2 profiling fixture | Matched profiler-only allocation changed to cache256/history127 for S128/G128 production fixture | Final matching capture is `profile_final_buffers`; earlier capture remains a labeled control. |
| Accidentally enabled tracker in performance launch | `prefill_trace_reduced.log`, KeyboardInterrupt during head upload | Instrumented smoke already passed; owned process interrupted and closed cleanly, then uninstrumented perf rerun | No performance result from interrupted run. Tracker is retained for correctness, excluded from headline timing. |
| Short qualitative runs truncate reasoning/code/story | `tt_qualitative.json`, HF256 controls and1024 extensions | Same template/token IDs; five prompts reach complete coherent answers with adequate budget. Generated Fibonacci passes functional cases. Both TT/HF story controls truncate at1024; TT uses more planning tokens | Matched-budget truncation is controlled; longer TT story probe completes coherently at1700 tokens and exactly matches the earlier1024-token prefix (`qualitative_metrics.json`). No claim that HF and TT use identical generation budgets in that extension. |

Profiler export also reports a missing optional wasm/UI trace-copy target and
pandas mixed-type inference. The required per-device CSV/gzip exports and all
phase reports exist; independent review verifies row alignment and reproduces
the accounting from those sources. These are controlled export warnings, not
a missing device timing interval.

The motherboard discovery warning only substitutes a tray identifier from its
PCI bus ID for this host motherboard. Physical four-chip topology discovery and
Ring initialization succeed; it is not a model, sampling or trace fallback.

No failure above is silently converted to eager decode or host sampling. The
selected warmed path is validated by explicit trace identities, nonblocking
replay order, device-only guards, unchanged counters, accuracy and output checks.
