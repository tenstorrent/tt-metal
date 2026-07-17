# Evidence provenance

All artifacts were produced on 2026-07-17 from repository checkpoint `64608f66cd8` plus the stage changes, on four local Blackhole p300c devices with firmware bundle 19.8.0. Exact commands and environment values are in `../work_log.md`.

| Artifact | Provenance and claim |
| --- | --- |
| `autofix_scalar_pcc.*` | corrected device-resident RoPE/current-position path; nonaligned prefill, decode, K/V, trace |
| `autofix_paged_control.*` | reversed page table plus identical paged/contiguous SDPA geometry; direct output and logical-cache PCC 1.0 |
| `autofix_mutable_trace.*` | one trace, nonuniform per-user positions updated over three replays, contiguous and paged caches, deterministic stress |
| `real_optimized_baseline_export.*` / `real_optimized_baseline_compare.*` | real layer-20 checkpoint; fresh optimized TP1 producer and final TP4 consumer processes |
| `autofix_capacity_32k.*` | all 40 decode-weight layers, 40 K/V pairs, vocabulary endpoints, shared RoPE, and 4 GiB/rank reserve resident during position-32767 decode |
| `perf_optimized.*` / `perf_multichip.*` | final exact-code warmed wall timings and 50-replay determinism |
| `geometry_sweep.csv` / `geometry_*.log` | bounded TP-local real-weight attention/MLP sweep, output PCC, three-run finalist medians |
| `collective_candidates.*` | replicated versus hidden-sharded RMSNorm/QKV consumer-chain comparison |
| `runtime_fallback_audit.xml` | static ownership/no-host-fallback gate |
| `runtime_contract_audit.xml` | owned-runtime audit plus explicit rejection of four-device 2x2 and 4x1 meshes |
| `final_full_gate.*` | exact final-code consolidated result: static ownership, nonaligned PCC/trace, paged parity, mutable positions, and 40-layer 32K capacity |
| `watcher_final_worker.*` | exact final worker-watcher run, ETH inspection disabled, fabric active, clean device log, process exit 0 |
| `watcher_default_failure*` | default active-Ethernet watcher kernel-size failure control |
| `watcher_retry*` | noinline full-ETH model pass followed by firmware teardown heartbeat failure and exit 134 |
| `final_device_health.log` | post-gate `tt-smi -ls --local`; all four p300c devices visible and reset-capable |
| `stage_review.md` | complete review/AutoFix/rereview sequence; final independent verdict `clean-pass` |
| `tracy_final_capture.log` | exact final-code profiler command transcript |
| `../tracy/final/ops.csv.gz` | complete processed multi-device operation stream, gzip integrity checked |
| `../tracy/final/*perf_report*` | human-readable and CSV merged/per-device TT performance tables |
| `../tracy/final/*summary*` | operation summary CSV/PNG for decode and prefill |

The direct baseline `.pt`, geometry reference `.pt`, expanded Tracy logs, and decompressed profiler CSV were temporary interchange/staging files. They were deleted after their consumers and archive integrity checks completed; the retained commands, compact op stream, reports, and tests regenerate them.
