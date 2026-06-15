---
name: kimi-61l-op2op-profiling
description: "DONE (9 layers) — per-layer/per-op op2op profiling w/+w/o H2D service; deep layers gap-free w/o service, per-layer tax w/ service"
metadata: 
  node_type: memory
  type: project
  originSessionId: 73e2e6db-60ce-49ca-9a6e-bf0b6caab1ac
---

DONE 2026-06-15 (user reduced scope to 9 layers). Profiled NUM_LAYERS=9 per-op + per-layer op2op on the
slowest device, warm kv=51200, WITH and WITHOUT the H2D service. RESULT:
- **No-service**: op2op gap is ENTIRELY a one-time pipeline-fill in **layer 0** (8.5ms); layers 1-8 incl.
  deep ones ~0 (perfectly pipelined). Standalone chunk is COMPUTE-BOUND. (Settles the earlier 3L-only doubt
  that deep layers might have gaps — they don't.)
- **With-service**: op2op recurs at EVERY layer (L0 14ms, L1-8 each 10-22ms; 146ms total / 302ms kernel =
  33% of wall). Per-op: gaps land on the small ops BEFORE RingJointSDPA each layer, collapse to ~0.5us after
  SDPA, then re-starve next layer. Mechanism: dispatcher run-ahead kills gaps (no-svc fills once at L0 SDPA
  & stays ahead); the H2D service's per-op dispatch coordination disrupts run-ahead every layer -> pre-SDPA
  small ops re-starve -> ~16ms/layer x 61 ~= the +1.4s/chunk service tax. => the fix lever is isolating the
  H2D service's dispatch domain (per [[kimi-perf-overnight]]), NOT per-op micro-opts on the standalone path.

DELIVERABLES (kimi_perf_overnight/): ops_slowest_device_9L_{noservice,service}.log (per-op detail, 3L-log
format), ops_9L_shm_{noservice,service}.perlayer.log (per-layer summary, grouped-in-3).

HOW IT WAS DONE (reusable profiling recipe for many layers, solved 2 hard blockers):
- Profiler DRAM marker buffer overflow (drops deep-layer markers at ~6 layer-execs): set env
  **TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=3000** (no rebuild; JIT recompiles kernels with bigger buffer).
- Raw profile_log_device.csv ~1.2GB/layer blows the 99%-full / disk: redirect env
  **TT_METAL_PROFILER_DIR=/dev/shm/ttprof** (/dev/shm = 284G tmpfs; /data = 9.7T NFS as backup).
- Tools: kimi_perf_overnight/profile_NL.sh (params NL, noservice|service, ITERS/PSC/PROFDIR/READ_EVERY env),
  parse_NL.py (per-layer), parse_NL_detail.py (per-op). DON'T use TT_METAL_PROFILER_MID_RUN_DUMP (still
  overflows + disk blowup) or PREFILL_PROFILE_READ_EVERY drains (cumulative CSV blowup).
- Postprocess of ~21GB profiler data is CPU/RAM heavy (~6min, ~40GB RSS) but fine on this box.

See [[kimi-perf-overnight]] (full investigation), [[firmware-host-mailbox-mismatch]] (infra gotcha).
