# Repository Guidance

## Active Qwen3.6 performance work

- `PERF_OPTIMIZATION_PLAN.md` is the authoritative active roadmap for the
  Qwen3.6-27B performance effort.
- The next work is Milestones 0-2: state-aware `T=1..3` GDN reference tests,
  device-side GDN state primitives, and a fused recurrent decode/verify
  operation. Continue from the first incomplete acceptance gate.
- Do not restart completed or rejected SDPA, dtype, gate-fusion, or scheduling
  experiments listed in the plan. `../RESULTS.md` is the evidence ledger.
- Preserve FP32 recurrent state unless long-decode model-level evidence proves
  an alternative safe.
- Hardware is shared. Obtain fresh explicit authorization for every silicon
  run, then follow `../hw-preflight.sh`; device-free and simulator work may
  proceed without hardware authorization.
