# Frontier compute sprint v2

Started 2026-09-18. User requested a deeper scheduling/overlap pass after v1.
No time or gain guarantee. Baselines are the frozen v1 winners, not the original
canonical kernels. All v1 evidence and sources remain untouched.

## Ownership

- `compensated/`: low-precision agent, shared B/E/G state pipeline candidates.
- `fp32/`: FP32 agent, C/D pipeline and state scheduling.
- `review/`: BF16 agent, independent phase profiling/ordering review and B
  transfer validation. A only if a specific exposed bottleneck warrants it.
- Root: hardware safety, independent evidence checks, provenance and report.

## Frozen contract

Noncausal Q256/K512/D128, original CB formats/capacities and KV buffer depths.
No input dataflow changes. Preserve fidelity, exp/subtraction/reciprocal,
compensation, rounding points and exact preprocessing per labeled variant.
Scheduling candidates require output-bit equality against the v1 winner.
Canonical recipe/prepare remain the numerical authorities.

## Experiment discipline

1. Source ordering argument and concrete hypothesis before kernel changes.
2. Phase instrumentation is diagnostic only; fine zones can perturb overlap.
   Uninstrumented alternating baseline/candidate trace timing decides wins.
3. Useful FLOPs count original QK/PV only. Report per-core compute throughput;
   no inferred chip or model speedup. Different fidelities have different roofs.
4. Resident repeated KV removes recurring external traffic but favors
   unchanged-max branches. Distinct KV, changing maxima and multiple Q jobs
   are separate correctness tests; retained changes need short/odd K loops,
   stress distributions and trace replay where applicable.
5. Frozen private source hashes, no edits/uploads during own compilation/run,
   and one queued device job per agent. Rejected/hanging candidates stay in
   the evidence; never describe numerical mismatches as acceptable speedups.
6. Do not enable these private shape-specialized paths in general production
   dispatch. Any broader use needs explicit guards and new qualification.

## Hardware protocol

Reuse IRD 223862 on bh-lb-08 if verified active. All tests exclusively lock
`/tmp/tt-device.lock` through v1 `run_locked.sh`, with its persistent dirty
guard. Do not nest safe_pytest. Only root coordinates failure clearance or
reserved-device reset after inspecting logs and remaining processes.
Host and container source root:
`/localdev/cglagovich/flux2-frontier-20260915/tt-metal`.
