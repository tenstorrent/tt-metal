# C/D arithmetic-reassociation probes

Scope: noncausal Q256/K512/D128, original one-slot K/V input buffering,
canonical C/D input formats, precision, exp, and CB geometry. The D control
is the v2 scoped-O2 winner; C is the v1 winner. Only private v3 sources change.

## Hypotheses

1. `full_pv`: row 0 currently accumulates four separately packed partial
   PV reductions through the FP32 L1 adder. Reduce all sixteen K tiles in
   DST and pack once. All other rows and recurrence remain unchanged.
   This removes three pack/ownership boundaries but stops overlapping this
   PV with the last Q row's exp. A speedup is not assumed.
2. Identity-only numerator fusion: load the prior FP32 numerator directly
   from L1 to DST, then accumulate the new PV products into it, skipping
   its separate state-update pack. First scope to unchanged-max rows 1–7;
   changed-max and row 0 retain the original recurrence. Direct unpack
   bypasses SrcA/B operand truncation. The hardware FPU still changes
   the summation order and rounding compared with completed PV plus L1
   addition; small increments into a large coherent numerator may be lost.
3. If identity fusion is worthwhile, investigate nonidentity rescaling
   before FPU accumulation with explicit SFPU/FPU ownership ordering.
   This must not move old state through a lower-precision Src operand.

The ISA documentation describes FP32 DST reads/writes as full width but
explicitly states that its floating-point pseudocode is approximate and
not an IEEE rounding model. Measurements against original-input FP64
reference, especially long coherent/common/constant-V cases, are decisive.

## Qualification

Use the parent v3 per-case +5% relative L2 plus 0.0001 percentage-point
floor; zero-reference outputs use the absolute gate. Log PCC, row
p95/p99/worst, absolute errors, output distance, source/input hashes, and
raw-bit eager/trace determinism. Numerical budget rejection is recorded
without converting a clean finite device run into a hardware fault.

Small first/odd-K and multi-Q precede sustained paired resident timing and
separate distinct-Q/K/V timing. No claims about generic performance from
the repeated resident, mostly identity-max workload.
