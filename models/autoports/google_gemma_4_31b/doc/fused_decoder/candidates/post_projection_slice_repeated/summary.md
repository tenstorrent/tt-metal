# Repeated post-projection slice A/B

## Verdict

Post-projection slicing is now selected. The first synchronized-host run showed
overlap and could not support the old 0.810 us ordering. A complete signposted
device-profiler run then retained 12 complete 40-op samples per arm. Its six
ABBA-cycle mean differences all favored post-projection slicing by 2.015--3.856
us. PCC between variants was 1.0 and every replay was bitwise deterministic.

## Repeated measurements

All differences below are candidate minus the former selected pre-projection
slice, so negative favors the post-projection candidate.

| timing | variant | n | median us | min us | max us | spread us |
|---|---|---:|---:|---:|---:|---:|
| synchronized host, ordinary | pre-projection | 12 | 2627.957 | 2603.097 | 2639.857 | 36.760 |
| synchronized host, ordinary | post-projection | 12 | 2626.502 | 2604.117 | 2630.897 | 26.780 |
| synchronized host, complete profile | pre-projection | 12 | 2664.742 | 2646.457 | 2679.547 | 33.090 |
| synchronized host, complete profile | post-projection | 12 | 2660.957 | 2642.657 | 2663.556 | 20.899 |
| signposted device-kernel sum | pre-projection | 12 | 2558.574 | 2555.949 | 2561.319 | 5.370 |
| signposted device-kernel sum | post-projection | 12 | 2556.062 | 2552.758 | 2558.353 | 5.595 |

- Ordinary host median difference: -1.455 us; ABBA-cycle differences crossed
  zero: `[0.751, -2.841, -1.820, -8.600, -5.775, 4.110]` us.
- Complete-profile host median difference: -3.785 us; ABBA-cycle differences:
  `[-9.560, -5.155, -3.815, -0.651, -6.490, -2.985]` us.
- Device median difference: -2.512 us. Device ABBA-cycle differences:
  `[-2.475, -3.856, -2.015, -2.451, -2.767, -2.742]` us.

The initial profiler run under `profile/` filled device-profiler buffers after
eight complete intervals. Its incomplete tail is not evidence. The rerun under
`profile_complete/` flushed warmup records before measurement and has 12/12
complete 40-op intervals for both variants with no buffer-full warning.

## Correctness and final verification

- Candidate versus former selection: PCC 1.0; both variants bitwise
  deterministic over all 12 measured replays.
- Final HF sliding-decode trace PCC: 0.9996291558392226; the existing test also
  proved eight bitwise-identical trace replays.
- Final signposted trace: 2557.519 us, 40 device ops, zero host ops in the
  signposted interval.

## Code disposition

Temporary instrumentation added an instance flag around slice placement and an
env-gated two-trace ABBA test. Both were removed after selection. The only final
runtime change moves the existing padded-batch slice from before `o_proj` to
after it:

```diff
-if output.shape[2] != batch_size:
-    padded = output
-    output = padded[:, :, :batch_size, :]
-    padded.deallocate(True)
 projected = ttnn.linear(output, weights.o_proj)
 output.deallocate(True)
+if projected.shape[2] != batch_size:
+    padded = projected
+    projected = padded[:, :, :batch_size, :]
+    padded.deallocate(True)
```

## Remaining uncertainty

The literal gain is small (about 0.1%) and the unpaired host distributions
overlap. Selection rests on same-process device timing and the paired ABBA
result, not on the original single-sample 0.810 us ordering. Firmware 19.9.0 is
newer than the fully tested 19.5.0. The smallest next graph improvement if a
larger margin is required is to eliminate the remaining standalone slice by
having the sharded-to-interleaved/head-concat boundary materialize only the
logical batch rows; no current local TTNN contract was found that safely fuses
that crop, so this experiment did not speculate beyond the proven relocation.
