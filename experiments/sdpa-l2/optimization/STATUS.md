# FP32 SDPA optimization status — September 9, 2026

Historical snapshot: the current work has advanced to the HiFi2 rounded-input
candidate in [../hifi2-rounding/STATUS.md](../hifi2-rounding/STATUS.md).
The source tree no longer matches final-hybrid.patch; that patch and the report
below preserve the previous experiment. The new candidate reaches about 0.49%
on three full-256K normal-input tail-row checks, with important limitations.

SSH is working. Reservation 214149 is on yyzo-bh-26 (Blackhole P100A).
Container: yyzo-bh-26-special-cglagovich-for-reservation-214149.
Remote repo: /localdev/cglagovich/tt-metal-blackhole-20260908.
The old reservation 212869 on yyzo-bh-08 expired.

See [REPORT.md](REPORT.md) for results, scope and reproduction commands.

The current three-file source patch is the fused hybrid candidate saved in
final-hybrid.patch. It improves normal 256K L2 from 8.83% to 1.94% at roughly
3% short-query overhead for D=128/K chunk=512. Full causal H=4 tail-row L2
improves from 8.72% to 1.95% with roughly unchanged runtime.
It does not reach 0.5%. Smaller chunks have substantial overhead (15–23%);
D=64/H=4 has about 8% overhead. This is not a production-ready universal fix.

The zero-overhead delta-only candidate was rejected for outlier regressions.
Direct-state and early hybrid prototypes had buffer/broadcast configuration
problems; use final-* and hybrid-fused results for final validation, not those
failed prototypes. Invalid restored-baseline results have an explicit
invalid-stale-host- prefix and must be excluded.

Final host build and on-device checks have been run. The repro includes FP64
reference self-checks, finite-output checks, relative-L2 thresholds, timing
through warmed trace replays, and trace/ordinary-output equality checks.
Black 23.10.1 and git-clang-format 19.1.4 were used.

Next optimization opportunity: a faster, more accurate logit exponential,
plus targeted HiFi2/TF32 error reduction. Accurate exp plus HiFi4 reached
0.244% in an earlier ablation but cost 3.46x baseline short-query runtime.
Further adversarial tests of the numerator's cancellation-based update and
broader operator-feature regression coverage remain necessary.
