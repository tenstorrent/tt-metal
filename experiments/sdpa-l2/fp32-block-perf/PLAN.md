# HiFi4 / approximate-exp / full-FP32-subtraction block tuning

Completed 2026-09-11. See [REPORT.md](REPORT.md) for final timings, profiles,
accuracy checks, limitations, and verified restoration of the retained build.

Keep mode 4 arithmetic unchanged: original BF16 Q/K/V, HiFi4 QK/PV and matched
denominator, FP32 online state and score subtraction, unbiased grid/cubic exp.
Noncausal B=1 H=10 D=128; N=32768, 65536, 131072, 262144. No fallback.

Temporarily enable additional block geometries with TT_SDPA_BLOCK_SWEEP=1.
Retain double-buffered K/V for Q<256 and K<=512, and the established
single-buffered layout for Q=256 or K>=1024. No reader/writer algorithm changes.
Sweep Q in 32/64/128/256 and K in 128/256/512/1024/2048, subject to L1 capacity.
Report actual tested candidates and unsupported/error results, not global optimality.

Screen full operators with 10 warmups and 5 blocking trace replays, then repeat
the best configurations with 40 warmups and 10 replays. Useful attention FLOP/s
is 4*H*N*N*D divided by time, not a raw activity-counter measurement.
References use the same BF16 inputs and FP64, initially 128 spread query rows
per head and 512 for final measurements. Seed 1236, matching prior performance
controls. Accuracy and trace-equality checks accompany timings; additional
distribution checks compare the tuned configuration to 128/1024.

Source arithmetic is a diagnostic patch, not a production API. Save patch and
hash provenance, build before device runs, and restore retained sources/build
after collecting measurements. No previous acceptance thresholds are relaxed.
