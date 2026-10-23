# The median check (CONTRACT section 3) under the contract's precision

The contract asks, next to `pcc_final >= 0.99`, that "every row's median must be within 1e-2 x its scale". The
round-0 device path meets every PCC floor with a wide margin, but not this bound:

| case | rows | pcc_normalized | pcc_final | worst median \|err\| / scale | evidence |
|---|---|---|---|---|---|
| model16, accuracy | 16 | 0.99992 | 0.99998 | 6.1e-2 | PCC suite, job 500 |
| multivar4, accuracy | 4 targets | 0.99997 | 0.999996 | 2.4e-2 | job 500 |
| single series (2048), accuracy | 1 | 0.99831 | 0.99835 | 4.3e-2 | job 500 |
| paper, realistic mix, accuracy | 1024 | 0.99991 | 0.99997 | 1.65e-1 | bench job 499 |
| paper, realistic mix, performance | 1024 | 0.99952 | 0.99984 | 4.4e-1 | bench job 499 |
| long_context (H = 928), realistic mix, accuracy | 32 | 0.99991 | 0.99999 | 7.3e-2 | bench job 499 |
| multivariate, realistic mix, accuracy | 128 targets | 0.99983 | 0.99998 | 1.7e-1 | bench job 499 |
| paper, first smooth input set, accuracy | 1024 | 0.99996 | 0.99999 | 1.8e-1 (p50 2.2e-2, p99 6.0e-2) | bench job 440 |

**The bound is below what bf16 weights allow.** A CPU simulation of the fp32 dataflow (validated against
upstream to 1e-7) with only the matmul weights rounded to bf16, all compute in fp32, already gives a worst
median error of **1.7e-2** on model16 (mean over rows 6e-3); adding bf16 rounding of the residual stream and of
every linear input gives 1.9e-2 (`tests/golden/sim_precision.py`). So no device path
with the contract's `accuracy` precision (bf16 weights) can pass 1e-2 on every row of model16, let alone the
worst of 1024 paper rows.

**What drives the device error.** Per-block comparison against the fp32 dataflow on the worst paper rows (618,
62, 382, 762; job 428) showed that the sLSTM input projection `W_in`, whose outputs feed the exponential gates,
was the largest single source: moving that one matmul from HiFi2 to HiFi4 cut row 618's median error from 0.200
to 0.050 of the scale (identical to running every matmul at HiFi4). That change is in round 0 (`gate_config`).
The remaining error is matmul fidelity (HiFi2 dense projections: bf16-output matmul error is 3.4e-3 mean
relative at HiFi2 vs 1.2e-3 at HiFi4, independent of M; probe in job 456) plus bf16 rounding of activations.
The worst row of the first (smooth) paper set is a series whose forecast is unusually sensitive (its CPU
bf16-weight simulation gives 1.6e-2, the device 1.8e-1). The collapsed output (every quantile replaced by the
quantile mean) scores a median error of 1.0 on the realistic paper set but only 0.14 on long_context, so the
median check is an accuracy bound, not a collapse detector.

**Request (G1):** replace "every row's median within 1e-2 x scale" with a bound the contract's precision can
meet, for example the worst row within 0.25 x scale (`accuracy`) and 0.5 x scale (`performance`) on the bench
shapes, and 0.1 x scale on the PCC-suite shapes, keeping `pcc_final >= 0.99` as the primary floor; or say which
precision exception (e.g. HiFi4 on every dense matmul in `accuracy`) should be bought to get closer.
