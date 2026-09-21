# Numerics-relaxed compute sprint v3

Started September 18, 2026. User approved exploring grouped compensation for
B/E/G and PV/state fusion for C/D with a 5% relative L2 allowance. This is not
permission to remove synchronization, tolerate races, or silently change inputs.

## Ownership and scope

- `compensated/`: shared grouped/hierarchical compensation, E/G implementation.
- `fp32/`: PV/state fusion and changed reduction grouping for C/D.
- `review/`: independent state/ordering review, B transfer and qualification.
- Root: common acceptance contract, hardware coordination, audits and report.

All v1/v2 code and preexisting tracked modifications are frozen. Only new
private v3 files may change. Noncausal Q256/K512/D128, reader/writer, input
buffering, CB formats/capacities, fidelity, preprocessing and exp remain fixed
initially. Any additional relaxation must be explicitly identified, not bundled
into an unexplained speedup. HiFi3 is a separate possible experiment, not part
of the initial fusion/compensation comparison.

## Baseline and numerical acceptance

Original BF16 inputs and the original-input FP64 reference are shared between
baseline and candidate. D's performance baseline includes v2's scoped O2
function attributes. C uses v1. B/E/G compare with both v1 general and v2 early
identity implementations; gains over only the slower control are insufficient.
The v2 early guard has the same qualified numerical recipe as v1.

For every nonzero-reference case, using L2 measured in percent:

`candidate_l2_pct <= 1.05 * baseline_l2_pct + 0.0001`

The additive floor is 0.0001 percentage points, not 0.01% or 5 percentage
points. Improvements have no lower bound. Undefined relative error on an
exactly zero reference must be reported explicitly; use a separate absolute
check `candidate_max_abs <= max(1e-6, 1.05 * baseline_max_abs)` there.
All outputs/references must be finite. Baseline/candidate bit equality is no
longer required; eager/trace and repeated-run bit equality remain required.

Also record PCC, row p95/p99/worst L2, maximum absolute error and normalized
candidate-versus-baseline distance. Material row or coherent-input regressions
must be highlighted and investigated rather than hidden by a global L2 pass.
Do not pool cases into an average acceptance score. Constant/common V, uniform
attention, changing maxima and repeated coherent KV are required diagnostics,
not optional exclusions. Passing finite tests is not a universal error bound.

## Experiment order

1. Source and mathematical argument: protected state, local accumulation,
   maximum changes, boundaries, publication fences and ownership.
2. Small host/state probes to reject invalid schemes; device first/odd-K and
   multi-Q checks before sustained timing. Numerical rejection is recorded as
   data after clean device closure, not confused with a hardware fault.
3. Matched normal 32K/256K, scaled QK, outliers, common Q/K/V, constant/zero V,
   uniform attention, max transitions and multiple Q jobs. Preserve input and
   preprocessing hashes. Hold-out seeds for promising candidates.
4. Fresh alternating resident controls measure useful QK+PV compute throughput;
   input preprocessing is excluded. Distinct-KV timing is separate and includes
   recurring DM. Do not credit extra products as useful FLOPs or infer model
   speedups. No promotion based on repeated resident inputs alone.

## Hardware

IRD 223862, bh-lb-08, container
`bh-lb-08-special-cglagovich-for-reservation-223862`, logical device 0.
Remote repository: `/localdev/cglagovich/flux2-frontier-20260915/tt-metal`.
All device jobs use frozen v1 `run_locked.sh`, `/tmp/tt-device.lock`, and its
persistent dirty marker. One queued job per agent; complete uploads before
compilation; no edits to a run's source while it executes. Only root inspects
and clears a dirty marker or coordinates a reset. No automatic recovery.
Host-only analysis that never opens a device need not acquire the device lock.
