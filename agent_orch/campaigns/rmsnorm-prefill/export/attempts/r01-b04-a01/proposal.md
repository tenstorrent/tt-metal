# r01-b04-a01: precompute x*gamma during the all-gather wait so POST is a single x*gamma*(1/rms) pass

## Motivation
Baseline device timeline (baseline_1 `profile_log_device.csv`, chip 1, zone markers; µs from kernel start),
20 workers × 1 tile-row each + 1 forwarder:

| shape | R_INPUT end | PRE→stick pushed | fabric AG (F_FABRIC) | W_AGWAIT end | W_DRAIN (POST + output) | kernel end |
|---|---|---|---|---|---|---|
| h3584 (28 tiles) | ~4.3–5.4 | ~5.6–6.6 | ~2.5–3.0 | ~8.9–10.2 | ~7.6 | ~17.0 |
| h7168 (56 tiles) | ~8.4–9.2 | ~9.7–10.4 | ~2.4–2.7 | ~12.8–13.2 | ~13.0 | ~26.2 |

POST+drain is the largest phase (45–50% of the kernel), and compute idles for the
whole ~2.5–3 µs all-gather window. POST for the default config (has_weight, no bias, no RoPE)
is two whole-row FPU passes, both after the AG:
  1. `intermediate = x * bcast_col(1/rms)` (fp32 pack)
  2. `out = intermediate * bcast_row(gamma)`
Pass 2's x*gamma part does not depend on the gathered stats.

## Mechanism
Compute kernel only (`device/kernels/compute/dit_rmsnorm_fused_compute.cpp`):
- New derived constexpr `prescale_weight` = broadcast weight && !bias && !rope && !per_head_norm
  && !streaming_low_l1 && !block_major_post && packed AG (the TP>1 default path the campaign runs).
- After PRE pushes the transposed stat tile to the writer (so the AG starts at the same time as
  before), and BEFORE waiting on the gathered stats, compute
  `intermediate_cb = x * bcast_row(gamma)` (fp32) for the whole row. This runs under the AG wait.
- POST then runs one pass: `out = intermediate * bcast_col(1/rms)` → output_cb, replacing
  sub-phases 1 and 2.
No host/CB changes: intermediate_cb already holds a padded whole row (fp32), weight is already
resident before PRE finishes (reader reads it right after the input row).

## Why this is not a repeat
Round 1, no prior nodes. Structural reordering of the POST dependency chain, not a
parameter tweak.

## Expected effect and risk
Saves ~min(AG window, one POST pass) ≈ 2–3 µs per shape: ~12–15% on h3584/h4096, ~8–10% on
h6144/h7168, unless the output writer (W_DRAIN) becomes the new bottleneck.
Accuracy: x*gamma (bf16×bf16) in fp32, then × fp32 1/rms, one bf16 rounding at output — the same
number of roundings as before; pcc/max_abs should stay ~baseline. Risks: CB-order deadlock if the
weight is not pushed before compute needs it (it is pushed right after the input row; the
writer/reader don't depend on compute in between), a missed reconfig (would show as a PCC fail).
