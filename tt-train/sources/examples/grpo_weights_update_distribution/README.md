# GRPO bf16 weights-update distribution

Diagnostic instrumentation for the GRPO trainer that answers a specific
question: **when the model weights are stored in bf16, how many of them
actually change on a given optimizer step, and what does the distribution of
those changes look like in fp32?**

`MorehAdamW` in tt-train stores parameters as bf16 and calls
`ttnn::moreh_adamw(..., param_out=<bf16 param>, ...)`, writing the updated
weights back into the same bf16 tensor. Any per-element update whose magnitude
is below the bf16 ULP at that weight's current value silently rounds to zero.
The exact fraction that survives that rounding — and the distribution of the
survivors — is the direct signal this instrumentation captures.

## How it works

The instrumentation lives directly in
`tt-train/sources/ttml/ttml/trainers/grpo_trainer.py`
(class `_WeightsUpdateLogger`, wired into `GRPOTrainer.train`). Around every
optimizer step that falls within the logging budget it:

1. Snapshots every trainable parameter as fp32 numpy
   (`param.to_numpy(FLOAT32, dp_composer)`, upcasting the bf16-stored value).
2. Runs the existing `optimizer.step()`.
3. Reads each param again as fp32 numpy and computes
   `diff_fp32 = new - old`.
4. Also computes a bf16-quantized view of the same diff:
   `diff_bf16 = diff_fp32.astype(ml_dtypes.bfloat16).astype(np.float32)`
   (round-to-nearest-even bf16 quantization, then upcast for host stats).
5. Emits per-parameter stats and a global histogram for both dtypes.

Because both views are logged with the same set of stats, every column carries
either a `fp32_` or a `bf16_` prefix so the two are trivially comparable.

## How to run

The instrumentation is behind a `--num_steps_to_log` flag on the existing
BoolQ GRPO example. All other flags of `boolq_training_example.py` work as
before.

```bash
# Log the first optimizer step only (fastest sanity check):
python3 tt-train/sources/examples/grpo/boolq_training_example.py --num_steps_to_log 1

# Log the first 5 steps:
python3 tt-train/sources/examples/grpo/boolq_training_example.py --num_steps_to_log 5

# Log every optimizer step for the whole run:
python3 tt-train/sources/examples/grpo/boolq_training_example.py --num_steps_to_log -1

# Disabled (the default, keeps behaviour identical to plain training):
python3 tt-train/sources/examples/grpo/boolq_training_example.py
```

Any other GRPO config (e.g. the DDP one at
`tt-train/configs/training_configs/grpo_boolq_llama_1b_ddp.yaml` or the FSDP
qwen3 one) works transparently — the logger picks up the same `dp_composer`
the trainer uses for checkpoints, so sharded params are gathered before the
fp32 diff is computed.

## Where the output lands

Each run writes into its own timestamped GRPO output directory, in a
`weights_update_distribution/` subfolder:

```
${TT_METAL_RUNTIME_ROOT}/generated/tt-train/grpo_run/<UTC-timestamp>/
  grpo_metrics.csv                              # existing GRPO trainer output
  weights_update_distribution/
    weight_update_distribution.csv              # per-(step, param) stats
    weight_update_distribution_histograms.csv   # per-(step, dtype) log10(|diff|) histogram
    weight_update_distribution.log              # human-readable per-step summary
```

The per-step summary is also mirrored to stdout via `logging.info(...)`, so
you see it live during the run.

## Per-param CSV columns

`weight_update_distribution.csv`, one row per `(step, param)`:

| column | meaning |
|--------|---------|
| `step` | 1-based optimizer step number (matches the trainer's on_step_end). |
| `param` | Named parameter (as `model.parameters()` yields). |
| `shape` | Python list of the fp32 diff's shape. |
| `numel` | Total number of elements in the parameter. |
| `fp32_n_changed` | Elements where `diff_fp32 != 0`. Below-ULP AdamW updates rounded to zero in bf16 will NOT count here — they are exactly the elements this stat measures. |
| `fp32_frac_changed` | `fp32_n_changed / numel`. |
| `fp32_abs_min_nonzero`, `fp32_abs_max`, `fp32_abs_mean_nonzero`, `fp32_abs_median_nonzero`, `fp32_q90_abs_nonzero`, `fp32_q99_abs_nonzero` | Magnitude stats over the nonzero fp32 diffs. |
| `fp32_signed_mean`, `fp32_signed_std` | Signed mean / std over the nonzero fp32 diffs. |
| `bf16_*` | Same stats computed on `diff_fp32` after a round-trip through bf16. Compare against `fp32_*` to see how much of the effective update further collapses if requantized to bf16. |

All non-count stats default to `0.0` when the corresponding `n_changed` is 0,
so downstream CSV readers don't have to special-case missing values.

## Histogram CSV columns

`weight_update_distribution_histograms.csv`, one row per `(step, dtype)`:

- `step`, `dtype` (`fp32` or `bf16`), `total_numel`, `total_n_changed`,
  `total_frac_changed`.
- 24 count columns, one per `log10(|diff|)` bin in `[-12, 0]`
  (i.e. `|diff|` in `[1e-12, 1)`). Header names carry the bin edges
  (`bin_i_[lo,hi)`). Only nonzero diffs contribute; below-ULP updates that
  bf16 rounded to zero do not appear in the histogram (they are already
  visible as `total_n_changed < total_numel`).

## Log file / stdout format

Per step the trainer appends a block like:

```
=== step 1 ===
  fp32: n_changed=1,234,567,890/1,235,000,000 (99.9650%) |diff| p50=3.05e-5 p90=1.22e-4 p99=6.10e-4 max=2.44e-3
  bf16: n_changed=1,211,000,000/1,235,000,000 (98.0567%) |diff| p50=3.05e-5 p90=1.22e-4 p99=6.10e-4 max=2.44e-3
  top-5 params by fraction UNCHANGED (fp32 view):
    model.embed_tokens.weight  numel=524,288,000  fp32_frac_changed=97.100%  fp32_abs_max=8.80e-4
    ...
```

The `top-5 params by fraction UNCHANGED` block surfaces where bf16 precision
is biting hardest — the tensors most likely to be stuck.

## Interpreting the numbers

- `fp32_n_changed == numel` on a step means bf16 precision was not a
  bottleneck for that parameter — every AdamW-computed update landed in a
  representable-different bf16 grid point.
- `fp32_n_changed < numel` measures the raw effect of bf16 storage on the
  parameter: the "missing" elements are AdamW updates so small the bf16 result
  matched the pre-step value bit-for-bit.
- `bf16_n_changed < fp32_n_changed` measures how much of the *already
  effective* update would be lost if the diff itself were further stored /
  transmitted as bf16 (e.g. by a hypothetical bf16 gradient sync). If the two
  are close, the effective update is well-conditioned for bf16.

## Overhead

The instrumentation is host-only: `to_numpy(FLOAT32)` gathers the params once
before and once after each logged step, and everything else is numpy on the
host. There are no extra device ops. For steps outside the logging budget the
cost is zero (the snapshot dict is never populated).
