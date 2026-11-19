# Raw recurrent accumulation-order repair

The cumulative raw recurrent candidate passes all batches 1–32 at 16 steps,
4K-prefix/128-step and 64K-prefix/128-step tests, but fails the distinct
1024-answer trajectory at `decode_79_30`, PCC 0.994103. The flat-input preparation
variant fails the same row/step at 0.992851. The threshold remains 0.995.

Component isolation on exactly that trajectory:

| Variant | Result | Traced stack ms |
| --- | --- | ---: |
| Partial Q/K normalization/RoPE only | Pass | 1.254255 |
| Tiled history only | Pass | 1.302955 |
| Partial Q/K + tiled history + chunk GDN | Pass | 1.233028 |
| Raw recurrence, original RM history | Fail | Not accepted |
| Cumulative raw recurrence | Fail | Not accepted |
| Cumulative raw recurrence with chunk accumulation order | Pass, minimum PCC 0.995991 | 1.164026 |

Independent source analysis found equivalent algebra but different rounding:
raw computes `q @ packed(S_new)`; chunk computes output from old state plus the
correction and accumulates each pair of state-update products in one DST pass.
The repair keeps raw external Q/K normalization, exp(g), padding, wave104,
HiFi4/exact FP32 and state aliases, but prepares one-token chunk operands and
calls the repository's actual `scan_step`. The only decoder-method change is
the recurrence call. This is compatibility with the accepted accumulation
order; it does not claim the old algebra is mathematically invalid.

Artifacts: `kernel_answer_isolation_jobs.json`, `ordered_recurrent_jobs.json`,
all corresponding command/source snapshots and logs, and
`ordered_extended_answer1024_32.json`. New implementation files are
`tests/multichip_raw_recurrent_ordered.py`, its candidate wrapper, and
`tests/kernels/raw_recurrent_ordered/`. The compute source documents every FP32
packing boundary. Python compilation/formatting and source comparison passed;
hardware short B32/B6 and the extended 1024-answer test passed. Further combined
flat-preparation/convolution, context, watcher and final-default gates remain
pending. No production selection is claimed here.
