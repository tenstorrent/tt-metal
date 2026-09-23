# Streaming SDPA integration plan

## Frozen numerical contract

The recipes and evidence are frozen at
[sdpa-recipes-20260921-v1](https://github.com/tenstorrent/tt-metal/tree/sdpa-recipes-20260921-v1).
The [API guide](sdpa_precision.md) defines A/B/C/D and the three E storage choices;
the [qualification report](sdpa_precision_qualification.md) records accuracy,
performance, limitations, and reproduction commands.

Preserve the selected arithmetic, input preparation, Q256/K512 blocking and
buffer depths. Legacy callers retain their existing defaults and feature dispatch.
Explicit recipes reject unsupported configurations rather than falling back.

## PR 1: production recipes and qualification

One implementation PR contains:

- The public precision enum, explicit input preparation, validation and dispatch.
- A dense streaming loop shared by B/C/D/E; A uses existing BF16 streaming.
- Focused exponential, compensated-state and FP32-state helpers. Component tests
  exercise the same FP32 state implementation as attention.
- Frozen-output, independent FP64, state-boundary, preparation, cache/trace,
  legacy-compatibility, Watcher and matched-geometry throughput tests.

Initial scope: single Blackhole, batch 1, matching heads, D128, dense noncausal
unmasked attention, full Q256/K512 blocks, tiled interleaved DRAM, BF16 output.
Detailed eligibility and rejection rules live in the API guide.

The cleanup must retain exact frozen outputs, rounding points, buffer ownership,
publication fences and packing transitions. Performance checks include resident,
distinct-input, changing-max and preparation-inclusive cases on the same hardware.
Historical model results do not qualify a new production implementation.

## Follow-on PRs

**PR 2:** expand feature/platform coverage and migrate callers where supported by
model evidence. Include masked tails, appropriate head dimensions, GQA, and
joint/ring/paged/chunked/MLA coverage as applicable; qualify multiple devices and
fresh pretrained-model quality/performance before changing model defaults.

PR2 is developed on `cglagovich/sdpa-streaming-pr2`, stacked on PR1. Its first
slice adds joint segment addressing with unchanged compute and shared accuracy
tests. It does not yet qualify PR2 for merge. Remaining integration gates:

- Dense/joint sub-tile tails, batch/GQA and uniform SPMD mesh execution now use
  the shared recipe implementation and pass release/Watcher/performance
  regression qualification. SPMD mesh coverage is not ring qualification.
- Separate Q-block release from final normalization before ring integration.
  Preserve raw maxima, denominators, numerators and pending compensated groups
  across ring steps, including state staging for multiple Q blocks.
- Reuse existing ring communication/scheduling, including skipped iterations and
  replicated versus sharded joint KV. Normalize only after the final active KV
  contribution, not separately per segment or ring step.
- Qualify single- and multiple-Q-block ring paths on real multi-device hardware;
  extend applicable prefill variants and migrate callers only with model evidence.

**PR 3:** delete non-streaming loops only after every in-scope supported
configuration has a qualified replacement. Retain utilities needed by sparse,
decode and CCL consumers. Decode and independent training attention are not part
of this deletion.

## Historical investigation

The [original staged plan](https://github.com/tenstorrent/tt-metal/blob/sdpa-pr1-evidence-20260921-v1/docs/sdpa_streaming_precision_plan.md)
and [bring-up log](https://github.com/tenstorrent/tt-metal/blob/sdpa-pr1-evidence-20260921-v1/docs/sdpa_streaming_precision_validation.md)
retain the infrastructure recovery and intermediate validation history.
They are historical evidence, not additional production contracts.
