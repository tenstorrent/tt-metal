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

PR2 is developed on `cglagovich/sdpa-streaming-pr2`, stacked on PR1:

- Dense/joint sub-tile tails, batch/GQA and uniform SPMD meshes reuse the shared
  recipe implementation and accuracy infrastructure.
- Ring separates Q release from final normalization, preserving raw state and
  pending compensated groups through single-Q residency or multi-Q checkpoints.
  Existing communication/scheduling, skipped iterations and replicated/sharded
  joint KV are retained. Real two-device release and Watcher tests check against
  dense attention in the same KV order.
- Wan self-attention has an explicit opt-in recipe/storage selector, with fresh
  pretrained attention-block accuracy and preparation-inclusive timing. No
  model defaults are changed; full-video quality is still a rollout gate.

The original coverage list is a rollout roadmap, not a claim that every prefill
configuration can switch now. D128 noncausal dense/joint/two-device ring are
qualified here. Other head dimensions, causal/masked/windowed/sink attention,
paged/indexed/chunked caches, MLA, Wormhole and larger ring topologies still
need separate coverage work. Explicit recipes reject them. They must be
qualified before the corresponding legacy implementation can be removed.

**PR 3:** delete non-streaming loops only after every in-scope supported
configuration has a qualified replacement. Retain utilities needed by sparse,
decode and CCL consumers. Decode and independent training attention are not part
of this deletion.

## Historical investigation

The [original staged plan](https://github.com/tenstorrent/tt-metal/blob/sdpa-pr1-evidence-20260921-v1/docs/sdpa_streaming_precision_plan.md)
and [bring-up log](https://github.com/tenstorrent/tt-metal/blob/sdpa-pr1-evidence-20260921-v1/docs/sdpa_streaming_precision_validation.md)
retain the infrastructure recovery and intermediate validation history.
They are historical evidence, not additional production contracts.
