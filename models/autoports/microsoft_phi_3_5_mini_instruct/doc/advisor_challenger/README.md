# Phi-3.5 Mini advisor challenger

Outcome: **no change**. The frozen incumbent remains the fastest publishable
decoder under the stage's tie rule.

The untouched shipped decoder was measured first with checkpoint weights,
recorded layer-0 target activations, and the matching nonzero prefix cache.
Three independent batch-1 traced-decode runs measured `0.466496`, `0.468089`,
and `0.466404` ms. The frozen incumbent is therefore the best repeat,
`0.466404` ms, with a `0.001685` ms spread/noise floor. All repeats passed the
incumbent PCC bar of `0.995` at PCC `0.9992643`.

## Capture

Phi-3.5 Mini has one meaningful dense decoder-layer kind across all 32 layers,
so one capture covers 100% of layers. It was constructed explicitly with the
executed shipped policy: BFP4 QKV/output/gate-up/down weights, LoFi compute,
BFP8 KV cache, and the shipped 8-core DRAM-sharded matmul programs.

The pinned advisor predates `paged_fused_update_cache`. For capture only, that
wrapper was lowered to its two already-supported `paged_update_cache` tracer
ops. This preserves candidate-bearing shapes, dtypes, layouts, attention, and
all four matmuls. The capture contains 42 ops, considered all four projection
matmuls for DRAM sharding, and advised all four; all four already ship
DRAM-sharded. The BF16-DS pipeline option was enabled because norm weights are
BF16, although all material projection weights traced as BFP4.

The `compute_config`/LoFi state visible in `final_ir.mlir` was treated as traced
state, not advice. Every challenger retained the incumbent's LoFi configs.

## Advice and measurements

The advisor found no new material projection or norm to shard: all projection
weights are already DRAM-sharded and the residual/norm path is already
L1-sharded. Its authoritative IR reproduced shipped QKV/output/gate-up block
widths `12/12/6`. It proposed down block width 32 instead of 16:

- Down block 32: `0.467055`, `0.466020`, `0.467480` ms. The best is only
  `0.000384` ms faster than the incumbent, inside the `0.001685` ms noise
  floor, so this is a tie and the incumbent wins. Real-weight PCC was
  `0.9992229`.
- Sharded SiLU output: `0.469572`, `0.468975`, `0.470447` ms. The best is
  `0.002571` ms slower than the incumbent, outside the noise floor, so it was
  rejected. Real-weight PCC was `0.9992643`.
- The remaining newly sharded helper occurrences are each below the 1% device
  window threshold. `reconciliation.json` records their measured shares; none
  is silently or prose-only rejected.

No topology rewrite changed an op shape, so no recapture was legal or needed.
The experimental SiLU switch was removed, leaving `tt/optimized_decoder.py`
byte-for-byte unchanged from the frozen incumbent. Thus `final_ms =
incumbent_ms = 0.466404`, satisfying the invariant.

`tracy/` contains only copied `tt-perf-report` CSV outputs, not raw Tracy
captures.

## Independent gate re-verification

The runner-side advisory failure was reproduced from
`02-02b-advisor-challenger.check-1.log`. It failed before inspecting any stage
artifact because the required positional argument was omitted:
`line 13: 1: model_dir`.

The documented invocation was then run from the repository root:

```text
bash .agents/prompts/model_bringup_multigoal/02b-advisor-challenger.check.sh models/autoports/microsoft_phi_3_5_mini_instruct
```

It exited 0 and reported all four substantive checks as `ok`, followed by
`02b-advisor-challenger gate PASSED for microsoft_phi_3_5_mini_instruct`.
Thus the advisory exit 1 is an invocation failure, not evidence against the
measured no-change result.
