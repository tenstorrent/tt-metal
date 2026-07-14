# Fixed-slot first-token sampler autofix

## Reproduction

The source-current reduced watcher run used real layers 0 and 5, `max_batch_size=2`, one active prefill row, and one inactive physical decode slot. The one-row prefill logits were initially sent to a common sampler whose local-index buffers were sized for the two-row trace batch:

```text
TT_FATAL: Input values and indices must have the same shape!
```

A later combined watcher run also invoked the already-rejected common force-argmax probe after the canonical greedy comparison and asserted during all-gather teardown. The isolated sampler test already retained that rejection evidence; executing it inside the selected-path gate was redundant.

## Fix

- Pad first-token device logits to the fixed physical trace slots and preserve the logical active-row count.
- Keep inactive rows at cache position `-1` with masked position increments.
- Remove the rejected force-argmax probe from the canonical combined gate; compare selected greedy output directly with host full-logit argmax.
- Replace the latency-dominant common greedy path with the exact custom TP4 reducer after isolated common-path testing justified custom code.

## Result

The selected sampler passes batch-one and batch-two exact boundaries, equal-score lower-global-token tie-breaking, three trace replays, reduced full-model token feedback, and watcher. Request reset and the next short-prompt request also pass under watcher. The gate is closed.
