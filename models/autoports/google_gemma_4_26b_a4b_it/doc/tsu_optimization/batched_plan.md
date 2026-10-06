# Remaining high-concurrency experiment

This is unfinished work, not an accepted optimization or a blocker. The selected
C1 checkpoint leaves batched execution unchanged.

`OptimizedDecoder.decode_forward` serializes every logical row when batch>1;
each row executes its own complete attention/router/expert/shared/tail path.
This explains why C32 median intertoken time is~677ms instead of amortizing
~20ms B1 work across users. The runtime trace removes host dispatch overhead,
but does not parallelize these row graphs.

Removing only that loop is invalid. `FusedAttention.decode` explicitly accepts
one slot; its fused RoPE repeats the first position, fused cache update uses
single-core buffers, and `_LocalAttention.project` reshapes to one row. Indexed
TP experts also reshape intermediate outputs around one logical row. These
are exact source contracts, not proof that native batched execution is impossible.

Proposed bounded investigation:

1. Profile a reduced two-real-layer batch32 graph, using the same full model
   context/table width and representative aggregate cache capacity, with no
   server or Watcher. Quantify row serialization and dominant repeated work.
2. Build a probe-only vectorized candidate. Reuse native per-user paged SDPA
   and cache ops, correct batch-major RoPE tables, existing local TP weights,
   and output projection/collective contracts. Canonical attention's
   `apply_rope_decode_peruser` provides a reference for the required axes.
3. Keep indexed expert semantics per row initially if required; separately
   measure existing EP active-union batching only after recording its actual
   dtype/fidelity difference. Do not call a changed precision policy equivalent
   solely because a synthetic test passes.
4. Compare to serialized controls on recorded target-layer activations at both
   layer kinds, batches2/3/8/32, heterogeneous positions and physical pages.
   Check per-slot outputs, cache writes, untouched slots and advancing replay.
   Do not enable an unsafe path for inactive-slot batches.
5. Only a verified reduced win advances to full generator/serving guards,
   shared qualitative checks, a new exact image and focused/full-matrix CI.
   Preserve the already measured C1 path and capacity throughout.

If an API/layout candidate fails, adapt legal layouts/packing before rejecting
the family. Hardware hangs require live triage before any process termination.
