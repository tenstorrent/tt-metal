# `mcast_pipe` migration guardrails

## 1. Chain arguments after the multicast block

Always derive the first compile-time and runtime argument after an `McastArgs` block with
`next_compile_time_args_offset()` and `next_runtime_args_offset()`, respectively. Never hard-code the current
multicast block width or resume parsing at a numeric index, because the helper's encoded argument layout may change
without changing the surrounding operation-specific arguments.

## 2. Treat host-generated argument blocks as opaque

Append complete helper blocks through `append_compile_time_args_to()` and
`append_runtime_args_to()` at one contiguous ABI boundary. Never extract helper
words with numeric indexing such as `[0]` or `[1]`, because callers must not
depend on the helper's internal encoding.

## 3. Let the helper own its protocol semaphores

When semaphores exist only for the multicast pipe, use `owned_semaphores()` and the standard creation bridge instead
of allocating them externally and passing their IDs back into the helper. Adopt external semaphore IDs only when
another protocol genuinely shares their ownership.

## 4. Use separate channels for different synchronization policies

If one phase requires a receiver-ready handshake and another does not, configure separate pipes instead of forcing
both phases through one pipe or retaining a duplicate raw handshake. Keep genuinely operation-owned synchronization,
such as an independent writer-done counter, outside the pipe.

## 5. Audit every downstream index after changing the argument ABI

Check TensorAccessor bases, optional bias and output-sharded tails, fused-operation fields, and program-cache override
indices whenever a multicast argument block moves or changes size. Derive every patch position from emitted range
sizes rather than preserving historical numeric indices.

## 6. Validate actual host-generated geometry

Inspect the multicast geometry produced by supported workloads instead of assuming every theoretical edge shape is
reachable. Add synthetic host coverage for defensive geometry, and benchmark any extra degenerate send paths that
remain in production configurations.

## 7. Optimize the final argument layout for clarity

Do not insert new arguments between existing arguments merely to preserve historical ordering. Place them at the
cleanest ABI boundary—often at the end, as with TensorAccessor arguments—and update the producer and parser together;
the goal is the clearest final code, not retaining the previous argument order.

## 8. Do not add preprocessor defines

Do not introduce any new preprocessor defines or `#ifdef` branches as part of a migration. Integrate the helper
directly into the final code path instead of retaining parallel legacy and migrated implementations.

## 9. Keep the migration diff focused

Do not rewrite, reformat, or remove comments or surrounding code that are unrelated to the migration. Preserve them
as-is so the diff exposes only the changes needed to adopt the helper.

## 10. Preserve the multicast receiver geometry

Do not expand a multicast rectangle merely because a larger uniform geometry is easier to express with the helper.
Every added destination increases data, signaling, and acknowledgement traffic and can change synchronization; preserve
the original receiver set, and treat an API that cannot represent it as a helper limitation to resolve.

## 11. Represent an absent helper with its tagged one-word block

When a shared kernel ABI has an inactive multicast role, append the helper-owned
false presence tag and no multicast runtime words. This is the deliberate
library-wide exception to a zero-width inactive helper: the tag lets ordinary
`McastArgs` select its absent compile-time specialization, derive its actual
one-word/zero-word boundaries, and reject sender or receiver construction at
compile time. Do not add an operation-owned presence flag, a synthetic multicast
geometry, or a separate optional decoder type.
