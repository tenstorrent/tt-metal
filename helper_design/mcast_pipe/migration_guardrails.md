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

For a runtime-sized operation tail, use the one deliberate exception to the
usual helper-tail layout: fixed operation prefix, complete helper block, then
the variable operation tail. Derive the tail start from
`next_runtime_args_offset()`. This keeps the helper's runtime base available as
a compile-time template argument without duplicating it in mutable decoder
state.

For fixed-width compile-time ABIs, keep operation arguments first and multicast
helper blocks last. A genuinely optional operation tail in a separately
compiled variant is the deliberate exception: emit the fixed operation prefix,
the complete helper block, then the optional operation tail only in the variant
that consumes it. Derive that tail from `next_compile_time_args_offset()`.
Do not emit zero addresses, null accessors, or other dummy operation fields
solely to preserve a helper-last layout.

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

## 12. Give `McastArgs` one runtime-base source of truth

Supply the runtime base only through `McastArgs<CT_BASE, RT_BASE>`. Do not add a
constructor/runtime override, store a mutable base, or expose parallel static
and instance base/end APIs. When a runtime-sized operation prefix prevents a
constant base, reorder that variable region after the opaque helper block and
derive its start from `next_runtime_args_offset()`.

## 13. Preserve operation-owned terminal drains

Do not remove a data-movement kernel's terminal write barrier merely because a
helper send has its own completion policy. The helper owns completion for its
transaction; the terminal drain covers every outstanding operation write. Check
all write-producing exits against the actual pre-migration kernel and ensure
they reach the original drain.

## 14. Add no migration-only source-lifetime synchronization

Preserve the operation's established flush and barrier policy. Do not add an
`async_writes_flushed()`, write barrier, persistence classification, or similar
source-lifetime synchronization unless the source buffer's reuse contract
requires it independently of migration. Compare with the actual pre-migration
implementation instead of inferring the contract from the helper default.

## 15. Derive helper roles and precisely name independent roles

When a scalar describes the sender or receiver for the same multicast family,
remove the duplicate wire and use `McastArgs::can_send()` or `can_receive()`.
When it describes independent operation ownership, retain it only under a name
that states that ownership, such as `has_sharded_input`; do not leave a second
ambiguous `is_sender_core` beside a helper role.

## 16. Derive dense ACK populations from geometry

Let `Mcast1D` derive `span - 1` and `Mcast2D` derive `area - 1` when every
non-sender landing core acknowledges. Pass an explicit ACK override only when
the landing and acknowledging populations diverge, and record why. Do not
silently carry raw geometric count ABIs into a future helper migration.

## 17. Chain offsets through an existing helper object

Keep `next_compile_time_args_offset()` and `next_runtime_args_offset()` static
constexpr because they describe the compile-time wire type. When a kernel
has a named constexpr `McastArgs` object, call the static member through that
object for subsequent parsing or helper composition, such as
`mcast_args.next_runtime_args_offset()`. For chained helpers, declare each
object before using its offsets to instantiate the next object. Do not
introduce or retain a type alias unless it independently names a nested
`SenderPipe` or `ReceiverPipe` type.

## 18. Construct each pipe face outside repeated work loops

Never instantiate a `SenderPipe` or `ReceiverPipe` inside a batch, block,
round, tile, or other repeated work loop. Construct each permitted pipe face
once after the `Noc` and multicast arguments are available, then reuse that
object for every `send()` or `receive()` call. In a mixed-role kernel, use
role-conditional optional storage outside the loops so a core constructs only
the faces allowed by `McastArgs::can_send()` and `can_receive()`.

Pipe construction is setup, not per-transfer work. `SenderPipe` precomputes
topology and fan-out state, while `ReceiverPipe` initializes its signal state
and retains its absolute receive round. Reconstructing either face in a loop
duplicates invariant setup and can reset protocol state, especially for a Flag
receiver or a Counter receiver whose round must remain monotone.
