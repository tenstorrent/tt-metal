# Group-attention rotating multicast — migrated at API v11

Date: 2026-08-16

## Verdict

Migrated in `6e8eb7638855ffc03948b34236b9219102215b49`. The prior v7 runtime-semaphore
design gap is obsolete: API v11 accepts the host-assigned semaphore IDs through helper compile-time
arguments while retaining the per-core runtime ACK override.

## Protocol mapping

- One independent `Mcast2D` covers the fixed dense receiver bounding rectangle. The first 32 logical
  cores rotate through the sender role by round.
- The host appends helper compile-time arguments after the TensorAccessor arguments and helper runtime
  arguments after operation slot 20. Runtime slot 20 remains the operation-owned, per-core divergent
  ACK count.
- The kernel constructs the sender face from the round's sender coordinates and runtime ACK count;
  receiver cores use `McastArgs::receiver()`.
- Sender identity is re-evaluated inside the `tile_row_id` loop at kernel lines 131 and 140-141 by
  calling `sender_x(tile_row_id)` / `sender_y(tile_row_id)`; those accessors index `RT_BASE + 4 +
  2*round` at `mcast_pipe.hpp:408-409`. Receivers pass the same `tile_row_id` at kernel line 211, and
  `ReceiverPipe::receive(round)` indexes `coords_[2*round]` at `mcast_pipe.inl:264-278`. No round-0
  sender value is latched.
- Loopback is inferred from source/destination aliasing and membership: sharded `CB2 -> CB1` includes
  an in-rectangle sender, while `CB1 -> CB1` and outside-rectangle sends exclude it.
- The historical post-flag barrier was not a required per-round completion point. Data and Flag form one
  linked NoC chain (`mcast_pipe.inl:146-148`), and `send()` does not return or reset the rotating
  sender's local Flag until `fence_()` has either completed loopback with a write barrier or flushed the
  remote chain (`mcast_pipe.inl:79-103,197-208`). On every receiver, round N+1 cannot ACK its sender
  until round N's Flag was observed and cleared, because `receive(N)` waits and clears before returning
  and `receive(N+1)` issues the next ACK (`mcast_pipe.inl:264-278`). Thus a later round's Counter wait
  cannot overtake an undelivered prior Flag. One final write barrier preserves last-send completion at
  kernel exit.

## Validation

- Production LOC: factory 20 additions / 55 deletions; kernel 24 additions / 132 deletions.
- `./build_metal.sh` passed before validation and after restoring the migrated source from the matched
  raw-baseline measurement.
- Exact q16 fully-sharded ROW_MAJOR node passed under `--dev` with fresh helper JIT artifacts.
- Complete `group_attn_matmul` inventory: 322 passed, 132 expected skips, 299 deselected; warm JIT
  351/351.
- The 132 skips are the suite's expected parameter exclusions: 96 optional-preallocated-output cases
  with sharded output and 36 remaining duplicate COL_MAJOR cases where both inputs are interleaved.
- Post-restore exact normal q16 node passed; `test_mcast_pipe.py` passed 80/80, the source audit passed
  17/17, and `McastHostFixture.*` passed 32/32.
- Matched Tracy at 800 MHz, three independent 25-iteration sessions per state, first five operation
  samples discarded in every session:
  - q16: raw 57,846.5 ns, migrated 39,219.5 ns, -32.20%.
  - q48: raw 405,652 ns, migrated 294,888 ns, -27.31%.
- The large win is attributable to the 32-round critical path: raw Blackhole code serialized each data
  multicast with `async_writes_flushed()`, then sent the Flag, then executed a full write barrier every
  round. The helper links data+Flag and uses one mode-appropriate completion fence. Profiles retained
  exactly 25 GroupAttn operation rows and 110 cores per row in both states; the full correctness matrix
  rules out omitted work. The 322 correctness passes exclude skipped work; the 132 skips are separately
  categorized above and are not counted as passes. NCRISC and waiting TRISC durations fell together,
  consistent with faster CB publication rather than omitted compute.

## Claude consultation

Three fact-complete consultations were attempted with the user-required Opus command (broad design,
compact design, and post-validation review). Each timed out without a verdict. A fourth final review
returned REVISE for evidence completeness only: explicitly prove per-round coordinate indexing and
cross-round ordering, attribute the large speedup, and categorize the expected skips. Those items are
resolved above; a bounded re-review returned PASS for ledger write-back. No code or API change was
requested. No API expansion was made.
