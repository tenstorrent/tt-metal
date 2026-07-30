# reader_single_row_multi_core.cpp — DEFERRED (design-gap)

Tier 3. Status: deferred. No code change.

## Role
Sort worker RECEIVER half of the coordinator STAR. Per Ht/substage: `up()` the
`cores_to_coordinator_sem` (atomic-barrier inc) to ack, then `wait(0)` on the inverted
`coordinator_to_cores_sem` GO flag, then `set(VALID)` to reset.

## Why deferred
- **Pairs with the coordinator**, which itself defers (runtime recipient count + split
  mcast/ack count). A receiver migration in isolation is meaningless if the sender stays
  raw.
- **Runtime sem ids.** Sem ids are runtime args (`get_arg_val(4)`/`(5)`), kernel builds
  `Semaphore<>(runtime_arg)`; v7 `ReceiverPipe` takes `DATA_READY_SEM_ID` /
  `CONSUMER_READY_SEM_ID` as compile-time template params. Runtime sem id inexpressible
  (same gap as group_attn_matmul).
- **Inverted flag polarity.** Worker waits for `0` and resets to `VALID(1)`; helper
  `receive_signal()` waits `VALID(1)` and clears to `INVALID(0)`. Re-wirable in principle
  but moot given the sender's binding gaps.

Helper untouched. Lines removed: 0.
