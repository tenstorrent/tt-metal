# writer_single_row_multi_core.cpp — DEFERRED (helper-neutral companion)

Tier 5 atomic unit: `sort-single-row-control`. Coupled ABI cleanup: `7337302b564`.

## Disposition

The writer is not a Pipe caller. It emits the operation-owned writer-done counter after each
processed pair; the coordinator waits and resets that counter independently from the helper-owned
coordinator-to-reader control channel. Counting this file as migrated would overstate helper reach.

The atomic production change removed dead doorbell and semaphore-ID runtime words, reducing the
writer runtime block from six words to four. Its done semaphore is constexpr ID 2. The functional
counter leg remains raw/object API by design.

## Coupled validation

- Host build passed.
- Exact fresh-cache `--dev` long-tensor node passed with the writer JIT artifact confirmed.
- Ht=2 deadlock regression: 2 passed.
- Full long-tensor inventory: 7 passed.

Standalone helper status remains deferred; no API gap is claimed.
