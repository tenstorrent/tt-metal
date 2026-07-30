# reader_interleaved.cpp — DEFERRED (Tier 2c)

**Kernel:** `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp`
**Status:** deferred (helper design gap — CHAIN topology)
**Validation:** none run (static deferral; no kernel edits, tree green)

## Verdict
The K+V mcast here is a **CHAIN / store-and-forward relay**, NOT a STAR rectangle broadcast.
This is the documented `chain_link.hpp` pattern and the **T2 CHAIN = GAP** in the topology matrix.

## Evidence
- Each `is_chain_participant` core is simultaneously a RECEIVER (L409-417 K, L601-609 V:
  `receiver_sem.set(INVALID); sender_sem.up(prev...); receiver_sem.wait(VALID)`) AND a
  SENDER (L471-513 K, L662-706 V: forwards to `next`).
- The forward signal is a **cross-id relay**:
  `Semaphore<>(valid_semaphore_id).relay_multicast(noc, Semaphore<>(receiver_semaphore_id), ...)`
  (L492-501, L680-689) — source sem (`valid_semaphore_id`, write-once VALID pinned in ctor L166)
  differs from dest sem (`receiver_semaphore_id`). The write-once VALID source is needed because a
  link's own doorbell (`receiver_semaphore_id`) is mutable (`receive()` drives it INVALID), so it
  cannot hold the VALID broadcast source at forward time (hazard H12/INV12).
- `SenderPipe` is a STAR primitive: it broadcasts its OWN `data_ready` cell via `set_multicast`
  (src sem == dst sem). It **structurally cannot** do cross-id relay
  (A5' `ASSERT(src != dst)`, noc_semaphore.h:192).

## Decision
Per KNOWN HELPER DESIGN GAPS (CHAIN / relay topology) and proposed_helpers.md Round-7 TOPOLOGY GAP:
the entire chain family (chain_link.hpp, reader_interleaved, exp_ring_joint_reader) is deferred until
the Pipe grows a relay/forwarding-link capability. **Helper NOT modified.** No revert needed (no edits made).
