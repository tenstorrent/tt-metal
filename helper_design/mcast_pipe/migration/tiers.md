# mcast_pipe API v9 production-port tiers — reconciled 2026-07-30

The July v8 worklist is complete for the current branch. There are no pending
production rows.

## Tier 0 — migrated and fully validated

- Matmul in1 sender/receiver pair
- Conv 1D weights sender/receiver pair
- Conv 2D weights sender/receiver pair
- GroupNorm legacy and Welford sender/receiver pairs

## Tier D1 — typed control values

- Matmul in0 1D sender/receiver

Required helper work: publish/consume typed values such as `IGNORE_BATCH`
without exposing the shared semaphore.

## Tier D2 — acknowledged control channels and streaming

- Pre-allgather LayerNorm sender/receiver
- Plain sharded LayerNorm sender/receiver

Required helper work: acknowledged signal-only traffic and, for plain sharded
LayerNorm, one-gate/multi-block mixed flag/counter streaming.

## Tier D3 — rotating and explicit-loopback protocols

- Block-sharded rotating matmul
- Width-sharded rotating Conv activation reader
- Post-allgather LayerNorm sender/receiver

Required work differs by row:

- block-sharded matmul needs independent data/signal loopback selection and
  sender-side ready participation;
- width-sharded Conv now has ACKed loopback support but needs a fresh full
  re-port/retest after its earlier migrated attempt failed numerically; and
- post-allgather LayerNorm needs explicit out-of-rectangle loopback plus
  explicit fan-out independent of rectangle area, or checked dense geometry.

## Tier D4 — startup ordering

- TopK reader/writer pair

Required helper work: a no-handshake receiver mode that trusts host-owned
initialization and cannot erase the first sender signal.

The exact test inventory is in `test_map.json`; failure evidence is in the
per-kernel migration logs.
