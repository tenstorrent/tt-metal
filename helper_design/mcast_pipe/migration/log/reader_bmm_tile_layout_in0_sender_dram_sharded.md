# reader_bmm_tile_layout_in0_sender_dram_sharded.cpp — DEFERRED (API-v11 design gap)

## 2026-08-16 API-v11 re-audit

API v11's runtime `consumer_ack_count` resolves the historical fan-out/ACK-count split documented
below, but the complete Tier-2.7 protocol is still not expressible:

- Type 1 is an in-rectangle sender-only core whose raw data and Flag multicasts both EXCLUDE source.
  Its sharded source and compute destination addresses differ, so `SenderPipe::send()` infers INCLUDE
  loopback. That adds an unowned local destination write and leaves the sender-local helper Flag VALID.
- Under `SKIP_MCAST`, type 2 must skip data while signaling INCLUDE source so its own local readiness
  dependency completes. `send()` always couples data and signal; `send_signal()` is always EXCLUDE.

A bounded Claude architecture review returned DEFER. The smallest proposed contract change is a
default-preserving explicit self-mode (`AUTO`, `INCLUDE`, `EXCLUDE`) available to both `send()` and
`send_signal()`. That is an API expansion and was not authorized. Production kernel and factory remain
untouched.

## Historical v7 verdict: deferred (helper design gap — NOT migrated, file untouched)

Worker-type-dispatched kernel (3 runtime core types via `worker_core_type` arg: 1=sender-no-compute,
2=sender+compute, 3=receiver+compute, 0=idle). The blocker is the **split between the mcast dest-count
and the consumer-ack count** — a known v7 design gap (one `NUM_ACTIVE_RECEIVER_CORES` template param
derives BOTH the mcast_dests and the ack-count, but this kernel needs them different):

Factory (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`):
- `in0_mcast_num_dests`  = num_worker_cores = **num_dram_banks**
- `in0_mcast_num_cores`  = **full compute grid** (x*y)
These are distinct quantities.

- **type 1** (sender, no compute): mcasts EXCLUDE_SRC to `in0_mcast_num_cores - 1` (grid-1) but
  pre-waits `sender_sem.wait(in0_mcast_num_dests)` (banks). dest-count = grid-1, ack-count = banks.
- **type 2** (sender + compute): mcasts INCLUDE_SRC to `in0_mcast_num_cores` (grid) but pre-waits
  `sender_sem.wait(in0_mcast_num_dests - 1)` (banks-1).

SenderPipe with `NUM_ACTIVE_RECEIVER_CORES = N` mcasts N (EXCLUDE) / N+1 (INCLUDE loopback) and waits N.
- type 1 needs N = grid-1 for the mcast AND wait = banks → requires banks == grid-1.
- type 2 (INCLUDE) needs N = grid-1 for the mcast AND wait = banks-1 → requires banks == grid.
Mutually contradictory ⇒ the dest-count and ack-count are genuinely independent here. Inexpressible in
v7 (split mcast-dest vs consumer-ack count). Needs helper change (out of scope).

## Action: no edit, ledger status=deferred, flag design-gap.
