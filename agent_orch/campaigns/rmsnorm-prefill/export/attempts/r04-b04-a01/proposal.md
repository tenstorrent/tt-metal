# r04-b04-a01: posted (no-ack) output-drain writes: BRISC issues each 2 KB output tile with NocOptions::POSTED and flushes posted-sent before each pop, instead of non-posted writes

## Motivation
The drain is the post-AG tail on the best node (r03-b02-a02): it ends 1.1-1.9 µs after pack's POST end, at
~100-109 ns/tile vs POST's ~64 ns/tile (r03-b03-a03 `drain.py`). Three nodes established the per-core drain cap
(~16 GB/s/core, ~147 cycles/tile) is not:
- aggregate DRAM/link bandwidth (r03-b04-a02: 10 lone writers in a wave drain at the same per-core rate),
- single-VC serialization (r03-b03-a03: 4 VCs, flat),
- command-buffer / issue overhead (r03-b02-a03: two cmd bufs, *slower*).
r03-b02-a03 concluded the cap looks like response/flow control for non-posted writes and named posted writes as
"the remaining cheap test of the per-core cap" (also suggested in r03-b04-a02 #2 and r01-b03-a04 #3). Never tried.

## Mechanism
Writer only (`dit_rmsnorm_fused_worker_writer.cpp`, W_DRAIN): both the default NoC and the alt-NoC output tile
writes use `async_write<NocOptions::POSTED>`; the per-block `async_writes_flushed()` becomes
`async_writes_flushed<NocOptions::POSTED>()` (data has left L1 before compute reuses the CB slots); at kernel end
a posted flush is added next to the existing write barrier on each NoC. Routing, order, VCs, CB handshakes unchanged.
Everything else (stick push, gathered reads, gamma) unchanged.

## Why this is not a repeat
Nearest: r03-b02-a03 (2 cmd bufs, flawed), r03-b03-a03 (VC round robin, neutral), r02-b0x dual-NoC routing.
None changed the write type; all kept non-posted (NOC_CMD_RESP_MARKED) writes with an ack per tile.

## Expected effect and risk
If the cap is ack/response bookkeeping, the drain rate rises toward the POST rate and the tail after POST end
shrinks: ~-0.5..-1.5 µs per shape, more on h6144/h7168. If neutral, the per-core cap is the NoC path/destination
ingress, which tells future nodes to stop on drain issue mechanics. Accuracy should be bit-identical (same bytes).
Risk: posted writes give no completion guarantee at kernel end (data in flight briefly); acceptable for the test
since the next consumer is far later, but a production version would need a final non-posted fence per bank.
Possible hang/garbage only if dynamic-NoC posted counters are mis-tracked (would show as hang).
