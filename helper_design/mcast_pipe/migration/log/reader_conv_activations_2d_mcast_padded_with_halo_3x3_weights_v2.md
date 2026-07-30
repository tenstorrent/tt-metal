# reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp — DEFERRED (out of scope: R4 streaming chunked send)

## Verdict: deferred (R4 streaming chunked send — out of scope this round, file untouched)

The data mcast goes through `mcast_block_chunked(...)` (lines 61-132, called at line 281), a
**producer-overlapped streaming chunked send**: it does NOT wait for the whole tilized block to be
ready before mcasting. Instead it loops over NOC_MAX_BURST_SIZE bursts, and for each burst calls
`src_cb_obj.wait_front(wait_tile_curr)` with a MONOTONICALLY GROWING tile count — i.e. it mcasts
chunk i while the upstream compute is still tilizing chunk i+1, overlapping the producer with the NoC.

The v7 `SenderPipe::send()` auto-chunks a *fully-ready* block transparently (one shot), but it cannot
express the interleaved `wait_front`-per-burst streaming: `send(src,dst,size)` takes a single ready
region and issues the whole mcast at once. Migrating would force a single `wait_front(block_tile_count)`
up front, destroying the producer/NoC overlap (perf regression) — and this is exactly the R4 pattern
that `proposed_helpers.md` explicitly DEFERS this round ("Streaming chunked send (R4) ... Deferred this
round (user decision). The Pipe handles only fully-ready blocks").

All 3 F3 loopback sub-cases (INCLUDE_SRC, EXCLUDE_SRC, degenerate self-write) ARE handled by the Pipe,
but they are reached *inside* `multicast_data`, called *per chunk* — so they cannot be lifted without
the per-burst streaming, which is the blocker.

## Action: no edit, ledger status=deferred, flag design-gap (R4 streaming chunked send, OOS this round).
