# reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp — DEFERRED (v7)

- Group: G3 conv (HS sender, 1D height-sharded)
- Status: DEFERRED — coverage-gap / helper design limitation (distinct mcast-dest vs ack count)
- Validation attempted: test_conv_features HEIGHT_SHARDED input_channels=16 output_channels=16
  input_height=256 input_width=256 config={'act_block_h':32} BFLOAT8_B HiFi4 fp32_accum=True (the
  same nodeid the already-migrated 1D receiver passed against). Co-compiled with the migrated receiver.

## Why deferred (HANG, A/B confirmed — single-count SenderPipe cannot represent this kernel)

The 1D height-sharded weights sender uses TWO DISTINCT counts in the raw block:
  * data + flag mcast target count = `weights_mcast_num_cores` = `total_num_cores - 1` (the FULL bounding-box
    rect — every core in the rect, including noop/inactive receiver cores).
  * consumer-ready ack wait count = `weights_mcast_num_dests` = `total_active_num_cores - 1` (only the ACTIVE
    receiver cores ack; noop receivers `if(noop) return;` before `receive()`, so they never `consumer_ready.up`).
For this parametrization `total_active_num_cores < total_num_cores` (proven by the A/B below — the hang site
flips, which is only possible if the two counts differ).

The v7 `SenderPipe` exposes a SINGLE `NUM_ACTIVE_RECEIVER_CORES` template param used for BOTH the
non-posted `async_write_multicast` dest count (drained by the kernel's trailing `noc.async_write_barrier()`)
AND the `consumer_ready.wait(N)` ack count. There is no value of N that satisfies both:

  * N = total_active-1: sender BRISC HANGS at `NWBW` (`noc_async_write_barrier`) — the non-posted data mcast
    physically covers all `total_num_cores-1` rect cores and expects that many write-acks, but N under-counts
    them, so the barrier never drains.
  * N = total_num_cores-1: sender BRISC HANGS at `NSW` (`consumer_ready.wait`) — waits for total-1 acks but
    only total_active-1 cores ack. Receivers stall at `GW` (data-ready flag never sent).

This is exactly the audit blocker: conv.md headline #2 / normalization.md blocker #6 — "F3 = INCLUDE vs
EXCLUDE + the num_dests off-by-one ... Pipe must encode this per F3 to avoid off-by-one." Closing it needs a
helper capability change (a separate mcast-dest count vs ack count, or a `num_active != num_rect` mode). Per
SUBAGENT_CONVENTIONS, the helper is NOT modified here.

Note: the migrated 1D RECEIVER (committed d93798c) PASSED only because it was co-compiled with the OLD RAW
sender, which still carries the two-count split. Migrating the sender to the single-count helper breaks it.

## Disposition
- Kernel + the host factory CT-arg edit (conv2d_op_sharded_program_factory.cpp:1046-1048 count derivation)
  REVERTED → tree green, raw sender retained, receiver migration unaffected.
- Ledger: status=deferred, flags add `coverage-gap` (here: helper design limitation, distinct counts).
