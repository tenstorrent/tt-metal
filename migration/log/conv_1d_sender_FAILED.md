# reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp (TIER 1 #5, sender) — FAILED / quarantined

## Reason: API mismatch (count divergence)
The 1D conv weights mcast uses TWO different counts (program factory
conv2d_op_sharded_program_factory.cpp:1330):
  mcast_num_dests = total_active_num_cores - 1   (handshake: how many receivers `up`)
  mcast_num_cores = total_num_cores       - 1   (data/flag mcast geometry: full rectangle)
These DIFFER when the receiver rectangle contains inactive (no-op) cores — true for the validated
256x256 HEIGHT_SHARDED case.

`dataflow_kernel_lib::Pipe` exposes a single `McastRect::num_dests` field that drives BOTH the
pre-handshake `consumed_.wait(num_dests)` AND the data/flag mcast `num_dests` param (INV8). It cannot
carry two distinct counts. Setting it to num_cores makes the sender over-wait (only active receivers
ack) -> HANG (observed: dispatch timeout on the HEIGHT_SHARDED node). Setting it to num_dests would
under-account the non-posted mcast ACKs.

## Action
Attempted migration (include + Pipe + 2 send()), validated, HUNG, then `git restore`d. Tree green.
NOT committed. Receiver counterpart (#6) migrated independently (count-independent) — see its log.

## Validation attempt
conv HEIGHT_SHARDED bf16/bf16 fp32_accum=True node -> HANG (SAFE_PYTEST_RESULT: HANG).

## Commit: none (reverted).
