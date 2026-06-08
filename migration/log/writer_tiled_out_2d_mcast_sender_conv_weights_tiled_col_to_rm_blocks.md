# writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp (TIER 1 #7, sender)

## Blocks migrated
Two send blocks (weights ~250-280, bias ~316-346): `sender_sem.wait(num_dests)/set(0)` +
`async_write_multicast(...,linked=true)` + `receiver_sem.set_multicast(...)` -> `weights_pipe.send()`.
The split-reader `reserve_done_sem`/`write_done_sem` CB-handshake channel is a SEPARATE local
coordination (no NoC mcast) and was left untouched (per conv audit).

## Pipe template args
`Pipe<>` = `<EXCLUDE_SRC, Flag, PRE_HANDSHAKE=true, LINK=true>`.
data_ready=receiver_sem, consumed=sender_sem. McastRect num_dests=weights_mcast_num_cores.
In the 2D BLOCK_SHARDED path mcast_num_dests == mcast_num_cores (factory line 1287-1288 = num_cores_x-1
for both), so the single-num_dests Pipe is correct here (unlike the 1D path, #5). Pipe adds a flush
the orig omitted — a no-op superset of the same-VC FIFO ordering. src==dst==CB write_ptr.

## Call-site diff
~32 lines (weights) + ~32 lines (bias) removed -> two 4-line send() (+ ~14-line construction).

## Validation
conv BLOCK_SHARDED bf16/bf8 fp32_accum=True node: PASS, PCC=0.99994 (1 passed in 3.11s).

## Commit
07ec88c29a3  "apply mcast_pipe to writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks"
