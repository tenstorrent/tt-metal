# reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp (TIER 1 #6, receiver)

## Blocks migrated
Two receive blocks (weights ~178-185, bias ~196-202): `receiver_sem.set(INVALID)` +
`sender_sem.up(noc, sx, sy, 1)` + `receiver_sem.wait(VALID)` -> `weights_pipe.receive()`.

## Pipe template args
`Pipe<>` = `<EXCLUDE_SRC, Flag, PRE_HANDSHAKE=true, LINK=true>`.
data_ready=receiver_sem, consumed=sender_sem. McastRect::single_core(sender_noc_x, sender_noc_y).
Receiver is count-independent (acks once via up) so it fits even though the matching 1D SENDER (#5)
does not (count divergence -> FAILED). Raw sender + Pipe receiver are wire-compatible.

## Call-site diff
~7 lines (weights) + ~7 lines (bias) removed -> two `weights_pipe.receive()` (+ ~7-line construction).

## Validation
conv HEIGHT_SHARDED bf16/bf16 fp32_accum=True node: PASS, PCC=0.9999993 (1 passed in 2.55s).
(Sender #5 still raw.)

## Commit
a18af0f1d42  "apply mcast_pipe to reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks"
