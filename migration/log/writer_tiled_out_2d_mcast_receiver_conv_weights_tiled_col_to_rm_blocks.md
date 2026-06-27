# writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp (TIER 1 #8, receiver)

## Blocks migrated
Two receive blocks (weights ~176-186, bias ~199-210): set(INVALID) + up + wait(VALID)
-> `weights_pipe.receive()`. reserve_done/write_done split-reader channel left untouched.

## Pipe template args
`Pipe<>` = `<EXCLUDE_SRC, Flag, PRE_HANDSHAKE=true, LINK=true>`.
data_ready=receiver_sem, consumed=sender_sem. McastRect::single_core(sender_noc_x, sender_noc_y).

## Call-site diff
~7 lines (weights) + ~7 lines (bias) removed -> two receive() (+ ~7-line construction).

## Validation
conv BLOCK_SHARDED node (same as #7), run together: PASS, PCC=0.99994.

## Commit
d8aaa9e8e6e  "apply mcast_pipe to writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks"
