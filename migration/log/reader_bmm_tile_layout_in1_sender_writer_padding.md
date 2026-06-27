# reader_bmm_tile_layout_in1_sender_writer_padding.cpp (TIER 1 #3, hybrid: reader-half sender)

## Blocks migrated (reader half; writer half untouched)
Two send blocks, identical canonical pattern (orig in1 lines 424-461, bias/in3 lines 557-592):
`sender_sem.wait(num_dests)/set(0)` + `async_write_multicast(...,linked=true)` + BH flush +
`receiver_sem.set_multicast(...)`. Both -> `in1_pipe.send(src,dst,bytes)` (one Pipe, two call sites).

## Pipe template args
`Pipe<>` = `<EXCLUDE_SRC, Flag, PRE_HANDSHAKE=true, LINK=true>`.
data_ready=receiver_sem (CT 11), consumed=sender_sem (CT 10). McastRect num_dests=in1_mcast_num_cores.
src==dst==in1_start_address / in3_start_address (static_cast<uint32_t> kept; orig stored as uint64).

## Call-site diff
~42 lines (in1) + ~41 lines (in3) removed -> two 4-line send() calls (+ ~11-line construction).

## Validation
matmul_1d node: PASS (in1 send path, has_bias=False -> bias send not exercised here).
Bias (in3) send path additionally confirmed PASS via the matmul_2d has_bias=True node (run under #4),
which dispatches this sender kernel with FUSE_BIAS.

## Commit
9370e0aed63  "apply mcast_pipe to reader_bmm_tile_layout_in1_sender_writer_padding"
