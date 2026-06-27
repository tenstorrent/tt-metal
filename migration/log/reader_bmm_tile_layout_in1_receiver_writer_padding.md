# reader_bmm_tile_layout_in1_receiver_writer_padding.cpp (TIER 1 #4, receiver; writer half untouched)

## Blocks migrated
Two receive blocks, canonical pattern (in1 lines 124-135, bias/in3 lines 142-153):
`receiver_sem.set(INVALID)` + `sender_sem.up(noc, sx, sy, 1)` + `receiver_sem.wait(VALID)`
-> `in1_pipe.receive()` (one Pipe, two call sites).

## Pipe template args
`Pipe<>` = `<EXCLUDE_SRC, Flag, PRE_HANDSHAKE=true, LINK=true>`.
data_ready=receiver_sem (CT 5), consumed=sender_sem (CT 4).
McastRect::single_core(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y).
H11 clear-before-ack: Pipe clears at end-of-receive; first iter safe (cell inits 0).

## Call-site diff
~7 lines (in1) + ~7 lines (in3) removed -> two `in1_pipe.receive()` (+ ~6-line construction).

## Validation
matmul_2d node (has_bias=True, transpose_mcast=True, grid (8,4)): SAFE_PYTEST_RESULT: PASS
(1 passed in 3.19s). Exercises both in1 and in3 receive paths (FUSE_BIAS dispatched).

## Commit
8a13611f3a3  "apply mcast_pipe to reader_bmm_tile_layout_in1_receiver_writer_padding"
