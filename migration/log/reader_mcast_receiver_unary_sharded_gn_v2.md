# reader_mcast_receiver_unary_sharded_gn_v2.cpp (TIER 1 #9, receiver)

## Block migrated
Reduce-mcast receive (lines ~75-81): `reduce_sender_sem.set(INVALID)` + `reduce_receiver_sem.up(noc, sx, sy, 1)`
+ `reduce_sender_sem.wait(VALID)` -> `reduce_pipe.receive()`. The wrapping CB ops
(cb_ex_partial.wait/pop, cb_ex_global.reserve/push) stay.

## Pipe template args
`Pipe<>` = `<EXCLUDE_SRC, Flag, PRE_HANDSHAKE=true, LINK=true>`.
NAMING FLIP vs matmul: data_ready = reduce_sender_sem (S->R flag, cleared+waited here);
consumed = reduce_receiver_sem (R->S counter, up'd here). McastRect::single_core(mcast_sender_noc_x/y).
Reordered reserve_back ahead of receive() (clear moved from before-ack to end-of-receive, H11) —
first iter safe (cell inits 0).

## Call-site diff
~3 lines (set/up/wait) removed -> 1 `reduce_pipe.receive()` (+ ~7-line construction).

## Validation
gn_v2 8x4 legacy node (C=1280 H16 W16 num_groups=32): SAFE_PYTEST_RESULT: PASS (1 passed in 4.72s).

## Commit
b07ce755ca8  "apply mcast_pipe to reader_mcast_receiver_unary_sharded_gn_v2" (clang-format re-stage)
