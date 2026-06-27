# reader_bmm_tile_layout_in0_receiver.cpp (TIER 1 #2, receiver)

## Block migrated
Steady-state receive (orig lines 77-84): `receiver_sem.set(INVALID)` + `sender_sem.up(noc, sx, sy, 1)`
+ `receiver_sem.wait(VALID)` -> `in0_pipe.receive()`.

NOT migrated: the sparsity branch (lines 51-58): `set(INVALID)` + `up` + `wait_min(VALID)` + 3-state
value readout (`*ptr == VALID`). Value-carrying / wait_min, not plain receive(). Left raw.

## Pipe template args
`Pipe<>` = `<EXCLUDE_SRC, Flag, PRE_HANDSHAKE=true, LINK=true>` (MCAST/LINK irrelevant on receiver path).
- data_ready=receiver_sem (CT arg 5), consumed=sender_sem (CT arg 4).
- McastRect::single_core(sender_noc_x, sender_noc_y) -> the consumed ack target.
- PRE_HANDSHAKE=true: receive() does the up() ack each iter (matches orig sender_sem.up).

## Ordering note (H11)
Orig: clear(set INVALID) BEFORE ack(up) each iter. Pipe: ack first, wait, clear AFTER wait. Equivalent
in steady loop (Pipe iter N's end-clear == orig iter N+1's pre-clear). First iter safe: receiver flag
cell inits to 0 (INVALID), so the missing first pre-clear is a no-op. Test confirms.

## Call-site diff
~7 lines removed (set/up/wait) -> 1 `in0_pipe.receive()` (+ ~9-line construction before loop).

## Validation
nodeid: same matmul_1d node as in0_sender_padding. SAFE_PYTEST_RESULT: PASS (1 passed in 2.45s).

## Commit
c9ac10b2e21  "apply mcast_pipe to reader_bmm_tile_layout_in0_receiver"
