# reader_final_topk.cpp (TIER 1 #10, "receiver" -> actually readiness-broadcast sender)

## Block migrated (partial — control half only)
The aggregator's readiness broadcast (orig lines ~39-47): `sender_sem.set(INVALID)` +
`receiver_sem.set(VALID)` + `receiver_sem.set_multicast<EXCLUDE_SRC>(...)` + `async_write_barrier`
-> kept `sender_sem.set(INVALID)`, replaced the rest with `ready_pipe.send_signal(VALID)`.

NOT a Pipe::receive() despite the "receiver" name: this core BROADCASTS readiness to a rect of
sender cores (R2 flag-only control), then waits a FAN-IN counter `sender_sem.wait(Wt_final)` for many
producers. The fan-in counter is a multi-producer channel the single-sender Pipe does not own (INV9)
— LEFT RAW. Only the broadcast maps (to send_signal).

## Pipe template args / fence note
`Pipe<>` = `<EXCLUDE_SRC, Flag, ...>`. data_ready=receiver_sem (the broadcast flag); consumed=sender_sem
(unused on this control path). McastRect{noc_start/end, num_dests}.
FENCE CHANGE: orig used async_write_barrier after the flag mcast; send_signal uses flush (Flag fence).
Validated PASS — the senders observe the readiness flag with flush, no barrier needed here.

## Call-site diff
~5 lines (set VALID + set_multicast + barrier) removed -> 1 `ready_pipe.send_signal(VALID)`
(+ ~8-line construction). The fan-in `sender_sem.wait(Wt_final)` and trailing barrier untouched.

## Validation
topk W=8192 k=50 BFLOAT16_B node: SAFE_PYTEST_RESULT: PASS (1 passed in 4.32s).

## Commit
da0ae515aa1  "apply mcast_pipe to reader_final_topk"
