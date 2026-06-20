# reader_argmax_interleaved_multicore.cpp — DEFERRED (design gap)

## Verdict: deferred (helper design gap — NOT migrated, file untouched)

The reduce core's mcast block (lines 362-374) is a FLAG-ONLY broadcast of `start_sem` to two
rectangles, paired with a per-page reset `done_sem` counter. Three independent design gaps, any one
fatal:

1. **Arbitrary-value monotone set (dominant).** The signal is `start_sem.set(k+1)` then
   `set_multicast` of that cell — broadcasting an arbitrary increasing value `k+1`, and the workers
   `start_sem.wait(k+1)` (exact wait on that specific value). The v7 helper Flag path broadcasts a
   FIXED `VALID` (`data_ready_.set(VALID)`); the Counter path uses `inc_multicast` (+1 atomic) with
   `wait_min`. Neither expresses "set_multicast an arbitrary value `k+1` + exact `wait(k+1)`". There is
   no "set_multicast(value)" verb. This is the value-carrying-flag gap (cf. moe_gpt deferral).

2. **Two rects, different loopback modes in one logical send.** rect0 = `MCAST_INCL_SRC` (count
   num_cores0, reduce core in box), rect1 = `DEFAULT`/EXCLUDE_SRC (count num_cores1), one shared
   `async_write_barrier`. `send_signal()` is always EXCLUDE_SRC; INCLUDE_SRC on a pure flag is
   inexpressible (loopback is inferred only for the data `send()` path). Per the known gaps: per-rect
   modes in one send.

3. **Mixed counters simultaneously.** monotone `start_sem` (no reset) AND reset-per-page `done_sem`
   (`up(...,1)` + `wait(num_cores)` + `set(0)`) live in the same kernel. F2 is per-slot, not per-pipe;
   a single SenderPipe count/signal can't serve both.

Counts (num_cores0/1) ARE compile-time, so the per-rect count is not itself the blocker — gap (1) is.

## Action: no edit, ledger status=deferred, flag design-gap.
