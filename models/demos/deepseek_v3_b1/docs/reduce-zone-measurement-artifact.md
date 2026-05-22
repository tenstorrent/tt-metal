# `reduce_sum` Profile Zone Variance — Likely Measurement Artifact (#43549)

> Analysis of the variable-duration `reduce_sum` zone Austin observed on `aho/sdpa-ops2`. The argument: the two reduce zones in `compute_sdpa_chunk` were instrumented **differently**, and that asymmetry — not anything intrinsic to the second reduce — is almost certainly the cause of the 100–500 ns spread.

---

## What I missed in the first pass

Compare the two zones' pre-zone setup in Austin's commit `2626e81a680` ("Profile test"):

**reduce_max** (the consistent ~140 ns one):

```cpp
// Profile testing, manually wait/clear the semaphores before reduce
for (uint32_t i = 0; i < chunk_size - 1; i++) {
    PACK((t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU)));
    PACK((t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU)));
}
PACK((t6_semaphore_wait_on_zero<p_stall::STALL_SFPU>(semaphore::FPU_SFPU)));
tensix_sync();
{
    DeviceZoneScopedN("reduce_max");
    PACK((... reduce_max with skip_signalling=true ...));
    tensix_sync();
}
```

Note he had to add `skip_signalling=true` here (it was `false` in the original) and lift `chunk_size` semaphore waits *out* of the reduce — each of those waits is `STALL_SFPU`, which is the actual stall on SFPU's pipe being empty. So by the time the `tensix_sync()` runs at zone entry, both the FIFO is drained AND the SFPU has been forced idle by the chain of STALL_SFPU waits.

**reduce_sum** (the variable 100–500 ns one):

```cpp
PACK((ckernel::sfpu::_init_sdpa_reduce_sum_row_8x32_replay_buffers_()));
tensix_sync();
{
    DeviceZoneScopedN("reduce_sum");
    PACK((... reduce_sum with skip_signalling=true ...));
    tensix_sync();
}
```

No `STALL_SFPU` waits. Just one `tensix_sync()`. And — critically — the replay buffer is loaded literally one instruction earlier.

## Why this matters

`_init_sdpa_reduce_sum_row_8x32_replay_buffers_` expands (via `load_replay_buf<NoExec>` → `lltt::record`) into one `__builtin_rvtt_ttreplay(start, 16, /*Exec=*/false, /*Record=*/true)` followed by the 16 SFPU instructions to record. Those 16 instructions go through the SFPU frontend in record-only mode. The `tensix_sync()` immediately after only blocks until TRISC2's instruction-issue FIFO is empty — *"some instructions may still be in flight"*. The SFPU's recording state can still be settling.

Then inside the zone, `_calculate_sdpa_reduce_row_8x32_` does:

```cpp
TT_SETC16(MATH_Offset, ...);
TTI_SETRWC(...);
// (no semaphore wait — skip_signalling=true)
TTI_SFPLOAD(LREG0, 0, ZERO_ADDR_MOD, 0);
TTI_SFPLOAD(LREG2, 0, ZERO_ADDR_MOD, 4);
lltt::replay(replay_start + 4, 12);   // ← uses the buffer just recorded
...
```

The `lltt::replay` issues a TTI_REPLAY(`Exec=false, Record=false`) that consumes from the buffer slot that may still be finalizing from the immediately-prior record. Whatever variable-latency interaction the SFPU has with that very-recently-recorded buffer slot ends up inside the measured zone. The reduce_max path, by contrast, recorded its buffer way back at `sdpa.h:265` — at the *top* of `compute_sdpa_chunk`, before MM1, the explicit semaphore drains, and a tensix_sync — so by the time reduce_max actually replays, the SFPU has had hundreds of cycles to fully retire the record.

The 100 ns lower bound is plausibly the "everything happens to be settled" case; 500 ns is "recording was still in flight at zone entry and the replay had to wait." Cross-core variance follows because each core's preceding SFPU work in the chunk (the `if (!first_chunk)` branch, fast_approx_exp loop, MM2 granularity-paced semaphore arrival) leaves the SFPU in different micro-states when the record-then-immediately-replay race resolves.

## Why swapping sum→max in slot 2 doesn't help

If Austin replaces `reduce_sum` with `reduce_max` in the second slot but **leaves the upstream `_init_sdpa_reduce_sum_row_8x32_replay_buffers_` (or its `_max_` counterpart) right before the tensix_sync**, the record-then-immediately-replay pattern is identical. The opcode in the buffer doesn't change the recording-latency variability.

## What to test to confirm

The cheapest A/B is to make the second zone use the **same** kind of pre-zone setup as the first:

```cpp
PACK((ckernel::sfpu::_init_sdpa_reduce_sum_row_8x32_replay_buffers_()));
// Force SFPU quiescence before measuring
PACK((TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK)));   // or equivalent
tensix_sync();
{
    DeviceZoneScopedN("reduce_sum");
    PACK((... reduce_sum ...));
    tensix_sync();
}
```

If the spread collapses to a tight band like reduce_max's ~140 ns, the hypothesis is confirmed and this is purely a measurement artifact: Austin's `tensix_sync()` is "FIFO empty" not "SFPU idle," and the two zones were not set up to start from the same SFPU state.

An even cleaner control: move the `_init_sdpa_reduce_sum_row_8x32_replay_buffers_()` call up to the top of `compute_sdpa_chunk` (right next to the `_init_sdpa_reduce_max_row_8x32_replay_buffers_()` at `sdpa.h:265`). Then the record-to-replay distance for sum matches max's, and there's no possibility of an in-flight record being measured.

## Where this leaves #43549

If the spread does collapse with either of the above, the answer to Austin's question is: **the second-reduce zone wasn't actually measuring reduce_sum — it was measuring (a) the tail of the replay-buffer record that happened one instruction earlier and (b) the small overhead of `tensix_sync()` not being an SFPU-quiescence barrier**. There may still be a real timing question about why reduce_sum sometimes takes 500 ns vs 100 ns of *kernel time*, but the right reframing is "how long does the SFPU take to finish recording a 16-entry replay buffer, with variable cross-thread arbitration?", not "why does the reduce itself vary?"

If the spread *doesn't* collapse, then we'd need to actually open the issue — but the artifact explanation should be ruled out first, because it's by far the most parsimonious read of what Austin's diff actually does.
