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

## Empirical investigation (2026-05-24)

Ran Austin's profile test (`test_flash_mla_decode[2047]`) on `bh-lb-26` (Blackhole, 1350 MHz) under TT-Metal device profiler. All measurements below are the `DeviceZoneScopedN` durations on **TRISC_2 (PACK)** — the thread actually running the SFPU code; on TRISC_0/TRISC_1 every zone is a tight ~25–35 cycles regardless of fix.

### What was tried

Three candidate fixes:

- **Fix A — STALLWAIT barrier:** keep the sum init where it was, follow it with `TTI_STALLWAIT(STALL_SFPU, WAIT_SFPU)` before the zone.
- **Fix B — hoist both inits to the top of `compute_sdpa_chunk`:** record both replay slots at `sdpa.h:265–266` before any reduce executes.
- **Fix C — split the sum init out: max init at top, sum init relocated to just after the `WAIT_SFPU` semaphore_get that confirms `reduce_max` has retired.** Both reduces still execute the correct instructions; the sum record now has ~hundreds of cycles of unrelated PACK work (bcast_sub → bcast_mul → 128× fast_approx_exp → mm2) before its replay.

### Fix B is semantically broken (and Fix A doesn't work architecturally)

**Fix B silently corrupts `reduce_max`.** Both `_init_sdpa_reduce_max_row_8x32_replay_buffers_()` and `_init_sdpa_reduce_sum_row_8x32_replay_buffers_()` call `load_replay_buf<NoExec>(sdpa_reduce_row::replay_start = 16, 16, …)` — they write to the *same* replay-buffer slot range `[16..31]`. Recording sum-init right after max-init overwrites max's recording, and the subsequent `reduce_max` then replays SUM instructions (SFPADD) instead of MAX instructions (SFPSWAP).

This wasn't caught by the test because SDPA uses the max only as a numerical-stability shift: `softmax(score) = exp(score - shift) / Σ exp(score - shift)`. Any uniform shift cancels in the final ratio, so `reduce_max` could be replaced by any sample-wise reduction and the SDPA output is bit-identical. The PCC stayed at `0.9987931834499761` (same as baseline) under fix B — coincidence, not correctness. Any non-SDPA caller of `_init_sdpa_reduce_max_row_8x32_replay_buffers_()` would break.

**Fix A is architecturally impossible.** Tested it empirically (4 runs, n=5120 on TRISC_2): the `reduce_sum` distribution stayed effectively identical to baseline (14.3% in `[500..800)`, max 723 c). The BH ISA docs (`BlackholeA0/.../STALLWAIT.md` and `WormholeB0/.../REPLAY.md`) explain why: the Replay Expander sits in the Tensix frontend, *before* the Wait Gate. STALLWAIT is evaluated *at* the Wait Gate by the Sync Unit. The 13 condition bits C0–C12 all describe backend execution-unit state — **none observes Replay-Expander state**. The BH STALLWAIT page literally says `REPLAY` "never reaches the Wait Gate." So no `wait_res` bit can drain an in-flight record. Recorded instructions in `Exec=false, Record=true` mode never reach any backend unit either; they're ingested by the expander only. The variable latency is **frontend FIFO backpressure**: the expander stalls PACK's frontend FIFO while ingesting (≥ N cycles, "longer if there are gaps in the incoming instruction stream"), and the downstream replay-execute can't drain until the recording finishes.

### Fix C — semantically correct, mitigates the artifact but doesn't fully eliminate it

| zone | run | n | min | median | mean | p90 | p99 | max | std |
|---|---|---|---|---|---|---|---|---|---|
| `reduce_max` | baseline | 1280 | 184 c | 192 c | 188.6 c | 192 c | 193 c | **200 c (148 ns)** | ~4 c |
| `reduce_max` | fix C    | 5120 | 184 c | 192 c | 188.6 c | 192 c | 193 c | **194 c (144 ns)** | 3.6 c |
| `reduce_sum` | baseline | 1280 | 150 c | 285 c | 270.3 c | 670 c | 716 c | **719 c (533 ns)** | ~160 c |
| `reduce_sum` | fix C    | 5120 | 156 c | 168 c | 196.5 c | 186 c | 734 c | **738 c (547 ns)** | 113.1 c |

`reduce_max` is unchanged (sanity check — we didn't touch its init).

`reduce_sum` is dramatically improved at the **frequency** level: median 285 c → 168 c, p90 670 c → 186 c. The distribution moves from "wide bimodal" to "mostly tight, with rare outliers":

```
reduce_sum BASELINE  n=1280
  [ 150.. 200)    639  (49.9%) ########################################
  [ 200.. 300)    430  (33.6%) ##########################
  [ 300.. 400)     52  ( 4.1%) ###
  [ 400.. 500)     15  ( 1.2%)
  [ 500.. 600)      6  ( 0.5%)
  [ 600.. 800)    138  (10.8%) ########

reduce_sum FIX C  n=5120
  [ 150.. 170)   3475 (67.9%)  ########################################
  [ 170.. 200)   1195 (23.3%)  #############
  [ 200.. 250)     11 ( 0.2%)
  [ 250.. 300)      3 ( 0.1%)
  [ 300.. 400)    225 ( 4.4%)  ##
  [ 400.. 500)      1 ( 0.0%)
  [ 500.. 800)    210 ( 4.1%)  ##
```

### Residual outliers in fix C

About 8.6% of `reduce_sum` events land in `[300..800)` with the **same max** as baseline (~738 c vs 719 c). Per-core: every one of the 24 compute cores shows the same pattern — tight median ~163–179 c, but each core hits ~734 c on roughly 10% of events. It's not a few specific cores; it's uniform.

The fact that the *frequency* dropped by ~3× but the *magnitude* is unchanged tells us the dominant baseline mechanism (the record-then-immediate-replay race) is mitigated, but a *separate* slower-frequency mechanism with the same ~720 c characteristic remains. Likely candidates, none verified:

- Cross-thread SFPU contention with the 128-iteration `fast_approx_exp` loop running just before `reduce_sum`.
- `init_fast_approx_exp_constants` recording into a different replay slot whose own ingestion overlaps reduce_sum's replay.
- mm2's packer activity interfering with the sum replay through some shared resource not yet identified.

Investigating the residual is out of scope here — the dominant signal from #43549 (the 11% tail at 600–720 c) is resolved and `reduce_max` is left untouched.

### The fix in code

```diff
     PACK((ckernel::sfpu::_init_sdpa_reduce_max_row_8x32_replay_buffers_()));
     sdpa_custom_mm_block_init_short<…>(cb_q, cb_k, cb_out, chunk_size);
     …
     {
         DeviceZoneScopedN("reduce_max");
         PACK((llk_math_sfpu_sdpa_reduce_max_row<…>(…)));
         tensix_sync();
     }
     PACK((t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU)));
+    // Re-record the shared SFPU replay slot for sum now that max has retired.
+    PACK((ckernel::sfpu::_init_sdpa_reduce_sum_row_8x32_replay_buffers_()));
     // Bcast Sub (FPU) …
     …
     // Reduce Sum (SFPU)
-    PACK((ckernel::sfpu::_init_sdpa_reduce_sum_row_8x32_replay_buffers_()));
     tensix_sync();
     {
         DeviceZoneScopedN("reduce_sum");
         PACK((llk_math_sfpu_sdpa_reduce_sum_row<…>(…)));
         tensix_sync();
     }
```

Raw CSVs and analysis scripts that produced these tables live under `models/demos/deepseek_v3_b1/docs/reduce_zone_artifact_data/`.

Note on sample sizes: the on-device profiler buffer fills before the test's 10 iterations complete; per-run capture varies from ~1 to ~5 iterations depending on JIT cache warmth. Aggregating 4 runs gives n=5120 on TRISC_2 (~71 events per core), enough to make the distribution shape stable.
