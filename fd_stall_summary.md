# Quasar Slow-Dispatch Prefetch/Dispatch Stall Summary

This document summarizes the layered hangs fixed in `QuasarSdPrefetchDispatchReproFixture.MinimalRelay32B_OneIter` (SD mode: `TT_METAL_SLOW_DISPATCH_MODE=1`). Prefetch (`cq_prefetch.cpp`, DM on core `{0,0}`) and dispatch (`cq_dispatch.cpp`) on the same worker never completed within 60s without watcher; bisect used phase markers in L1 and host post-mortem reads.

**Repro command:**

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_SD_REPRO_TIMEOUT_SEC=60 \
  ./build_Debug/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher \
  --gtest_filter=QuasarSdPrefetchDispatchReproFixture.MinimalRelay32B_OneIter
```

**Architecture constraint (Quasar):** NOC reads/writes use **cached** L1 offsets. The DM CPU must use the **uncached alias** (`+ MEM_L1_UNCACHED_BASE`) for polling host-written TL1 and for semaphore/cmd decode reads. Host NOC reads physical TL1 directly (no host-side cache coherency issue).

---

## Stall progression (bisect)

| Observed phase | Interpretation |
|----------------|----------------|
| `FETCH_Q_HQW` (0x50490003) | Device poll saw empty FetchQ; host saw non-zero entry |
| `FETCH_Q_RETIRE` … `RETIRE_POST_SYNC` (0x50490020–25) | Hang inside retire sub-step (before full chain existed) |
| `RETIRE_FENCE_ADVANCED` / `RETIRE_BACK` (0x50490028 / 0x50490027) | Retire finished; hang before `fetch_q_get_cmds` return |
| `AFTER_FETCH_Q` (0x50490001) | Fetch returned; hang in redundant post-fetch sync or decode |
| *(none — test passes)* | All fixes applied; retire path has no phase breadcrumbs |

---

## Fix 1: Dispatch CB credit semaphore bootstrap

**Problem:** On Quasar, `CreateSemaphore` initializes to 0. Host `cluster.write_core` primes **cached** L1 only. `CBWriter::acquire_pages` in the prefetch relay path reads the **uncached** alias and saw 0 credits → dispatch never consumed.

**Symptom:** Post-mortem showed `credit=128` on host read but prefetch stuck in `RELAY_INLINE` at `acquire_pages` (when reached).

**Code changed:** `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp`

```229:233:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
FORCE_INLINE void quasar_bootstrap_prefetch_downstream_credits() {
    if constexpr (is_d_variant != 0) {
        Semaphore<fd_core_type>(my_downstream_cb_sem_id).set(downstream_cb_pages);
    }
}
```

Called at kernel entry before the HD loop:

```3102:3102:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
    quasar_bootstrap_prefetch_downstream_credits();
```

---

## Fix 2: FetchQ poll visibility (uncached alias)

**Problem:** PrefetchQ entries written by host via NOC land in TL1 but do not invalidate the DM L1 D$/L2. Polling via cached port + `invalidate_l1_cache()` left the device spinning in HQW while host saw valid entries.

**Symptom:** Phase `FETCH_Q_HQW` with aux=0; host FetchQ slot non-zero.

**Code changed:**

1. **`prefetch_q_rd_ptr` uses uncached base** — `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp` lines **686–699**:

```686:699:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
    // Read via the uncached L1 alias on Quasar so polled host updates bypass the L1/L2 cache
    // hierarchy. invalidate_l1_cache() alone doesn't flush L2, so cached reads of TL1 locations
    // written by an external NOC source (host fetch-queue write) can otherwise stall forever.
    ...
    static constinit volatile tt_l1_ptr prefetch_q_entry_type* prefetch_q_rd_ptr = prefetch_q_uncached_base_;
```

2. **`fetchq_poll_load`** — direct uncached load + fence — `tt_metal/impl/dispatch/kernels/cq_common.hpp` lines **113–123**:

```113:123:tt_metal/impl/dispatch/kernels/cq_common.hpp
// Quasar: prefetch_q_rd_ptr is the uncached alias; load TL1 directly so host NOC-filled entries
// are visible without stale L1 D$/L2 on the cached port (see quasar-noc-l1.mdc).
template <typename T>
FORCE_INLINE T fetchq_poll_load(volatile T tt_l1_ptr* rd_ptr) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    asm volatile("fence" ::: "memory");
    return *rd_ptr;
#else
    return *rd_ptr;
#endif
}
```

Used throughout `fetch_q_get_cmds` (e.g. lines **768**, **882**, **966**, **968**, **973** in `cq_prefetch.cpp`).

3. **Supporting helpers** (pre-existing, used by above) — `cq_common.hpp` lines **64–87** (`l1_uncached_addr`, `uncached_l1_ptr`).

---

## Fix 3: Cmddat retire sync chain after NOC fetch

**Problem:** NOC async reads fill cmddat at **cached** offsets. DM command decode uses **uncached** `uncached_l1_ptr`. Without tr_ack ordering, NOC barrier, and cmddat invalidate before decode, the device saw zeros or stale data.

**Symptom:** Valid `cmddat[0..3]` on host NOC read but device hung before/during cmd decode; repro kernels (`quasar_prefetch_retire_repro`, `quasar_tr_ack_tl1_visibility`) isolated each sub-step.

**Code changed:** `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp` lines **173–214**

| Helper | Lines | Role |
|--------|-------|------|
| `cmddat_invalidate_after_noc_read` | 177–184 | L1 D$ + L2 invalidate per 64B cmddat line |
| `quasar_cmddat_post_fetch_sync` | 186–192 | Fence only (`quasar_cmddat_post_sync_iters = 0`, line **174**) |
| `quasar_cmddat_pre_retire_barrier_sync` | 194–198 | Fence + `tr_ack_trid` read + fence (no L2 flush loop) |
| `quasar_cmddat_retire_fetch_read` | 203–214 | Full production chain (see below) |

Production retire chain (no phase markers — see Fix 7):

```203:214:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
FORCE_INLINE void quasar_cmddat_retire_fetch_read(uint32_t trid, uintptr_t read_start, uint32_t size_bytes) {
    while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
    }

    quasar_cmddat_pre_retire_barrier_sync(trid);

    noc_async_read_barrier_with_trid(trid);

    cmddat_invalidate_after_noc_read(read_start, size_bytes);

    quasar_cmddat_post_fetch_sync();
}
```

Minimal sync helpers:

```174:198:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
static constexpr uint32_t quasar_cmddat_post_sync_iters = 0;
...
FORCE_INLINE void quasar_cmddat_post_fetch_sync() {
    asm volatile("fence" ::: "memory");
    for (volatile int delay = 0; delay < static_cast<int>(quasar_cmddat_post_sync_iters); ++delay) {
        (void)delay;
    }
    asm volatile("fence" ::: "memory");
}

FORCE_INLINE void quasar_cmddat_pre_retire_barrier_sync(uint32_t trid) {
    asm volatile("fence" ::: "memory");
    (void)__builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(trid);
    asm volatile("fence" ::: "memory");
}
```

Invoked from `fetch_q_get_cmds` retire path at line **924**.

**Bisect note:** Early bring-up used a 6-line `flush_l2_cache_line(scratch_db_base + …)` loop in `quasar_cmddat_pre_retire_barrier_sync` and a 32-iteration delay in `quasar_cmddat_post_fetch_sync`. Those were empirical RTL workarounds (delay / L2 activity between tr_ack clear and TL1 visibility). Once the full chain, FetchQ uncached poll, early return, and phase-marker removal were in place, both were **removed entirely** — the scratch_db flush loop deleted, `quasar_cmddat_post_sync_iters` set to **0** — and `MinimalRelay32B_OneIter` still passes. Load-bearing pieces are: tr_ack spin, pre-barrier `tr_ack_trid` + fences, `noc_async_read_barrier_with_trid`, and cmddat invalidate.

**Isolated repro:** `tests/.../kernels/quasar_prefetch_retire_repro.cpp` still uses optional pre-retire L2 flush + delay for waveform bisect (`TT_METAL_QUASAR_PREFETCH_REPRO_PRE_RETIRE`); production kernel does not.

---

## Fix 4: Early return from `fetch_q_get_cmds` after commit

**Problem:** After retiring the oldest in-flight read with committed bytes, continuing the fetch loop with a consumed FetchQ cursor and other reads still in-flight hung Quasar RTL.

**Symptom:** `RETIRE_FENCE_ADVANCED` / valid cmddat but no progress into `process_cmd`; FetchQ showed 2/3 entries consumed.

**Code changed:** `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp` lines **898–957**

- Snapshot inflight metadata **before** retire (lines **898–902**) so post-retire cache ops cannot corrupt stack reads.
- After fence advance and inflight decrement, **return immediately** when `committed_bytes != 0` (lines **953–957**); handle `STALL_AFTER` via `stall_state` without a separate ASSERT/return path.

```953:957:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
                // Committed data is available; return immediately (do not continue the loop with a
                // consumed FetchQ cursor and other reads still in-flight — Quasar RTL hangs there).
                if (committed_bytes != 0U) {
                    if (retire_flags == InflightFlags::STALL_AFTER) {
                        stall_state = StallState::STALLED;
                    }
                    return;
                }
```

---

## Fix 5: Remove L2 flush from phase marker publish

**Problem:** `prefetch_publish_phase` originally called `tl1_publish_flush` on the marker line. After ~8–9 L2 flushes during one retire, Quasar RTL stalled even on a single additional flush/write to the same line.

**Symptom:** Stall at `RETIRE_POST_SYNC` or immediately after retire breadcrumb publishes.

**Code changed:** `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp` lines **89–98**

```89:98:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
FORCE_INLINE void prefetch_publish_phase(PrefetchPhaseMarker phase, uint32_t aux = 0) {
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    if constexpr (prefetch_phase_marker_l1 != 0) {
        // Host NOC reads physical TL1; uncached stores are sufficient for post-mortem probes.
        // flush_l2 on this line during retire sub-phases stalls Quasar RTL (even once per publish).
        volatile uint32_t* marker = uncached_l1_ptr<uint32_t>(prefetch_phase_marker_l1);
        marker[0] = static_cast<uint32_t>(phase);
        marker[1] = aux;
        asm volatile("fence" ::: "memory");
    }
```

`tl1_publish_flush` remains in use for FetchQ consume (`read_from_pcie`, lines **637–639**) where host visibility of rd_ptr updates is required.

---

## Fix 6: Single post-fetch sync (retire only on Quasar DM)

**Problem:** Retire chain already runs `quasar_cmddat_post_fetch_sync()`. Duplicate sync in `kernel_main_hd` and `process_cmd` triggered RTL stalls (`AFTER_FETCH_Q` without reaching decode).

**Code changed:**

1. **`kernel_main_hd`** — removed redundant sync and duplicate cmd decode; call `process_cmd` directly after `AFTER_FETCH_Q` — lines **3059–3065**:

```3059:3065:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
        fetch_q_get_cmds<preamble_size>(fence, cmd_ptr, pcie_read_ptr);
        prefetch_publish_phase(PrefetchPhaseMarker::AFTER_FETCH_Q, static_cast<uint32_t>(cmd_ptr));

        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);

        uint32_t stride;
        done = process_cmd<false, false>(cmd_ptr, downstream_data_ptr, stride, l1_cache, exec_buf_state);
```

2. **`process_cmd`** — skip entry `post_fetch_sync` on Quasar DM (retire already synced); keep phase markers for decode bisect — lines **2298–2306**:

```2298:2306:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
#if !(defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM))
    // Fetch retire chain already invalidated + post_fetch_sync'd committed cmddat on Quasar DM.
    quasar_cmddat_post_fetch_sync();
#endif
    volatile CQPrefetchCmd tt_l1_ptr* cmd = uncached_l1_ptr<CQPrefetchCmd>(cmd_ptr);
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    prefetch_publish_phase(PrefetchPhaseMarker::PRE_CMD_DECODE, static_cast<uint32_t>(cmd->base.cmd_id));
    prefetch_publish_phase(PrefetchPhaseMarker::PROCESS_CMD_ENTER, static_cast<uint32_t>(cmd->base.cmd_id));
#endif
```

---

## Fix 7: Remove retire-path phase marker publishes (final unblock)

**Problem:** Even uncached phase-marker writes **after** L2 flush/invalidate in the retire hot path (~7 publishes per retire) caused Quasar RTL to stop forward progress. Last observed phase before fix: `RETIRE_BACK` (0x50490027).

**Symptom:** Retire functionally complete (host cmddat valid); 60s timeout with phase stuck at retire breadcrumb.

**Code changed:**

- Removed all `prefetch_publish_phase` calls from `quasar_cmddat_retire_fetch_read` (formerly at lines that published `FETCH_Q_RETIRE`, `RETIRE_TR_ACK`, `RETIRE_PRE_RETIRE`, `RETIRE_BARRIER`, `RETIRE_INVALIDATE`, `RETIRE_POST_SYNC`). Production function is now lines **203–214** (see Fix 3).
- Removed post-retire breadcrumbs (`RETIRE_BACK`, `RETIRE_FENCE_ADVANCED`, `FETCH_Q_RETURN`, `RETIRE_ZERO_COMMIT`) from the commit path in `fetch_q_get_cmds` (lines **924–957**); plain `return` on commit.

**Phase enum values retained** for post-mortem decoder and repro kernels — `cq_prefetch.cpp` lines **66–87**.

**Phased retire bisect** remains in `tests/.../kernels/quasar_prefetch_retire_repro.cpp` (uses `publish_phase` + L2 flush on result buffer, not production marker line).

---

## Test / diagnostics changes

**Post-mortem on SD repro timeout** — `tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/test_prefetcher.cpp`:

| Item | Lines |
|------|-------|
| Phase name decoder | 2911–2935 |
| Phase aux hints | 2937–2956 |
| `dump_quasar_sd_stall_post_mortem` | 2959–3186 |
| FetchQ prelaunch snapshot | 2849–2908 |
| Phase marker L1 addr (last 64B of scratch_db) | 3574–3583 |
| Timeout thread + post-mortem call | 3725–3753 |

**Removed:** `tt_metal/impl/dispatch/kernels/cq_prefetch_debug.hpp` (misleading debug tooling; per debug session).

---

## Phase marker reference (prefetch DM)

Defined in `cq_prefetch.cpp` lines **66–87**. Host reads marker at `PREFETCH_PHASE_MARKER_ADDR` (test: scratch_db last 64B).

| Phase | Value | Typical aux |
|-------|-------|-------------|
| `HD_LOOP` | 0x50490000 | cmd_ptr |
| `AFTER_FETCH_Q` | 0x50490001 | cmd_ptr |
| `PROCESS_CMD_ENTER` | 0x50490002 | cmd_id |
| `FETCH_Q_HQW` | 0x50490003 | polled FetchQ entry |
| `PRE_CMD_DECODE` | 0x50490004 | cmd_id |
| `POST_FETCH_SYNC_DONE` | 0x50490005 | committed bytes |
| `RELAY_INLINE_*` | 0x50490010–13 | relay sub-steps |
| `FETCH_Q_RETIRE` … `RETIRE_POST_SYNC` | 0x50490020–25 | *(repro only in production kernel)* |
| `FETCH_Q_RETURN` | 0x50490026 | inflight_count |
| `RETIRE_BACK` … `RETIRE_ZERO_COMMIT` | 0x50490027–29 | *(removed from production retire path)* |

---

## Files touched (production fix)

| File | Summary |
|------|---------|
| `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp` | Bootstrap, retire chain, fetch early return, phase publish policy, `kernel_main_hd` / `process_cmd` flow |
| `tt_metal/impl/dispatch/kernels/cq_common.hpp` | `fetchq_poll_load`, uncached L1 helpers |
| `tests/.../dispatch/test_prefetcher.cpp` | Quasar SD repro fixture, post-mortem, phase decoder |
| `tests/.../kernels/quasar_prefetch_retire_repro.cpp` | Isolated retire-phase repro (unchanged pattern for RTL) |

---

## Notes

- `rd_ptr_addr` is written **before** `++prefetch_q_rd_ptr` → post-mortem `entries_consumed=2` with 3 host entries can be expected after partial consume, not necessarily “stuck before first consume.”
- Kernel JIT recompiles from `cq_prefetch.cpp` at launch; rebuild `test_prefetcher` for host-side test changes only.
- With all fixes, `MinimalRelay32B_OneIter` completes without watcher.
- Production retire sync is intentionally minimal: no scratch_db L2 flush loop, no post-retire delay iterations. Repro kernels may still use flush/delay knobs for RTL bisect.
