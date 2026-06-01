# Quasar Slow-Dispatch Prefetch/Dispatch Stall Summary

This document summarizes the layered hangs fixed in `QuasarSdPrefetchDispatchReproFixture.MinimalRelay32B_OneIter` (SD mode: `TT_METAL_SLOW_DISPATCH_MODE=1`). Prefetch (`cq_prefetch.cpp`, DM on core `{0,0}`) and dispatch (`cq_dispatch.cpp`) on the same worker never completed within 60s without watcher.

**Repro command:**

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build_Debug/test/tt_metal/perf_microbenchmark/dispatch/test_prefetcher \
  --gtest_filter=QuasarSdPrefetchDispatchReproFixture.MinimalRelay32B_OneIter
```

**Architecture constraint (Quasar):** NOC reads/writes use **cached** L1 offsets. The DM CPU must use the **uncached alias** (`+ MEM_L1_UNCACHED_BASE`) for polling host-written TL1 and for semaphore/cmd decode reads. Host NOC reads physical TL1 directly (no host-side cache coherency issue).

---

## Stall progression (bisect, historical)

| Observed symptom | Interpretation |
|------------------|----------------|
| `FETCH_Q_HQW` | Device poll saw empty FetchQ; host saw non-zero entry |
| Retire sub-phases / `RETIRE_BACK` | Hang in retire or before `fetch_q_get_cmds` return |
| `AFTER_FETCH_Q` | Fetch returned; hang before uncached cmd decode |
| *(test passes)* | Minimal sync chain + no phase-marker L1 writes |

Phase-marker L1 probes and host post-mortem tooling were used during bisect and have been **removed** from production and `test_prefetcher.cpp`.

---

## Fix 1: Dispatch CB credit semaphore bootstrap

**Problem:** On Quasar, `CreateSemaphore` initializes to 0. Host `cluster.write_core` primes **cached** L1 only. `CBWriter::acquire_pages` reads the **uncached** alias and saw 0 credits.

**Code:** `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp`

```187:191:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
FORCE_INLINE void quasar_bootstrap_prefetch_downstream_credits() {
    if constexpr (is_d_variant != 0) {
        Semaphore<fd_core_type>(my_downstream_cb_sem_id).set(downstream_cb_pages);
    }
}
```

Called at `kernel_main` entry (line **3044**) before `kernel_main_hd` (line **3047**).

---

## Fix 2: FetchQ poll visibility (uncached alias)

**Problem:** Host NOC writes to PrefetchQ do not invalidate DM L1 D$/L2; cached polling spun in HQW while host saw valid entries.

**Code:**

- Uncached `prefetch_q_rd_ptr` in `fetch_q_get_cmds` (comment **643–646**, init **653–656**)
- `fetchq_poll_load` in `tt_metal/impl/dispatch/kernels/cq_common.hpp` lines **98–108**
- `tl1_publish_flush` after FetchQ consume in `read_from_pcie` (lines **585–587**)

---

## Fix 3: Cmddat retire sync chain after NOC fetch

**Problem:** NOC fills cmddat at **cached** offsets; DM decode uses **uncached** `uncached_l1_ptr`. Need tr_ack ordering, NOC barrier, and per-line invalidate before decode.

**Current production helpers** — `cq_prefetch.cpp` lines **129–173**:

| Symbol | Lines | Role |
|--------|-------|------|
| `quasar_cmddat_pre_retire_iters` | 129 | Delay loop count (8) before `tr_ack_trid` |
| `quasar_cmddat_pre_cmd_decode_iters` | 130 | Delay loop count (32) before cmd decode (see Fix 4) |
| `cmddat_invalidate_after_noc_read` | 133–140 | L1 D$ + L2 invalidate per 64B cmddat line |
| `quasar_cmddat_pre_retire_barrier_sync` | 142–149 | Fence + delay + `tr_ack_trid` + fence |
| `quasar_cmddat_pre_cmd_decode_sync` | 151–159 | Fence + delay + fence (used in `process_cmd`, not retire) |
| `quasar_cmddat_retire_fetch_read` | 164–173 | Retire chain (no post-retire delay) |

Quasar-only block ends at line **179** (`#endif`).

```142:173:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
FORCE_INLINE void quasar_cmddat_pre_retire_barrier_sync(uint32_t trid) {
    asm volatile("fence" ::: "memory");
    for (volatile int delay = 0; delay < static_cast<int>(quasar_cmddat_pre_retire_iters); ++delay) {
        (void)delay;
    }
    (void)__builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(trid);
    asm volatile("fence" ::: "memory");
}

FORCE_INLINE void quasar_cmddat_pre_cmd_decode_sync() {
    asm volatile("fence" ::: "memory");
    for (volatile int delay = 0; delay < static_cast<int>(quasar_cmddat_pre_cmd_decode_iters); ++delay) {
        (void)delay;
    }
    asm volatile("fence" ::: "memory");
}

FORCE_INLINE void quasar_cmddat_retire_fetch_read(uint32_t trid, uintptr_t read_start, uint32_t size_bytes) {
    while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
    }

    quasar_cmddat_pre_retire_barrier_sync(trid);

    noc_async_read_barrier_with_trid(trid);

    cmddat_invalidate_after_noc_read(read_start, size_bytes);
}
```

**Retire flow (per in-flight DRAM→cmddat read):**

1. Spin until `ncrisc_noc_read_with_transaction_id_flushed(trid)`.
2. **`quasar_cmddat_pre_retire_barrier_sync`:** fence → **8** volatile delay iterations → read `scmdbuf_tr_ack_trid` → fence. Bridges tr_ack vs TL1 visibility before the NOC barrier.
3. **`noc_async_read_barrier_with_trid`:** complete the tagged read.
4. **`cmddat_invalidate_after_noc_read`:** invalidate cached cmddat range so a later uncached decode sees host/NOC-filled TL1.

Invoked from `fetch_q_get_cmds` when retiring the oldest in-flight read (line **880**). Inflight metadata is snapshotted at lines **855–858** before retire.

**Bisect history:** Early bring-up used scratch_db L2 flush loops and longer delays inside retire and after invalidate. Those were removed or folded into the minimal delay + `tr_ack_trid` pattern above; repro kernels (`quasar_prefetch_retire_repro`) still expose optional heavier pre-retire chains via env knobs.

---

## Fix 4: Pre-cmd-decode sync in `process_cmd` (replaces phase markers)

**Problem:** After `fetch_q_get_cmds` returns with committed cmddat, the kernel hung on the first **uncached** read of `cmd->base.cmd_id` (and when debug used `prefetch_publish_phase` writes to scratch L1). Uncached marker stores after L2 invalidate also stalled Quasar RTL.

**Where it runs:** `process_cmd` on Quasar DM (lines **2242–2251**), after `uncached_l1_ptr` setup and **before** the first `cmd->base.cmd_id` read (`switch` at line **2257**). Called from `kernel_main_hd` after `fetch_q_get_cmds` returns (lines **3003–3007**).

**Name:** `quasar_cmddat_pre_cmd_decode_sync` — fence + **`quasar_cmddat_pre_cmd_decode_iters` (32)** delay iterations + fence. Same structural pattern as the pre-retire delay, but at the **decode** boundary instead of the tr_ack boundary.

**Code:**

```2248:2251:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
    volatile CQPrefetchCmd tt_l1_ptr* cmd = uncached_l1_ptr<CQPrefetchCmd>(cmd_ptr);
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    quasar_cmddat_pre_cmd_decode_sync();
#endif
```

**`kernel_main_hd` loop (no extra sync at loop top):**

```3000:3009:tt_metal/impl/dispatch/kernels/cq_prefetch.cpp
    while (!done) {
        DeviceZoneScopedN("CQ-PREFETCH");
        constexpr uint32_t preamble_size = 0;
        fetch_q_get_cmds<preamble_size>(fence, cmd_ptr, pcie_read_ptr);
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);

        uint32_t stride;
        done = process_cmd<false, false>(cmd_ptr, downstream_data_ptr, stride, l1_cache, exec_buf_state);
        DPRINT << "hd:st=" << stride << " ncp=" << (cmd_ptr + stride) << ENDL();
        cmd_ptr += stride;
    }
```

**What this replaced:**

| Removed | Replaced by |
|---------|-------------|
| `prefetch_publish_phase(PRE_CMD_DECODE / PROCESS_CMD_ENTER)` | `quasar_cmddat_pre_cmd_decode_sync()` |
| `quasar_cmddat_post_fetch_sync()` at retire tail | *(removed — invalidate is sufficient)* |
| Duplicate sync in `kernel_main_hd` after fetch | Sync only at `process_cmd` entry |

**Minimum timing (current):** Pre-retire **8** iters + `tr_ack_trid`; pre-decode **32** iters. Counts were reduced from longer bring-up values; zero-iter versions failed on RTL without watcher.

---

## Fix 5: Early return from `fetch_q_get_cmds` after commit

**Problem:** Continuing the fetch loop after retire with a consumed FetchQ cursor and other reads still in-flight hung RTL.

**Code:** Snapshot inflight at lines **855–858**, call `quasar_cmddat_retire_fetch_read` at line **880**, then when `committed_bytes != 0` (lines **909–913**) update `cmd_ptr`/`fence`/inflight and **`return`** immediately.

---

## Fix 6: Remove L1 phase-marker publishes (historical)

**Problem:** `prefetch_publish_phase` (uncached stores to scratch_db marker line) and `tl1_publish_flush` on that line during retire caused RTL stalls.

**Resolution:** All `prefetch_publish_phase` / `PrefetchPhaseMarker` / `PREFETCH_PHASE_MARKER_ADDR` plumbing removed from production `cq_prefetch.cpp` and SD test harness. Timing gap before decode is covered by **`quasar_cmddat_pre_cmd_decode_sync`** (Fix 4).

`tl1_publish_flush` remains in `read_from_pcie` for host-visible FetchQ rd_ptr updates.

---

## Files touched (production)

| File | Summary |
|------|---------|
| `tt_metal/impl/dispatch/kernels/cq_prefetch.cpp` | Credit bootstrap, retire chain, pre-cmd-decode sync, FetchQ early return |
| `tt_metal/impl/dispatch/kernels/cq_common.hpp` | `fetchq_poll_load`, uncached L1 helpers |
| `tests/.../dispatch/test_prefetcher.cpp` | Post-mortem / phase-marker debug removed; direct kernel launch |
| `tests/.../kernels/quasar_prefetch_retire_repro.cpp` | Isolated retire bisect (optional heavy pre-retire) |

---

## Notes

- `rd_ptr_addr` is written **before** `++prefetch_q_rd_ptr` → `entries_consumed` can lag visible slot zeroing by one entry.
- Kernel JIT recompiles from `cq_prefetch.cpp` at launch.
- `MinimalRelay32B_OneIter` passes without watcher with the minimal delay counts above.
- Tuning: `quasar_cmddat_pre_retire_iters` and `quasar_cmddat_pre_cmd_decode_iters` are the primary RTL timing knobs in the production path.
