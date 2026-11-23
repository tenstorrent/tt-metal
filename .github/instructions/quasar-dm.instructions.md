---
description: 'PR review rules for code that runs on Quasar data-movement (DM) cores'
applyTo: 'tt_metal/hw/**,tt_metal/fabric/hw/**,tt_metal/impl/*/kernels/**,ttnn/**/kernels*/**,ttnn/**/dataflow/**,ttnn/cpp/ttnn/kernel_lib/**,ttnn/cpp/ttnn/operations/kernel_helper_functions/**,tests/**/kernels/**,tests/tt_metal/tt_metal/**'
excludeAgent: "cloud-agent"
---

# Quasar DM Review

These rules apply to code built for Quasar DM cores (`ARCH_QUASAR` with `COMPILE_FOR_DM`). Code that breaks them is incorrect on Quasar, where it can read stale data, lose another agent's writes, or hang.

## 🔴 CRITICAL

- **L1 shared with other agents needs cache handling**: DM cores reach L1 through write-back caches. Host writes and other cores' NoC writes do not update these caches, and host and NoC reads do not see what is in them. A word that the host, dispatch or another core writes must be read through the uncached alias (`MEM_L1_UNCACHED_BASE`), through the cache after its line is invalidated, or through the cache when the writer's NoC packet sets the snoop bit (`noc_set_packet_tags`). A word that the host or another core reads must be written through the alias or written back before it is read. `invalidate_l1_cache()` does nothing on Quasar DM and `volatile` does not bypass the cache, so a WH/BH-style raw read of such a word is stale. Prefer an API that does this:
  - `Semaphore` and `GlobalSemaphore`. `get_semaphore()` returns the cached address on Quasar DM, so a raw pointer from it, or `noc_semaphore_wait()` on that pointer, is wrong.
  - `scoped_lock()` on `CoreLocalMem`, `Scratchpad` and `LocalTensorAccessor`: acquire invalidates the range and release writes it back. `operator[]` outside a lock goes through the cache.
  - `DataflowBuffer::scoped_read_lock()` / `scoped_write_lock()`. `get_read_ptr()` / `get_write_ptr()` already return the uncached alias, so do not add it again.
- **Switching between cached and uncached access**: a word that only this core's DM threads write can stay cached (`SemScope::DM_LOCAL_CACHED` semaphores do this). The uncached alias does not see this core's dirty lines, and the cache does not see uncached stores, so a word read through one view after a store through the other needs a write-back or an invalidate in between (`scoped_lock()` release and acquire do these). A line this core dirties must not hold a word another agent writes, because the write-back of the whole line overwrites the other agent's update.

## 🟡 IMPORTANT

- **Caches persist across launches**: the DM firmware invalidates only the kernel's text before a launch (see `device_setup()` in `tt_metal/hw/firmware/src/tt-2xx/dm.cc`), so lines cached by one program can still be valid in the next, and a persistent kernel is never relaunched. A cached read is not safe just because the kernel has just started. A host test that reads L1 mid-run sees a DM core's cached stores only after they are written back.
- **Fence before a NoC command reads a stored word**: the NoC reads L1 directly, so nothing orders a CPU store before a later NoC command that sends the same word, for example a multicast source. Put a `fence` between them, as `noc_semaphore_set()` does.
- **More than one DM thread**: a DM kernel can run on several harts (`get_num_threads()`, `get_my_thread_id()`); on WH/BH it runs on one. Work that only one thread must do needs a thread check. Synchronize a kernel's DM threads with `sync_threads()`, and DM threads with TRISCs with `sync_dm_compute_threads()`, not with an L1 counter: DM and TRISC atomics are not coherent with each other. `GlobalSemaphore::down()` allows only one thread per semaphore.
- **Transaction IDs**: user kernels may use IDs 0 to `USER_TXN_ID_MAX` (7), although WH/BH allow up to 15. Higher IDs belong to the runtime: the DFB pool, and fast-dispatch kernels (`DISPATCH_KERNEL`), which may go up to `HW_TXN_ID_MAX` using the ranges in `tt_metal/impl/dispatch/kernels/dispatch_noc_trids.hpp`.
- **Zeroing local L1 borrows command buffer 0**: between the local-L1 overload of `noc.async_write_zeros()` and `write_zeros_l1_barrier()`, no other NoC write may run on the same core; it would write zeros instead of its data. The DRAM overload uses ordinary NoC writes and `write_zeros_dram_barrier()`.

## 🟢 SUGGESTION

- Prefer the APIs above to open-coded `addr + MEM_L1_UNCACHED_BASE`. The macro is defined only for Quasar, so each use needs `#ifdef ARCH_QUASAR`, and the alias is easy to apply twice.

## Review Checklist

- [ ] Each L1 word that the host, dispatch or another core writes is read uncached, after an invalidate, or from a snooped write
- [ ] Each word that the host or another core reads is written uncached or written back first
- [ ] A word stored through one view and read through the other has a write-back or invalidate in between, and no line this core dirties holds a word another agent writes
- [ ] No cached read relies on a cold cache at kernel start
- [ ] A `fence` separates a store from a NoC command that sends the stored word
- [ ] Multi-threaded DM kernels split or guard single-thread work, and synchronize with the thread-sync APIs
- [ ] User-kernel transaction IDs stay within `USER_TXN_ID_MAX`; dispatch kernels take theirs from `dispatch_noc_trids.hpp`
- [ ] No other NoC write between the local-L1 overload of `async_write_zeros()` and `write_zeros_l1_barrier()`
