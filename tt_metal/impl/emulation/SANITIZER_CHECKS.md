# Emule ASAN-Style Sanitizer Checks

This branch adds a set of ASAN-style runtime checks to the **emule** software
emulator. This doc is organized **by check**: for each one it explains what it
catches, where it fires, **which file(s) it lives in**, and — in plain terms —
how it's implemented. (A test under `tests/tt_metal/tt_metal/api/` exercises each
check; it's listed at the end of each section for reference.)

Paths are prefixed with the repo: **`[metal]`** = `tt-metal/`, **`[emule]`** =
`tt-emule/`.

**Common mechanics**
- All checks are gated by one env var, **`TT_METAL_EMULE_ASAN=1`**, off by
  default (one exception: *CB Reservation Overflow* is always on — see §8).
- Each fires a single `[ASAN ERROR] <Category>: …` line and then `abort()`s.
- When the flag is off, every check is a no-op: host helpers return early, and
  the runner leaves the kernel's thread-local state pointers null so the in-kernel
  checks short-circuit.

**Where checks live (the three layers)**
- **Host-side** — `[metal] tt_metal/impl/emulation/host_sanitizers.hpp` + call
  sites in `[metal] tt_metal/impl/host_api/tt_metal.cpp`.
- **Kernel-side** — the JIT headers in `[emule]`:
  `include/jit_hw/api/dataflow/dataflow_api.h`, `include/jit_hw/jit_kernel_stubs.hpp`,
  `include/jit_hw/api/cb_api.h`. Every kernel L1 access flows through
  `__emule_local_l1_to_ptr` (duplicated in `dataflow_api.h` and the stub; the JIT
  wrapper includes the stub first, so the **stub's copy is the one that runs** —
  kernel-side edits must be made in both).
- **Runner / post-launch** — `[metal] tt_metal/impl/emulation/emulated_program_runner.cpp`
  (runs after all kernel threads join). The live-buffer extents it relies on are
  registered in `[metal] tt_metal/impl/emulation/emule_live_ranges.{hpp,cpp}`,
  wired in from `[metal] tt_metal/impl/buffers/buffer.cpp`.

## Checks at a glance

| Check | Layer | Lives in | Fires when |
|---|---|---|---|
| Use-After-Free | host | `[metal] host_sanitizers.hpp` + `host_api/tt_metal.cpp` | host access through a deallocated `Buffer` |
| Host L1 / DRAM Alignment | host | `[metal] host_sanitizers.hpp` + `host_api/tt_metal.cpp` | host poke address not aligned to the transfer's real requirement |
| Metadata Overflow | host | `[metal] host_api/tt_metal.cpp` | program's static CB region overruns the reserved L1 window |
| Out-of-Bounds Write (L1/DRAM) | kernel + runner | `[emule] jit_kernel_stubs.hpp` & `dataflow_api.h` (L1); `[metal] emulated_program_runner.cpp` (`__emule_dram_ptr`) | access lands in no live buffer extent |
| Tensor Padding Violation | kernel | `[emule] jit_kernel_stubs.hpp` & `dataflow_api.h` | write into a buffer's `[logical_end, physical_end)` pad band |
| Illegal Semaphore Access | kernel | `[emule] jit_kernel_stubs.hpp` & `dataflow_api.h` | scalar access into the reserved semaphore region |
| CB Boundary Violation | kernel | `[emule] jit_kernel_stubs.hpp` & `dataflow_api.h` (counters in `api/cb_api.h`) | access to a CB page outside an **active** reserve/wait window |
| CB Reservation Overflow | kernel | `[emule] include/jit_hw/api/cb_api.h` | `cb_reserve_back(n)` with `n` > the CB's total pages |
| NoC-read-pending on push | kernel | `[emule] api/cb_api.h` (push) + `api/dataflow/dataflow_api.h` (read counter) | `cb_push_back` while a `noc_async_read` is unbarriered |
| NOC Transfer Alignment | kernel | `[emule] include/jit_hw/api/dataflow/dataflow_api.h` | a NoC endpoint isn't aligned to its own memory-type alignment |
| Dirty CB Detected | runner | `[metal] emulated_program_runner.cpp` (counter in `[emule] tt_emule/cb_sync_state.hpp`) | a consumed CB left non-empty at exit (push > pop) |
| Object Intent Violation | runner | `[metal] emulated_program_runner.cpp` | a kernel changed a buffer it never resolved a pointer into |
| Fabric Access Violation | runner | `[metal] emulated_program_runner.cpp` (`__emule_resolve_noc_addr`) | a NoC access targets an unallocated core coordinate |

---

## Host-side checks

### 1. Use-After-Free
**Lives in:** `[metal] tt_metal/impl/emulation/host_sanitizers.hpp`
(`check_buffer_allocated`); called from the host entry points in
`[metal] tt_metal/impl/host_api/tt_metal.cpp`.
**What it catches:** a host read/write through a `Buffer` whose device memory was
already deallocated — on silicon this touches reclaimed memory and corrupts
unrelated allocations.
**How it works:** `check_buffer_allocated(buffer, op)` calls `buffer.is_allocated()`
at the top of each host entry point (`WriteToBuffer`, `ReadFromBuffer`, `ReadShard`,
the core-subset and shared-ptr overloads). Not allocated → abort. One boolean,
before any access touches memory.
*Diagnostic:* `Use-After-Free: <op> called on Buffer … not currently allocated`.
*Exercised by:* `test_tensor_bad_acess.cpp`.

### 2. Host L1 / DRAM Alignment
**Lives in:** `[metal] host_sanitizers.hpp` (`check_host_l1_alignment` /
`check_host_dram_alignment`); called from `WriteToDeviceL1`/`ReadFromDeviceL1` and
`WriteToDeviceDRAMChannel`/`ReadFromDeviceDRAMChannel` in
`[metal] tt_metal/impl/host_api/tt_metal.cpp`.
**What it catches:** a host→device poke at an address the underlying transfer
can't satisfy.
**How it works:** the helpers take the **required alignment as a parameter** and
fire only when `alignment > 1 && address % alignment != 0`. The call sites pass
`Cluster::get_alignment_requirements(device, size)`, which returns the DMA
alignment when a DMA engine backs the transfer, otherwise **1**. On emule
(memory-backed I/O, no DMA) it resolves to 1, so the check is a no-op — matching
the framework's own contract that host pokes accept any byte address; on a real
DMA build it catches a genuinely misaligned DMA transfer.
*Diagnostic:* `L1 Alignment: … must be N-byte aligned` / `DRAM Alignment: …`.
*Exercised by:* no dedicated death test in this set (resolves to a no-op under
emule). `test_alignment_writes.cpp` covers the *kernel*-side NOC alignment (§10).

### 3. Metadata Overflow
**Lives in:** `[metal] tt_metal/impl/host_api/tt_metal.cpp` (the
`ConfigureDeviceWithProgram` path).
**What it catches:** a program whose static circular-buffer region grows down past
the lowest occupied L1 address — CB metadata colliding with tensor space.
**How it works:** during `ConfigureDeviceWithProgram` the runner computes the
program's static CB region extent and compares it against the reserved L1 window;
an overrun throws, which the host turns into the `[ASAN ERROR] Metadata Overflow`
abort.
*Diagnostic:* `Metadata Overflow: Program metadata exceeds reserved L1 region`.
*Exercised by:* `test_metadata_size.cpp`.

---

## Kernel-side checks

> All of these run inside `__emule_local_l1_to_ptr` (the chokepoint every kernel
> L1 access passes through — duplicated in `[emule] jit_kernel_stubs.hpp` and
> `[emule] include/jit_hw/api/dataflow/dataflow_api.h`) or inside the CB/NoC API
> shims (`[emule] include/jit_hw/api/cb_api.h`, `…/dataflow/dataflow_api.h`). The
> host snapshots the relevant state at launch and threads it into per-kernel
> thread-locals; with the flag off those pointers are null and the checks vanish.

### 4. Out-of-Bounds Write (L1 and DRAM)
**Lives in:** L1 — `__emule_local_l1_to_ptr` in `[emule] jit_kernel_stubs.hpp` &
`[emule] include/jit_hw/api/dataflow/dataflow_api.h`. DRAM — `__emule_dram_ptr` in
`[metal] tt_metal/impl/emulation/emulated_program_runner.cpp`. Live extents:
`[metal] emule_live_ranges.{hpp,cpp}`, fed from `[metal] tt_metal/impl/buffers/buffer.cpp`.
**What it catches:** a kernel writing to L1/DRAM at or above the unreserved base
but inside no live buffer — scribbling on memory it never legitimately addressed.
**How it works:** at launch the host snapshots every live buffer's extent
(`LiveL1Ranges` / `LiveDramRanges`, fed from `Buffer::allocate`/`deallocate`) into
per-kernel range arrays. On each access the kernel first **normalizes the address
to a buffer-relative offset** via `__emule_addr_to_offset` — necessary because
sharded / CB / `l1_alloc` accesses arrive as absolute bridge pointers, not
offsets, and a raw absolute value would never match a relative range — then checks
that offset against the live extents. In none → abort.
*Diagnostic:* `Out-of-Bounds Write: Attempted to access address 0x… which is not part of any allocated tensor`.
*Exercised by:* `test_write_outside_tensor.cpp` (L1 + DRAM).

### 5. Tensor Padding Violation
**Lives in:** `__emule_local_l1_to_ptr` in `[emule] jit_kernel_stubs.hpp` &
`dataflow_api.h`; padding bands registered in `[metal] emule_live_ranges.{hpp,cpp}`
(`LiveL1PaddingRanges`) from `Buffer::set_logical_size` in `[metal] buffers/buffer.cpp`.
**What it catches:** a kernel writing into the padding gap
`[logical_end, physical_end)` of a buffer whose logical size is smaller than its
allocated physical size.
**How it works:** `set_logical_size` registers the padding band; the runner threads
those `(logical_end, physical_end)` pairs into the kernel. After the OOB check, the
normalized offset is tested against each padding band — inside one → abort.
*Diagnostic:* `Tensor Padding Violation: Attempted to write to a padded memory region at 0x…`.
*Exercised by:* `test_padded_write.cpp`.

### 6. Illegal Semaphore Access
**Lives in:** `__emule_local_l1_to_ptr` in `[emule] jit_kernel_stubs.hpp` &
`dataflow_api.h`; the reserved range is set by the runner
(`[metal] emulated_program_runner.cpp`).
**What it catches:** a kernel doing a raw scalar L1 access into the reserved
semaphore region (semaphores must go through the semaphore API).
**How it works:** the runner passes the semaphore L1 range
(`__emule_sem_l1_range_start/end`) to the kernel. It's the **first** test in
`__emule_local_l1_to_ptr`: if the address is in that range, abort.
*Diagnostic:* `Illegal Semaphore Access: Offset 0x… is inside the reserved Semaphore region [start, end)`.
*Exercised by:* `test_semaphore_write.cpp`.

### 7. CB Boundary Violation
**Lives in:** `__emule_local_l1_to_ptr` in `[emule] jit_kernel_stubs.hpp` &
`dataflow_api.h`; the `reserved`/`waited` page counters are maintained in
`[emule] include/jit_hw/api/cb_api.h`.
**What it catches:** a kernel reading/writing a circular-buffer page **outside the
window it actually reserved/waited for**.
**How it works:** when an access lands inside a CB's backing memory, the check
computes the accessed page's distance from `write_idx`/`read_idx` and compares
against the pages currently `reserved`/`waited` (`__emule_cb_reserved_pages` /
`__emule_cb_waited_pages`, maintained by `cb_reserve_back`/`cb_wait_front`). It
fires **only when a window is active** (`reserved > 0 || waited > 0`) and the page
is outside both — so legitimate raw `get_write_ptr`/`get_read_ptr` addressing with
no active handshake (globally-allocated/sharded CBs, single-buffered scratch,
output CBs written then DMA'd) is not flagged. A write *past* the CB's allocated
region is caught by the OOB check (§4) instead.
*Diagnostic:* `CB Boundary Violation: Attempted to access CB <id> at offset 0x… outside the write/read window`.
*Exercised by:* `test_write_beyond_res_pages.cpp` (write side, read side, wraparound, and a no-active-window control).

### 8. CB Reservation Overflow  *(always on)*
**Lives in:** `cb_reserve_back` in `[emule] include/jit_hw/api/cb_api.h`.
**What it catches:** `cb_reserve_back(cb, n)` asking for more pages than the CB
holds in total.
**How it works:** before blocking to wait for free space, compare `n` against the
CB's `num_pages`; if it exceeds capacity, abort. **Always on** (not gated by the
env var) because gating it would let an over-reserve *deadlock* on the space wait
instead of reporting a clear error.
*Diagnostic:* `CB Reservation Overflow: CB <id> has <N> total pages, …`.
*Exercised by:* `test_cb_pages.cpp`.

### 9. NoC Read Pending on `cb_push_back`
**Lives in:** `cb_push_back` in `[emule] include/jit_hw/api/cb_api.h`; the pending
counter is incremented/cleared in `noc_async_read` / `noc_async_read_barrier` in
`[emule] include/jit_hw/api/dataflow/dataflow_api.h`.
**What it catches:** pushing a CB page while a `noc_async_read` filling it hasn't
been barriered — the consumer could read half-filled data on silicon.
**How it works:** `noc_async_read` increments a thread-local
`__emule_pending_noc_reads`; the read barrier clears it. `cb_push_back` aborts if
that counter is still > 0 (a barrier was skipped before the push).
*Diagnostic:* `Race Condition: cb_push_back(cb_id=…) called while a NoC read is still pending`.
*Exercised by:* `test_noc_without_barrier.cpp`.

### 10. NOC Transfer Alignment
**Lives in:** `__emule_check_noc_read_alignment` / `__emule_check_noc_write_alignment`
in `[emule] include/jit_hw/api/dataflow/dataflow_api.h`, called at the top of
`noc_async_read`/`noc_async_write` in the same file. (DRAM-vs-L1 is decided by
`__emule_noc_addr_is_dram` in `[metal] emulated_program_runner.cpp`.)
**What it catches:** a NoC read/write whose endpoint isn't aligned to its memory
type's requirement.
**How it works:** each endpoint is checked **against its own alignment** —
absolute, per-side, *not* a relative "low bits of src and dst must match" rule:
L1 = 16 B; DRAM read = 32 B (WH) / 64 B (BH); DRAM write = 16 B. So a DRAM read
from a 32-aligned source into a 16-aligned L1 destination is legal even though
their low bits differ.
*Diagnostic:* `NOC Transfer Alignment: <L1|DRAM> <source|destination> 0x… must be N-byte aligned`.
*Exercised by:* `test_alignment_writes.cpp` (4 misalignment death tests + a positive control).

---

## Runner / post-launch checks

> These run after all kernel threads join (in
> `[metal] tt_metal/impl/emulation/emulated_program_runner.cpp`) and catch
> program-level invariants that no single per-access check can see.

### 11. Dirty CB Detected
**Lives in:** the post-join sweeps (`sweep_per_kernel_dirty_cbs` /
`sweep_program_dirty_cbs`, abort in `abort_if_dirty_cb`) in
`[metal] emulated_program_runner.cpp`. The `occupied` + `total_popped` counters
live on `CBSyncState` in `[emule] include/tt_emule/cb_sync_state.hpp` (incremented
in `cb_sync_pop`; reset in `[emule] include/tt_emule/device.hpp`).
**What it catches:** a circular buffer left non-empty by a **real producer/consumer
imbalance** — a consumer that popped fewer pages than were pushed, which can
back-pressure a later (cache-hit) reuse of the program.
**How it works:** `CBSyncState` tracks `occupied` (pushes − pops) plus a cumulative
`total_popped` counter. After join, both sweeps abort only when
`occupied > 0 && total_popped > 0`. The `total_popped > 0` guard is the key: a CB
with **no consumer at all** (a globally-allocated/sharded output CB, or a
producer-only single-kernel program that DMAs its result out) ends non-empty *by
design*, so it must not be flagged.
*Diagnostic:* `Dirty CB Detected: Core (x, y) CB <id> … pages remain after program exit; the consumer popped <M> (pushed > popped)`.
*Exercised by:* `test_cb_leak.cpp` (a real under-drain + a producer-only control).

### 12. Object Intent Violation
**Lives in:** `[metal] tt_metal/impl/emulation/emulated_program_runner.cpp` (the
pre-launch byte snapshot + post-join `memcmp`); the per-kernel "resolved set"
(`__emule_l1_resolved_ranges`) is recorded inside `__emule_local_l1_to_ptr`
(`[emule] jit_kernel_stubs.hpp` & `dataflow_api.h`).
**What it catches:** a kernel that scribbles on *another* buffer's bytes — valid,
allocated L1, but a buffer it never took a pointer into via the public API (a
provenance/aliasing bug).
**How it works:** before launch the runner snapshots every live L1 tensor's bytes
per core. Each kernel L1 access records the `(start,end)` of the buffer it resolved
into — its "intended write set". After the kernel exits, the runner `memcmp`s the
snapshot vs current L1 for every buffer **not** in the resolved set; any change is
an unintended write. (Exact attribution requires one kernel per core; multi-kernel
programs hit a friendlier early-out.)
*Diagnostic:* `Object Intent Violation: Attempted to modify memory belonging to an adjacent object context — L1 buffer [start, end) … changed but no pointer was resolved into it`.
*Exercised by:* `test_valid_mem_wrong_alloc.cpp` (adjacent + non-adjacent violations + a control).

### 13. Fabric Access Violation
**Lives in:** `__emule_resolve_noc_addr` in
`[metal] tt_metal/impl/emulation/emulated_program_runner.cpp`.
**What it catches:** a NoC transfer whose target NOC X/Y doesn't map to any
allocated core (an off-grid / unallocated-coordinate access).
**How it works:** `__emule_resolve_noc_addr` decodes the NOC X/Y from the address
and looks it up in the core map. No core registered at those coordinates → abort,
instead of dereferencing a null/garbage backing pointer.
*Diagnostic:* `Fabric Access Violation: Attempted to access unallocated Core at NOC coordinates (x, y)`.
*Exercised by:* `test_fabric_allocation.cpp`.

---

## Each check's core mechanism + home, in one line

| Check | Lives in | The trick |
|---|---|---|
| Use-After-Free | `[metal] host_sanitizers.hpp` | `buffer.is_allocated()` at host entry |
| Host L1/DRAM Alignment | `[metal] host_sanitizers.hpp` | `address % get_alignment_requirements(device, size)` (1 ⇒ no-op on emule) |
| Metadata Overflow | `[metal] host_api/tt_metal.cpp` | static CB region vs lowest L1 alloc, at configure time |
| Out-of-Bounds Write | `[emule] jit_kernel_stubs.hpp`/`dataflow_api.h`; `[metal] emulated_program_runner.cpp` | normalized offset ∉ any live `LiveL1Ranges`/`LiveDramRanges` extent |
| Tensor Padding | `[emule] jit_kernel_stubs.hpp`/`dataflow_api.h` | offset ∈ `[logical_end, physical_end)` padding band |
| Illegal Semaphore | `[emule] jit_kernel_stubs.hpp`/`dataflow_api.h` | offset ∈ reserved semaphore L1 range |
| CB Boundary | `[emule] jit_kernel_stubs.hpp`/`dataflow_api.h` | accessed page outside an **active** reserve/wait window |
| CB Reservation Overflow | `[emule] api/cb_api.h` | `cb_reserve_back(n)` with `n > num_pages` (always on) |
| NoC pending on push | `[emule] api/cb_api.h` + `dataflow_api.h` | `cb_push_back` while `__emule_pending_noc_reads > 0` |
| NOC Transfer Alignment | `[emule] api/dataflow/dataflow_api.h` | each endpoint vs its own absolute alignment (16 / 32 / 64 B) |
| Dirty CB | `[metal] emulated_program_runner.cpp` (+ `[emule] cb_sync_state.hpp`) | `occupied > 0 && total_popped > 0` after join |
| Object Intent | `[metal] emulated_program_runner.cpp` | post-launch `memcmp` of buffers never resolved into |
| Fabric Access | `[metal] emulated_program_runner.cpp` | NOC X/Y not in the core map |
