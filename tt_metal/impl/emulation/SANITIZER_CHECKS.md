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
- Each fires a single `[ASAN ERROR] <Category>: …` line, then a unified
  diagnostic trace (see *Diagnostic trace* below), then `abort()`s — every site
  calls `__emule_asan_panic()` instead of a bare `abort()`.
- When the flag is off, every check is a no-op: host helpers return early, and
  the runner leaves the kernel's thread-local state pointers null so the in-kernel
  checks short-circuit.
- The `abort()` writes **no core dump** by default (the emulated process maps ~13 GB of
  L1+DRAM *virtual* address space, but only touched pages are dumped, so a real core is
  ~1.4 GB). Set **`TT_METAL_EMULE_ASAN_ALLOW_CORE=1`** to capture
  one for debugging instead — see *Diagnostic trace* below.

**Where checks live (the three layers)**
- **Host-side** — `[metal] tt_metal/impl/emulation/host_sanitizers.hpp` + call
  sites in `[metal] tt_metal/impl/host_api/tt_metal.cpp`.
- **Kernel-side** — the JIT headers in `[emule] include/jit_hw/`. Every kernel L1
  access flows through `__emule_local_l1_to_ptr`, which has a **single** definition
  in `internal/emule_l1_to_ptr.h` (included by both `jit_kernel_stubs.hpp` and
  `api/dataflow/dataflow_api.h`). It early-outs when ASAN is off; otherwise it
  dispatches to the per-access check bodies in `asan/asan_l1_checks.h`. The CB-op
  checks live in `asan/asan_cb.h` (called from `api/cb_api.h`), NoC alignment in
  `api/dataflow/asan/asan_dataflow.h`, and the master switch + diagnostic trace in
  `asan/emule_asan.h`.
- **Runner / post-launch** — `[metal] tt_metal/impl/emulation/emulated_program_runner.cpp` (+ `emule_sanitizers.cpp` for Dirty CB / Object Intent)
  (runs after all kernel threads join). The live-buffer extents it relies on are
  registered in `[metal] tt_metal/impl/emulation/emule_live_ranges.{hpp,cpp}`,
  wired in from `[metal] tt_metal/impl/buffers/buffer.cpp`.

## Diagnostic trace

Every `[ASAN ERROR]` is followed by a unified trace, emitted by
`__emule_asan_panic()`. libtt_metal carries its own libc/POSIX-only definition in
**`[metal] tt_metal/impl/emulation/emule_asan_panic.cpp`** — a faithful mirror of
the tt-emule runtime's copy (declared, with the standalone-runtime definition, in
`[emule] include/jit_hw/asan/emule_asan.h`). The host-API checks call it directly and
JIT kernel `.so` files resolve it at dlopen, so metal pulls nothing from `jit_hw`.
It prints:

- **Which kernel + where:** when a kernel is on the stack, the kernel source
  path (`__emule_kernel_name`, threaded in per launch by the runner) plus the
  logical/physical core and processor/neo/trisc ids (from the existing identity
  thread-locals). Host-API checks (no kernel running) omit this block.
- **A symbolized backtrace:** `backtrace()` captures the stack; each frame is
  mapped to its module via `dladdr`, and frames inside a JIT kernel `.so` are
  resolved to **kernel-source `file:line`** by shelling out to `llvm-symbolizer`
  (falling back to `addr2line`, then to raw `backtrace_symbols`). The walk stops
  at the JIT entry point `__emule_kernel_entry` so libc/runner internals aren't
  dumped.

To make kernel frames resolvable, `jit_compile_kernel` adds `-g
-fno-omit-frame-pointer -funwind-tables` to the kernel compile **when ASAN is
on** (kept at `-O2` so numerics match normal runs); the ASAN flag is folded into
the JIT cache key (`:asan_g`) so these debug `.so` files never collide with the
lean non-ASAN cache. The kernel-frame `file:line` refers to the JIT-generated
kernel source. For post-execution checks (Dirty CB, Object Intent) the kernel
has already returned, so the trace shows kernel identity + the runner stack, not
the offending kernel line.

**Core dumps (`TT_METAL_EMULE_ASAN_ALLOW_CORE`).** `__emule_asan_panic` decides what
happens to the core the `abort()` would otherwise produce:
- **Unset (default) — no core.** The process is marked non-dumpable
  (`prctl(PR_SET_DUMPABLE, 0)`). The emulated process maps ~13 GB of L1+DRAM virtual
  address space (only touched pages dump, so a real core is ~1.4 GB), and on hosts whose
  `core_pattern` pipes to a crash handler (e.g. apport)
  `ulimit -c 0` / `RLIMIT_CORE` is *ignored* — `PR_SET_DUMPABLE=0` the kernel honors
  regardless. So the suite is safe to run on any machine with no `LD_PRELOAD` shim or
  external setup, and the trace above already captures what a core would.
- **Set — capture a core.** A real core of the process is written to
  `./emule_asan_core.<pid>` (CWD) via `gcore`. emule dumps it itself rather than relying
  on the kernel `core_pattern`, because that global pipe (apport) silently drops cores
  from non-package binaries. Best-effort: if `gcore`/ptrace is unavailable the process is
  left dumpable (a plain-file `core_pattern` still works) and a note is printed. The core
  is a standard ELF core — read it with `gdb <test-binary> emule_asan_core.<pid>`. Given
  its ~1.4 GB size, use it on a single `--gtest_filter` test, not the whole suite. Done
  once, under the panic lock, so a multi-core kernel bug never spawns more than one dump.

## Checks at a glance

| Check | Layer | Lives in | Fires when |
|---|---|---|---|
| Use-After-Free | host | `[metal] host_sanitizers.hpp` + `host_api/tt_metal.cpp` | host access through a deallocated `Buffer` |
| Host L1 / DRAM Alignment | host | `[metal] host_sanitizers.hpp` + `host_api/tt_metal.cpp` | host poke address not aligned to the transfer's real requirement |
| Metadata Overflow | host | `[metal] host_api/tt_metal.cpp` | program's static CB region overruns the reserved L1 window |
| Out-of-Bounds Write (L1/DRAM) | kernel + runner | `[emule] asan/asan_l1_checks.h` (L1); `[metal] emulated_program_runner.cpp` (`__emule_dram_ptr`) | access lands in no live buffer extent |
| Tensor Padding Violation *(test skipped — see §5)* | kernel | `[emule] asan/asan_l1_checks.h` | write into a buffer's `[logical_end, physical_end)` pad band |
| Illegal Semaphore Access | kernel | `[emule] asan/asan_l1_checks.h` | scalar access into the reserved semaphore region |
| CB Boundary Violation | kernel | `[emule] asan/asan_l1_checks.h` (window counters in `asan/asan_cb.h`) | access to a CB page outside an **active** reserve/wait window |
| CB Reservation Overflow | kernel | `[emule] asan/asan_cb.h` | `cb_reserve_back(n)` with `n` > the CB's total pages |
| NoC-read-pending on pop | kernel | `[emule] asan/asan_cb.h` (pop) + `api/dataflow/dataflow_api.h` (read counter) | `cb_pop_front` while a `noc_async_read` is unbarriered |
| NOC Transfer Alignment | kernel | `[emule] api/dataflow/asan/asan_dataflow.h` | a NoC endpoint isn't aligned to its own memory-type alignment |
| Dirty CB Detected | runner | `[metal] emule_sanitizers.cpp` (counters in `[emule] asan/asan_cb.h`) | a kernel left a `cb_reserve_back` un-pushed or a `cb_wait_front` un-popped |
| Object Intent Violation | runner | `[metal] emule_sanitizers.cpp` | a kernel changed a buffer it never resolved a pointer into |

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
*Exercised by:* `test_tensor_bad_access.cpp` (five deallocated-Buffer death tests
across every host entry point + a live-buffer round-trip positive control).

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
*Exercised by:* `test_host_alignment.cpp` — no death test is possible on emule
(the requirement resolves to 1, so the check cannot fire); instead two
positive-control round-trips (unaligned host→L1 and host→DRAM pokes that must
NOT abort) guard the no-op contract against re-hardcoding a fixed alignment.
`test_alignment_writes.cpp` covers the *kernel*-side NOC alignment (§10).

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
*Exercised by:* `test_metadata_size.cpp` (an overrunning CB death test + a
fitting-CB positive control, plus a self-calibrating direct test of the emule
`check_program_metadata_size` throw — it inflates a no-tensor program's static
config with runtime args to overflow the KERNEL_CONFIG window, and SKIPs on arches
where that window equals the finalize ring buffer so the check is not
independently reachable, as on WH/BH today).

---

## Kernel-side checks

> The per-access L1 checks run inside `__emule_local_l1_to_ptr` (the chokepoint
> every kernel L1 access passes through, single definition in
> `[emule] include/jit_hw/internal/emule_l1_to_ptr.h`), which dispatches to the
> check bodies in `[emule] include/jit_hw/asan/asan_l1_checks.h`. The CB-op checks
> live in `[emule] include/jit_hw/asan/asan_cb.h` (called from `api/cb_api.h`) and
> the NoC alignment check in `[emule] include/jit_hw/api/dataflow/asan/asan_dataflow.h`. The
> host snapshots the relevant state at launch and threads it into per-kernel
> thread-locals; with the flag off the chokepoint early-outs and the checks vanish.

### 4. Out-of-Bounds Write (L1 and DRAM)
**Lives in:** L1 — `__emule_asan_check_oob_tensor` in
`[emule] include/jit_hw/asan/asan_l1_checks.h` (dispatched by the
`__emule_local_l1_to_ptr` chokepoint, `internal/emule_l1_to_ptr.h`). DRAM — `__emule_dram_ptr` in
`[metal] tt_metal/impl/emulation/emulated_program_runner.cpp`. Live extents:
`[metal] emule_live_ranges.{hpp,cpp}`, fed from `[metal] tt_metal/impl/buffers/buffer.cpp`.
**What it catches:** a kernel writing to L1/DRAM at or above the unreserved base
but inside no live buffer — scribbling on memory it never legitimately addressed.
**How it works:** at launch the host snapshots every live buffer's extent
(`LiveL1Ranges` / `LiveDramRanges`, fed from `Buffer::allocate`/`deallocate`) into
per-kernel range arrays. Each L1 buffer is registered with its **per-core
footprint** (`aligned_size_per_bank()`, what the allocator reserves on one core),
not the aggregate `size_`: `size_` spans all banks, so a sharded buffer's full size
at a single core's local offset would register a range many times the core's L1
(e.g. a 64-core shard → a 4.6 MB range on a 1.5 MB L1). That over-permits this OOB
check and makes the Object Intent (§12) snapshot read past the core's L1 into
adjacent memory, false-positiving unrelated cores' writes.

Both buffer-creation paths must register the extent: owning buffers in
`allocate_impl()`/`deallocate_impl()`, and **explicit-address (non-owning)** buffers —
e.g. the per-physical-device L1 buffers `MeshBuffer::initialize_device_buffers` builds,
which never run `allocate_impl()` — in the `Buffer::create(address, …)` overload and
`Buffer::deallocate()`. Without the latter, the runner's physical-`device->id()`
snapshot is empty for every mesh sharded-L1 buffer and each legitimate access
false-positives. The non-owning `deallocate()` removes the range under an
`allocation_status_` guard so the explicit-call + destructor double-deallocate drops it
exactly once.

On each access the kernel first **normalizes the address
to a buffer-relative offset** via `__emule_addr_to_offset` — necessary because
sharded / CB / `l1_alloc` accesses arrive as absolute bridge pointers, not
offsets, and a raw absolute value would never match a relative range — then checks
that offset against the live extents. In none → abort.
*Diagnostic:* `Out-of-Bounds Write: Attempted to access address 0x… which is not part of any allocated tensor`.
*Exercised by:* `test_write_outside_tensor.cpp` (L1 + DRAM gap death tests +
in-bounds L1 and DRAM positive controls, a host-poke fallback accept control
confirming raw L1 the host designated via `WriteToDeviceL1` is not flagged, and a
just-past-the-poke death test confirming the fallback is not a blanket whitelist).

#### Host-poked L1 regions (false-positive fix)
The L1 OOB check assumes every legitimately-accessed L1 address ≥ the unreserved
base is inside a tracked `Buffer` (the only thing that feeds `LiveL1Ranges`, via
`Buffer::allocate_impl`). That assumption breaks for code that uses **raw L1**: the
`tests/tt_metal/.../data_movement/` micro-benchmarks take the L1 scratch base from
`get_l1_address_and_size()` (= `hal.get_dev_addr(DEFAULT_UNRESERVED)`), fill it with
`WriteToDeviceL1`, and hand that bare address to a kernel. No `Buffer` is ever
allocated, so `LiveL1Ranges` has no extent, and a kernel read of that address (e.g.
the `noc_async_write_multicast_loopback_src` source in `sender_multicast_sem.cpp`)
**false-positives** as an OOB write — the memory is valid, just untracked.

Fix (two tiers, both feeding the add-only/deduped `LiveL1HostPokeRanges` registry
in `[metal] emule_live_ranges.{hpp,cpp}`, snapshotted per launch into the kernel
thread-local `__emule_l1_host_ranges`, which `__emule_asan_check_oob_tensor` scans
**after** a tensor miss and **before** aborting):

1. **Precise per-poke (general).** `WriteToDeviceL1` / `ReadFromDeviceL1`
   (`[metal] host_api/tt_metal.cpp`) register the exact `[addr, addr+size)` they
   touch when ASAN is on. This covers any host-declared raw L1 — sources, and
   destinations the host pokes before launch — with no loss of precision, in any
   context (not just DM tests).

2. **DM-suite scratch extent (timing gap).** Tier 1 can't cover an output the
   kernel *writes* and the host only `ReadFromDeviceL1`s **after** the launch (e.g.
   the `all_from_all` / `transaction_id` / `one_packet` requestors): at
   launch-snapshot time that region was never poked. So the DM test helper
   `get_l1_address_and_size()` (`[metal] tests/.../data_movement/dm_common.cpp`)
   registers the whole unreserved-L1 extent it hands out, ASAN-gated, when emule
   ASAN is on. Every raw-L1 DM benchmark carves its src/dst from this extent, so
   one registration covers the entire suite — including outputs read back
   post-launch — pre-launch.

   *Why this doesn't weaken the checker.* The registration lives in DM-test code,
   so it only affects the `unit_tests_data_movement` binary, which allocates **no**
   tracked `Buffer`s — the L1 OOB check there has no tensor to protect and could
   only ever fire as a false positive. Production ttnn ops never call
   `get_l1_address_and_size`, so their OOB precision is untouched. It is
   `if constexpr (kEmuleAsanBuild)`-guarded so non-emule builds don't link the
   registry.

Two properties make this safe:
- **Precise, not a blanket whitelist.** Only the exact `[addr, addr+size)` the host
  poked is accepted — a kernel writing *past* that region, or into untouched L1,
  still aborts. Real OOB bugs (e.g. the untilize / repeat / nd_reshard kernel
  overruns) are not masked.
- **Invisible to Object Intent (§12).** Host-poke ranges live in their own array,
  are **not** appended to `__emule_self->san_resolved_log`, and are **not** snapshotted
  by Object Intent — so a host-NOC write into a poked destination can't be
  misread as an "unintended write."

This is L1-only. The DRAM OOB check (`__emule_dram_ptr`) uses a per-bank/flattened
address convention distinct from the per-channel address `WriteToDeviceDRAMChannel`
takes, so the same registration is not yet wired for DRAM; if a `dram_*` raw-DRAM
benchmark surfaces the analogous false positive, it needs that convention resolved
first.

### 5. Tensor Padding Violation
**Lives in:** `__emule_asan_check_padding` in
`[emule] include/jit_hw/asan/asan_l1_checks.h`; padding bands registered in `[metal] emule_live_ranges.{hpp,cpp}`
(`LiveL1PaddingRanges`) from `emule::register_logical_size` in `[metal] host_sanitizers.cpp`.
**What it catches:** a kernel writing into the padding gap
`[logical_end, physical_end)` of a buffer whose logical size is smaller than its
allocated physical size.
**How it works:** `register_logical_size` registers the padding band; the runner threads
those `(logical_end, physical_end)` pairs into the kernel. After the OOB check, the
normalized offset is tested against each padding band — inside one → abort.
*Diagnostic:* `Tensor Padding Violation: Attempted to write to a padded memory region at 0x…`.
*Exercised by:* `test_padded_write.cpp`.
**Status — currently skipped:** `test_padded_write.cpp` is `GTEST_SKIP`'d. The
check as implemented (a flat `[logical_end, physical_end)` band test) is too
coarse to be correct in general — real padding is laid out per-row / per-tile-face,
not as one trailing block, so the simple band both misses interior pad bytes and
risks false-positiving legitimate writes. The check needs to be reworked to model
the actual (row-major / tiled face-aware) pad layout before the test is re-enabled.

### 6. Illegal Semaphore Access
**Lives in:** `__emule_asan_check_semaphore` in
`[emule] include/jit_hw/asan/asan_l1_checks.h`; the reserved range is set by the runner
(`[metal] emulated_program_runner.cpp`).
**What it catches:** a kernel doing a raw scalar L1 access into the reserved
semaphore region (semaphores must go through the semaphore API).
**How it works:** the runner passes the semaphore L1 range
(`__emule_sem_l1_range_start/end`) to the kernel. It's the **first** test in
`__emule_local_l1_to_ptr`: if the address is in that range, abort.
*Diagnostic:* `Illegal Semaphore Access: Offset 0x… is inside the reserved Semaphore region [start, end)`.
*Exercised by:* `test_semaphore_write.cpp` (an in-region write death test + an
outside-region positive control).

### 7. CB Boundary Violation
**Lives in:** `__emule_asan_cb_resolve` in
`[emule] include/jit_hw/asan/asan_l1_checks.h`; the `reserved`/`waited` page
counters are maintained by the `__emule_asan_cb_on_*` helpers in
`[emule] include/jit_hw/asan/asan_cb.h`.
**What it catches:** a kernel reading/writing a circular-buffer page **outside the
window it actually reserved/waited for**.
**How it works:** when an access lands inside a CB's backing memory, the check
computes the accessed page's distance from `write_idx`/`read_idx` and compares
against the pages currently `reserved`/`waited` (`__emule_cb_reserved_pages` /
`__emule_cb_waited_pages`, maintained by `cb_reserve_back`/`cb_wait_front`). It
fires **only when a window is active** (`reserved > 0 || waited > 0`) and the page
is outside both. Two classes of legitimate raw `get_write_ptr`/`get_read_ptr`
addressing are exempt so they are not false-positived:
- **No active handshake** (`reserved == 0 && waited == 0`) — single-buffered
  scratch, output CBs written then DMA'd.
- **Globally-allocated / sharded CBs** (`set_globally_allocated_address`) — these
  are addressed across the whole backing buffer via computed offsets and only call
  `reserve_back` *nominally* (e.g. matmul `cb_in0_sharded` reserves 1 but reads the
  full shard). The `CBSyncState::globally_allocated` flag (set in
  `init_core_cb_sync` from `cb_impl->globally_allocated()`) skips the window
  sub-check for them even when a nominal window is active.
- **Reuse of produced data** — an access into the already-produced-but-unconsumed
  region `[read_idx, write_idx)` holds valid data the kernel may legitimately
  re-read or re-derive (e.g. conv `activation_reuse` writes back into earlier rows
  at `cb_start + pixel_row*reuse_offset`). Only an access outside the active window
  **and** outside this produced region reaches not-yet-produced free space — the
  real over-reach hazard (#46843). This is what keeps the genuine catch alive while
  dropping the reuse false positive.

A write *past* the CB's allocated region is caught by the OOB check (§4) instead.
*Diagnostic:* `CB Boundary Violation: Attempted to access CB <id> at offset 0x… outside the write/read window`.
*Exercised by:* `test_write_beyond_res_pages.cpp` (write side, read side, a
wraparound positive control + a wraparound violation that confirms the modular
window stays active through a wrap, a no-active-window control, a produced-region
reuse control, and a globally-allocated-CB exemption control confirming a
sharded/global CB accessed outside its nominal window is not flagged).

### 8. CB Reservation Overflow  *(always on)*
**Lives in:** `__emule_asan_cb_on_reserve` in `[emule] include/jit_hw/asan/asan_cb.h` (called from `cb_reserve_back`).
**What it catches:** `cb_reserve_back(cb, n)` asking for more pages than the CB
holds in total.
**How it works:** before blocking to wait for free space, compare `n` against the
CB's `num_pages`; if it exceeds capacity, abort. **Always on** (not gated by the
env var) because gating it would let an over-reserve *deadlock* on the space wait
instead of reporting a clear error.
*Diagnostic:* `CB Reservation Overflow: CB <id> has <N> total pages, …`.
*Exercised by:* `test_cb_pages.cpp` (an over-reserve death test, an
exact-capacity positive control for the `>` boundary, and an always-on death
test with the master switch explicitly cleared).

### 9. NoC Read Pending on `cb_pop_front`
**Lives in:** `__emule_asan_cb_on_pop` in `[emule] include/jit_hw/asan/asan_cb.h`
(called from `cb_pop_front`); the pending counter is incremented/cleared in
`noc_async_read` / `noc_async_read_barrier` in
`[emule] include/jit_hw/api/dataflow/dataflow_api.h`.
**What it catches:** popping (freeing) a CB page while a `noc_async_read` hasn't
been barriered — the pop releases the page for the producer to refill, and a read
still landing into it would race the refill (consumer reads stale/torn data on
silicon). The check sits on **pop**, not push: a pop frees the page, so all reads
must have completed first; before a push only writes precede it, so an unbarriered
read there is harmless.
**How it works:** `noc_async_read` increments a thread-local
`__emule_pending_noc_reads`; the read barrier clears it. `cb_pop_front` aborts if
that counter is still > 0 (a barrier was skipped before the pop).
*Diagnostic:* `Race Condition: cb_pop_front(cb_id=…) called while a NoC read is still pending`.
*Exercised by:* `test_noc_without_barrier.cpp` (a missing-barrier death test on
the raw-read path + a barrier-present positive control, an addrgen-path
missing-barrier death test via `noc_async_read_page` confirming that increment
site is covered, and a multi-read/single-barrier control pinning the
clear-to-zero — not decrement — semantic).

### 10. NOC Transfer Alignment
**Lives in:** `__emule_check_noc_read_alignment` / `__emule_check_noc_write_alignment`
in `[emule] include/jit_hw/api/dataflow/asan/asan_dataflow.h`, called at the top of
`noc_async_read`/`noc_async_write` in `dataflow_api.h`. (DRAM-vs-L1 is decided by
`__emule_noc_addr_is_dram` in `[metal] emulated_program_runner.cpp`.)
**What it catches:** a NoC read/write whose endpoint isn't aligned to its memory
type's requirement.
**How it works:** each endpoint is checked **against its own alignment** —
absolute, per-side, *not* a relative "low bits of src and dst must match" rule:
L1 = 16 B; DRAM read = 32 B (WH) / 64 B (BH); DRAM write = 16 B. So a DRAM read
from a 32-aligned source into a 16-aligned L1 destination is legal even though
their low bits differ.
*Diagnostic:* `NOC Transfer Alignment: <L1|DRAM> <source|destination> 0x… must be N-byte aligned`.
*Exercised by:* `test_alignment_writes.cpp` (7 misalignment death tests + 3 positive controls), covering each per-side branch: L1 destination/source on read, L1 source/destination on write, DRAM source on read, DRAM destination on write, plus an L1-16B positive control. The DRAM-read tests are split per arch — `_WH` variants bake in the 32 B rule, `_BH` variants the 64 B rule. Each queries `device->arch()` at runtime and `GTEST_SKIP`s when it doesn't match, so the bare `Noc*` filter is safe on either cluster (the wrong-arch DRAM variants skip). The regression runners additionally pre-exclude the wrong-arch variant by filter (`Noc*:-*_BH` on wormhole, `Noc*:-*_WH` on blackhole).

---

## Runner / post-launch checks

> These live in the runner
> (`[metal] tt_metal/impl/emulation/emulated_program_runner.cpp`, with the Dirty-CB / Object-Intent logic in `emule_sanitizers.cpp`) and catch
> program-structure invariants that no single per-access check can see. Most run
> after all kernel threads join; **Dirty CB** runs at each kernel's exit (it reads
> per-kernel thread-locals that are cleared on teardown).

### 11. Dirty CB Detected
**Lives in:** `sweep_per_kernel_dirty_cbs` (abort in `abort_if_dirty_cb`) in
`[metal] emule_sanitizers.cpp`. Reads the per-kernel thread-local
*trailing-dangling* flags `__emule_cb_reserve_dangling[]` /
`__emule_cb_wait_dangling[]` (maintained by the `__emule_asan_cb_on_*` helpers in
`[emule] include/jit_hw/asan/asan_cb.h`, called from `cb_reserve_back`/`cb_push_back`
and `cb_wait_front`/`cb_pop_front` in `api/cb_api.h`).
**What it catches:** a kernel that exits with a `cb_reserve_back` that **no**
`cb_push_back` ever followed, or a `cb_wait_front` that **no** `cb_pop_front` ever
followed — the producer/consumer claimed the handshake but never handed off, so on
silicon the matching `cb_wait_front` hangs.
**How it works:** `__emule_cb_reserve_dangling[cb]` is set by `cb_reserve_back` and
cleared by **any** following `cb_push_back`; `__emule_cb_wait_dangling[cb]` is set by
`cb_wait_front` and cleared by **any** following `cb_pop_front`. A flag still set at
kernel exit is the leak; the page count reported is the window counter
(`__emule_cb_reserved_pages[]` / `__emule_cb_waited_pages[]`) at exit, which for a
genuine dangling reserve is the unpushed amount.
**Why a flag, not a net count (the faithful-to-silicon fix):** on silicon
`cb_reserve_back(n)` is a **non-cumulative free-space wait** — it claims nothing,
moves no pointer, and creates no obligation to push exactly `n`. So a net
`reserved − pushed` count (the original model) **false-positives** on legitimate
**lookahead / double-buffer producers**: the DRAM-sharded matmul in1 reader
(`reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`) reserves 2 blocks of headroom
but pushes 1 per iteration to cover the previous block's still-in-flight reads
(invisible to the free-space count), yet pushes **every** block it produces — leaving
a net residual of ≈ `num_blocks × in1_block_num_tiles` "unpushed" pages that are
nothing but speculative headroom. The dangling flag is decoupled from the window
counters precisely so the **CB Boundary** check (§7) can keep using the cumulative
window while this check stops mis-reading it as a leak. **Trade-off:** a
`reserve;reserve;push` (one *intermediate* push forgotten) clears the flag and is
**not** caught here — that pattern overwrites the first block in place and surfaces
via the Object-Intent (§12) / OOB (§4) checks or a PCC mismatch instead. This is a
per-kernel property (reserve pairs with push inside the producer, wait with pop
inside the consumer), so it is checked at **each kernel's exit** (right after its
variants run, before the thread-locals are cleared) — not post-join. Note this is
**not** about leftover occupancy: a producer that reserves+pushes but is never
consumed ends with pages occupied yet fully handed off (dangling flag clear) and is
correctly **not** flagged (globally-allocated/sharded output CBs, producer-only
programs that DMA their result out).
*Diagnostic:* `Dirty CB Detected: Core (x, y) CB <id> was not flushed! Kernel (processor P): <N> page(s) reserved via cb_reserve_back at <file:line> were never committed with cb_push_back. … <M> page(s) waited … were never released with cb_pop_front.`
*Per-check opt-out:* set `TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB=1` (non-empty, not `0`)
to suppress **only** this check while every other sanitizer stays active under the
master switch. `sweep_per_kernel_dirty_cbs` returns early when
`dirty_cb_check_skipped()` (host_sanitizers.hpp) is true. Use it to run a full
regression past a kernel with a known un-flushed-CB bug without losing OOB /
Padding / Object-Intent / CB-Boundary coverage. The `test_cb_leak.cpp` death tests
`unsetenv` it so they still validate the check even when it is exported globally.
**Why this is the only check with its own switch:** un-flushed/leaked CBs are by
far the most common *known-but-not-yet-fixed* violation surfaced when sweeping the
whole op suite (real kernels are routinely mid-fix for a CB-handoff bug), so a
dedicated escape hatch lets the rest of the suite keep running at full coverage.
The other checks rarely need blanket suppression — the master switch
(`TT_METAL_EMULE_ASAN`) already covers the all-off case. The env var is re-read on
every call (not cached) for the same reason as the master switch: a static cache
would stick to the first value observed across a combined gtest binary that toggles
it between test cases.
*Exercised by:* `test_cb_leak.cpp` (reserve-without-push, wait-without-pop, a
lookahead-reserve no-violation control that pins the false-positive fix, a balanced
no-violation control, and the per-check opt-out env test).

### 12. Object Intent Violation
**Lives in:** `[metal] tt_metal/impl/emulation/emule_sanitizers.cpp` (the
pre-launch byte snapshot + post-join `memcmp`); the per-kernel "resolved set"
(the fiber ctx `__emule_self->san_resolved_log`) is recorded by `__emule_asan_check_oob_tensor`
(`[emule] include/jit_hw/asan/asan_l1_checks.h`), reached via the
`__emule_local_l1_to_ptr` chokepoint.
**What it catches:** a kernel that scribbles on *another* buffer's bytes — valid,
allocated L1, but a buffer it never took a pointer into via the public API (a
provenance/aliasing bug).
**How it works:** before launch the runner snapshots every live L1 tensor's bytes
per core. Each kernel L1 access records the `(start,end)` of the buffer it resolved
into — its "intended write set". After the kernel exits, the runner `memcmp`s the
snapshot vs current L1 for every buffer **not** in the resolved set; any change is
an unintended write. (Exact attribution requires one kernel per core; multi-kernel
programs hit a friendlier early-out.)
**Concurrency — the resolved-set append is single-kernel-only.** Each kernel thread
records its resolved ranges into a thread-local stack array, then at exit appends them to
the per-core `resolved_acc_` vector. That append is gated on `snapshots_` being non-empty,
which is true **only** for single-kernel cores (the pre-launch snapshot bails when
`num_kernels != 1`, and `verify_post_launch` early-returns with no snapshot anyway). The
gate is load-bearing: on a multi-kernel core all kernel threads exit concurrently, so an
ungated append would have every thread mutating the shared `resolved_acc_` vector at once
— an unsynchronized `std::vector::insert` (UB, heap corruption) whose result is never read.
Gating on `snapshots_` confines the append to the single-thread case at no behavioral cost.
**Exempt buffers (never snapshotted, so writes to them never flag):** (1) *persistent
buffers* — globally-allocated CB backing buffers (`cb_impl->globally_allocated()`); the
CB *is* the tensor, so the kernel owns it. (2) *I/O tensors handed to this kernel* — any
live-tensor whose L1 start address appears in the kernel's runtime args. A buffer passed
in as a runtime arg is one the kernel was explicitly told to operate on (in-place ops,
fused producers/consumers), even if it "belongs" to another kernel's context, so writing
to it is legitimate. The base address passed as a runtime arg equals the buffer's start
offset (same address space, no normalization), so the match is exact.
*Diagnostic:* `Object Intent Violation: Attempted to modify memory belonging to an adjacent object context — L1 buffer [start, end) … changed but no pointer was resolved into it`.
*Exercised by:* `test_valid_mem_wrong_alloc.cpp` (adjacent + non-adjacent
violations, a resolve-both control, an I/O-arg-exemption positive control
that confirms a buffer handed to the kernel via runtime args is NOT flagged, a
globally-allocated-CB exemption control confirming the kernel's own persistent CB
is not flagged, and a multi-kernel-core control confirming the check cleanly
no-ops when a core runs more than one kernel — the single-kernel attribution gate).
Note: a violation test must pass the victim's *byte offset*, never its absolute
address — passing the address would exempt the victim as an I/O tensor and mask
the violation.

---

## Each check's core mechanism + home, in one line

| Check | Lives in | The trick |
|---|---|---|
| Use-After-Free | `[metal] host_sanitizers.hpp` | `buffer.is_allocated()` at host entry |
| Host L1/DRAM Alignment | `[metal] host_sanitizers.hpp` | `address % get_alignment_requirements(device, size)` (1 ⇒ no-op on emule) |
| Metadata Overflow | `[metal] host_api/tt_metal.cpp` | static CB region vs lowest L1 alloc, at configure time |
| Out-of-Bounds Write | `[emule] asan/asan_l1_checks.h`; `[metal] emulated_program_runner.cpp` | normalized offset ∉ any live `LiveL1Ranges`/`LiveDramRanges` extent |
| Tensor Padding *(test skipped — see §5)* | `[emule] asan/asan_l1_checks.h` | offset ∈ `[logical_end, physical_end)` padding band |
| Illegal Semaphore | `[emule] asan/asan_l1_checks.h` | offset ∈ reserved semaphore L1 range |
| CB Boundary | `[emule] asan/asan_l1_checks.h` (counters in `asan/asan_cb.h`) | accessed page outside an **active** reserve/wait window |
| CB Reservation Overflow | `[emule] asan/asan_cb.h` | `cb_reserve_back(n)` with `n > num_pages` (always on) |
| NoC pending on pop | `[emule] asan/asan_cb.h` + `dataflow_api.h` | `cb_pop_front` while `__emule_pending_noc_reads > 0` |
| NOC Transfer Alignment | `[emule] api/dataflow/asan/asan_dataflow.h` | each endpoint vs its own absolute alignment (16 / 32 / 64 B) |
| Dirty CB | `[metal] emule_sanitizers.cpp` (+ `[emule] asan/asan_cb.h`) | trailing-dangling flag: a `reserve_back` with no following `push_back` (or `wait_front` w/o `pop_front`) at kernel exit — decoupled from the cumulative window count so lookahead producers aren't false-flagged; opt out with `TT_METAL_EMULE_ASAN_SKIP_DIRTY_CB=1` |
| Object Intent | `[metal] emule_sanitizers.cpp` | post-launch `memcmp` of buffers never resolved into |
