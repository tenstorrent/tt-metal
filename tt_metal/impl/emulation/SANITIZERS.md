# Emule ASAN-Style Sanitizers — Reviewer Guide

Reviewer-facing summary of the host-side sanitizer surface this PR adds. For
the kernel-side mechanics (especially how unintended writes are detected),
see `tt-emule/docs/ASAN.md`.

## What this PR adds

A set of ASAN-style runtime checks that fire on common host/kernel bugs when
running under the emule software emulator. Each check fires its own
`[ASAN ERROR] <Category>: ...` diagnostic and `abort()`s the process.

## Activation

Single environment variable, off by default:

```
TT_METAL_EMULE_ASAN=1
```

The flag is read on every check via
`tt::tt_metal::emule::emule_asan_enabled()` ([host_sanitizers.hpp](host_sanitizers.hpp)).
No caching — flipping the variable between subprocesses (e.g. `EXPECT_DEATH`
gtests, see below) works correctly. When unset, every sanitizer is a no-op:
the helpers return early at the top of each function and the runner's
thread-local state pointers stay null, so the in-kernel checks short-circuit.

The flag is independent of `TT_METAL_EMULE_MODE` (which selects the emulator
backend vs hardware) — sanitizers can be turned on/off independently of
emulator selection.

## Checks added

| # | Category | Where it fires | Trigger |
|---|---|---|---|
| 1 | Use-After-Free | host entry points | Access through a Buffer whose backing memory was deallocated |
| 2 | L1 Alignment | host entry points | `WriteToDeviceL1`/`ReadFromDeviceL1` address not aligned to the transfer's `Cluster::get_alignment_requirements()` (DMA alignment when DMA-backed; `1` — i.e. no-op — for host/UMD pokes such as emule's memory-backed I/O) |
| 3 | DRAM Alignment | host entry points | `WriteToDeviceDRAMChannel`/`ReadFromDeviceDRAMChannel` address not aligned to the transfer's `Cluster::get_alignment_requirements()` (DMA alignment when DMA-backed; `1` — i.e. no-op — for host/UMD pokes such as emule's memory-backed I/O) |
| 4 | Metadata Overflow | `ConfigureDeviceWithProgram` | Program's static CB region overruns lowest occupied L1 address |
| 5 | Out-of-Bounds Write (L1) | kernel (in `__emule_local_l1_to_ptr`) | Kernel L1 access at-or-above `l1_unreserved_base` doesn't hit any live tensor extent |
| 6 | Out-of-Bounds Write (DRAM) | kernel (in `__emule_dram_ptr`) | Kernel DRAM access doesn't hit any live DRAM tensor extent |
| 7 | Tensor Padding Violation | kernel (in `__emule_local_l1_to_ptr`) | Kernel access falls inside `[logical_end, physical_end)` of a buffer with declared logical size |
| 8 | Illegal Semaphore Access | kernel (in `__emule_local_l1_to_ptr`) | Kernel scalar access into reserved Semaphore L1 region |
| 9 | CB Boundary Violation | kernel (in `__emule_local_l1_to_ptr`) | Kernel L1 access inside a CB's backing memory but outside its active write/read window |
| 10 | CB Reservation Overflow | kernel (`cb_reserve_back`) | `cb_reserve_back(cb_id, n)` reserves more pages than the CB has total (always on — see below) |
| 11 | NoC Barrier Missing | kernel (`cb_push_back`) | `cb_push_back` called while `noc_async_read` is still in flight |
| 12 | NOC Transfer Alignment | kernel (`noc_async_read`/`write`) | Source and dest low bits don't match (3 variants: DRAM→L1, L1→L1, L1→DRAM) |
| 13 | Dirty CB Detected | runner, post-launch | Kernel exited with reserved/waited pages on a CB |
| 14 | Object Intent Violation | runner, post-launch | Kernel modified L1 bytes belonging to a buffer it never resolved a pointer into |
| 15 | Fabric Access Violation | runner, host kernel-launch path | Access to unallocated NOC coordinates |

Checks 1–4 + 13–15 are host-side and live in tt-metal. Checks 5–12 are
kernel-side and live in tt-emule (the JIT-compiled headers under
`include/jit_hw/`).

## What the PR touches in tt-metal

- **[host_sanitizers.hpp](host_sanitizers.hpp)** (new, 80 lines) —
  three host-side checks (UAF, L1/DRAM alignment) plus the master
  `emule_asan_enabled()` switch.
- **[emule_live_ranges.{hpp,cpp}](emule_live_ranges.hpp)** (new) — three
  per-device registries (`LiveL1Ranges`, `LiveDramRanges`,
  `LiveL1PaddingRanges`) wired into Buffer allocate/deallocate so the
  runner can snapshot which buffer extents are live at launch time.
- **[emulated_program_runner.cpp](emulated_program_runner.cpp)** — host-side
  pre/post-launch sanitizer work: live-range snapshot, populating the
  kernel-thread thread-locals, post-join dirty-CB sweep, post-join object-
  intent byte-diff.
- **tt_metal/tt_metal.cpp** — 9 call sites of the host sanitizers + one
  emule-side gate on Metadata Overflow.
- **buffer.hpp / buffer.cpp** — `set_logical_size` now registers the padding
  range in `LiveL1PaddingRanges`; allocate/deallocate update the live range
  registries.

## Why each check exists (priorities)

- **UAF, alignment, metadata overflow**: silent on silicon today; produce
  garbage data or corrupt unrelated allocations. Host-only — cheap.
- **Out-of-bounds, padding, CB boundary, semaphore, NOC alignment, NoC
  barrier missing**: silent or fatal-but-confusing on silicon. The kernel
  is the right place to catch them because the firmware-style L1 offset
  is only meaningful in the kernel's translation context.
- **Dirty CB, Object Intent, Fabric**: detect program-level errors that
  the per-access checks structurally can't see (a CB that gets pushed
  more times than it gets popped is well-formed on every individual
  access; an object-intent violation is by definition through valid
  bytes of L1).

## How "unintended writes" (Object Intent Violation) is detected — short version

The interesting case the reviewer called out. Long version in
`tt-emule/docs/ASAN.md`; here is a 6-step summary.

1. Before launch, the runner snapshots every live L1 tensor's bytes on
   each core.
2. Each kernel thread gets a small thread-local array
   (`__emule_l1_resolved_ranges`, capacity 64) backed by stack storage.
3. Every kernel L1 access goes through `__emule_local_l1_to_ptr`. When
   it resolves to address inside a live tensor extent, the (start, end)
   pair is appended to that array — this is the kernel's "intended write
   set."
4. After the kernel exits, the runner merges all threads' resolved sets
   on that core.
5. For each pre-launch snapshot whose (start, end) is NOT in the merged
   set: `memcmp` snapshot bytes vs. current L1 bytes.
6. Any mismatch is an unintended write — the kernel modified an
   adjacent object's bytes without ever taking a pointer into that
   object via the public API.

Limitations (acknowledged in the diagnostic): exact attribution only
works when one kernel runs on a core per launch. Multi-kernel programs
(producer + consumer) hit an earlier, friendlier abort that tells the
user to keep their ASAN runs single-kernel-per-core.

## Why one check (CB Reservation Overflow) is always on

`cb_reserve_back` blocks in a CV wait until pages are free. Gating the
overflow check would cause an over-reserve to deadlock on the wait
rather than abort with a useful message. The check is structurally
load-bearing for the test runner's responsiveness — keeping it always
on costs one comparison on the slow path and matches the silicon
behavior (a real hardware over-reserve also wedges the FIFO).

## Test coverage

Each sanitizer has a corresponding gtest under
`tests/tt_metal/tt_metal/api/`, each using `EXPECT_DEATH` to capture the
abort message and confirm the right category fired:

```
test_tensor_bad_acess.cpp        — UAF (4 entry points)
test_alignment_writes.cpp        — L1/DRAM alignment
test_metadata_size.cpp           — Metadata Overflow
test_write_outside_tensor.cpp    — OOB L1
test_valid_mem_wrong_alloc.cpp   — Object Intent
test_padded_write.cpp            — Tensor Padding
test_semaphore_write.cpp         — Illegal Semaphore Access
test_write_beyond_res_pages.cpp  — CB Boundary
test_cb_pages.cpp                — CB Reservation Overflow (always on)
test_noc_without_barrier.cpp     — NoC Barrier Missing
test_cb_leak.cpp                 — Dirty CB
test_fabric_allocation.cpp       — Fabric Access
```

All run via the standard `unit_tests_api` gtest binary in `build_emule`.
