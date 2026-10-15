# X280 device images

How X280 firmware images are written and built in this tree: the build function, the boot
code and linker scripts, the lifecycle harness every active image runs and the obligations it
leaves to the image, NoC access from firmware, the images the build produces and the checks
that guard them. The runtime that loads them is in [runtime.md](runtime.md).

## Building an image

An image is a freestanding C program compiled when tt-metal builds, with the
SFPI compiler; nothing is compiled for the X280 at run time. Add one with
`x280_add_image` (`x280_image.cmake`) in this directory's `CMakeLists.txt`:

```cmake
x280_add_image(NAME my-image SOURCES src/my_image.c DEFINES MY_IMAGE_MODE=1)
```

It builds `build/x280/my-image.{elf,bin}` as part of `x280_images`. An active
image, the default, links the lifecycle harness and `ld/x280.ld` at
`0x08001000`; its `main` follows the active-image contract (§ The active-image contract). `LIM_IDLE`
links the per-hart entry and `ld/x280-lim-idle.ld` at `0x08000000` instead;
only the resident idle is built that way.

## Writing firmware

### Layout

| File | Purpose |
|---|---|
| `boot/entry.S` | Reset vector → init gp/sp/mtvec → call `main(hartid)`, or `x280_active_start(hartid)` when assembled with `X280_ACTIVE_HARNESS`. Trap handler has IPI + probe modes. |
| `ld/x280.ld` | Linker script for active code running from LIM (`0x08001000`). |
| `ld/x280-lim-idle.ld` | Linker script for resident idle, at the start of LIM (`0x08000000`). |
| `include/x280_address_map.h` | Canonical LIM, DDR, CLINT, image, trap, sentinel, and ECC addresses; safe for C, C++, and preprocessed assembly. |
| `include/x280_hart_protocol.h` | Canonical per-hart commands, phases, record offsets, wire structs, and layout assertions; safe for C, C++, and preprocessed assembly. |
| `include/x280.h` | Umbrella over `x280_address_map.h` and `x280_hart_protocol.h`, plus `HART_STACK_SIZE`. |
| `include/noc.h` | NOC TLB programming and address encoding helpers. |
| `src/x280_active_start.c` | The lifecycle harness every active image runs (`entry_active.o` calls it; it calls the image's `main`). |
| `src/main.c` | Runtime-owned active success image (`firmware.bin`). |
| `x280_image.cmake` | `x280_add_image()`: compiles and links one image with SFPI (§ Images and build checks). |

### Linker script picks

- Current active images use `ld/x280.ld` and the lifecycle harness.
- Resident idle uses `ld/x280-lim-idle.ld` (`x280_add_image(... LIM_IDLE ...)`).

### Multi-hart

`entry.S` puts each hart at `sp = _stack_top - hartid * 0x8000` and calls
`main(hartid)` with `mhartid` in `a0`. All four harts run it. A standalone
image's `main` must check `hartid` and decide what unused harts do; use IPI
(`msip`) from hart 0 to wake them.

An **active image** — one the runtime loads into the LIM active-FW region and
reaches by `JUMP` from the resident idle FW — does not decide that. It links
the active harness (every `x280_add_image()` without `LIM_IDLE`), so `entry.S` enters `x280_active_start(hartid)`
instead, and a hart with no work just returns from `main`.

### The active-image contract

The harness (`src/x280_active_start.c`) owns the host↔hart lifecycle:

```text
_start -> x280_active_start(hartid)
              publish RUNNING
              -> main(hartid)
              commit terminal success
              wait for the host's STOP
              take the acknowledged return to the idle FW
```

It carries no hart-to-hart dependency — no director hart, no launch barrier,
no termination broadcast — so every hart stays independently observable. Five
obligations are left to the image, and `X280ActiveContract` in
`tests/device/test_x280_active_hw.cpp` enforces them:

1. **`main` returns once `x280_hart_stop_requested(hartid, run_value)` is
   true — on every hart, not only the coordinator.** Any loop that can wait
   indefinitely (a config wait, a command wait, a worker's dispatch poll) must
   poll it, or the host cannot end that hart at all. This is the load-bearing
   one: it also makes obligation 2 non-fatal, because a coordinator that
   forgets to release its workers does not strand them.
   The `x280_stop_poll` CTest checks it statically, with no hardware;
   `X280ActiveContract` catches what the static check cannot.
   The `p2p_latency` X280 images break it in their timed poll, which checks
   STOP only when built with `-DP2P_LATENCY_POLL_CHECKS_STOP`: a token that
   never arrives there strands the hart until `tt-smi -r`.
2. **Every hart reaches `return`.** One hart that never returns blocks every
   aggregate wait.
3. **No WFI.** Host NOC writes raise no interrupt on this part, so a
   WFI-parked hart never wakes.
4. **No `.bss`.** The harness zeroes none, and the link asserts it. Mutable
   globals need `__attribute__((section(".data")))` and a non-zero
   initialiser.
5. **Cross-hart coordination is the image's own** — its mailboxes, values and
   dispatch shape. The harness provides none, which is why the per-image
   armed/gate rendezvous in `atomic_bench.c` stays. A
   rendezvous is a wait under obligation 1, on the coordinator's side as much
   as the participants'. Once any hart can leave on STOP, a barrier that spins
   on a peer's flag must poll STOP too, or the coordinator strands on a flag
   nobody remains to set.

Publish a result with `x280_hart_set_result(hartid, value)`. The harness seeds
nothing: `enqueue` clears the record first, so an image that computes nothing
publishes zero. Only the sentinel echoes the run value, so
`hart_result_ready` does not check the payload — `active_result_matches` is
the echo check.

Each harness step advances a free-running `stage` word in the hart's state
record. No predicate reads it; it exists so that a hart stuck in an image is
distinguishable from one waiting on the host, which `phase` alone cannot say.
`describe()` prints every hart's record, so read the stage of whichever hart is
in the wrong state:

| `stage` | Name | Meaning when a wait times out here |
|---|---|---|
| 0 | — | the hart never entered the harness |
| 1 | `ENTERED` | RUNNING published, `main` not yet reached |
| 2 | `MAIN` | inside `main` — a loop there is not polling STOP (obligation 1) |
| 3 | `POSTED` | terminal success committed; the host side is at fault |
| 4 | `STOPPING` | awaiting the host's STOP |
| 5 | `RETURNING` | awaiting `RETURN_ACK` |

## NoC from firmware

### Outbound paths from X280 (best to worst)

| Path | Target | Peak BW | Mechanism |
|---|---|---|---|
| **System Port vec write** | Tensix L1 | **14.4 GB/s** | 64B vector stores via TLB |
| System Port vec write | DRAM | 4.0 GB/s | Same, DRAM controller backpressure |
| System Port scalar write | Tensix L1 | 3.6 GB/s | 8B stores |
| Memory Port (cached) | DDR via L3 | local-only | Stays in L3 cache |
| **DMA M2M** | DRAM/Tensix | 2.34 GB/s | Synopsys DW_ahb_dmac at `0x2FF80000` |
| System Port read | NOC target | 0.15 GB/s | Non-posted round-trip — avoid |

### Inbound paths to X280

| Path | Source | Peak BW | Notes |
|---|---|---|---|
| Front Port write | Tensix or external | **10.24 GB/s** | Posted writes, dest-bound |
| MemPort cached read | DDR | 7.92 GB/s | 128 KB cache-miss |

### TLB programming (System Port outbound)

```c
#include "noc.h"
// 2 MB window 0 → NOC tile (X=1, Y=11), local addr 0x0
volatile void *win = noc_configure_tlb_2m(/*win=*/0, /*x=*/1, /*y=*/11,
                                          /*local_addr=*/0x0,
                                          /*posted=*/1, /*strict_order=*/0);
volatile uint32_t *dst = (volatile uint32_t *)win;
for (int i = 0; i < 16; i++) dst[i] = 0xDA7A0000 + i;
__asm__ volatile("fence ow, ow");
```

There are **224 × 2 MB** windows + **32 × 128 GB** windows. Pre-program
once for scatter-to-N-cores; switching between pre-configured windows
costs zero cycles.

`noc_configure_tlb_2m` applies no bound: x and y each occupy a six-bit field, so
64 carries into the neighbouring field and the window then aims at a real tile
that is the wrong one. Where the coordinate is not a constant the firmware chose
— a host-supplied value, a peer address out of a mailbox — use
`noc_configure_tlb_2m_checked`, which returns `NULL` and programs nothing rather
than truncating. A benchmark image opens its window with `noc_open_window_2m`,
which takes a `noc_ordering_t` (Default, Strict AXI or Posted Writes, never
Counted Writes) and a request VC (`NOC_VC_DYNAMIC`, or a static unicast VC 0-3)
and reads the entry back. It never clears a window whose
readback differs (`NOC_WINDOW_UNVERIFIED`): whether that mapping is the image's
to release is the image's policy, stated where it opens the window.
`noc_clear_tlb_2m` zeroes a window's configuration and
confirms the zeroing took, which an image owes on every exit path: a window left
programmed is a standing path from this tile into whatever now owns the address
it names. Zero is not a disabled entry — the bitfield has no valid bit, so an
all-zero window still routes to NOC0 (0, 0) — but it names no tile the caller
was reaching, which is the property that matters.

### Vector writes for peak bandwidth

```c
asm volatile("vsetvli zero, %0, e32, m1, ta, ma" : : "r"(16));
asm volatile("vle32.v v0, (%0)" : : "r"(src) : "memory");
asm volatile("vse32.v v0, (%0)" : : "r"(dst) : "memory");
```

VLEN=512 → 64 bytes per `vse32.v` store. This is what gets you to
14.4 GB/s into Tensix L1.

### Host → X280 NOC writes

A host NOC access to an L2CPU takes the L2CPU-local address (for example, `X280_LIM_BASE` =
`0x08000000` and `X280_LOCAL_DDR_TILE_BASE` = `0x400030000000`, from
`x280_address_map.h`).

## Images and build checks

Built by `ninja -C build x280_images` ([README](../README.md#2-build-boot-and-run-a-benchmark)):

| Image (`build/x280/`) | Source | Use |
|---|---|---|
| `firmware-lim-idle.bin` | `src/lim_idle.c` | resident idle, the runtime prerequisite |
| `firmware.bin` | `src/main.c` | active session success image (the sentinel) |
| `firmware-pinned-input-prepare.bin` | `src/x280_pinned_input.c` | pinned-input preparation, run by `X280Tile::prepare_pinned_input` ([runtime.md](runtime.md#prepared-pipeline-inputs-x280-only)) |
| `firmware-atomic-bench.bin` | `src/atomic_bench.c` | cross-hart atomics micro-benchmark (§ Cross-hart atomics micro-benchmark); in the active-image contract set |
| `firmware-active-session-fault.bin` | `src/active_session_fault.c` | traps and CEASEs hart 2, for `x280_active_failure_test` only |
| `firmware-transport-bandwidth-x280-to-tensix.bin` | `src/transport_bandwidth_x280_to_tensix.cpp` | X280 -> Tensix member of transport_bandwidth, run by `x280_transport_bandwidth_x280_to_tensix`; the Tensix -> X280 member (`x280_transport_bandwidth_tensix_to_x280`) and its idle-ERISC variant (`x280_transport_bandwidth_erisc_to_x280`) submit no X280 image |
| `firmware-p2p-latency-x280-same-tile.bin` | `src/p2p_latency_x280_same_tile.cpp` | X280 hart <-> X280 hart on one tile, run by `x280_p2p_latency_x280_same_tile` |
| `firmware-p2p-latency-x280-peer-tile.bin` | `src/p2p_latency_x280_peer_tile.cpp` | X280 <-> X280 on two L2CPU tiles, both participants, run by `x280_p2p_latency_x280_peer_tile` |
| `firmware-p2p-latency-metal-peer-tile.bin` | `src/p2p_latency_metal_peer_tile.cpp` | X280 half of the Tensix, idle-ERISC and free-DRISC peers (`x280_p2p_latency_{tensix,erisc,drisc}_peer_tile`) |
| `firmware-pipeline-stage-route-compute.bin` | `src/pipeline_stage_route_compute.cpp` | X280 L2CPU route_compute over the pinned input, run by `x280_pipeline_stage_route_compute` |
| `firmware-pipeline-stage-route-forward.bin` | `src/pipeline_stage_route_forward.cpp` | X280 L2CPU route_forward over the first pinned header, run by `x280_pipeline_stage_route_forward` |
| `firmware-core-sync-director.bin` | `src/core_sync_director.cpp` | the X280 participant or director of `core_sync_barrier`, run by `x280_core_sync_director` and `x280_core_sync_characterization` (built only when named) |
| `firmware-core-sync-release-probe.bin` | `src/core_sync_release_probe.cpp` | test-only release of one or more X280 multicast groups, run by `x280_core_sync_director` |
| `firmware-packet-injection-{latency-include,latency-exclude,throughput,backpressure-include,backpressure-exclude}.bin` | `src/packet_injection_x280.cpp` | the X280 consumer of packet_injection, one image per mode and creation flag; `x280_packet_injection_tensix_to_x280` runs each point's variant and `x280_packet_injection_erisc_to_x280_header` the throughput image. The latency variants link the pinned-input operations |
| `firmware-eth-credit-x280-fwd.bin` | `src/eth_credit_x280_fwd.cpp` | the X280 forwarders of eth_credit on both chips, run by `x280_eth_credit` |
| `firmware-eth-credit-x280-fwd-measure.bin` | `src/eth_credit_x280_fwd.cpp` with `EC_FWD_MEASURE=1` | the same forwarders with their measurement modes (W1, W2 and R1 alone; the round trip), run by `x280_eth_credit`'s `W1AloneBySlotAndP`, `W2AloneByP`, `R1AloneByP` and `RoundTripBySlot` |

Every image compiles with `-mcpu=sifive-x280 -march=rv64gc -mabi=lp64d
-mcmodel=medany -O2 -Werror -fno-zero-initialized-in-bss`, freestanding, and
links with `X280_REQUIRE_NO_BSS` defined, so the linker scripts refuse a `.bss`.
`x280_image_layout` checks resident idle's `.text` at `0x08000000` and the
sentinel's at `0x08001000`. The `x280_abi_headers` build target compiles each
ABI header alone as C, C++ and, for the two `entry.S` includes, assembly.
It also compiles each benchmark suite header under `kernels/` alone, as C and
C++, or as C++ only for the templates.

`x280_stop_poll` scans every source an active image links and fails on any
unbounded `fence ir, ir` wait that cannot reach `x280_hart_stop_requested`. The
scan needs no toolchain and no hardware. Its source list is what
`x280_add_image` records for every image linking the active harness, including
the source behind each prebuilt object it links, so no image can link the harness
without being scanned.

A wait that genuinely must not poll STOP carries
`/* STOP-POLL-EXEMPT(released-by=<agent>): <justification> */` immediately
above it; no wait in this tree carries one. Both fields are mandatory, the
marker binds to the loop that follows it, and a marker on a loop that already
polls or on no wait at all is an error, so it cannot be sprinkled to silence
the check. Every honoured exemption prints on every run.

## Active-image contract suite

`X280ActiveContract` runs one sequence — `enqueue` → `wait_until_running` →
`cleanup`, then `active_all_idle` and `terminal_status == SUCCESS` on all four
harts — against every image named in `X280_ACTIVE_FW_IMAGES`, a colon-separated
list. Adding a conforming image means adding a path, not writing a test. The
list falls back to the singular `X280_ACTIVE_FW_IMAGE` when it is unset, and
with both unset to the build's sentinel and atomic-bench in `build/x280/`.

A path that does not resolve costs its case but is never dropped silently.
`X280ActiveContractImages.EveryRequestedImageResolves` compares what was
requested against what resolved:

| Request | Outcome |
|---|---|
| every path resolves | one case per path |
| some resolve | those cases run; each unresolved path prints as `[  DROPPED ]`; exit 0 |
| none resolve | **fail**, naming every path — the suite would otherwise instantiate nothing and pass |
| the build's images, one missing | **fail**, naming it |
| `X280_ACTIVE_FW_IMAGES` names no path | skip: the suite is opted out |

The request is `X280_ACTIVE_FW_IMAGES`, else a non-empty `X280_ACTIVE_FW_IMAGE`,
else the build's sentinel and atomic-bench. The guard needs no hardware, so a
broken path list fails on a laptop.

Every runtime device test takes its image from `build/x280/` unless its
variable names another (`X280_ACTIVE_FW_IMAGE` for `X280ActiveHw`, which runs
the sentinel; `X280_ATOMIC_BENCH_FW_IMAGE`; `X280_ACTIVE_FAULT_FW_IMAGE`;
`X280_PINNED_INPUT_PREPARE_FW_IMAGE`; `X280_IDLE_FW_IMAGE` for the resident idle the active fixtures boot), so a
run from the build needs none of them. Set `X280_ACTIVE_FW_IMAGES` to exercise
the guard:

```bash
export TT_VISIBLE_DEVICES=0                              # UMD chip id
export TT_METAL_RUNTIME_ROOT="$PWD"
B=$PWD/build/x280
export X280_ACTIVE_FW_IMAGES="$B/firmware.bin:$B/firmware-atomic-bench.bin"

build/test/tt_metal/x280/x280_runtime_device_tests --gtest_filter='*X280ActiveContract*'
```

**Scope every device invocation with `TT_VISIBLE_DEVICES`.** Unscoped, UMD
opens all eight chips, and a fault on any one of them surfaces as a fault on
the one under test. `TT_VISIBLE_DEVICES` and `tt-smi -r` take UMD chip ids;
`/dev/tenstorrent/N` takes PCIe ids. UMD prints the mapping at open
— `Opening local chip ids/PCIe ids: {0..7}/[2,3,1,0,6,7,5,4]` on an 8-chip
p150b host, where UMD chip 0 is `/dev/tenstorrent/2`. Record that line
from each log; a bare "card 0" is ambiguous.

A failure here is a firmware obligation (§ The active-image contract), not a runtime bug. The failure
message renders every hart, so read the `stage` of whichever one is in the
wrong state: `MAIN` means a loop in `main` does not poll
`x280_hart_stop_requested`; `ENTERED` means the hart never reached `main`;
`POSTED` or `STOPPING` means the host side is at fault.

It does **not** prove that a self-completing image holds in `RUNNING` long
enough for the host to read its result — that needs `wait_until_result_ready`,
which an image gated on a config it never receives cannot satisfy. The sentinel
covers that mechanism through `expect_self_completion`.

The contract set is the sentinel (`firmware.bin`) and the atomic benchmark.
Other active images are absent from the list above by decision rather than by
a silent skip:

| Image | Why it is excluded |
|---|---|
| `firmware-active-session-fault.bin` | the deliberate non-conformer, with its own gated binary and its own test |

## Cross-hart atomics micro-benchmark

Models all 4 harts on one L2CPU tile pushing into a single downstream
queue, where each must bump a shared write pointer without colliding. The correctness primitive is an atomic
read-modify-write; this ubench measures its cost and contention.

- Firmware: [`src/atomic_bench.c`](../src/atomic_bench.c) — a **ping-pong
  active FW** (LIM mode, `ld/x280.ld`). All four harts participate in the
  session; hart 0 orchestrates and harts 1–3 are workers coordinated through
  per-hart LIM mailboxes.
- The C++ suite checks this image's lifecycle conformance only, not its
  benchmark protocol or results.

### What it does

Phased (N = 1,2,3,4 harts), started together via a LIM gate so the
contention windows overlap. Each run does **two sections**:

**Section A — constant work per hart.** Phase *N* has *N* harts each
atomically increment **one shared LIM counter** a fixed number of times
(default 1,000,000). Total work grows with N (`N × iters`).
- **Phase 1** = uncontended baseline (cost of one atomic op).
- **Phases 2/3/4** = same address hammered concurrently; the memory
  subsystem serializes the atomics so per-op cost rises. The increase is
  the contention/stall overhead. Headline metric: worst-hart c/op vs the
  1-hart baseline (how much it *degrades*).

**Section B — fixed total work split evenly (strong scaling).** The *same*
total is divided across the N harts (so 4 harts do 250k each,
total still 1M). Answers the real question: *if you have 1M pushes to do,
is it faster to split them across 4 harts, or does atomic contention on the
one write pointer eat the parallelism?* Headline metric: speedup vs 1 hart
doing all of it (ideal = N×; less means contention is winning; a flat
wall-time across hart counts means the atomic is fully serialized).

**Correctness** is checked every phase in both sections two ways:
1. *Final counter* must equal the section's expected total (no lost updates).
2. *Slot-uniqueness.* Each atomic add returns the **old** pointer — the slot
   that producer just claimed (`my_slot = atomic_add(&wptr, 1)`), which is
   what the real code consumes. If every slot `0..total-1` was issued
   exactly once, the sum of all returned slots equals `total*(total-1)/2`.
   The firmware sums the returned slots per hart and publishes each sum in
   its phase record; that total is a direct proof that **no two harts got the same slot** (the exact
   guarantee the write pointer needs), stronger than the counter check alone.

The non-atomic mode is the racy `ld/add/sd` control — you can watch it
*lose* updates (final < expected) **and** fail the slot-sum check (a slot
handed to two producers). That's the "why atomics are required" demo.

### Two hardware facts it exploits

1. **Coherent cluster, uncached LIM.** The 4 harts are cache-coherent, but
   LIM is uncached scratchpad (see tt-isa-documentation
   `BlackholeA0/L2CPUTile`). So an atomic to LIM never bounces a cache
   line between harts — every AMO goes straight to the LIM/L3 controller,
   which serializes them. Clean place to measure raw atomic serialization.
2. **Timing = CLINT `mtime` only.** Each hart brackets its loop with the
   CLINT `mtime` counter, a true 50 MHz wall-clock (20 ns/tick) that ticks
   *through* stalls — so the measured cost includes time blocked waiting on
   the atomic subsystem, which is what a producer actually pays. We
   deliberately do **not** use `rdcycle`: on this X280 it freezes during
   memory stalls (see `x280_rdcycle()` in `rt/x280_hw.h`), so it would hide the very atomic-latency
   cost we want to see.

### Safety: AMO-to-LIM probe

Whether AMOs work against *uncached* LIM is silicon-specific. Before any
untrapped atomic runs, hart 0 arms `entry.S` **probe mode** and issues a
single test op: if it faults, the handler skips it (instead of `CEASE`,
which would permanently halt the hart and strand the L2CPU) and the run
aborts cleanly, writing 0 to its AMO-supported word and status
`AB_STATUS_AMO_UNSUP`.

## Constants

```c
// From x280_address_map.h
#define X280_LOCAL_DDR_TILE_BASE 0x400030000000ULL
#define X280_LIM_BASE            0x08000000ULL
#define X280_CLINT_BASE          0x02000000ULL
```

Host-only reset, PLL, and L3 register addresses are private to
`src/device/x280_hw_addrs.hpp`; callers use `X280Runtime` rather than
duplicating them.
