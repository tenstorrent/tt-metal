# X280 Bare-Metal Getting Started

End-to-end guide to running real work on the SiFive X280 RISC-V cores
inside the L2CPU tiles on a Tenstorrent Blackhole card. Covers boot,
reset, kernel launch, clock programming, NOC traffic, and I/O paths.

The supported host boundary is the C++ `X280Runtime` described in § 1.

---

## Using the X280 runtime

### Production code

Metal opens the device; the runtime binds to that mesh. A caller opens Metal,
then `X280Runtime::open` with the default `AutoPrime` policy and the LIM shape
it needs. When the open reports `requires_reset()`, the caller closes the mesh,
resets the chip with `tt::umd::WarmReset::warm_reset`, recreates the mesh and
opens again. It then boots the tiles it uses with `boot_tiles`, and runs X280
work beside its Metal workloads:

```cpp
#include "x280/device/x280_runtime.hpp"
#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-umd/warm_reset.hpp>

using tt::tt_metal::distributed::MeshDevice;

auto mesh = MeshDevice::create_unit_mesh(chip);
const x280::RuntimeOpenConfig config{.lim_shape = x280::LimShape{.lim_ways = 15}};  // 10 to 15 LIM ways
auto opened = x280::X280Runtime::open(mesh, config);
if (opened.requires_reset()) {
    // The PCIe device number (/dev/tenstorrent/<N>); an empty list resets every device.
    const int pcie = tt::tt_metal::MetalContext::instance().get_cluster().get_chips_with_mmio().at(chip);
    mesh->close();
    mesh.reset();
    tt::umd::WarmReset::warm_reset({pcie});
    mesh = MeshDevice::create_unit_mesh(chip);
    opened = x280::X280Runtime::open(mesh, config);
}
if (!opened.ready()) {
    throw std::runtime_error(opened.diagnostic);
}
x280::X280Runtime& runtime = *opened.runtime;

const x280::TileId id = runtime.tile_ids().front();
const x280::MultiTileBootRequest boot{
    .tiles = {id}, .idle_image_path = image_dir + "/firmware-lim-idle.bin", .allow_reset_release = true};
if (const auto booted = runtime.boot_tiles(boot); !booted.ready()) {
    throw std::runtime_error(booted.diagnostic);
}

x280::X280Tile& tile = runtime.tile(id);
const uint64_t args = tile.available_lim().begin;
tile.write_available_lim(args, args_block);  // whole 64-byte lines, rewritten for every run
const auto enqueued = runtime.enqueue(
    tile,
    x280::ActiveRunRequest{
        .image_path = image_dir + "/my-image.bin",
        .run_value = ++generation,
        .args_addr = args,
        .args_bytes = args_block.size()});
if (!enqueued.session) {
    throw std::runtime_error(x280::describe(enqueued));
}
EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, /*blocking=*/false);
const auto result = runtime.session(enqueued.session).wait_until_result_ready(std::chrono::seconds(1));
Finish(mesh->mesh_command_queue());
const auto cleaned = runtime.cleanup(enqueued.session, std::chrono::seconds(3));
```

`chip`, `image_dir` (the build's `build/x280/`), `args_block`, `generation` and
`workload` are the caller's.

- **The reset is the caller's choice.** `warm_reset` resets the whole chip:
  Tensix, Ethernet and every L2CPU tile, not just the one being prepared. The
  runtime never resets anything; it reports `ResetRequired` and stops.
- **What survives a reset.** LIM ECC priming persists across a warm reset, so
  the reset after a prime is needed once per device, not per process. Booted
  tiles do not: every reset returns them to reset-held, and each process that
  finds a tile held boots it again. `boot_tiles` boots one batch per chip-reset
  lifetime, so boot every tile you need in one call. It leaves PLL4 at
  `kSafePllMhz`; `set_pll` sets the operating clock under the PLL4 ownership
  rules in `x280_runtime.hpp`.
- **One execution per tile.** A second `enqueue` while a session is live
  returns `Busy`. `cleanup` ends a session once; the runtime destructor cleans
  up any still live.
- **Arguments.** `args_addr` must be cache-line aligned and lie wholly inside
  the tile's `available_lim()`; `enqueue` checks that placement and nothing
  else. LIM keeps the previous run's block, so the image must check that the
  block is this run's, for example by a run value the host writes into it,
  before using it (`x280_hart_args`, `rt/x280_idle.h`).
- **Device-side synchronization is the caller's.** The runtime orders nothing
  between an X280 image and Tensix or Ethernet kernels. `wait_until_result_ready`
  and `Finish` each wait for their own side only; any handoff between the two
  goes through flags or counters the caller places in L1 or LIM.
- **Slow dispatch for idle-ERISC and DRISC peers.** A Metal program on idle
  Ethernet or DRAM RISC-V cores that the X280 exchanges data with needs
  `TT_METAL_SLOW_DISPATCH_MODE=1`.
- **Watcher off.** With `TT_METAL_WATCHER` set, the host NoC sanitizer rejects
  the ARC register reads `open` starts with (§ 1, C++ host runtime boundary).

TODO: a helper that runs this open, reset, reopen and boot sequence and returns
a ready runtime, with the reset still an explicit caller opt-in.

### Tests

Reversible device tests open with `LimEccPrimingPolicy::RequirePrepared`, so
they never commit `WayEnable`, and their fixtures boot resident idle when the
tile is not already running it (`tests/device/x280_test_images.hpp`). On an
unprimed device they skip and name the fix. Prime each device once, then reset
it:

```bash
X280_ALLOW_LIM_ECC_PRIME=1 TT_VISIBLE_DEVICES=0 build/test/tt_metal/x280/x280_prime_test
env -u TT_VISIBLE_DEVICES tt-smi -r 0
```

Tests take images from the build's `build/x280/` (`X280_IMAGE_DIR`, compiled in
by CMake); an `X280_*_FW_IMAGE` variable names another file (§ 6). A test that
changes the chip for the rest of a reset lifetime, or can wedge it, lives in
its own executable and runs only when its `X280_ALLOW_*` variable is `1`:
`x280_prime_test` (`X280_ALLOW_LIM_ECC_PRIME`), `x280_shape_test`
(`X280_ALLOW_LIM_SHAPE_CHANGE`), `x280_pinned_input_test`
(`X280_ALLOW_PINNED_INPUT_PREPARE`), `x280_active_failure_test`
(`X280_ALLOW_ACTIVE_SESSION_FAILURE`) and `x280_pll_test`
(`X280_ALLOW_PLL_SLEW`).

### Building an X280 image

An image is a freestanding C program compiled when tt-metal builds, with the
SFPI compiler; nothing is compiled for the X280 at run time. Add one with
`x280_add_image` (`x280_image.cmake`) in this directory's `CMakeLists.txt`:

```cmake
x280_add_image(NAME my-image SOURCES src/my_image.c DEFINES MY_IMAGE_MODE=1)
```

It builds `build/x280/my-image.{elf,bin}` as part of `x280_images`. An active
image, the default, links the lifecycle harness and `ld/x280.ld` at
`0x08001000`; its `main` follows the active-image contract (§ 3). `LIM_IDLE`
links the per-hart entry and `ld/x280-lim-idle.ld` at `0x08000000` instead;
only the resident idle is built that way.

---

## 1. Hardware overview

A Blackhole chip exposes **four L2CPU tiles**, each containing a SiFive
**X280 cluster of 4 harts** (16 RISC-V cores total per chip). Each
L2CPU tile is a NOC endpoint with its own LIM SRAM, L3 cache, DDR
front port, and three outbound NOC paths.

| Resource | Per-tile | Notes |
|---|---|---|
| X280 harts | 4 | rv64gcv (vector unit, VLEN=512) |
| L3 cache / LIM | 2 MB | LIM = L3 way 0 exposed as SRAM (WayEnable=0 default) |
| DDR via Front Port | shared | Cache-coherent path through L2CPU |
| NOC TLB windows | 224 × 2 MB + 32 × 128 GB | Outbound translation |
| DMA engine | 1 (Synopsys DW_ahb_dmac) | At periph offset `0x2FF80000` |

**L2CPU NOC coordinates on Blackhole:**

| Index | (NOC X, Y) |
|---|---|
| 0 | (8, 3) |
| 1 | (8, 9) |
| 2 | (8, 5) |
| 3 | (8, 7) |

**Memory map (X280 view):**

| Range | Region |
|---|---|
| `0x0000_0000 – 0x0000_FFFF` | Boot trampoline / vectors |
| `0x0200_0000 – 0x020F_FFFF` | CLINT (mtime/msip) |
| `0x0800_0000 – 0x081F_FFFF` | LIM SRAM (when WayEnable=0) |
| `0x4000_3000_0000 +` | DDR (cache-coherent) |
| `0x2FF8_0000` | DMA engine registers |
| `0xFFFFF7FEFFF10000` | L2CPU local register block (reset vectors, etc.) |

**Clock domains:**

| Clock | Source | Frequency | Drives |
|---|---|---|---|
| AICLK | ARC PLL, ramped via `arc_msg(0x52)` | ~1.35 GHz | NOC fabric, Tensix BRISC, NIU |
| X280 PLL | PLL4 (programmed by `X280Runtime`) | 1750 MHz typ. | X280 cores, `rdcycle` |
| l2sysclk | X280 PLL / 2 | 875 MHz | L2CPU AXI fabric, DMA engine |
| mtime | reference | 50 MHz | CLINT timer |

### C++ host runtime boundary

The reusable C++ path is rooted at `X280Runtime::open(mesh, config)`.
`RuntimeOpenConfig` defaults to automatic ECC priming and the reset cache/LIM
partition: fifteen 128 KiB LIM ways (`WayEnable=0`, 1.875 MiB backed). A caller
can select `LimEccPrimingPolicy::RequirePrepared` and a smaller `LimShape`.
The component reservation ends at the per-hart protocol block (`0x08131B00`) and
requires at least ten LIM ways.

`open` is Metal's stateful X280 initialization entry point, not a read-only
factory: it binds the caller-owned Metal mesh and discovers and inspects every
selected chip. `AutoPrime` runs the host PLL/LIM-ECC bootstrap when
hardware state requires it. `RequirePrepared` performs the same readiness path
but returns `PreparationRequired` instead of committing `WayEnable` when LIM is
unprimed.

Opening is all-or-nothing across the selected chips. Its compact aggregate
result is `RuntimeOpenResult`:

- `RuntimeOpenOutcome::Ready`: every chip is usable and the optional `runtime`
  is present;
- `RuntimeOpenOutcome::PreparationRequired`: the selected policy found
  unprepared hardware and made no one-way change; no runtime is present;
- `RuntimeOpenOutcome::ResetRequired`: bootstrap made progress or hardware is
  committed to a state that needs an ASIC reset; no runtime is present;
- `RuntimeOpenOutcome::Failed`: initialization could not safely complete; no
  runtime is present.

`RuntimeOpenResult` contains only the aggregate outcome, optional runtime,
`std::vector<ChipOpenResult>`, and aggregate diagnostic, plus structured
`requires_preparation()` and `requires_reset()` queries.
Each `ChipOpenResult` contains `chip_id`, outcome, and diagnostic. Open results
do not expose action vectors or tile snapshots.

A successful automatic ECC prime commits `WayEnable`, so that invocation
returns `ResetRequired` without a runtime. The caller closes Metal, resets the
device, recreates the mesh, and calls `open` again.

Reversible hardware tests call the public entry point with `RequirePrepared`;
production callers and the destructive prime test use the default `AutoPrime`.

Before exposing a runtime, `open` preflights every selected chip and applies the
configured shape only while every L2CPU on each affected chip is held in reset.
WayEnable is increase-only: a request that would expand LIM returns
`ResetRequired`. Reopening a live runtime with a different shape returns
`Failed` without touching hardware. Live repartitioning after reset release is
unsupported because the runtime does not own caller LIM allocations and
resident idle has no protocol to quiesce every hart and cache master.

### Prepared pipeline inputs (X280 only)

`X280Tile::prepare_pinned_input` pins a fixed 4288-byte cached Zero Device region at
`[0x0A000000, 0x0A0010C0)` on each participating L2CPU tile. Way 1 holds the
reserved lines; way 0 remains general-purpose. The required partition is
`WAYENABLE=1` (14 LIM ways). Preparation does not create this partition, prime
ECC, boot resident idle, reset the ASIC, or flush caches.

An operator must first establish that partition and resident idle, and
exclusive device ownership covering the whole prepared sequence. Neither matching input bytes nor mask readback proves fresh cache
placement for the reserved addresses, so the runtime records preparation in the
tile's 64 B general-purpose scratch (`0x2001_0100`), which survives Metal and
runtime opens and is cleared by an ASIC reset (observed; the hardware
documentation gives no reset value). An empty record shows that
no preparation has run since the reset; it does not show that nothing else
cached those addresses, which exclusive ownership covers.

Prime, open and boot as in "Using the X280 runtime", opening with
`LimEccPrimingPolicy::RequirePrepared` and `LimShape{14}` so shape provisioning
cannot silently run the ECC primer. The runtime applies an increase-only shape
only while every L2CPU remains in reset; it refuses a request that would expand
LIM.

`X280Tile::prepare_pinned_input` prepares once per ASIC-reset lifetime, taking
the path of `firmware-pinned-input-prepare.bin` from its caller. It refuses
without writing unless the runtime is at 14 LIM ways, the tile's `WAYENABLE` is
1 and the runtime owns no live session on any tile; while its own session is
live, `X280Runtime::enqueue` refuses every tile. From an empty record it
records an aborted attempt, runs the image as an active session on the tile in
resident idle, requires hart 0 to report prepared and harts 1-3 no-op, restores
resident idle, then records the prepared region. A record already prepared at
this shape returns `AlreadyPrepared`, so each later executable calls it and
reuses the region. An aborted, corrupt or unreadable record, or one prepared at
another shape, returns `ResetRequired`. `X280Tile::pinned_input_state` reads the
record.
Concurrent calls on one tile in one process are serialised; nothing serialises
across processes.

The operation is compiled into the image instead of supplied through a
firmware control block. The trusted
artifact path remains an operator responsibility; the operation-specific hart
result is the final mismatch check.

Preparation requires active mask bits `11` for every master, temporarily limits
Hart 0's population to way 1, touches all 67 lines, and leaves active bits `01`
for all 38 masters. Inputs can then be updated coherently through the pinned
hierarchy. Individual configurations do not allocate cache leases, invalidate
the region, change masks, or restore cache state when their fixtures close.
The ordinary active-session STOP/cleanup and NoC-window cleanup still run.

Prepare every participating tile under the same external ownership.

There is no teardown: an ASIC reset (`tt-smi -r <device>`) clears the pinning,
the 14-way partition and the record together. Do not run cache-flush, repartition, ECC-priming, or
conflicting Zero Device experiments inside this prepared sequence.

A preparation that fails after its image may have run, including a failed
cleanup or a process crash, leaves the aborted record, and every later
`prepare_pinned_input` returns `ResetRequired` until an ASIC reset. Closing a
benchmark process neither unpins the region nor preserves a device lock for the
next executable.

### Runtime ownership

Runtime ownership follows the live `MeshDevice`. Reopening the same live mesh
reuses its initialized `RuntimeImpl`. Opening a different live mesh or submesh
while that owner exists returns `RuntimeOpenOutcome::Failed` with a busy-owner
diagnostic. After the original mesh closes or drops, a recreated mesh gets
fresh contexts and hardware is inspected again.

The public surface has no `Runtime::inspect`, `X280Chip`, or `X280Hart`.
`TileId` aliases `TileIdentity`; `X280Runtime::tile_ids()` returns ordered
enabled-tile identities and `X280Runtime::tile(id)` returns the same
runtime-owned `X280Tile` reference for every lookup of an exact identity. Tile
references are valid only while the runtime remains alive.

The static `X280Runtime::virtual_core_from_logical_core(scope, logical,
core_type, noc)` resolves one virtual (UMD TRANSLATED) coordinate common to all
chips of a live `MeshDevice` scope, for a consumer on NoC `noc`. It needs no
open runtime, so Metal-only callers use it too.
Tensix, Ethernet, and DRAM use Metal's public mesh resolver; L2CPU uses the live
SoC descriptors, without a Metal dependency change. Missing mappings,
unsupported types, disagreement across chips, a coordinate outside six bits,
and a NoC 1 pair firmware leaves untranslated (Y < 2, X != 8) throw. The scope
is borrowed and does not open another execution runtime. An overload taking a
`MeshCoordinate` resolves on that one chip of the scope, for a selection that
differs per chip, such as each end of an Ethernet link.
`virtual_core_from_tile(id, noc)` validates an exact runtime tile identity and uses
UMD to translate its NOC0 position directly to a virtual coordinate on that
chip. Both return a `TranslatedCoord` (`x280/device/translated_coord.hpp`),
which nothing else constructs, so a logical or NOC0 value cannot reach a device
argument; device code carries it as `x280_noc_coord_t` (`include/x280_noc_coord.h`). Hardware tile indices are not logical coordinates. These queries perform
no device I/O, and the caller must not close or reshape the mesh concurrently.
The ISA's L2CPU translated range and UMD's mapping differ; the software lookup
does not establish NIU configuration or prove traffic delivery on silicon.

The Metal-linked `x280_runtime_coordinate_tests` target exercises the production
transport without opening an X280 execution runtime or accessing silicon.
CTest prepares a test-owned descriptor overlay for unharvested, harvested,
and disagreeing two-chip mock meshes; it does not edit the Metal checkout.
With the X280 build enabled (§ 6, Build targets):

```sh
cmake --build build --target x280_runtime_coordinate_tests
ctest --test-dir build --output-on-failure -R '^x280_coordinates'
```

`X280Runtime::lim_shape()` reports the effective partition. `X280Tile` exposes
`id`, read-only `inspect`, `reserved_lim`, `available_lim`, `read_noc_window`,
`healthy`, and `last_failure`. `read_available_lim()` and
`write_available_lim()` provide bounded block I/O only inside the current
available range; writes must be 64-byte aligned whole cache lines so high LIM
with uninitialised ECC never takes a partial-line merge path.
`boot()` explicitly prepares resident idle,
establishes the safe reset-release clock while the chip is still quiescent, and
may release reset when requested. `X280Runtime::set_pll()` owns operating-clock
changes after that, and the caller coordinates the selected frequency across
tiles.

Routine silicon tests create a `MeshDevice` first and enter exclusively through
`X280Runtime::open(...RequirePrepared)`. Raw ARC messaging remains in
`x280_arc_test`, and adversarial divider-slew characterization remains in
the opt-in `x280_pll_test`; neither is part of the runtime API.

`X280Runtime::enqueue(tile, request)` accepts only a healthy runtime-owned tile
from that runtime. The C++ runtime uses four per-hart control, state, and trap
records at migration-control `+0x1800..+0x1AFF`; no tile-global mailbox
addresses are reserved. Each hart has one cache
line for each record. All four harts run the same image, entry, run value, and
argument-block address and byte extent.

Enqueue requires released reset and four versioned, committed resident-IDLE
records before it writes anything. It reserves the tile's single execution slot
first, so a racing submission loses the slot rather than the records, then
validates and reads back the active image, zeroes every per-hart record by full
cache-line writes, publishes and verifies run value, entry and argument-block
address and byte extent for all harts,
registers the runtime-owned session, and publishes four `JUMP` commands last,
returning that session's `SessionId` without polling. An uncertain JUMP write
retains the session so bounded cleanup can report the complete four-hart state.

`X280Runtime::sessions()` lists every live execution and
`X280Runtime::session(id)` returns the one it names, for as long as that
execution is live. The session's `wait_until_running()` and
`wait_until_result_ready()` return bounded `WaitResult` values whose
`hartstates` array contains every hart. Success requires all four harts at the
requested generation; one stale, invalid, trapped, or uncommitted record blocks
aggregate success and remains visible in that array.
`X280Runtime::cleanup(id, timeout)` sends four STOP commands, waits for four
committed returns, sends four return acknowledgements, and requires four
versioned IDLE records for that generation. It then reaps the session, so it is
single-shot and a later call with that id returns `UnknownSession`. The runtime
destructor runs the same bounded cleanup for every session still live, with each
request's `cleanup_timeout`, and logs a warning for any it cannot restore. A
closed `MeshDevice` leaves it nothing to write through: it names the abandoned
session and the tile in a warning, and the tile runs active firmware until an
ASIC reset. A running or result timeout does not change tile health. Failed restoration marks the runtime-owned tile unhealthy, and later
enqueue attempts return `Unhealthy` without hardware writes.

Topology, register, and LIM traffic stays on Metal's `tt::Cluster` wrapper.
Raw UMD is restricted to ARC telemetry, which the wrapper does
not expose. With Metal watcher enabled, `X280MetalTransport::bind()` warns and
continues: the host-NOC sanitizer admits no ARC or L2CPU coordinate, so the
first such access fails. It does not bypass the
wrapper. Wrapper I/O also scopes UMD's thread-local physical fabric selector
to NOC0; coordinate translation alone does not select the fabric.

The host PLL/ECC bootstrap is private to `open()`: runtime open may establish
the safe probe clock and, under `AutoPrime`, prime LIM ECC. A cold `boot()`
establishes the safe reset-release clock, and runtime owners use `set_pll()`
for later operating-clock changes. There is no
tile `launch()`, public hart facade, or writable
outbound-NOC API.

The reversible success image is the sentinel, `build/x280/firmware.bin`. Every active image shares one lifecycle harness
(`src/x280_active_start.c`, § 3): each hart publishes its own run-correlated
heartbeat, runs the image's `main`, reports terminal success, and waits for
its own acknowledged return. The sentinel's `main` adds only the run-value
echo. The test-only `firmware-active-session-fault.bin` image makes hart 2
record an `ebreak` trap and CEASE while the other harts report success; only
the separately gated `x280_active_failure_test` uses it.

---

## 2. Reset semantics

| What you want | How |
|---|---|
| Stop or restart a running L2CPU | No per-tile reset path; use runtime cleanup or reset the ASIC |
| Run a new image on a running L2CPU | `X280Runtime::enqueue` through resident idle; never touches reset |
| Full board recovery (clears PLL, NOC TLBs, etc.) | Reset the selected ASIC before reopening Metal |

Harts leave reset once per ASIC reset. Clearing a tile's bit in `L2CPU_RESET`
does not put its harts back — only an ASIC reset does — and a second `0 → 1`
transition does not restart them. So `assert_l2cpu_reset → load → release` on a
tile that is already running writes a new image underneath code that never
stopped, and neither call reports a problem. The C++ runtime therefore admits
reset release only through its cold-boot path and refuses state that cannot
safely support it. Source: tt-isa-documentation
`BlackholeA0/L2CPUTile/README.md`, § Reset.

---

## 3. Writing firmware

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
| `x280_image.cmake` | `x280_add_image()`: compiles and links one image with SFPI (§ 6, Build targets). |

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
   `X280ActiveContract` checks it on hardware.
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

---

## 4. NOC: TLBs, addressing, paths

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
Counted Writes) and reads the entry back. It never clears a window whose
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

---

## 5. Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Memory Port write reaches L3 but not physical DDR | Memory Port is a cache-fill path, not a store-through path | Use System Port for outbound NOC writes; only use MemPort for cached DDR reads |
| 16 KB `noc_async_write` from Tensix hangs | NIU outstanding-write queue overflow | Either keep barriers, batch them, or use `<posted=true>` template parameter |
---

## 6. Reference: tests, fixtures, scripts

### Build targets

The X280 tree builds only in a tt-metal build configured with the tests and
`TT_METAL_BUILD_X280`. The images are compiled at build time with the SFPI
compiler tt-metal fetches (`runtime/sfpi`, or `/opt/tenstorrent/sfpi` with
`--use-system-sfpi`); no other RISC-V toolchain is involved.

```bash
./build_metal.sh --build-metal-tests --configure-only
cmake -B build -DTT_METAL_BUILD_X280=ON
ninja -C build x280_images x280_abi_headers x280_runtime_host_tests x280_runtime_coordinate_tests
ctest --test-dir build -R x280      # image placement, pinned-input guards, host and mock-cluster tests; no device
```

| Image (`build/x280/`) | Source | Use |
|---|---|---|
| `firmware-lim-idle.bin` | `src/lim_idle.c` | resident idle, the runtime prerequisite |
| `firmware.bin` | `src/main.c` | active session success image (the sentinel) |
| `firmware-pinned-input-prepare.bin` | `src/x280_pinned_input.c` | pinned-input preparation, run by `X280Tile::prepare_pinned_input` (§ 1) |
| `firmware-atomic-bench.bin` | `src/atomic_bench.c` | cross-hart atomics micro-benchmark (§ 7); in the active-image contract set |
| `firmware-active-session-fault.bin` | `src/active_session_fault.c` | traps and CEASEs hart 2, for `x280_active_failure_test` only |

Every image compiles with `-mcpu=sifive-x280 -march=rv64gc -mabi=lp64d
-mcmodel=medany -O2 -Werror`, freestanding, and links with
`X280_REQUIRE_NO_BSS` defined, so the linker scripts refuse a `.bss`.
`x280_image_layout` checks resident idle's `.text` at `0x08000000` and the
sentinel's at `0x08001000`. The `x280_abi_headers` build target compiles each
ABI header alone as C, C++ and, for the two `entry.S` includes, assembly.

### Active-image contract suite

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

A failure here is a firmware obligation (§ 3), not a runtime bug. The failure
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

## 7. Cross-hart atomics micro-benchmark

Models all 4 harts on one L2CPU tile pushing into a single downstream
queue, where each must bump a shared write pointer without colliding. The correctness primitive is an atomic
read-modify-write; this ubench measures its cost and contention.

- Firmware: [`src/atomic_bench.c`](src/atomic_bench.c) — a **ping-pong
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

## 8. Useful constants cheat-sheet

```c
// From x280_address_map.h
#define X280_LOCAL_DDR_TILE_BASE 0x400030000000ULL
#define X280_LIM_BASE            0x08000000ULL
#define X280_CLINT_BASE          0x02000000ULL
```

Host-only reset, PLL, and L3 register addresses are private to
`src/device/x280_hw_addrs.hpp`; callers use `X280Runtime` rather than
duplicating them.
