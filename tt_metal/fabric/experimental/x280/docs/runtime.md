# X280 runtime

The C++ host runtime, `X280Runtime`, binds the X280 (L2CPU) tiles of a Blackhole chip to a
caller's Metal mesh: it opens and prepares the chips, boots resident idle, runs active images
and cleans them up. The hardware overview and the build are in the [README](../README.md);
writing and building images, and the obligations an active image carries, are in
[device_images.md](device_images.md).

## Production use

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
#include <umd/device/warm_reset.hpp>

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
- **Watcher off.** With `TT_METAL_WATCHER` set, the host NoC sanitizer rejects
  the ARC register reads `open` starts with (§ C++ host runtime boundary).

TODO: a helper that runs this open, reset, reopen and boot sequence and returns
a ready runtime, with the reset still an explicit caller opt-in.

## Tests

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
by CMake); an `X280_*_FW_IMAGE` variable names another file
([device_images.md](device_images.md#images-and-build-checks)). A test that
changes the chip for the rest of a reset lifetime, or can wedge it, lives in
its own executable and runs only when its `X280_ALLOW_*` variable is `1`:
`x280_prime_test` (`X280_ALLOW_LIM_ECC_PRIME`), `x280_shape_test`
(`X280_ALLOW_LIM_SHAPE_CHANGE`), `x280_pinned_input_test`
(`X280_ALLOW_PINNED_INPUT_PREPARE`), `x280_active_failure_test`
(`X280_ALLOW_ACTIVE_SESSION_FAILURE`) and `x280_pll_test`
(`X280_ALLOW_PLL_SLEW`).

## C++ host runtime boundary

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

### Release destinations

Release destinations are resolved in input order through
`X280Runtime::virtual_core_from_logical_core` by the shared host release
selector, `select_release`. It releases a Tensix selection whose logical
coordinates fill one rectangle by one multicast and every other participant by
its own unicast endpoint. Core-sync's multicast cases require the multicast
(`ReleaseMode::RequireMulticast`) and fail before launch without it; packet
injection takes the selector's choice (`ReleaseMode::Auto`).

## Prepared pipeline inputs (X280 only)

The X280 pipeline benchmarks share a fixed 4288-byte cached Zero Device region at
`[0x0A000000, 0x0A0010C0)` on each participating L2CPU tile. Way 1 holds the
reserved lines; way 0 remains general-purpose. The required partition is
`WAYENABLE=1` (14 LIM ways). Preparation does not create this partition, prime
ECC, boot resident idle, reset the ASIC, or flush caches.

Whoever prepares must first establish that partition and resident idle, and
exclusive device ownership covering the whole prepared sequence. Neither
matching input bytes nor mask readback proves fresh cache placement for the
reserved addresses, so the runtime records preparation in the tile's 64 B
general-purpose scratch (`0x2001_0100`), which survives Metal and runtime opens
and is cleared by an ASIC reset (observed; the hardware documentation gives no
reset value). An empty record shows that no preparation has run since the
reset; it does not show that nothing else cached those addresses, which
exclusive ownership covers.

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
record. Run the compatible pipeline configurations on the prepared tile.
Concurrent calls on one tile in one process are serialised; nothing serialises
across processes.

The pipeline_stage suite calls `prepare_pinned_input` before every device case,
so its executables run in any order and any number of times within one reset
([pipeline_stage.md](benchmarks/pipeline_stage.md#suite-prepared-input-x280-only)).

ID 7 uses the first 96-byte header in this region at `0x0A000AC0`. Its
`decide_plain` configuration reseeds that complete header before warmup; it
does not allocate a source region or change the prepared policy.

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

Both pipeline executables always require this prepared profile. The latency
members and both transport-bandwidth executables run at the reset shape, so a
chip left at 14 LIM ways skips them until an ASIC reset.

There is no teardown: an ASIC reset (`tt-smi -r <device>`) clears the pinning,
the 14-way partition and the record together. Do not run cache-flush, repartition, ECC-priming, or
conflicting Zero Device experiments inside this prepared sequence.

A preparation that fails after its image may have run, including a failed
cleanup or a process crash, leaves the aborted record, and every later
`prepare_pinned_input` returns `ResetRequired` until an ASIC reset, so in the
suite the failing case fails and every later case skips. Closing a benchmark
process neither unpins the region nor preserves a device lock for the next
executable.

## Runtime ownership

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
With the X280 build enabled ([README](../README.md#2-build-boot-and-run-a-benchmark)):

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
(`src/x280_active_start.c`, [device_images.md](device_images.md#the-active-image-contract)): each hart publishes its own run-correlated
heartbeat, runs the image's `main`, reports terminal success, and waits for
its own acknowledged return. The sentinel's `main` adds only the run-value
echo. The test-only `firmware-active-session-fault.bin` image makes hart 2
record an `ebreak` trap and CEASE while the other harts report success; only
the separately gated `x280_active_failure_test` uses it.

## Reset semantics

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

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Memory Port write reaches L3 but not physical DDR | Memory Port is a cache-fill path, not a store-through path | Use System Port for outbound NOC writes; only use MemPort for cached DDR reads |
| 16 KB `noc_async_write` from Tensix hangs | NIU outstanding-write queue overflow | Either keep barriers, batch them, or use `<posted=true>` template parameter |
