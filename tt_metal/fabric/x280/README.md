# X280 Bare-Metal Getting Started

End-to-end guide to running real work on the SiFive X280 RISC-V cores
inside the L2CPU tiles on a Tenstorrent Blackhole card. Covers boot,
reset, kernel launch, clock programming, NOC traffic, and I/O paths.

The supported host boundary is the C++ `X280Runtime` described in § 1.

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

Release destinations are resolved in input order through
`X280Runtime::virtual_core_from_logical_core` by the shared host release
selector, `select_release`. It releases a Tensix selection whose logical
coordinates fill one rectangle by one multicast and every other participant by
its own unicast endpoint. Core-sync's multicast cases require the multicast
(`ReleaseMode::RequireMulticast`) and fail before launch without it; packet
injection takes the selector's choice (`ReleaseMode::Auto`).

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

The approved reusable C++ path is rooted at `X280Runtime::open(mesh, config)`.
`RuntimeOpenConfig` defaults to automatic ECC priming and the reset cache/LIM
partition: fifteen 128 KiB LIM ways (`WayEnable=0`, 1.875 MiB backed). A caller
can select `LimEccPrimingPolicy::RequirePrepared` and a smaller `LimShape`.
The current component and socket reservation requires at least eleven LIM ways.

`open` is Metal's stateful X280 initialization entry point, not a read-only
factory: it binds the caller-owned Metal mesh and discovers and inspects every
selected chip. `AutoPrime` runs the temporary host PLL/LIM-ECC bootstrap when
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

`RuntimeOpenConfig::shape_change_policy` defaults to
`LimShapeChangePolicy::ApplyWhileReset`, preserving this behavior.
`RequireExact` instead requires the observed shape to match `lim_shape` on every
selected chip before ECC priming or shape application. Pair it with
`RequirePrepared` when opening hardware that must already have the requested
partition; a mismatch is refused rather than provisioned by the benchmark.
Unreadable shape observations are refused without attempting PLL recovery.
This admission policy does not itself allocate or pin cached Zero Device storage.

### Prepared pipeline inputs (X280 only)

The X280 pipeline benchmarks share a fixed 4288-byte cached Zero Device region at
`[0x0A000000, 0x0A0010C0)` on each participating L2CPU tile. Way 1 holds the
reserved lines; way 0 remains general-purpose. The required partition is
`WAYENABLE=1` (14 LIM ways). Preparation does not create this partition, prime
ECC, boot resident idle, reset the ASIC, or flush caches.

An operator must first establish that partition and resident idle, fresh cache
placement for the reserved addresses, and exclusive device ownership covering
the entire sequence of benchmark executables. Neither matching input bytes nor
mask readback proves fresh placement. `--fresh-cache` and `--exclusive-device`
are assertions of those external prerequisites, not mechanisms that establish
them.

After LIM ECC has already been prepared for the current reset lifetime, an
authorized operator can provision the required shape and boot resident idle in
one normal runtime invocation:

```bash
TT_VISIBLE_DEVICES=0 build/test/tt_metal/x280/x280_runtime_cli \
  --chip 0 --tile 0 --lim-ways 14 --require-prepared \
  --boot --pll 1750 --timeout 5000
```

The resident-idle image is `--idle-image`, else `$X280_IDLE_FW_IMAGE`, else the
build's `build/x280/firmware-lim-idle.bin`. `--lim-ways` defaults to 15 and
accepts 0 through 15. An explicit value requires
`--require-prepared`, so shape provisioning cannot silently run the ECC primer.
The runtime applies an increase-only shape only while every L2CPU remains in
reset; it refuses a request that would expand LIM. Pinned-input commands reject
`--lim-ways` because their admission profile is fixed at 14.

The `x280_images` target builds the two operation-specific images. On an already
prepared, exclusively owned device, the command sequence is:

```bash
build/test/tt_metal/x280/x280_runtime_cli --tile 0 \
  --prepare-pinned-input --fresh-cache --exclusive-device \
  --prepare-pinned-input-image build/x280/firmware-pinned-input-prepare.bin
# Run the compatible pipeline configurations sequentially on the same tile.
build/test/tt_metal/x280/x280_runtime_cli --tile 0 \
  --teardown-pinned-input --exclusive-device \
  --teardown-pinned-input-image build/x280/firmware-pinned-input-teardown.bin
```

ID 7 uses the first 96-byte header in this region at `0x0A000AC0`. Its
`decide_plain` configuration reseeds that complete header before warmup; it
does not allocate a source region or change the prepared policy.

The operation is compiled into each image instead of supplied through a
firmware control block. The CLI resolves preparation only from
`--prepare-pinned-input-image` or `X280_PINNED_INPUT_PREPARE_FW_IMAGE`, and
teardown only from `--teardown-pinned-input-image` or
`X280_PINNED_INPUT_TEARDOWN_FW_IMAGE`. This costs two artifacts but prevents an
automatic default from selecting the opposite operation. The trusted artifact
path remains an operator responsibility; the operation-specific hart result is
the final mismatch check.

Preparation requires active mask bits `11` for every master, temporarily limits
Hart 0's population to way 1, touches all 67 lines, and leaves active bits `01`
for all 38 masters. Inputs can then be updated coherently through the pinned
hierarchy. Individual configurations do not allocate cache leases, invalidate
the region, change masks, or restore cache state when their fixtures close.
The ordinary active-session STOP/cleanup and NoC-window cleanup still run.

Both pipeline executables always require this prepared profile. The latency
members and both transport-bandwidth executables run at the reset shape, so a
chip left at 14 LIM ways skips them until an ASIC reset. Prepare every
participating tile under the same external ownership.

Final teardown requires confirmed resident idle and the pinned policy. It
re-enables only way 1's replacement bit, preserving other bits, without flushing
the region or changing `WAYENABLE`. The device remains at 14 LIM ways; restoring
the default partition requires separately authorized reset/provisioning.
Do not run cache-flush, repartition, ECC-priming, or conflicting Zero Device
experiments inside this prepared sequence.

A failed preparation, unconfirmed cleanup, or process crash aborts the prepared
sequence. Do not continue or use normal teardown as recovery: retain exclusive
ownership and perform explicit recovery/repreparation before reuse. Closing a
CLI or benchmark process neither unpins the region nor preserves a device lock
for the next executable.

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
P2P, transport-bandwidth and pipeline forwarding resolve their Tensix and X280
destinations through these methods,
as do core-sync's L2CPU sites, participant arrival and release-probe groups,
and the eth_credit and packet-injection L2CPU sites.
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
records at migration-control `+0x1800..+0x1AFF`; the older tile-global mailbox
addresses remain reserved but are not part of this path. Each hart has one cache
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

Topology, register, and LIM traffic stays on Metal's `tt::Cluster` wrapper so
the component remains a staging boundary for eventual Metal runtime
integration. Raw UMD is restricted to ARC telemetry, which the wrapper does
not expose. With Metal watcher enabled, `X280MetalTransport::bind()` warns and
continues: the host-NOC sanitizer admits no ARC or L2CPU coordinate, so the
first such access fails until upstream adds a route. It does not bypass the
wrapper. Wrapper I/O also scopes UMD's thread-local physical fabric selector
to NOC0; coordinate translation alone does not select the fabric.

The private host PLL/ECC bootstrap is temporary. When system firmware or Metal
owns those operations, the safe-clock and prime branches are removed from
private `initialize_chip()`. Until then, runtime open may establish the safe
probe clock, a cold `boot()` establishes the safe reset-release clock, and
runtime owners use `set_pll()` for later operating-clock changes. There is no
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
| `include/socket_layout.h` | Migration and socket mailbox placement, values, and wire-layout guards. |
| `include/x280.h` | Compatibility umbrella for characterization firmware constants. |
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
   forgets to release its workers no longer strands them.
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

```python
chip.noc_write32(0, noc_x, noc_y, addr, value)        # 32-bit
chip.noc_write(0, noc_x, noc_y, addr, byte_data)      # bulk
chip.noc_read(0, noc_x, noc_y, addr, buf)             # bulk
```

`addr` is the L2CPU-local address (for example, `X280_LIM_BASE` =
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
ninja -C build x280_images          # build/x280/<image>.{elf,bin}
ctest --test-dir build -R x280      # image placement and header checks; no device
```

| Image (`build/x280/`) | Source | Use |
|---|---|---|
| `firmware-lim-idle.bin` | `src/lim_idle.c` | resident idle, the runtime prerequisite |
| `firmware.bin` | `src/main.c` | active session success image (the sentinel) |
| `firmware-pinned-input-prepare.bin`, `firmware-pinned-input-teardown.bin` | `src/x280_pinned_input.c` | pinned-input preparation and teardown (§ 1) |

Every image compiles with `-mcpu=sifive-x280 -march=rv64gc -mabi=lp64d
-mcmodel=medany -O2 -Werror`, freestanding, and links with
`X280_REQUIRE_NO_BSS` defined, so the linker scripts refuse a `.bss`.
`x280_image_layout` checks resident idle's `.text` at `0x08000000` and the
sentinel's at `0x08001000`; `x280_abi_headers` compiles each ABI header alone
as C, C++ and, for the two `entry.S` includes, assembly.

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

| `X280_ACTIVE_FW_IMAGES` | Outcome |
|---|---|
| every path resolves | one case per path |
| some resolve | those cases run; each unresolved path prints as `[  DROPPED ]`; exit 0 |
| none resolve | **fail**, naming every path — the suite would otherwise instantiate nothing and pass |
| unset, or naming no path | skip: nothing was requested |

The guard keys on `X280_ACTIVE_FW_IMAGES` alone and needs no hardware, so a
broken path list fails on a laptop. `X280_ACTIVE_FW_IMAGE` is a fallback for
instantiating `X280ActiveContract` when `X280_ACTIVE_FW_IMAGES` is unset, but
it does not count as a request for this guard test.

Every runtime device test takes its image from `build/x280/` unless its
variable names another (`X280_ACTIVE_FW_IMAGE` for `X280ActiveHw`, which runs
the sentinel; `X280_ATOMIC_BENCH_FW_IMAGE`; `X280_ACTIVE_FAULT_FW_IMAGE`), so a
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
— `Opening local chip ids/PCIe ids: {0..7}/[2,3,1,0,6,7,5,4]` on the p150b host
these suites ran on, where UMD chip 0 is `/dev/tenstorrent/2`. Record that line
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

Prototype for the case where all 4 harts on one L2CPU tile push into a
single downstream (erisc) queue and each must bump a shared write
pointer without colliding. The correctness primitive is an atomic
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
   The firmware sums the returned slots per hart and the host verifies that
   total — a direct proof that **no two harts got the same slot** (the exact
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
   memory stalls (see `socket_layout.h` "UNITS NOTE"), so it would hide the very atomic-latency
   cost we want to see.

### Safety: AMO-to-LIM probe

Whether AMOs work against *uncached* LIM is silicon-specific. Before any
untrapped atomic runs, hart 0 arms `entry.S` **probe mode** and issues a
single test op: if it faults, the handler skips it (instead of `CEASE`,
which would permanently halt the hart and strand the L2CPU) and the run
aborts cleanly with `AMO-to-LIM supported: False`.

## 8. Useful constants cheat-sheet

```c
// From x280_address_map.h
#define X280_LOCAL_DDR_TILE_BASE 0x400030000000ULL
#define X280_LIM_BASE            0x08000000ULL
#define X280_SENTINEL_ADDR       0x08100000ULL
#define X280_CLINT_BASE          0x02000000ULL
```

Host-only reset, PLL, and L3 register addresses are private to
`src/device/x280_hw_addrs.hpp`; callers use `X280Runtime` rather than
duplicating them.
