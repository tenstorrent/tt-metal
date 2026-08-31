# Phase 2.2 — Idle-Eth Aggregator

Companion to [`PLAN.md`](PLAN.md). Addresses the remote-chip transport cost and the
structural sample-loss ceiling documented in `util_sampler.h`.

---

## 1. Problem

Two independent problems, one fix.

**(a) Remote chips are expensive to sample.** A WH chip with no PCIe path is reached
through UMD's ethernet tunnel. Per `remote_communication_legacy_firmware.cpp`, one
`read_from_device` to a remote chip is:

1. acquire `MutexType::NON_MMIO` — a named `RobustMutex`, i.e. **interprocess**, so
   ttnvtop-collector contends with the workload process itself;
2. write a 32-byte `routing_cmd_t` into an eth core's request queue (MMIO over PCIe);
3. bump the request write pointer;
4. spin-poll the response queue pointers over PCIe;
5. spin-poll the response flags;
6. read the payload.

The ERISC servicing that command does so via `run_routing()` →
`internal_::risc_context_switch()` (`tt_metal/hw/inc/internal/ethernet/tunneling.h:178`),
called from ~15 sites in the eth `dataflow_api.h` wait loops. On a T3K, dispatch to
remote chips is *itself* tunneled over ethernet, so telemetry reads inject latency
directly into the dispatch path.

UMD round-robins remote transfers across **all** active eth channels
(`cluster.cpp:781`). `Cluster::configure_active_ethernet_cores_for_mmio_device()`
exists to restrict that set — tt-metal never calls it.

Current cost: 64 Tensix × 4 reads = **256 tunnel transactions per remote chip per tick**,
each on the worst path (`use_host_dram` requires `size > 256 * DATA_WORD_SIZE` = 1 KiB;
4-byte reads never qualify).

**(b) The host drain rate caps sampling fidelity — on every chip.** From
`util_sampler.h`:

> host drain at 50 Hz × 62 slots × 64 cores = ~198k/sec drainable per chip, while
> 100 µs sampling on 64 cores × 2 producer threads generated ~1.28M samples/sec/chip
> → ~84% structural sample loss.

The period was raised 100 µs → 1 ms to work around this. The ring is 62 entries; the
host cannot drain faster than it polls, so resolution is bounded by PCIe round-trips.

## 2. Approach

Put a persistent kernel on an **idle ethernet core** on each chip. It NOC-reads every
Tensix sampler ring intra-chip and accumulates into a large journal in its own eth L1.
The host then does **one** block read per chip per tick.

Why an idle eth core and not a Tensix:

- On a real workload the Tensix grid is typically fully allocated; reserving one either
  fails or perturbs exactly what is being measured. Idle eth cores are capacity the
  workload was never going to use.
- `HalProgrammableCoreType::IDLE_ETH` is a first-class programmable core type
  (`wh_hal_idle_eth.cpp`, `kernel.hpp:415`). No bare-metal ERISC firmware needed.
- Eth L1 is `MEM_ETH_SIZE = 256*1024 - 32` — room for a journal ~200× the size of one
  Tensix ring.
- The gather is intra-chip NOC. No ethernet involved in the gather itself.

### Idle-core availability (verified)

Union of channels ever active across all shipped WH cluster descriptors
(`tt_metal/third_party/tt-cluster-descriptors/wormhole/`), `eth_harvesting_mask: 0`
everywhere:

| Chip role | Channels ever active | Never routed | Idle (worst case) |
|---|---|---|---|
| N300 local | `0,1,6,7,8,9,14,15` | `2,3,4,5,10,11,12,13` | 8 |
| **N300 remote** | `0,1,6,7` | `2,3,4,5,8,9,10,11,12,13,14,15` | **12** |
| T3K remote (measured config) | `0,1,6,7` | — | **12** |

Corrected 2026-08-29 against the physical board map (corsix.org/content/tt-wh-part4),
which the T3K cluster descriptor agrees with. Per-ASIC wiring on an n300:

| | E0,E1 | E6,E7 | E8,E9 | E14,E15 | used | free |
|---|---|---|---|---|---|---|
| local (MMIO) | QSFP-DD #1 | QSFP-DD #2 | internal -> 2nd ASIC | Warp 100 Bridge | 8/16 | **8** |
| remote | internal <- E8/E9 | Warp 100 Bridge | — | — | 4/16 | **12** |

The remote row previously listed `14,15` as ever-active. That was wrong: on the
second ASIC the Warp 100 Bridge lands on E6/E7, and E14/E15 go nowhere. Remote chips
have **12** free channels, not 10, even with every QSFP-DD and TFly populated.

Provisioning is therefore comfortable at both ends: the gatherer needs one free core
on a remote chip (12 available) and the journal landing spot needs one on the MMIO
chip (8 available), in the maximally-cabled case.

Note the free-channel count is not binning-dependent on WH: eth harvesting is
impossible there, enforced by a UMD throw (see 2.1).

CAVEAT: "no recabling can claim them" is an inference from board wiring and shipped
descriptors, not a schematic. Probing those cores' ERISC mailboxes directly
(2026-08-29) shows them FIRMWARE-ALIVE but peer-less -- magic `0xabcd1234` present,
sensible per-channel identity, `0xffffffff` in the remote-info words, zero traffic
counters. That distinguishes has-peer from no-peer, NOT has-PHY from no-PHY.

SECOND CAVEAT: free of links != free of users -- but ONLY under ETH dispatch.
Corrected 2026-08-29: `dispatch_core_manager.cpp:268` adds every
`get_inactive_ethernet_cores()` core to the FD pool inside
`if (resolve_dispatch_core_type(...) == CoreType::ETH)`. The default is WORKER
(`DispatchCoreConfig()` is `DispatchCoreType::WORKER`, and on WH/BH resolve returns
whatever the config says), so on a stock run dispatch never touches idle eth cores at
all. Under ETH dispatch it takes ALL of them. See 3.4.

6U Galaxy is the only WH topology with zero idle channels, and it has **32 MMIO-capable
chips and zero remote chips** — it does not need this feature. Scope the aggregator to
`is_remote` chips and the constraint never binds.

### 2.1 Harvesting invariants (verified 2026-08-29)

**On Wormhole, harvesting cannot touch the transport.** This is enforced in the
driver, not merely absent from the descriptors --
`third_party/umd/device/coordinates/coordinate_manager.cpp:80`:

```cpp
if (harvesting_masks.dram_harvesting_mask != 0)
    UMD_THROW(error::RuntimeError, "DRAM harvesting is supported only for Blackhole.");
if (harvesting_masks.eth_harvesting_mask != 0)
    UMD_THROW(error::RuntimeError, "ETH harvesting is supported only for Blackhole.");
```

A WH part with a fused-off eth channel, DRAM bank, or PCIe tile cannot be constructed
by UMD at all. All 278 chip entries across every shipped WH cluster descriptor carry
`eth_harvesting_mask: 0`, `dram_harvesting_mask: 0`, `pcie_harvesting_mask: 0`.

So the 2 free-eth budget (8 MMIO / 12 remote, fully cabled) is **not binning-dependent
on WH**. Likewise the PCIe landing tile and the DRAM-BW axis's 6 banks x 3 NIUs.

#### Tensix row harvesting IS real, and differs per chip

Every nonzero WH `harvest_mask` in the descriptor set has popcount 2 -- 2 rows of 10
fused off, 8x8 = **64 live Tensix**. Live T3K agrees: `armed FPU counter on 64/64` on
all 8 chips. Critically the mask **differs chip to chip inside one system** (dual_t3k:
132, 520, 514, 192, 96, 72, 130, 272 -- eight distinct masks), so no two chips have the
same physical NOC0 rows alive.

Three consequences:

**(a) Collector enumeration -- already correct.** `soc.get_cores(TENSIX, NOC0)`
dispatches to `CoordinateManager::get_tensix_cores()`, which skips harvested rows.
The 64/64 is the evidence.

**(b) The gatherer kernel must walk TRANSLATED space, not NOC0.** Harvesting removes
whole rows (WH) or whole columns (BH), so the live set is always the CROSS PRODUCT of a
live-x list and a live-y list. `nx + ny` coordinates therefore describe all `nx * ny`
cores -- WH 8+8 = 16 numbers for 64 cores, BH 12+10 = 22 for 120 -- instead of a
per-core address table.

CORRECTED 2026-08-29. This section previously claimed the live cores form a CONTIGUOUS
translated rectangle, so a fixed `for y in 18..25 / for x in 18..25` loop would do. That
is true on WH only. `WormholeCoordinateManager` synthesises translated coords as
`x + tensix_translated_coordinate_start_x`, which is contiguous; but
`BlackholeCoordinateManager::fill_tensix_noc0_translated_mapping()` takes them from the
NOC0 core list, which skips the non-Tensix columns -- so BH's live translated x values
have GAPS. The generalisation was wrong and a host-side assertion caught it on the first
Blackhole run. Pass coordinate lists, not an origin and a width, and assert the
cross-product property rather than trusting either shape.

Deriving the NOC address in-kernel is safe here ONLY because translated coordinates are
NOC-independent by construction: `NOC_XY_ADDR` is `(y << 42) | (x << 36) | addr`,
byte-identical on WH and BH, with no `NOC_X_PHYS_COORD` flip. Do not substitute
`get_noc_addr()`, which resolves against the kernel's own `noc_index`.

CAVEAT: holds only when `noc_translation_enabled`. `CoordinateManager::initialize()`
runs `identity_map_noc0_cores()` unconditionally and inserts the translated maps only
under `if (noc_translation_enabled)`, so with translation off TRANSLATED degenerates to
NOC0 and the compaction disappears. The gatherer must read the flag and fall back to an
explicit core list. T3K has translation on (the collector uses translated coords today).

**(c) Journal core count is not a constant.** 64 is a T3K value; 224 descriptor entries
have `harvest_mask: 0` -> 80 cores. `util_agg_msg_t.num_cores` (3.1) already carries it;
never bake 64 into the host drain or the viewer.

Nothing here moves eth core coords, the PCIe tile, the fabric route, the VC assignment,
or AICLK. **The push transport is untouched by WH harvesting.**

#### Blackhole: none of the above transfers

| mask | BH value | meaning |
|---|---|---|
| `eth_harvesting_mask` | **288** on 250/251 entries, 320 on 1 | channels 5 & 8 fused off -- always 2 of 14 |
| `pcie_harvesting_mask` | **1** (224) or **2** (27) | two PCIe tiles, NOC0 `(2,0)` and `(11,0)`; exactly one is ALWAYS harvested, and which one varies |
| `harvest_mask` | single bits 1..8192 | **column** harvesting, 1 of 14 -- different geometry from WH's rows |
| `dram_harvesting_mask` | 0 in all shipped BH | but legal on BH, unlike WH |

Two hard requirements when this design reaches BH:

1. **The landing tile must come from `get_cores(CoreType::PCIE)`, never a constant.**
   A hardcoded `(2,0)` is wrong on ~11% of the BH descriptor set, and it fails the way
   the fabric bug in 5d failed: silently, into a tile that is not there.
2. **The spare-eth budget must be computed, not asserted.** BH always loses 2 channels,
   so the "structurally free" argument in 2 is **WH-only**.

## 3. Design

### 3.1 Journal format

New header `tt_metal/hw/inc/util_aggregator.h`, sibling to `util_sampler.h`.

```
struct util_agg_entry_t {          // 20 B
    uint16_t core_id;              // index into the chip's Tensix core list
    uint16_t seq;                  // per-core wrap counter, for loss detection
    util_sampler_entry_t sample;   // 16 B, verbatim from the Tensix ring
};

struct util_agg_msg_t {
    volatile uint32_t magic;       // 'TTAG'
    volatile uint32_t version;
    volatile uint32_t head;        // monotonic entry count written
    volatile uint32_t capacity;    // entries in journal[]
    volatile uint32_t num_cores;   // cores this aggregator sweeps
    volatile uint32_t sweep_count; // aggregator liveness heartbeat
    volatile uint32_t lost;        // entries dropped: ring overrun observed
    volatile uint32_t reserved;
    volatile util_agg_entry_t journal[];
};
```

Sizing: 192 KiB journal / 20 B = **~9,800 entries**. At 64 cores × 1 kHz = 64k
entries/sec that is ~150 ms of buffering, comfortably ahead of a 10 Hz host drain.
Restoring the 100 µs sample period raises it to 640k/sec → ~15 ms of buffering, which
needs a 50–100 Hz host drain. Both are viable; the 100 µs case is the point of the
exercise.

### 3.2 Aggregator sweep loop

```
for each core c in cores:
    head = noc_read_u32(c.ring_base + offsetof(head))   // 4 B
    if head == last_head[c]: continue
    n = min(head - last_head[c], RING_SIZE)
    if head - last_head[c] > RING_SIZE: lost += (head - last_head[c]) - RING_SIZE
    read the n new slots, append to journal with core_id + seq
    last_head[c] = head
sweep_count++
```

Cost per sweep when idle: 64 × 4 B NOC reads. At a 10 kHz sweep rate that is 2.6 MB/s
of intra-chip NOC traffic — negligible against NOC bandwidth, and it never touches
ethernet.

### 3.3 Host side — PUSH, not pull  (revised 2026-08-28, see §5c)

The original design had the host **pull** the journal: one `read_from_device` of
`header + new journal entries` per remote chip per tick. **That is wrong, and the
reason is measured, not theoretical.**

A host read of a remote chip is `RemoteCommunicationLegacyFirmware::read_non_mmio`,
which is not a memory access: it **writes a command into the ethernet core's firmware
queue on the MMIO chip and polls for completion**, holding UMD's `NON_MMIO` mutex for
the whole round trip. Under a workload that saturates the ETH links, those cores do not
service the command promptly and the host polls for tens of seconds. Captured under
Llama-3.3-70B on 2026-08-28 (§5c).

Pulling therefore takes the transaction count from 256 to 1 but leaves the **failure
mode unchanged** — one blocked read is all it takes. ~256x rarer is still "eventually",
and a 70B run saturates ETH for minutes at a time.

**The aggregator must PUSH.** The remote chip's eth core writes the journal across its
own ethernet link into the **MMIO chip's** L1/DRAM. The host then reads it from the
LOCAL chip over plain PCIe.

| | pull (original) | push (this design) |
|---|---|---|
| host transport | ETH tunnel (`read_non_mmio`) | local PCIe |
| takes `NON_MMIO` | yes | **no** |
| ETH saturated | host **blocks** tens of s | journal goes **stale** |
| can stall the workload | yes | **no** |

The category change is the point: **for a monitor, stale data is acceptable and blocking
is not.** Push also removes the collector's ability to interfere with the workload at
all, because it never touches the tunnel — which no host-side mitigation can achieve.

Concretely:

- **Landing spot: an IDLE ETHERNET CORE's L1 on the MMIO chip**, at
  `hal.get_dev_addr(HalProgrammableCoreType::IDLE_ETH, HalL1MemAddrType::UNRESERVED)`.
  Not Tensix L1 and not DRAM -- both are the workload's, and colliding with its
  buffer allocator is the class of problem this design exists to avoid. An
  unlinked eth core is claimed by nobody: on a T3K, chip 0 has links only on
  channels 6,7,8,9,14,15, leaving 10 idle cores. One core holds a slot per remote
  chip; 4 x ~1 KB against a 256 KB ERISC L1 is nothing.
  NOTE the `ERISC_L1_UNRESERVED_BASE` / `ROUTING_FW_ENABLED` split in
  `eth_l1_address_map.h` governs ACTIVE eth cores. Idle eth derives its base from
  `MEM_IERISC_MAP_END` (`wh_hal_idle_eth.cpp`) with no such branch, so there is no
  host/kernel divergence on this path.
- Host probes that slot on the LOCAL chip: if `magic == 'TTAG'` and the sequence number
  advanced, decode; if it has not advanced, the journal is stale — publish the last
  values with a staleness flag rather than blocking.
- Sequence number + checksum in the header so the host can detect a torn write without
  any handshake.
- **Fallback:** existing per-core ring drain, unchanged.

The per-entry decode logic is unchanged — `util_sampler_entry_t` is forwarded verbatim,
so the delta/EWMA/kernel-attribution code needs no edit beyond the `core_id` demux.

**Transport delta per remote chip per tick: 256 tunnelled transactions → 0.**

**Push all the way to PCIe / host memory: TESTED AND BLOCKED (2026-08-29).**
Not on design grounds -- it does not work today. Three tests in
`tests/tt_metal/tt_fabric/fabric_data_movement/test_fabric_pcie_host_target.cpp`
plus kernels `test_fabric_pcie_target.cpp` / `test_direct_pcie_write.cpp`
isolate it to one component:

| test | result | establishes |
|---|---|---|
| `TestSysmemNocAddressEncoding` | PASS | KMD returns `0x8_8000_0000` -- PCIe marker + offset, NO XY. Consumer ORs in the tile, as `cq_prefetch.cpp` does. Do NOT decompose and rebuild it: an attempt to do so produced a meaningless address that HUNG the fabric router. |
| `TestDirectPcieWriteFromMmioChip` | PASS | Address is right (host-computed == kernel-computed bit for bit) and the PCIe tile IS reachable on NOC1 from an idle eth core. Both writes land. |
| `TestFabricWriteReachesHostMemory` | **FAIL** | Sender does everything right -- connection opens from an ERISC, `sends_done=16` -- and nothing arrives. |

Same address, same NOC, same chip, same core type. The only difference is who
issues the write:

    plain noc_async_write   ->  NOC_UNICAST_WRITE_VC = 1                    lands
    EDM receiver            ->  NOC_CMD_VC_STATIC | STATIC_VC(2 or 3)       dropped

`DEFAULT_NOC_VC = 2` and `edm_noc_vc = DEFAULT_NOC_VC + (link_idx % NUM_EDM_NOC_VCS)`,
so the EDM pins its local write to a static VC 2/3. **A PCIe-tile destination is
silently dropped on that VC** -- no fault, no error, the same failure shape as
mis-addressed DRAM NIU registers.

Question for the fabric owners: *can the EDM's local-write VC be configurable, or
use VC 1, for PCIe-tile destinations?* Until then the journal lands in L1, which
costs one cheap local PCIe read -- see below for exactly which L1.

**Why not push all the way to PCIe / host memory (design argument, still true):** The remote chip
has no PCIe -- `CoreType::PCIE` is a NOC endpoint only on the MMIO chip -- so any route to
host memory crosses the ethernet link first. tt-metal's own device->host path for a
remote device relays through cores that live on the MMIO chip:
`dispatch_core_manager.hpp` notes "remote device command queue interface cores are on the
associated MMIO device". A remote eth core *might* be able to address the MMIO chip's
PCIe tile directly in one hop, but tt-metal does not do it and eth routing support for
arbitrary destination tiles is unverified.

It also buys nearly nothing. Once the journal is in the MMIO chip's L1 the host reads it
over plain local PCIe: no `read_non_mmio`, no ethernet command queue, no `NON_MMIO`, no
polling a saturated core. Every property that causes the hang is already gone. The extra
hop would save one cheap local read in exchange for a hugepage, PCIe address-translation
setup, and a dependency on undocumented routing.

### Does this reintroduce the NON_MMIO problem?  No -- in steady state.

The only interprocess mutexes in UMD's I/O path are `PCIE_DMA` (on `dma_*` calls
only) and `NON_MMIO` (in `remote_communication_legacy_firmware`). A local
`read_from_device` goes straight to `device_protocol_->read_data` -- a TLB-window
read taking **no mutex at all**. TLB windows are allocated per-process by the KMD,
so there is no cross-process contention either.

| hop | crosses tunnel | host-side lock |
|---|---|---|
| Tensix -> gatherer eth core | no (local NOC) | none |
| gatherer -> MMIO chip L1 | fabric (chip-side) | none |
| host reads MMIO L1 | no (local PCIe) | **none** |

Per-sample host remote transactions: ~770/chip -> **0**. The failure that killed
the collector under Llama-70B -- one `read_non_mmio` blocking 30 s while holding
`NON_MMIO` -- becomes structurally impossible, because the host never calls it.

Interference is bounded and fails in our direction: under saturated fabric the
aggregator blocks on `wait_for_empty_write_slot()`, so OUR journal goes stale
rather than the workload stalling.

**Attach is NOT fixed.** Discovery still performs the remote ARC handshake (300 s,
no knob) and loading the kernel is a burst of remote writes taking `NON_MMIO`.
Push fixes steady state only, so the start-before-the-workload rule and the
systemd unit still stand. Worse: tt-metal's device init clobbers armed Tensix
counters and plausibly resets idle eth cores too, so **every relaunch re-enters
that exposure**. The aggregator therefore needs a heartbeat in the journal and a
relaunch policy that backs off rather than retrying into a running workload.

Two things to verify before building:

1. The push consumes ETH bandwidth the workload wants. Keep it a journal, not a stream —
   size and rate must be bounded and measured against a 70B TP run.
2. Idle-core availability was verified (§2) but NOT under a 70B tensor-parallel workload,
   which may occupy more links than the cases tested. Re-check there.

**ARC telemetry is not an alternative.** Reading a remote chip's ARC telemetry is an ARC
message over the same tunnel — `get_clock()` -> `write_to_arc_apb` -> `write_to_non_mmio`
— which is exactly the call that blocked the collector's publish thread in §5c. It has
the identical failure mode and would need the same push treatment.

### 3.4 Placement and launch  (rewritten 2026-08-29, see 3.5)

- Core selection: `host/agg_core_select.hpp`. Corrected 2026-08-29 -- there is NO
  reservation API to claim through on Wormhole:
    - `ServiceCoreManager` looks like the right registry (`dispatch_core_manager`
      already drops its `claimed_cores()` from the pool, and `service_active` is an
      escape in the `LaunchProgram` assert) but `claim()` opens with
      `TT_FATAL(cluster.is_ubb_galaxy() || arch == BLACKHOLE, "Service core claims are
      only supported on Blackhole and UBB Galaxy clusters.")`. Unavailable on exactly
      the system that has the problem.
    - So the implemented behaviour is: resolve the dispatch core type; under WORKER
      take an idle eth core, preferring the never-routed set `{2,3,4,5,10,11,12,13}`;
      under **ETH dispatch, refuse to start** with a reason, rather than gamble that
      dispatch will not reach the core we picked. A monitor that silently corrupts a
      dispatch kernel is a far worse failure than one that declines to start.
    - The fix that lifts the ETH-dispatch restriction is the pattern the real-time
      profiler already uses fifteen lines below the eth branch in
      `dispatch_core_manager.cpp`: reserve from the BACK of the pool at construction
      time, because dispatch consumes from the FRONT. That is an upstream change, and
      it is 7.6.
- Journal base: `hal::get_erisc_l1_unreserved_base()` + fixed offset, passed to the
  kernel as a compile-time arg and recomputed host-side. Avoids touching
  `dev_mem_map.h` and the cross-layer lockstep it implies (host C++, dev_msgs codegen,
  RISC-V linker scripts) -- the reason the Tensix reservation was painful in Phase 2.1.a.
- Env gate `TTNVTOP_ETH_AGGREGATOR=1`, default off.
- **Launch mechanism: the idle-eth HOST-dispatch launch-message path (3.5).** The
  earlier `CreateKernel(..., EthernetConfig{.eth_mode = Eth::IDLE})` sketch is
  withdrawn: 5d established that fast dispatch cannot launch to IDLE_ETH, so that
  form only ever worked under slow dispatch, before the workload.

### 3.5 Why the launch path is a launch MESSAGE, not a reset  (2026-08-29)

The mid-workload attach requirement made `CreateKernel` unusable and pointed at raw
`assert_risc_reset`/`deassert_risc_reset` over UMD. Reading the firmware shows a third
option that is strictly better, and it is the mechanism slow dispatch already uses.

**Idle eth cores are ALWAYS in host dispatch mode**, even when the workload runs fast
dispatch -- `risc_firmware_initializer.cpp:1225`:

```cpp
launch_msg.kernel_config().mode() = (!rtoptions_.get_fast_dispatch() or is_idle_eth)
                                        ? dev_msgs::DISPATCH_MODE_HOST
                                        : dev_msgs::DISPATCH_MODE_DEV;
```

That is *why* fast dispatch cannot reach IDLE_ETH -- not an unimplemented feature, a
deliberate mode split. And the idle-erisc firmware is already running and already
polling, `hw/firmware/src/tt-1xx/idle_erisc.cc:145`:

```c
mailboxes->go_messages[0].signal = RUN_MSG_DONE;
while (1) {
    while (mailboxes->go_messages[0].signal != RUN_MSG_GO) { ... }
    launch_msg_t* launch_msg = &mailboxes->launch[mailboxes->launch_msg_rd_ptr];
    ... kernel_init(kernel_config_base + launch_msg->kernel_config.kernel_text_offset[i])
    mailboxes->go_messages[0].signal = RUN_MSG_DONE;
    if (launch_msg->kernel_config.mode == DISPATCH_MODE_DEV) { /* notify, advance rd_ptr */ }
}
```

So the aggregator launches in three host writes over UMD, mid-workload, from any PID:

1. write the aggregator ELF's spans into the core's kernel-config region
2. write a `launch_msg_t` naming `kernel_text_offset` / `enables`
3. write `RUN_MSG_GO` into `go_messages[0].signal`

**This also resolves the kernel-lifetime open question (7.1) by construction.** The
firmware only regains control when the kernel *returns*. An aggregator that never
returns owns the core outright: `RUN_MSG_DONE` is never written, the `DISPATCH_MODE_DEV`
notify/rd-ptr-advance branch is not taken in host mode, and nothing else is looking.
This is exactly how the fabric EDM kernels persist -- it is a supported pattern, not a
trick.

#### Verified: program dispatch cannot disturb it

Every ethernet reference in `impl/program/dispatch.cpp` -- launch messages, kernel
config addresses, go-signal unicast, worker counts -- is
`HalProgrammableCoreType::ACTIVE_ETH`. `IDLE_ETH` appears nowhere in the dispatch path.
A kernel on an idle eth core is invisible to program dispatch, so it needs no
"persistent across dispatch" mechanism.

#### What DOES kill it: device init, not dispatch

`RiscFirmwareInitializer::assert_inactive_ethernet_cores()` resets `RiscType::ALL` on
every inactive eth core. It is called unconditionally from `assert_cores()`, and from
the init path when `FabricManagerMode::INIT_FABRIC` is set. The aggregator's lifetime
is therefore **one device-open epoch**: attach after init, and expect to be killed by
the next device open or close. The host must detect this (`sweep_count` stops
advancing, 3.3) and re-attach, not assume permanence.

#### Two implementation hazards

- **`RiscType::ERISC0` and `ERISC1` are aliases for `BRISC` and `TRISC0`**
  (`umd/device/types/risc_type.hpp`: `ERISC0 = 1ULL << 3` == `BRISC`, with a standing
  "Consider having separate entries" TODO). Any reset code that gets copy-pasted onto a
  Tensix core silently means something else. Prefer `RiscType::ALL`, and if the
  launch-message path is used as designed, no reset call is needed at all.
- **The collector needs the ELF and the `launch_msg_t` layout, not tt-metal.**
  `llrt::get_risc_binary(path)` is a pure ELF-to-`ll_api::memory` parse with no cluster
  dependency; the span writes can go straight through UMD `write_to_device`, and the
  dev_msgs layout is already generated into `tt_metal/hw/inc` (the collector includes
  `util_sampler.h` from there today). This keeps 7.2 answered in favour of the
  collector staying standalone. The build-system task is to emit the aggregator ERISC
  ELF as a fixed artifact at tt-metal build time rather than JIT-compiling it.

### 3.6 The launcher already exists: `LaunchProgram(force_slow_dispatch=true)`

3.5 concluded we would write the launch message ourselves. We do not have to.
`detail::LaunchProgram(dev, program, wait_until_cores_done=false,
force_slow_dispatch=true)` already does exactly the three writes 3.5 describes --
`ConfigureDeviceWithProgram` (binaries), `WriteRuntimeArgsToDevice`, then
`llrt::write_launch_msg_to_core`, which forces `DISPATCH_MODE_HOST` and writes the
launch message and go signal with `cluster.write_core_immediate`. No command queue,
no `EnqueueProgram`.

`wait_until_cores_done` MUST be false. A kernel that never returns never writes
`RUN_MSG_DONE`, so waiting on it hangs forever. That is the persistence mechanism,
not a bug.

**The one blocker was an assert, and it already had a carve-out for our exact case.**
`LaunchProgram` guards force-slow-dispatch with:

```cpp
TT_ASSERT(!(fd_active && rt_done) || service_active || dram_only,
          "Cannot force slow dispatch while fast dispatch firmware is active ...");
```

`dram_only` exists for the persistent tensor-prefetcher DRISC senders, on the stated
grounds that DRAM cores are "disjoint from the FD worker grid and dispatch column, so
launching them via slow dispatch does not perturb an active FD session."

That reasoning applies verbatim to IDLE_ETH, and we verified it independently: every
ethernet reference in `impl/program/dispatch.cpp` is ACTIVE_ETH, and the three sites
that meet IDLE_ETH skip it ("Fast dispatch not supported on IDLE_ETH yet"). So the
predicate was generalised from `program_targets_only_dram_cores` to
`program_targets_only_fd_disjoint_cores`, accepting DRAM and IDLE_ETH. DRAM behaviour
is unchanged.

That is the whole production launch path: a three-line predicate change, not a
bespoke ELF loader. It keeps `ConfigureDeviceWithProgram`'s kernel-config base and
rta-offset computation instead of duplicating it -- the fragile part.

## 4. Phasing  (re-cut 2026-08-29 around the launch-message path, 3.5)

> The original a/b/c split predates both the push revision (3.3, 5c) and the launch
> mechanism (3.5). It is superseded by the milestones below. The 5d verdict stands:
> build the drain-and-accumulate directly; the old "2.2.a transport only" survives only
> as the subset of work needed to stand the kernel up.

**The ordering principle: M1 deliberately dodges the one unanswered external
question.** Joining an EDM from a separate PID (7.4) is unresolved and owned by the
fabric team. Launching the aggregator in the same process that initialised fabric makes
that question moot, and still delivers the entire fidelity win -- which is the actual
justification per 5d. Mid-workload attach is a capability, not a prerequisite.

### M1 — Aggregator up, same-process launch  (~1–1.5 weeks)

Aggregator kernel on one idle eth core per remote chip, launched by the workload process
after device init, sweeping the Tensix L1 rings that `util_sampler.h` already publishes
and pushing the journal over fabric into an idle-eth L1 slot on the MMIO chip (3.3).
Host reads that slot over plain PCIe.

Build: aggregator ERISC kernel; ELF emitted as a fixed build artifact; host-side
launch-message writer (3.5); core claim through the dispatch core manager; journal
decode in the collector; `sweep_count` staleness detection and fallback to the existing
per-core drain.

Walks TRANSLATED space (2.1b), so one kernel covers every harvest pattern.

Exit criteria:
- `NON_MMIO` tunnel transactions per tick for remote chips: 256 -> **0**
- remote-chip fidelity no worse than today
- workload slowdown from telemetry at noise on the 5b dispatch-bound workload, not
  just on large matmuls
- kernel survives a full Llama-3.3-70B run without the collector stalling (the 5c
  failure, reproduced as a negative control)

### M2 — Restore 100 µs sampling  (~1 week)

Drop `UTIL_SAMPLER_DEFAULT_PERIOD_CYCLES` back toward 100 µs on aggregated chips and
size the journal for it (3.1: ~15 ms of buffering, needs a 50–100 Hz host drain).

Exit criteria: 100 µs sampling with `lost` at **zero** and <1% structural loss on an
aggregated chip, against the ~59% loss measured in Result 2 and the ~84% that forced
the 1 ms period in the first place. **This is the milestone that justifies the feature.**

### M3 — Mid-workload attach  (gated on 7.4)

Launch the aggregator from the `tt-coremon` process against a device another PID is
driving. The launch-message path (3.5) already works from any PID; what is unresolved
is whether that PID can join an EDM whose flow-control state the workload owns.

**Blocked until the fabric team answers 7.4.** File it now; do not design around it
speculatively.

### M4 — Local chips, opt-in  (~3 days)

Enable on MMIO chips too. The transport win is smaller (PCIe is cheaper than the tunnel)
but the sample-loss win is identical. Gate separately so a regression on local chips
cannot take out the common path.

### Explicitly out of scope

- **Blackhole.** 2.1 documents why the WH invariants do not transfer: BH harvests eth
  and PCIe, and exactly one of its two PCIe tiles is always fused off. Revisit only if a
  BH topology with remote chips ships.
- **Push all the way to host memory.** Blocked on the EDM static-VC finding (3.3, 5d)
  and unnecessary: landing in MMIO-chip L1 removes the tunnel, which is the whole point.

## 5. Evaluation — homelab-1

homelab-1 is a **T3K: 8 WH chips, 4 local (0–3) + 4 remote (4–7), 64 Tensix each**,
firmware bundle 19.13.2. Confirmed by running `ttnvtop-collector` there 2026-08-27.

This gives a built-in control: chips 0–3 (local, unaffected) vs 4–7 (remote,
aggregated). Same silicon, same workload, same tick.

Measurement matrix — a fixed multi-chip workload, wall-clock end to end:

| Run | Collector | Expected |
|---|---|---|
| A | off | baseline |
| B | on, current per-core remote reads | slowdown — magnitude is the thing to establish |
| C | on, Phase 2.2.a aggregator | slowdown → noise |
| D | on, Phase 2.2.b @ 100 µs | noise, with ~10× the samples of B |

Also record per run: `[ring-drain]` `entries` / `lost` / `drain_hz` from collector
stderr, and aggregator `sweep_count` / `lost`.

**Run A/B first, before building anything.** The mechanism in §1 is verified from
source; the *magnitude* of the workload impact is not. If B is already in the noise for
realistic workloads, Phase 2.2.a is not worth building and only 2.2.b (sample fidelity)
justifies the work.

## 5a. Measured results — homelab-1, 2026-08-27

Ran via `scripts/ab_remote_cost.sh --reps 5` (15 runs, 3 arms interleaved and rotated).
Workload: 2x4 mesh, 2048^2 matmul, 50,000 iters (~19-28 s timed region), 366 TFLOP/s
across 8 chips at start (~70% of LoFi peak).

### Result 1 — workload impact: NOT measurable

> **SUPERSEDED — see §5b.** This result holds only for large sustained matmuls. On a
> dispatch-bound workload (Llama decode) the collector stalls the workload outright.
> Do not cite this section without §5b.

Thermal steady state only (runs >= 24 s), mean wall-clock:

| Arm | n | mean | sd |
|---|---|---|---|
| A (no collector) | 4 | 28.194 s | 1.664 s (5.9%) |
| B_local | 3 | 28.274 s | 0.200 s (0.7%) |
| B_all | 3 | 27.486 s | 1.379 s (5.0%) |

`B_all - B_local` = **-2.79%** of baseline. Negative sign, i.e. noise, not signal.

**Caveat that bounds this result:** the sweep drifted **+48.9%** first run to last
(19.04 s -> 28.36 s) as the chips heated and AICLK drooped. Arm rotation spread the
drift across arms, but it left only n=3-4 per arm at steady state with sd up to 5.9%.
**This experiment can only resolve effects larger than ~6%.** A smaller real cost is
not excluded. The workload is also compute-bound (~2,600 dispatches/sec); a
dispatch-heavy workload would stress the path where the interference actually lands.

### Result 2 — telemetry quality: severe degradation

The validity check — confirming the collector really was reading remote chips — is where
the effect showed up. `[ring-drain]` stats for **chip 0, a LOCAL chip, in both arms**:

| Arm | n | drain_hz | samples lost | loss |
|---|---|---|---|---|
| B_local (chips 0-3) | 5 | **79.8** | 0 | **0.0%** |
| B_all (chips 0-7) | 5 | **16.5** | 1,857,871 | **58.6%** |

Zero overlap between the distributions (B_local 72.9-83.0 Hz; B_all 12.3-22.0 Hz).
Adding the 4 remote chips collapses collector drain **4.8x** and loses **~59% of
samples** — and it does so *on the local chips*, which have a direct PCIe path. The
remote tunnel is serializing the whole collector: the `NON_MMIO` mutex is held across
each multi-round-trip remote transaction, so local-chip sampling stalls behind it.

## 5b. Result 1 SUPERSEDED — 2026-08-28, dispatch-bound workload

Result 1 concluded workload impact was "NOT measurable" and the Verdict below
deprioritized Phase 2.2.a on that basis. **That conclusion does not hold.** §5a's own
caveat named the reason in advance:

> *"This experiment can only resolve effects larger than ~6%. A smaller real cost is not
> excluded. The workload is also compute-bound (~2,600 dispatches/sec); a dispatch-heavy
> workload would stress the path where the interference actually lands."*

and the follow-up list asked for exactly *"a dispatch-bound arm — many small ops rather
than 50k large matmuls."* That arm has now run.

**Evidence.** A Llama-3.1-8B decode on homelab-1's T3K, with the collector attached,
repeatedly logged:

```
UMD | Waiting for lock 'NON_MMIO_2_PCIe' which is currently held by ... PID <collector>
```

roughly every 9 s. The collector was stalling the inference workload outright — not a
sub-6% effect that the matmul A/B merely failed to resolve.

**Mechanism.** UMD acquires the `NON_MMIO` mutex *per*
`read_from_non_mmio_device()` call, not across a batch, and documents the acquisition as
non-trivial (`remote_communication_legacy_firmware.cpp`, "NON_MMIO_MUTEX Usage"). A
64-core sweep is therefore ~770 back-to-back acquire/release cycles per remote chip.
This is **starvation by transaction volume**, not one long hold — which is why a
compute-bound matmul issuing few remote transactions of its own showed nothing, while a
dispatch-heavy decode loses the race repeatedly.

Note this is the *workload-side* mirror of Result 2: the same serialization that
collapses collector drain 4.8x also starves any other process on that tunnel.

**Interim mitigation shipped (not a fix):** `--remote-budget` caps NOC transactions per
second per remote chip while another process holds a chip (`CHIP_IN_USE` peeked
read-only, never acquired), visiting cores round-robin so coverage rotates rather than
freezing. This costs no accuracy — each core's busy% is a ratio of its own counter
deltas over its own interval — but it does cost resolution: ~1.95 Hz per core on remote
chips vs ~80 Hz on local, permanently, and it divides further as chip count grows. It
rations the contention; it does not remove it.

## 5c. Why the transport must PUSH — Llama-3.3-70B, 2026-08-28

§5b established that a dispatch-bound workload is stalled by host-side remote polling.
A tensor-parallel 70B run shows something stronger: **the host itself gets stuck**, and
that is what dictates the aggregator's transport direction.

**Setup.** tt-coremon fully armed on all 8 chips of a T3K (discovery complete, 13.0 Hz
sweep), then Llama-3.3-70B batch-1 started. The collector's own watchdog fired after 30 s
of zero sampler progress and captured backtraces before aborting.

**Thread 2 (sampler)** — holds `NON_MMIO`, state `R`, >30 s inside ONE remote read:

```
memcpy_from_device
  SiliconTlbWindow::read_block
  PcieProtocol::read_data
  RemoteCommunicationLegacyFirmware::read_non_mmio
```

**Thread 1 (publish)** — queued behind it on the same mutex:

```
futex_wait / __pthread_mutex_lock_full
  RobustMutex::lock
  LockManager::acquire_mutex(NON_MMIO)
  RemoteCommunicationLegacyFirmware::write_to_non_mmio
  WormholeTTDevice::write_to_arc_apb
  WormholeArcMessenger::send_message
  WormholeTTDevice::get_clock          <-- AICLK telemetry, once per publish tick
```

**Mechanism.** `read_non_mmio` writes a command into the ethernet core's firmware queue
on the MMIO chip and polls for completion. Under 70B tensor-parallel those cores are
saturated moving model traffic, so the command is not serviced and the host polls for
tens of seconds while holding `NON_MMIO`.

**Consequences that shaped the design:**

- Transaction count is not the lever. The lean cross-tick path had already halved
  ops/core (12 -> 6) and taken 8B workloads to zero waits; 70B still wedged it. **Any
  single remote read can block**, so reducing their number lowers probability, not risk.
- `get_clock()` per publish tick was a real bug (fixed: AICLK is now rate-limited, 1 s
  local / 5 s remote) but it was the VICTIM here, not the cause — it was merely the next
  caller to want the lock. Worth recording, because two host-side "fixes" that day were
  aimed at symptoms.
- The workload was never harmed in any monitor-ready-first run: 70B reported
  `1 passed`, 10.33 tok/s/user, zero `NON_MMIO` waits. The watchdog kills the monitor,
  not the job. Containment works; prevention does not exist host-side.

**Therefore §3.3 pulls no data over the tunnel.** See the pull-vs-push table there.

## 5d. Transport verified end-to-end except one hop — 2026-08-29

Everything the push design needs is now proven on homelab-1's T3K, by test or by
shipped code, with a single exception:

| link | status |
|---|---|
| spare (unlinked) eth core runs a kernel | `Eth::IDLE` + `get_inactive_ethernet_cores()` |
| ERISC can be a fabric client | VC2 runtime-arg path, `edm_fabric_worker_adapters.hpp` |
| idle-eth fabric kernel runs on this T3K | `TestSetUnicastRouteIdleEth` PASSES |
| sender opens connection and completes sends | markers: `alive=0x09E00000 sends_done=16` |
| fabric header carries an arbitrary 64-bit dest | `NocUnicastCommandHeader { uint64_t noc_address; }` |
| receiver passes dest through unvalidated | straight into `noc_async_write_one_packet_with_trid` |
| a kernel can reach host memory over PCIe | shipped (`cq_realtime_profiler_push.cpp`) and re-verified |
| EDM delivers to **L1** | ordinary destination -- this is the design |
| EDM delivers to a **PCIe tile** | **NO** -- static VC 2/3, see 3.3 |

Also settled along the way:

- **The 12 "free" eth channels per remote chip are firmware-alive, not dead
  silicon.** Probed directly: their ERISC mailbox is populated (magic `0xabcd1234`,
  sensible per-channel identity) but reports no peer (`0xffffffff` in the remote-info
  words) and zero traffic counters. Section 2's claim that "no recabling can claim
  them" is an inference from shipped cluster descriptors, NOT evidence of an absent
  PHY; the mailbox distinguishes has-peer from no-peer, not has-PHY from no-PHY.
  Either way they cannot transmit today -- which is why the aggregator hands its
  journal to FABRIC rather than needing a wire of its own.
- **Fabric claims every link**: `num_links = min(chans_dir1.size(), chans_dir2.size())`
  builds a router pair per link index, so on a T3K both `ch0`/`ch1` toward the MMIO
  chip are taken. There is no conflict-free private link, and that is fine: joining
  fabric as one more small client is the design, not a compromise.
- **Fast dispatch cannot launch to IDLE_ETH** (`impl/program/dispatch.cpp`:
  "Fast dispatch not supported on IDLE_ETH yet"; matching `TT_ASSERT` in
  `worker_config_buffer.cpp`). The aggregator must be launched under slow dispatch
  before the workload, or loaded directly over UMD
  (`assert_risc_reset`/`deassert_risc_reset` with `RiscType::ERISC0/1`).

### Verdict

- ~~**Phase 2.2.a (transport) is NOT justified by workload impact.** Deprioritize.~~
  **REVISED 2026-08-28 (see §5b): workload impact IS real on dispatch-bound workloads.**
  The original finding was an artifact of testing only large matmuls.
- **The transport must PUSH (§3.3, §5c).** A host pull of the journal keeps the exact
  failure mode that wedges the collector today; it only makes it ~256x rarer. Building
  2.2.b as originally specified would not fix Llama-70B.
- **Phase 2.2.b (drain-and-accumulate) is justified, and more urgently than assumed.**
  The remote tunnel is not merely slow; it poisons telemetry fidelity chip-wide. The
  ~59% loss here is on top of the sampling-period compromise already documented in
  `util_sampler.h`.
- Recommended re-phasing: build 2.2.b directly. 2.2.a survives only as the subset of
  2.2.b needed to stand the kernel up.
- **Both justifications now point the same way.** Result 2 (collector fidelity: 4.8x
  drain collapse, ~59% sample loss) and §5b (workload impact: outright stalls on
  dispatch-bound work) are the same serialization seen from either side. The host-side
  mitigation rations it and permanently halves remote-chip resolution; only moving the
  fine-grained reads on-chip removes it.

### Redoing this measurement better

1. **Thermal soak** — 60-120 s of the workload discarded before the first timed run, so
   all arms measure from the same plateau. This is the single biggest fix.
2. ~~**Add a dispatch-bound arm** — many small ops rather than 50k large matmuls.~~
   **DONE 2026-08-28** — Llama-3.1-8B decode; see §5b. This was the decisive one.
3. **Log AICLK per run** (ARC `TAG_AICLK`) to confirm the plateau rather than infer it.

---

## 5e. M1 transport verified on Blackhole -- 2026-08-29

`TestEthAggregatorLaunchesWithoutDispatch` (fabric_unit_tests) on the 4x p150a box:

```
landing on chip 2 eth 0-6 translated (26,25) dest=0x000065a000015740
launch: t1 state=0x09e00000 sweeps=2698 head=2701 cores=1
launch: t2 state=0x09e00000 sweeps=5396 head=5402 cores=1
landed: magic=0x47415454 head=5404 sweeps=5398 lost=0 cores=1
```

Established, on hardware:

| link | status |
|---|---|
| aggregator launches with NO dispatch | `LaunchProgram(force_slow_dispatch=true)`, 3.6 |
| kernel persists (never returns) | `sweeps` advancing 2698 -> 5396 across 2 s |
| runtime args read correctly | `cores=1` matches what was passed |
| fabric connection opens from idle ETH | marker `0x09E00000` |
| Tensix ring sweep produces entries | `head` advancing with `sweeps` |
| sweep keeps up with the producer | `lost=0` |
| journal LANDS in peer idle-eth L1 | `magic=0x47415454` ('TTAG') read back over PCIe |
| header checksum survives the wire | verified host-side |

NOT yet covered: the REMOTE-chip case. This box is all-MMIO, so
`TestEthAggregatorJournalLands` skips ("no remote chip") and the WH T3K on homelab-1
is still required for the case the feature exists for. Multi-core sweep (64 cores
rather than 1) and the collector-side decode are also outstanding.

Two bugs found by construction, both worth recording because neither would have
failed loudly:
- **L1 NOC reads need 16 B alignment at both ends** (`NOC_L1_READ_ALIGNMENT_BYTES` 16
  on WH and BH). Reading the 4 B ring `head` at offset 8 is misaligned. The sweep now
  pulls the aligned 16 B chunk containing it. This also forced `util_agg_entry_t` to
  put the 16 B sample at offset 0 rather than after `core_id`/`seq`, so a ring slot
  can be read straight into it.
- **`MEM_IERISC_STACK_MIN_SIZE` is 128 BYTES.** A per-core `seq[]` array of 128 u32
  is 512 B and silently smashes the ERISC stack. It lives in L1.

## 5f. Multi-core sweep, journal wrap, and three lifecycle findings -- 2026-08-29

5e ran `num_cores=1` against a 6142-entry journal, which left three kernel paths
never executed: the mid-core flush, the wrap-split write, and the `lost` accounting.
`TestEthAggregatorMultiCoreAndWrap` forces all three by shrinking `capacity` and
`stage_entries_max` (both runtime args) and sweeping the whole grid:

```
multicore: fabric link_idx 1
multicore: 120 cores, capacity=256 stage=8 landing chip 2 eth 0-3
multicore: head=972249 sweeps=73298 lost=0 cores=120 cap=256
multicore: 120 distinct core_ids, 0 unwritten slots, 0 bad core_ids
```

120 Tensix cores swept, journal wrapped **3798 times**, mid-core flush firing ~15x per
sweep, all 120 core_ids present, no unwritten slots, no out-of-range core_ids, and a
1 KiB poison guard band immediately after the journal came back untouched -- so no
wrap-split write ran off the end of the ring. No bug found in the paths that had
never run.

### Finding 1 -- UTIL_AGG_MAX_CORES was too small for an unharvested Blackhole

It was 128. `blackhole::TENSIX_GRID_SIZE` is `{14, 10}` = **140** unharvested; the
p150a measured here has 2 columns harvested and reports 120, which is why it passed.
Raised to 160, sized from the unharvested grids of both arches (WH 8x10 = 80).

### Finding 2 -- a persistent kernel CANNOT be relaunched over

The kernel never returns, so the firmware loop that consumes launch messages is not
running. A second `LaunchProgram` on the same core is therefore never picked up --
AND `ConfigureDeviceWithProgram` writes the new binary and runtime args straight over
the live kernel, which keeps running on corrupted state and keeps pushing
plausible-looking WRONG telemetry:

```
sender markers state=0x00000000 sweeps=0 head=0 cores=0            <- 2nd never started
landed magic=0x00000000 sweeps=5400 lost=5586 cores=0 cap=2240     <- 1st corrupted
```

This is the "two armers" hazard from the collector work, in the aggregator. A launcher
must detect an existing aggregator and refuse, or stop it first -- never launch onto a
live core. `host/agg_core_select.hpp` grows `rank_aggregator_eth_cores()` so
independent callers take distinct cores deterministically instead of colliding.

`stop_aggregator()` asserts `RiscType::ALL` and stops there. Deasserting does NOT
bring the core back: an assert+deassert pair leaves the next launch silently never
starting. Restarting an ERISC needs its reset vector pointed at the firmware, which is
device-init's job. "Stopped until the device is reopened" is the intended semantics --
the aggregator's lifetime is one device-open epoch anyway (3.5).

### Finding 3 -- a dead fabric client does NOT release its EDM connection (7.4)

The strongest result here, and it is 7.4 measured rather than assumed.

The first aggregator was stopped by asserting its reset. It never reached
`sender.close()` -- it cannot, the kernel does not return. A second aggregator on a
DIFFERENT core, pushing to a DIFFERENT landing slot, then starved:

| link_idx | sweeps in 6 s |
|---|---|
| 0 (held by the dead client) | **5** |
| 1 | **73,284** |

Same kernel, same everything else. ~15,000x, from the link index alone. The EDM still
believes a client is connected on link 0 and the new client blocks in
`wait_for_empty_write_slot()`.

Consequences for the design:

- **An aggregator holds its EDM connection for the whole device-open epoch.** If it
  dies -- reset, crash, a workload tearing the device down -- that link is degraded
  until fabric re-init. This needs saying to the fabric owners alongside 7.4.
- **Two fabric clients must not share a `link_idx`.** On a T3K each remote chip runs
  one aggregator pushing to its own MMIO chip, so the production topology does not hit
  this; a second monitoring client on the same chip would.
- It makes 7.4 sharper: the question is not only whether a separate PID can *join* an
  EDM, but whether it can ever *safely leave* one.

## 5g. M1 on the T3K -- data path PROVEN on a remote chip, launch path is not -- 2026-08-29

First run on real Wormhole hardware (homelab-1, T3K: chips 0-3 MMIO, 4-7 remote).

### The M1 exit criterion is met on the data path

```
aggregator on chip 4 eth ch2, landing chip 3 eth ch2
aggregator: sweeping 64 Tensix cores on remote chip 4
aggregator: landing on chip 3 eth 0-2 translated (24,16) dest=0x000041800000e200
aggregator: t1 magic=0x47415454 head=191461 sweeps=2953 lost=0 cores=64
aggregator: t2 magic=0x47415454 head=317645 sweeps=4898 lost=0 cores=64
  entry[0] wall=0x3a8c7327 kid=0 fpu=0 core_id=5 seq=4894
```

An aggregator on a REMOTE chip, sweeping all 64 of its Tensix rings, pushing over
fabric to an MMIO chip, read back by the host over plain PCIe -- the tunnel never
touched in steady state. `head` advanced 191461 -> 317645 in 2 s = ~63k entries/s,
which is exactly 64 cores x 1 kHz. `lost=0`.

64 cores confirms 2.1 on real WH silicon: 8x10 with 2 rows harvested. The selector
picked eth **ch2**, one of the never-routed channels, so the WH preference works.

Multi-core and wrap also pass on WH: 64 cores, journal wrapped 1478x, 64 distinct
core_ids, 0 unwritten slots, 0 bad core_ids, guard band clean.

### The launch path is NOT reliable, and it is not our kernel

Launching onto a remote chip wedges the NON_MMIO tunnel roughly half the time, in
`RemoteCommunicationLegacyFirmware::wait_for_non_mmio_flush` after its fixed 5 s
`NON_MMIO_RW_TIMEOUT` (`umd/device/utils/timeouts.hpp`, compile-time, no env knob --
the same shape as the `ARC_STARTUP_TIMEOUT` problem):

```
wait_for_non_mmio_flush            <- times out
Cluster::write_core_immediate
llrt::write_launch_msg_to_core
detail::LaunchProgram
```

The flush spins until every tunnel eth core on the MMIO chip has drained its request
queue (`erisc_q_ptrs[0] == erisc_q_ptrs[4]`) and matched its write-response counters.

What is established, and what is not:

| claim | evidence |
|---|---|
| Not leftover aggregators / stale state | fails 3/3 immediately after a full `tt-smi -r` |
| Not remote idle-eth launch in general | `TestFabricWriteReachesRemoteL1_Control` launches a remote idle-eth kernel the same way: **4/4 PASS** |
| Not transient within a process | when it wedges, 5 retries all fail identically -- retrying inside the process is useless |
| Aggravated by remote-write volume | with the 512 B ring-table pre-write 2/4 launch; without it 3/4. Aggravates, does not solely cause |
| Root cause | **NOT established.** UMD/eth-firmware tunnel behaviour, same family as 5c |

Two theories were tested and are WRONG, recorded so nobody re-runs them: leftover
aggregators from prior runs (killed by the reset test), and pass/fail alternation
(7 runs gave FAIL PASS FAIL FAIL PASS FAIL FAIL).

### What this means for the design

**The aggregator removes the tunnel from steady-state telemetry. It does not remove
the tunnel from its own launch.** Getting a kernel onto a remote chip is inherently a
tunnel operation -- binary, runtime args, launch message -- and that is exactly the
path 5c showed collapsing under load. On an idle T3K it already fails half the time.

Actions:

1. ~~**Drop the ring-address table.**~~ **DONE 2026-08-29, and it helped measurably.**
   The kernel now addresses cores from a live-x / live-y cross product (2.1b) instead of
   a host-written table, removing a `num_cores * 8` B L1 write to the remote chip and
   its `WriteToDeviceL1` call -- WH 512 B, BH 960 B, replaced by 16 and 22 runtime-arg
   words respectively.

   Remote launch success on the T3K went from **2 of 7 to 4 of 6**. An improvement, not
   a fix: the tunnel still wedges about a third of the time, so 7.7 stands. Correctness
   held on both arches -- BH `translated grid 12x10` = 120 cores, WH `8x8` = 64, all
   core_ids present, `lost=0`.
2. Raise with the UMD owners as 7.7.
3. It sharpens M3 (mid-workload attach) considerably: if launch is unreliable on an
   IDLE T3K, launching into a saturated one is worse. M3 may need the aggregator
   started before the workload regardless of the EDM question (7.4).

The test SKIPs rather than fails when the launch wedges, since the data path is what
it covers and that is proven.

## 5h. 7.7 RESOLVED -- the tunnel was waiting on channels it never uses -- 2026-08-29

7.7 (remote launch wedging the NON_MMIO tunnel) is fixed, and the fix was already
sitting in the plan as 7.3.

### Cause

`cluster.cpp:780`, the default when nobody overrides it:

```cpp
const std::set<uint32_t> active_channels = cluster_desc->get_active_eth_channels(chip_id);
remote_communications_.at(chip_id)->set_remote_transfer_ethernet_cores(... active_channels ...);
```

UMD uses **every active eth channel on the MMIO chip** for remote transfers, and
tt-metal **never calls** `configure_active_ethernet_cores_for_mmio_device` to narrow it.
On a T3K MMIO chip that is six channels, confirmed on hardware:

```
tunnel: mmio chip 3 active eth channels: 6 7 8 9 14 15
```

Per the board map (2): 6,7 -> a QSFP-DD cage, 8,9 -> the internal trace to this chip's
own remote ASIC, 14,15 -> the Warp 100 bridge. Only **8,9** can carry a transfer to
chip 4. The other four link to entirely different boards.

`wait_for_non_mmio_flush` nevertheless iterates ALL of them and waits for each to drain
its request queue and match its write-response counters. Four of those six are busy
carrying fabric traffic for other chips and do not drain inside the fixed 5 s
`NON_MMIO_RW_TIMEOUT`. UMD also round-robins `active_eth_core_idx` across the whole set,
so transfers get issued on channels that cannot reach the target at all.

### Measurement

Remote launch of the aggregator on the T3K, varying only the tunnel's channel set:

| tunnel channels | pass rate |
|---|---|
| default (6,7,8,9,14,15) | **1/6** |
| **8,9 -- the link pair to the remote chip** | **14/14** |
| 8 alone | 0/4 |
| 9 alone | 0/6 |
| 6,7 (QSFP -- wrong link) | 0/3 |

Both channels of the pair, and nothing else. Deriving the set from the cluster
descriptor rather than hardcoding it -- `get_directly_connected_ethernet_channels_
between_chips(mmio_id, remote_id)` -- gives **8/8**, with the aggregator's own numbers
unchanged and correct (64 cores, `lost=0`).

### This is bigger than the aggregator

Nothing here is specific to this feature. It applies to **every remote-chip operation
on a T3K**: the tunnel waits on four ethernet cores that are irrelevant to the transfer
and contended by fabric. That is very likely a contributor to 5c, where the collector
held NON_MMIO for tens of seconds under Llama-3.3-70B and stalled the workload.

**Recommendation for the tt-metal/UMD owners:** call
`configure_active_ethernet_cores_for_mmio_device()` at device init with the channels
that actually link each MMIO chip to its remote chips, instead of leaving UMD to
default to all active channels. It is a one-call change with a measured 1/6 -> 14/14
effect on an idle machine.

Until that lands, the aggregator's launcher applies the pinning itself.

## 5i. Collector-side journal discovery verified -- 2026-08-29

The last M1 piece is the collector reading the landed journal. Discovery and header
decode are done and verified against a live aggregator on the T3K; entry demux into the
attribution path is not.

`ttnvtop-collector --journal-probe`, run from a SEPARATE process while the workload
process held all 8 chips:

```
journal probe: landing base 0xe200
  chip 3 eth (24,16)  src_chip=4 cores=64 capacity=6142 head=1249692 sweeps=19072
```

That confirms, on hardware:

- **The landing-address formula is right.** The collector does not link tt-metal (7.2),
  so it mirrors `wh_hal_idle_eth.cpp`'s
  `((MEM_IERISC_MAP_END + 25 KiB - 1) | 31) + 1` -> `0xe200`, matching the launcher's
  `dest=0x000041800000e200`.
- **Discovery needs no IPC.** The header is self-describing, so the collector finds the
  journal by scanning the MMIO chip's 16 ethernet cores for `TTAG` and validating the
  checksum. In M1 the launcher is the workload process and the collector is a separate
  one, with no channel between them; this closes that gap without inventing one.
- **It never touches the tunnel.** The probe skips remote chips outright -- journals
  land on the MMIO side by design, so this is plain PCIe.
- **`util_aggregator.h` being dependency-free paid off**: the collector includes it
  directly, so the journal layout has ONE definition shared with the kernel, unlike the
  ring layout which is hand-mirrored as `ttnvtop_ring`.

Two bugs the first live run exposed, both fixed:

- **`src_chip` was the fabric node id, not the physical chip id.** A journal from remote
  chip 4 announced `src_chip=0`, because the fabric node id is mesh-local. The collector
  thinks in physical chip ids and has no mesh map, so it could not have demuxed. The
  launcher now stamps the physical id.
- **`--journal-probe` was not actually read-only.** It armed the perf counters, wrote
  `period_cycles` on every core, and ran the FPU_OUT_L liveness probe -- all device
  WRITES, and on a remote chip they cross the tunnel. The flag claimed read-only in its
  help text while perturbing every chip it looked at. Now genuinely read-only.

**Remaining for M1:** ingest journal entries into the existing per-kernel attribution
path -- demux `core_id` to the source chip's `CoreState` using the same live-x/live-y
ordering the kernel walks, track `last_head` and `lost`, and fall back to the per-core
tunnel drain when no journal is present or `sweep_count` stops advancing.

## 5j. 7.4 answered -- a separate PID CAN start the aggregator and join a live EDM -- 2026-08-29

The requirement is an aggregator independent of the workload: separate PID, started
before or during the run. Two tt-metal processes cannot share a device (CHIP_IN_USE),
so the second process must be UMD-only. Tested by splitting the launch.

Process A (tt-metal): initialises fabric, compiles the kernel, writes the binary, the
runtime args and the launch message -- then deliberately withholds the go word
(`llrt::write_launch_msg_to_core(..., send_go=false)`) -- and holds the device.

Process B (`ttnvtop-collector --launch-go`, **UMD only, no tt-metal**): writes the go
word. Four bytes.

```
launch-go: chip 0 eth (19,16) addr 0x2490 4 bytes
launch-go: go word written
journal probe:  chip 1 eth (19,16)  src_chip=0 cores=64 head=540608 sweeps=8219
after hold:     markers state=0x09e00000 sweeps=26666 head=1754176 cores=64
```

`0x09E00000` is the fabric-connection-opened marker. So a kernel started by a DIFFERENT
process opened a connection on an EDM this process owns, swept 64 Tensix cores, and
pushed its journal -- which B then read back over PCIe.

### Why tt-metal is not the blocker

The device-side launch contract is four plain L1 writes (3.5). Nothing in it needs
tt-metal. Two practical dependencies remain, and neither is a device requirement:

1. **The kernel ELF is JIT-compiled.** Fix: emit it as a fixed build artifact and write
   its spans over UMD. `llrt::get_risc_binary()` is a pure ELF parse with no cluster
   dependency, and `launch_msg_t` is a generated header, not a library.
2. **The fabric connection args.** These are **DERIVED, not allocated** --
   `fabric.cpp` builds the whole `SenderWorkerAdapterSpec` from static `edm_config`
   addresses plus the control plane, and for ETH clients the sender channel is
   hardcoded:

   ```cpp
   // Sender channel 0 is always for local worker in the new design
   const auto sender_channel = 0;
   ```

   There is no per-process allocator to contend on, so a second process can compute
   identical args without coordination. That is engineering, not an unknown.

### The real remaining constraint is CONTENTION, not permission

`sender_channel = 0` is the ONLY local-worker channel per EDM, and the EDM's
`worker_location_info` holds one worker's location. So **one connected worker at a time
per (router channel, link_idx)**. That is exactly the 5f measurement -- a second client
on a link whose slot was still held ran at 5 sweeps vs 73,284.

Consequence for running alongside a real workload: the aggregator needs a `link_idx`
the workload is not using. On a T3K internal trace `num_links = 2`. If the workload's
own fabric clients occupy both, there is no slot for telemetry.

**This is the question for the fabric owners, and it is now specific:** can a low-rate
telemetry client share an EDM sender channel, or can one be reserved? It is a design
constraint, not a bug, and it is what gates 7.4-for-real-workloads even though the
mechanism itself is proven.

### Not yet covered

The staged handoff proves the EDM join and the cross-process start. It does NOT yet
prove B can compute the connection args itself (derived, so expected to work) or write
the binary itself (needs the ELF artifact). And the test staged only ONE aggregator, so
coexistence with an ACTIVE competing client on the same EDM is still untested.

## 5k. Fabric push REMOVED; host-pull of a local journal -- 2026-08-30

The push design is withdrawn. The aggregator now sweeps into a journal in its OWN eth
L1 and uses NO FABRIC AT ALL; the host reads the journal where it lies.

### Why the reversal

Reading the EDM code settled 7.4's remaining half:

- `EDMChannelWorkerLocationInfo` (fabric_edm_types.hpp:69) holds exactly ONE worker --
  `worker_semaphore_address`, `worker_teardown_semaphore_address`, `worker_xy`,
  `edm_read_counter`. Not an array.
- `WorkerToFabricEdmSender::open_start()` writes that struct UNCONDITIONALLY: three
  `noc_inline_dw_write`s, no test-and-set, no wait-for-free, no arbitration. A second
  worker silently overwrites the first, after which the EDM sends credits to the new
  worker while the old one waits forever.
- The handshake semaphore is tri-state (`unused=0 / open=1 / close=2`), not a counter.
  It cannot represent two connected workers.

So a persistent telemetry client does not "share" sender channel 0 -- it clobbers
whatever CCL op connects next, and gets clobbered in turn. That is the mechanism behind
5f's 5-sweeps-vs-73,284, and it means transient open/close does not help either: with no
mutual exclusion, being brief only narrows the race.

Reserving a dedicated sender channel or taking a `link_idx` both remove interconnect
resource from the workload, which is not acceptable for monitoring. Hence: no fabric.

### What that buys

The aggregator consumes ZERO fabric resource. What leaves the chip is one host-initiated
journal read instead of 64 per-core reads -- attacking the mechanism 5b identified,
starvation by transaction VOLUME (~770 NON_MMIO acquire/release cycles per sweep).

Verified on Blackhole: 120 cores, `lost=0`, journal wrapped 3797x, all 120 core_ids
present, guard band clean.

### Two hard findings from the T3K

**1. A remote read is NOT a coherent snapshot at any granularity.** The first layout put
`head` at offset 8 and a checksum at offset 32 -- different 16 B chunks -- and every
remote read tore, with every field individually sane. Moving `head` and a `head_xor`
tear-detector ADJACENT in chunk 0 did not fix it either: they disagreed too
(`head=621121`, `head_xor` decoding to 625337). A remote read samples words across the
read's duration; it is not atomic even within 16 bytes. Any host-pull scheme must
tolerate that -- a single-word `head` read, or a header held stable for longer than a
read takes.

**2. The aggregator's own NOC sweep degrades the host's tunnel reads.** Measured on
remote chip 4, varying only the sweep interval:

| sweep interval | 64 B header read | result |
|---|---|---|
| 1 ms (64k NOC reads/s) | **2512 ms** | torn, failed |
| 20 ms (3.2k NOC reads/s) | **634 ms** | **passed** |

The eth core servicing remote IO competes with the sweep for the same chip. This is a
direct tension with M2's 100 us goal: the finer we sample on-chip, the slower and less
reliable the host's readout of it becomes. The tradeoff curve is real and needs
characterising before M2 is scoped.

### Status

- Blackhole (local journal, multi-core, wrap, guard): **passing**
- T3K remote chip at a 20 ms sweep: **passing**
- T3K remote chip at a 1 ms sweep: **fails** -- read cost and tearing, per finding 2

## 5l. Llama-3.3-70B on the T3K: monitoring impact is UNDETECTABLE -- 2026-08-30

Ran `models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"`
against Llama-3.3-70B on the T3K, three arms, one run each:

| arm | remote txns/s | avg speed | throughput | NON_MMIO lock waits |
|---|---|---|---|---|
| baseline, no monitoring | 0 | 96.63 ms | 10.35 tok/s | -- |
| collector at DEFAULTS | ~12,800 | 96.71 ms | 10.34 tok/s | 1 |
| `--journal-transport` (2 reads/tick) | ~20 | 97.11 ms | 10.30 tok/s | 0 |

Total spread 0.5%, and **the ordering is backwards** -- the arm with 640x less remote
traffic came out marginally slower. That is noise dominating, not signal.

### What this does and does not say

It does NOT say host-pull is safe. It says **this experiment cannot distinguish the
designs**, because no monitoring load produced a measurable effect. 5c's stall does not
reproduce on this workload.

The likely reason is in the log: `Done Capturing Decode Trace`. This demo runs TRACED
decode, so the steady-state loop replays a captured trace with minimal host involvement
and there is little for the collector's tunnel traffic to collide with. That makes 5c's
stall **workload-shape dependent** rather than a property of the tunnel -- a materially
different conclusion from the one this plan carried, and the one that motivated the
whole push design.

### The one real datum

In `--journal-transport` mode the collector's own drain ran at **0.5 Hz**: ~2 s per tick
for ~8 remote reads across 4 remote chips, i.e. **~250 ms per remote read while Llama
runs**. Remote reads under fabric load are slow. They simply do not hurt THIS workload.

That number matters for the aggregator's readout budget, and it compounds with the 5k
finding (our own sweep pushing a 64 B read to 2512 ms). Reading a journal is cheap in
transaction COUNT but each transaction is expensive in LATENCY, which bounds drain rate
and therefore how much on-chip buffering is required.

### What to do instead

Chasing transport tuning against an undetectable effect is not worth more runs. The
useful questions now:

1. **Find the execution shape that actually stalled in 5c.** Prefill? Non-traced
   dispatch? Multi-CQ? The push-vs-pull argument was built on that measurement, and it
   needs to be reproduced before it justifies anything.
2. **Characterise remote-read latency** (~250 ms observed) against sweep rate and
   workload, since that -- not transaction count -- is what bounds the design.
3. Repeats. One run per arm; 0.5% spread is well inside uncharacterised variance.

## 5m. Remote-read latency MEASURED -- 5k and 5l were wrong -- 2026-08-30

`ttnvtop-collector --read-latency-probe`, remote chip 4, median of 3, read-only:

| transfer | idle | under Llama-3.3-70B |
|---|---|---|
| 64 B | 0.0 ms | 0.1 ms |
| 256 B | 0.0 ms | 0.1 ms |
| 1 KB | 0.1 ms | 0.2 ms |
| 2 KB | 0.2 ms | 0.2 ms |
| 4 KB | 0.3 ms | 0.6 ms |
| 8 KB | 0.6 ms | **1.1 ms** |

Size-proportional, ~13 MB/s idle and ~7 MB/s under full load. Llama in that same run:
97.05 ms @ 10.3 tok/s, unchanged from every other arm.

### Two retractions

**5l claimed "~250 ms per remote read while Llama runs." That is WRONG.** It was not
measured -- it was a 2 s drain-tick duration divided by ~8 reads, reported as if it were
a per-transfer cost. An 8 KB read under the same load is **1.1 ms**, ~230x faster. The
2 s tick was spent elsewhere: almost certainly waiting on the NON_MMIO mutex held by
tt-metal's own remote traffic, plus the per-core work still running for LOCAL chips in
that loop. Drain rate under load is a LOCK CONTENTION story, not a transfer-cost story.

**5k's conclusion "the finer we sample on-chip, the slower and less reliable the host's
readout becomes" is WITHDRAWN.** The 2512 ms figure behind it was a header read with 8
RETRIES against a header republishing at 1 kHz -- a self-inflicted retry storm from the
publish-ordering bug, not transfer cost. A single 64 B read is 0.0-0.1 ms. There is no
sampling-rate-versus-readout tension; that was an artifact of a bug that is now fixed.

Both errors ran the same direction: inferring per-transfer latency from a loop time that
contained something else. Measure the transfer.

### What this means for the drain rate

10 Hz is not merely reachable, it is far below the ceiling:

| payload per chip per tick | at ~7 MB/s loaded | 4 remote chips | max drain |
|---|---|---|---|
| aggregated per-core state, ~2 KB | 0.2 ms | 0.8 ms | **>1000 Hz** |
| raw samples, M1 rate (204 KB) | ~29 ms | ~116 ms | ~8 Hz |
| raw samples, M2 100 us (2 MB) | ~286 ms | ~1.1 s | ~1 Hz |

So on-chip aggregation is still the right design -- it is what makes M2's 100 us
sampling free, since a fixed per-core state table does not grow with sample rate,
whereas raw samples are already marginal at M1 and hopeless at M2. But it is an
optimisation for headroom, not a rescue from a broken transport.

### Operational finding

The collector must not be SIGKILLed. Killing it mid-transaction (`pkill -9`) left UMD
topology discovery wedging in `wait_arc_core_start` on the next open, unrecoverable
without a board reset. A monitoring daemon that bricks the tunnel when killed is a
defect in its own right: it needs a signal handler that completes the in-flight
transaction and releases NON_MMIO before exiting.

## 5n. v2: on-chip aggregation -- fixed-size state table -- 2026-08-30

The journal is no longer a ring of raw samples. The aggregator now does the delta
arithmetic on-chip and publishes a FIXED per-core state table, overwritten in place.

```c
struct util_agg_core_state_t {   // 32 B
    uint32_t busy_cycles;  // accumulated FPU/SFPU deltas, monotonic
    uint32_t wall_cycles;  // accumulated wall-clock deltas over the same interval
    uint32_t samples; uint32_t kernel_id; uint32_t resets; uint32_t seq;
    uint8_t counter_sel; uint8_t flags; uint16_t rsvd16; uint32_t rsvd;
};
```

64 cores = 2112 B total, and that does not change with the sample rate.

Two properties matter:

- **The payload stops scaling with sampling.** Raw samples at 100 us are ~2 MB per chip
  per drain tick; at the measured ~7 MB/s under load (5m) that caps the drain near 1 Hz.
  A 2 KB table reads in 0.2 ms, so 10 Hz has three orders of magnitude of headroom. This
  is what makes M2's 100 us goal free rather than fatal.
- **The accumulators are MONOTONIC, so the readout is loss-immune.** A host that misses
  a read gets a bigger delta next time. There is no ring to lap, no wrap handling on the
  host side, and `lost` now counts only what the ON-CHIP sweep missed.

The host's job shrinks to: read the table, diff against the previous read with unsigned
arithmetic, `util = d(busy) / d(wall)`.

### Verified

| | Blackhole | T3K |
|---|---|---|
| cores advancing | 120/120 | 64/64 |
| cores with new samples | 120 | 64 |
| `lost` | 0 | 0 |
| remote-chip header read | -- | **59-69 ms** (was 2512 ms in v1) |

The read-cost collapse independently confirms 5m: the old figure was a retry storm
against a 1 kHz-republishing header, not transfer cost.

### NOT verified: the utilization numbers themselves

`max util 0.000` on both arches, because both were idle -- the FPU counters do not
advance, so `d(busy)` is legitimately zero. The arithmetic is exercised, the values are
not.

Validating against a real workload is blocked on the standalone launcher: the aggregator
is currently launched by a tt-metal gtest, which cannot coexist with Llama (CHIP_IN_USE).
That makes the ELF-as-build-artifact work (3.5, 5j) the gating item for M1 acceptance,
not an optimisation. With it, the launch is four plain L1 writes from a UMD-only process
-- and 5j already proved the last of those works cross-process.

## 5o. The wedge: a persistent eth kernel breaks UMD topology discovery -- FIXED 2026-08-30

Four times today, and twice needing a board reset, a process opening the device hung for
5-8 minutes in UMD topology discovery. Root cause found, and it was ours.

### Mechanism

`TopologyDiscovery::eth_heartbeat_running()` polls EVERY ethernet core, waiting for the
firmware heartbeat word to change. That word is incremented by the idle-erisc FIRMWARE.
Our aggregator occupies ERISC0 and never returns (3.5), so the firmware never runs and
the heartbeat freezes permanently. Discovery then waits out its timeouts on every core we
have claimed -- for EVERY subsequent device open, by ANY process: ours, tt-metal's,
tt-smi's.

This was not a test artifact. A persistent kernel on an idle eth core degraded the whole
machine, and it presented as an intermittent unkillable hang.

### Fix: the kernel maintains the heartbeat it displaced

Each sweep the aggregator writes `(0xABCD << 16) | (sweep_count & 0xFFFF)` to
`ETH_HEARTBEAT_ADDR` (0x1C on WH) -- the signature UMD checks, and a low half that
changes. Free: the loop already runs at ~1 kHz.

| | discovery time |
|---|---|
| clean baseline, nothing resident | 0.474 s |
| **aggregator resident, heartbeat maintained** | **0.475 s** |
| aggregator resident, pre-fix | 90 s timeout (5-8 min hangs observed) |

Wormhole only: Blackhole's address differs and its discovery skips the check
("Temporary - heartbeat check disabled for Blackhole").

### Defence: a discovery watchdog

UMD's own timeouts are compile-time (ETH_STARTUP 10 s/core, ARC_STARTUP 300 s), so a
degraded chip can hold a process for minutes with no output. The collector now bounds
discovery (default 60 s, `TTNVTOP_DISCOVERY_TIMEOUT_S`) and exits with a diagnostic
naming this exact cause. Verified: 20.046 s and a clear message, against an 8-minute
hang. The SIGTERM handler can also exit during discovery -- safe there specifically
because discovery is READ-ONLY, so there is no half-finished write to strand.

### RETRACTION

5m recorded "the collector must not be SIGKILLed... pkill -9 mid-transaction left UMD
wedging in wait_arc_core_start". **That explanation was wrong.** The final wedge involved
no kill at all -- a fresh process met eth cores whose heartbeats had been frozen since
the moment the aggregators started. The aggregators were the cause throughout. Clean
shutdown is still good hygiene, but it was never the mechanism.

### 3.5 CORRECTED

3.5 argued the persistent-kernel approach was safe because "IDLE_ETH appears nowhere in
impl/program/dispatch.cpp". That reasoning was right about DISPATCH and never considered
DISCOVERY, which is where the hazard actually lived. The general form:

**Displacing firmware means inheriting its contracts.** The heartbeat is the one we
tripped over; anything else the idle-erisc firmware maintains for the platform is a
candidate, and deserves an audit rather than waiting to be bitten.

### Two stale assumptions from the push era, also fixed

Removing the push (5k) left debris in code that looked untouched:

- the journal probe skipped REMOTE chips ("journals land on the MMIO side by design") --
  exactly backwards once the journal lives in the aggregator's own L1;
- the journal sat AFTER variable-size scratch, making discovery circular: its size
  depends on `num_cores`, which is a field inside it. It now sits at `base`, so a
  UMD-only reader finds it with one 64 B read at a fixed address.

### End to end, cross-process, on a remote chip

```
chip 4 eth (24,16)  src_chip=4 cores=64 capacity=64 head=415278 sweeps=6208
real 0m0.520s
```

An aggregator on remote chip 4, discovered and read by a process that never linked
tt-metal, in half a second, with discovery unharmed.

## 5p. Artifact launch: works on all 8 chips, but two real bugs and one regression -- 2026-08-30

`--launch-aggregator` starts the aggregator from an emitted artifact with four UMD
writes and no tt-metal. On the T3K it launched on **all 8 chips**, remote included:

```
chip 0-3 (mmio)   eth (24,16) 8x8 = 64 cores, journal 0xe200
chip 4-7 (remote) eth (24,16) 8x8 = 64 cores, journal 0xe200
launched 8 aggregator(s) — no tt-metal, no dispatch, no fabric.
```

That is the deployment model working: emit once with tt-metal, launch any time from a
process that cannot take CHIP_IN_USE.

### Bug 1 -- the launcher displaced an ACTIVE ethernet core (cost a board reset)

Core selection took "the first core that reads successfully". Every core reads
successfully. So it claimed an ACTIVE link-carrying core on all four MMIO chips,
overwrote ERISC0, killed the firmware servicing the NON_MMIO tunnel, and wedged
discovery machine-wide.

**Readable is not the same predicate as unused.** The gtest never hit this because it
asks the control plane via `get_inactive_ethernet_cores()`; the substitute predicate
written for the collector was wrong in the most damaging way available. Fixed: exclude
`cluster_desc->get_active_eth_channels()`, prefer never-routed channels (2), and REFUSE
to launch rather than displace a live link.

### Bug 2 -- the artifact image overlapped the journal

`image_bytes` was sized `max(64 KB, ...)`, a number invented with no reference to the
device. Measured on WH:

```
image spans 0x7df0..0x17e70   journal at 0xe200   -> OVERLAP
```

So replaying the image wrote 65 KB through the journal and every scratch array. Fixed by
bounding at `eth_l1_unreserved`: 65664 B -> 23056 B, stopping exactly at the journal.

Adjacent, found by the same arithmetic: **nothing checked that the runtime args fit**
between `rta_offset` and `kernel_text_offset`. They are 31 words = 124 B against a 128 B
gap on WH -- one argument from silently overwriting the kernel. Now asserted at emit.

### REGRESSION -- the heartbeat write makes the aggregator hostile to tt-metal startup

With aggregators resident, Llama would not start AT ALL:

```
RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID: 14521831458,
ETH core e8-0 (NOC0) to advance. Stuck at 0xabcd5904
```

`0xabcd5904` is OUR value -- correct signature, counter 0x5904 = 22788 sweeps ~= 23 s of
running. So the aggregator maintained the heartbeat for 23 s and then stopped.

This is WORSE than the pre-fix behaviour. A frozen heartbeat with no valid signature made
discovery slow; a frozen heartbeat WITH a valid signature is a hard error, and tt-metal
refuses to start.

**Why it stopped at 23 s is NOT established.** Two candidates, undistinguished:
- the kernel died or hung after sustained running -- nothing in the 5 s tests would catch
  it, and bugs 1 and 2 above were both live during that run;
- something reset the core, freezing the last written value.

**Recommendation: do not ship the heartbeat write until this is understood.** It should
be re-tested on Blackhole (where a wedge is free) with the `dbg` markers sampled over
minutes to see whether the kernel is alive. If the kernel is dying, the heartbeat is not
the bug -- it is the symptom, and the aggregator has a stability problem that matters far
more.

## 5q. Requirement: survive BOTH before and during a workload -- NOT MET -- 2026-08-30

### What works

**Attach DURING a workload: yes.** Aggregators attached while Llama-3.3-70B was decoding
left it at 96.72 ms / 10.34 tok/s against a 10.35 control -- indistinguishable.

**The kernel is stable.** A 90 s soak on Blackhole: 112,039 sweeps, perfectly linear,
`lost=0`, no stall. The "dies at 23 s" theory is WRONG -- there is no inherent instability.

**The tt-metal launch path works on the T3K**: 64 cores, head advancing, `lost=0`.

### What does not work

**The ARTIFACT REPLAY path does not start the kernel.** All 8 chips "launch" without
error and with a correctly-bounded image, but no journal ever gets its magic:

```
launched 8 aggregator(s) — no tt-metal, no dispatch, no fabric.
t+10s: 0 chips reporting
t+30s: 0 chips reporting
t+60s: 0 chips reporting
```

The same kernel launched through tt-metal on the same board works immediately. So the
defect is in emit/replay, not in the kernel, and not in the four-write launch contract
(5j proved the go-word handoff works when tt-metal staged the binary on that core).

The difference is that the artifact snapshots one core's kernel-config region and
replays it onto a DIFFERENT core, possibly on a different chip. Something in that is
insufficient -- state outside the snapshotted range, or a field in the launch message
that is core-specific. NOT diagnosed.

### Why this blocks the requirement

Launching BEFORE a workload needs the aggregator resident and its heartbeat advancing
when the workload opens the device, so discovery passes. The chain then works:
discovery passes -> firmware init resets our core -> `initialize_firmware(IDLE_ETH)`
reloads the eth firmware so the heartbeat resumes -> a supervisor relaunches us.

But the only launcher that can run without tt-metal is the artifact path, and it does not
start the kernel. The tt-metal path cannot be used before a workload because two tt-metal
processes cannot share a device (CHIP_IN_USE).

So: **during = works, before = blocked on the artifact replay defect.**

### The 23-second freeze, revisited

`0xabcd5904` on the T3K was NOT the kernel dying of its own accord -- the soak disproves
that. During that run the artifact image was overlapping the journal and scratch (5p bug
2) and active eth cores had been clobbered (5p bug 1). Either explains it. With both
fixed the kernel does not freeze; it simply never starts from the artifact.

### Next

1. **Diagnose the artifact replay.** Compare, byte for byte, the kernel-config region
   after a working tt-metal launch against the same region after an artifact replay on
   the same core. The diff is the answer, and it is a cheap experiment.
2. Then the supervisor: watch `sweep_count` per chip and relaunch when it stops.
   `assert_inactive_ethernet_cores()` means NO kernel on an idle eth core survives a
   device init, by design -- so "survives" has to be a property of the system, not of
   the kernel.

## 5r. Artifact replay DIAGNOSED -- the artifact was never the problem -- 2026-08-30

The byte-for-byte diff 5q asked for, run on Blackhole (desktop-0, p150a, 120 cores) where
a wedged eth core costs nothing. Two processes, the SAME eth core, because
`stop_aggregator` asserts the RISC reset and leaves the core with no firmware until the
next device init -- so a same-process second launch could not start whatever the bytes
said.

    run 1  TTNVTOP_DUMP_DIR=<d> --gtest_filter=*EthAggregatorGoldDump*    emit + LaunchProgram
    run 2  TTNVTOP_DUMP_DIR=<d> --gtest_filter=*EthAggregatorReplayDiff*  replay the artifact

Both dump the kernel-config region and the whole mailbox block (MAILBOX up to
kernel_config_base: the launch ring, both go-message arrays, `launch_msg_rd_ptr` and the
go index -- every field `idle_erisc.cc` consults before it jumps).

### The diff

```
gold:   marker=0xa66e0000 sweeps=2494 head=318647
replay: marker=0xa66e0000 sweeps=2495 head=318649
diff kernel-config: 0 of 23056 bytes differ
  mailbox differs at +0x40 (L1 0x140) len 1: gold word 0x00000001 replay 0x00000000
diff mailbox: 1 of 64560 bytes differ
REPLAY VERDICT: running=yes cfg_diff=0 mbox_diff=1
```

**The replay started the kernel, and 23056 of 23056 kernel-config bytes were identical.**
The single differing byte is `launch[0].kernel_config.mode` at `launch_addr + 0x30`: gold
`DISPATCH_MODE_HOST` (1), replay `DISPATCH_MODE_DEV` (0). `llrt::write_launch_msg_to_core`
sets that field at write time, and the emitter snapshots `kg->launch_msg` before ever
calling it. It is harmless: `idle_erisc.cc` reads `kernel_config.mode` only in the
post-kernel block, and this kernel never returns.

So the artifact content is correct. **The defect was never in emit or replay.**

### What it actually was

`idle_erisc.cc`'s wait loop is the ONLY thing anywhere that polls
`go_messages[0].signal` for `RUN_MSG_GO`. That firmware gets onto an inactive ethernet
core from exactly one place: a tt-metal device init
(`risc_firmware_initializer.cpp` writes the IDLE_ETH launch message and a `RUN_MSG_INIT`
go word). A bare board runs the base eth firmware, which knows nothing about launch
messages.

A/B on the same board and the same core, toggling only whether a tt-metal process holds
the device open (`TestEthAggregatorHoldDevice`, which opens the devices and sleeps):

    tt-metal process holding the device   RUNNING     (sweeps 624 -> 1232, magic republished)
    no tt-metal process                   NOT RUNNING (magic 0x0, sweeps 0 -> 0)

The four writes land in both arms. In arm B nothing executes them, because nothing is
polling.

### Consequence: the operating model inverts, and it is FINE

5q had it backwards. The artifact launcher was built to run BEFORE the workload, and that
is the one thing it structurally cannot do -- not for want of a missing write, but because
before the workload there is no firmware on the core to receive the go signal. The
sequence that works is the other order:

    workload starts -> its tt-metal device init loads idle-erisc firmware on every
    inactive eth core -> the UMD-only collector replays the artifact -> kernel starts

which is exactly what a monitor wants, and what 5q already measured as free (attach
during Llama-3.3-70B: 10.34 vs a 10.35 control). Two tt-metal processes cannot share a
device, but a UMD-only collector and a tt-metal workload can -- that is the whole point of
the artifact.

The residual limitation is narrow and worth stating plainly: **the aggregator cannot be
pre-staged on an idle board, and any subsequent device init kills it** (measured -- arm
A's aggregator was gone once the holding process closed the device). Liveness is a
supervisor's job, not the kernel's, exactly as 5q's item 2 said.

### Two reporting bugs fixed on the way

1. **`--launch-aggregator` reported success it had not checked.** Four writes returning
   meant "launched". That is what let this defect stand for a day across all 8 chips. It
   now reads the journal header twice and reports `RUNNING (sweeps a -> b)` or
   `NOT RUNNING`, and exits non-zero if nothing started.
2. **The stale-journal trap bit again, and then a second, opposite way.** A journal from
   an earlier aggregator keeps a valid magic and a valid header checksum forever with its
   count simply frozen -- the first version of the verify above read `2512 -> 2512` and
   correctly said NOT RUNNING. But a RESTARTED aggregator counts from ZERO, so the
   working arm read `2512 -> 624` and was reported NOT RUNNING too. "Did the count go up"
   is wrong in BOTH directions unless the baseline is known. The launcher now zeroes the
   journal header before the go word, which is the only baseline that cannot lie.

### Also found, not fixed

- **`send_reset_go_signal` is a no-op on IDLE_ETH.** 5p added it as "a real gap". It is
  not one: `idle_erisc.cc` has no `RUN_MSG_RESET_READ_PTR_FROM_HOST` branch at all --
  unlike `brisc.cc`, `active_erisc.cc` and `dm.cc`, which all handle it. Its wait loop
  tests only `signal != RUN_MSG_GO`. Kept in the artifact because LaunchProgram sends it
  and byte-identity is the point, but it never reset any read pointer.
- **`LaunchProgram` passes an address where `write_launch_msg_to_core` takes a `bool`.**
  `tt_metal.cpp:1010` passes `hal.get_dev_addr(programmable_core_type, LAUNCH)` as the
  5th argument, whose parameter is `bool send_go = true`. It is always nonzero so it
  always means `true`, which is the intended behaviour -- but the function derives the
  launch address itself and ignores what was passed. Latent, upstream, unrelated to this
  defect.

## 5s. FIDELITY MEASURED -- the surviving justification HOLDS -- 2026-08-30

The one measurement this feature actually rests on. The original justification --
"monitoring stalls workloads" -- did not reproduce (5l/5q: three Llama-70B arms spanned
0.5% with the ordering backwards), so fidelity was the only argument left, and it was
unmeasured.

homelab-1 T3K, 8 chips, 2x4 mesh, 2048^2 bf16 matmul, 60 s window,
`scripts/fidelity_ab.sh`. **Both arms run CONCURRENTLY against one workload.** Both read
the same 62-entry per-core rings and neither consumes, so this removes "were the two arms
even producing the same samples" as a variable -- which separate runs cannot, since
production is driven by the workload's kernel-launch rate.

### Aggregator arm

```
chip  loc     cores  sweeps      folded      lost   loss%   samples/s   torn  fail
   0  mmio       64     32320     7886899         0    0.00      131434      0     0
   1  mmio       64     46336     8236042         0    0.00      137253      0     0
   2  mmio       64     46896     8294957         0    0.00      138234      0     0
   3  mmio       64     46912     8274535         0    0.00      137894      0     0
   4  remote     64     47712     8364861         0    0.00      139399      0     0
   5  remote     64     48416     8401634         0    0.00      140012      0     0
   6  remote     64     47408     8349721         0    0.00      139147      0     0
TOTAL folded=57808649 lost=0 loss=0.00%  aggregate 963374 samples/s
```

### Host per-core drain arm, same window

```
chip     entries       lost   loss%   entries/s
   0     4079104    3780578   48.10       65075
   1     4079104    4169238   50.55       65075
   2     4079104    4223857   50.87       65075
   3     4079104    4205280   50.76       65075
   4     4079104    4287972   51.25       65075
   5     4079104    4324129   51.46       65075
   6     4079104    4269851   51.14       65075
   7     4079104    4281665   51.21       65075
TOTAL entries=32632832 lost=33542570 loss=50.69%
```

**Host loses 50.7% of samples. The on-chip sweep loses 0.00%.** Replicated in an earlier
run the same day (`runs/fidelity-20260830-115656`, 6 chips): host 54.70%, aggregator 0.00%.

### Why these numbers are trustworthy, and not just two tools disagreeing

1. **`entries` is bit-identical across all eight chips: 4,079,104 = 1028 ticks x 64 cores
   x 62 slots.** Every core on every chip hit the ring-capacity clamp on every single
   tick. The host drain is not sampling the rings, it is **saturated** -- it only ever
   gets the most recent 62 entries per core per pass, and everything else is gone. That
   the number is exactly `ticks x cores x ring_size` is what makes this a structural
   ceiling rather than a tuning problem.

2. **The two arms independently agree on the PRODUCTION rate to within 0.4%.** Sum what
   each arm accounts for:

        chip 4   host (4079104 + 4287972)/60 = 139451/s    aggregator 139414/s   0.03% apart
        chip 0   host (4079104 + 3780578)/60 = 130995/s     aggregator 131448/s   0.35% apart

   Two completely different mechanisms -- 64 PCIe/tunnel reads per tick vs a NOC sweep on
   an idle eth core -- measuring the same underlying stream and landing on the same total.
   One keeps 100% of it, the other 49%.

3. **The tunnel is irrelevant to the aggregator.** Remote chips 4-6 folded slightly MORE
   than the MMIO chips (139k vs 131-138k samples/s) at `lost=0`. The sweep never crosses
   ethernet, so "remote" costs nothing -- which is the entire architectural claim.

### What the loss actually costs

Not just resolution. The host drain's attribution model is **fires x period**
(`collector/main.cpp`: each observed ring entry contributes `hdr->period_cycles` to its
`kernel_id`). A lost sample is therefore a lost *fire*, i.e. lost attributed cycles,
roughly proportionally. 50.7% sample loss means per-kernel TIME% is under-attributed by
about half -- not merely coarser. That is the difference between a monitor you can quote
and one you cannot.

### VERDICT

**Fidelity holds. Build 2.2.b.** The host-pull design has a hard structural ceiling at
~65k entries/s/chip (1028 drain passes x 62 slots x 64 cores in 60 s) against a ~139k
samples/s/chip producer, and no amount of drain tuning closes a 2x gap that is set by ring
capacity. The on-chip sweep runs at ~790 sweeps/s and loses nothing, on local and remote
chips alike, for a fixed 2 KB read per chip per tick.

Note this does NOT revive 2.2.a (fabric transport): the journal is read where it lies and
the tunnel is only ever asked for 2 KB.

### Three defects this run exposed

1. **`[ring-drain]`'s 5 s throttle was GLOBAL to the drain thread, not per chip.**
   `last_debug_us` / `last_debug_ticks` were `static thread_local` locals inside the
   per-chip loop, so whichever chip the loop reached first after the interval elapsed
   logged and reset the timer, silently suppressing the other seven. The first T3K run
   printed 15 lines for chip 0 and one each for chips 4, 5 and 7 over 75 s -- which reads
   exactly like the drain starving 7 of 8 chips, and is nothing of the kind. It also made
   `drain_hz` meaningless (tick baseline taken from whichever chip logged last). Moved
   into `ChipState`. **Without this fix the host arm could only be measured on one chip.**

2. **Remote header writes are UNORDERED, and the launcher's zeroing raced the kernel.**
   The launcher zeroes the journal header before the go word so a stale journal cannot
   read as live. On chip 7 the zeroing landed *after* the kernel had already stamped the
   magic, leaving a live aggregator publishing an advancing `sweep_count` under
   `magic 0x0` -- invisible to `probe_landings` (which gates on the magic), so the
   launcher reported `NOT RUNNING (magic 0x0, sweeps 352 -> 736)` (note: sweeps
   *advancing*, which is the tell) and `--stop-aggregator` could not reach it either. An
   unstoppable aggregator on a core nobody could see. Fixed by republishing the magic in
   the publish block instead of once at startup, which makes the journal self-healing
   against any late or reordered header write.

3. **Remote aggregator launch is still flaky: 2 of 4 remote chips failed in the first run,
   1 of 4 in the second.** Unchanged cause (5h): the collector cannot pin the tunnel to
   the channels that reach the target chip, because
   `configure_active_ethernet_cores_for_mmio_device` is a `umd::Cluster` method and a
   monitoring process deliberately never constructs a Cluster. Open question 7.3.

### And one operational rule, learned the expensive way

**Never SIGKILL a tt-metal process that has initialised fabric.** It leaves the fabric
ERISC firmware stopped mid-loop on an ACTIVE ethernet core, whose heartbeat word then
holds `FABRIC_HEARTBEAT_SIGNATURE` (0xAABB) with a frozen counter. UMD's
`eth_heartbeat_running` throws on a valid-but-frozen signature (an *invalid* signature
merely returns false with a warning), so the next tt-metal device open dies outright:

    RuntimeError: Timed out waiting for ETH heartbeat on ... ETH core e8-0 ...
                  Stuck at 0xaabb2d45

Recovered with `python3 -m tt_smi -r all` (~40 s, discovery back to 0.505 s). Note the
signature: **0xAABB is fabric's, not ours** -- this failure has nothing to do with the
aggregator and would happen to any hard-killed fabric workload. `ab_workload.py` gained
`--seconds` so the workload is time-bounded and exits through its own `finally`, and
`fidelity_ab.sh`'s teardown now waits 60 s for a clean exit and **refuses to escalate to
SIGKILL**, reporting instead.

## 5t. The 50.7% was a SINGLE-THREADED DRAIN. Scope collapses to remote chips -- 2026-08-30

This section RETRACTS the central claim of 5s. 5s reported a "structural ceiling ... no
amount of drain tuning closes a 2x gap that is set by ring capacity". That was a
measurement of one implementation, not a property of host-pull.

### What was actually wrong

`ring_drain` was a SINGLE thread running `for (auto& chip : chips)` with 64 serial 1 KiB
reads inside each -- 512 sequential reads per pass on a T3K. It asks for
`kRingDrainHz` = 200 and achieved 17.1. The disproving experiment needed no code change
and I should have run it first: restrict the same single-threaded drain to ONE chip.

    host drain, 8 chips, 1 thread    50.8% loss
    host drain, 1 chip,  1 thread     0.00% loss      <- unchanged binary, unchanged workload

### One drain thread per chip

Measured twice on the T3K, 2x4 mesh, 2048^2 matmul, 60 s and 45 s windows:

    MMIO chips 0-3      0.00% loss, both runs
    remote chips 4-7    10.0 / 16.6 / 16.7 / 18.7 / 19.3 / 20.7 / 30.9 / 33.2%
    overall             9.7% and 11.3%   (was 50.7%)

**On MMIO chips host-pull is sufficient and the on-chip aggregator earns nothing.** The
feature's scope, if it ships at all, is remote chips on multi-chip Wormhole systems.

### Why remote still loses -- measured, not inferred

A probe that times the drain's exact shape (1 KiB from each of 64 different Tensix cores),
on a remote chip, idle:

    drain pattern (64 cores x 1KiB): 4.94 ms/pass -> 202.6 passes/s
    same-core x64: 4.98 ms  |  one 64KiB call: 4.45 ms  |  per-call overhead: 8.47 us

Three results, two of which killed hypotheses that were about to become work:

- **Per-call overhead is 8.47 us** -- 0.53 ms of a 4.94 ms pass. UMD's own comment on the
  interprocess NON_MMIO mutex warns that fine-granularity locking "would be more
  detrimental to performance than acquiring it for a large block", and the drain does
  exactly that 64 times per pass. Batching under one acquisition would buy ~10%. Not it.
- **Target switching is free**: 64 different cores costs the same as 64 reads of one core.
- **A remote chip sustains 202 passes/s idle and needs only 52** -- 4x headroom. Under load
  it collapses to 34.7-46.7. So remote loss is neither bandwidth, nor per-call cost, nor
  the mutex in isolation: it is **contention with the workload's own tunnel traffic**,
  since on a T3K dispatch to the remote chips is itself tunneled.

That is [[tt-coremon-n300-remote-contention]] read the other way round. The plan was built
on "monitoring stalls workloads", which did not reproduce (5l). The real effect is
**workloads stall monitoring** -- and an on-chip sweep is immune to it because it never
crosses ethernet. That is the honest justification for the feature, and it is narrower and
better founded than anything claimed before it.

### The cheap alternative, and why it is BLOCKED

With 4x idle headroom the drain is contention-bound, so ring capacity trades directly
against the shortfall (earlier arithmetic had dismissed this on a bandwidth-bound
assumption):

    ring   needed passes/s   margin vs worst remote chip (34.7)
      62            51.9     0.67x   <- current, lossy
     126            25.5     1.36x
     254            12.7     2.74x

126 slots (`MEM_UTIL_SAMPLER_SIZE` 1024 -> 2048) should make the host drain lossless on
remote chips and remove the need for an aggregator entirely. It does not work:

**Raising `MEM_UTIL_SAMPLER_SIZE` to 2048 builds cleanly and then HANGS Tensix firmware
init, reproducibly.** Every core times out in `waiting for physical cores to finish` ->
`Device 0 init: failed to initialize FW`. Established by bisection against a stashed tree:

    ring 62,  precompiled fw   PASS
    ring 62,  JIT fw           PASS
    ring 126, precompiled fw   PASS   (stale fw -- host/device disagree, not a valid test)
    ring 126, JIT fw           FAIL   (twice, second failing 12 s after the first)

Not a compile-time timeout (the second attempt is immediate), not a link or size error
(the build is silent). `MEM_MAP_END` is derived from this reservation and is the Tensix
**KERNEL_CONFIG base**, so growing it shifts that base and every
`*_INIT_LOCAL_L1_BASE_SCRATCH` above it. Which of those movements breaks init is NOT
diagnosed. Reverted; the constant now carries a warning comment.

**DO NOT PURSUE THIS. Corrected below -- an earlier revision of this section called it "the
highest-value open item", which was wrong on footprint grounds.**

### Why growing the ring is the WORST option, not the cheapest

The ring lives in **Tensix** L1 (`wh_hal_tensix.cpp` registers `UTIL_SAMPLER`; the writers
are `brisc.cc`'s `init()`, `trisc.cc`'s `force_kernel_start_sample()` and the LLK math
hooks). The ethernet core is only ever a CONSUMER. So the ring size is the one lever that
reaches back into the address space every workload runs in:

    Tensix L1                1464 KB
    reserved (MEM_MAP_END)   35.2 KB
    sampler ring              1024 B  = 0.068% of L1, 2.8% of the reserved region

Nothing overlaps or corrupts -- the ring is below `MEM_MAP_END` and workload buffers are
allocated from `UNRESERVED` up, with `MEM_MAP_END` as the watcher's enforced boundary. But:

1. **It is a permanent global tax.** 1 KiB (or 2) of EVERY Tensix core on EVERY chip,
   reserved whether or not anyone is monitoring. The aggregator is 2 KB on ONE idle eth
   core, only while monitoring. On footprint the ring is strictly MORE invasive.
2. **`MEM_MAP_END` is an ABI boundary**, not merely a size: the Tensix `KERNEL_CONFIG` base
   and the base of every `*_INIT_LOCAL_L1_BASE_SCRATCH`. Moving it invalidates precompiled
   firmware -- whose `build_key` does not hash header contents -- and anything with baked
   L1 addresses. A fleet-wide compatibility event for an optional monitor.
3. **L1-tight ops.** tt-metal kernels are tuned against available L1 to the byte; a hard
   boundary moving by 1 KiB can push a tuned config over.

Read that way, the init hang is plausibly the platform saying this boundary is not meant to
move, rather than a bug to be fixed.

### Corrected ranking for the remote-chip gap

    option                              permanent device footprint        while monitoring
    parallel host drain (DONE)          none                              host threads only
    eth aggregator, remote chips only   none                              2 KB + 1 idle eth core
    bigger ring                         1 KiB x every Tensix core, ALWAYS  --
                                        + moves an ABI boundary

So the aggregator is the option with ZERO permanent footprint, scoped to the only case that
still needs it. Its hazards are real but bounded, Wormhole-specific, and the mitigations are
known (invalid-signature exit so a dead kernel cannot hard-block a device open, plus a
host-side watchdog on a stalled `sweep_count`).

Note the producer side is committed either way: the 1 KiB ring exists today and the whole
telemetry feature rests on it. This was only ever a CONSUMER-side decision -- which is
exactly why the parallel drain was a pure host change with no device footprint at all.

Note also that the precompiled-firmware `build_key` is derived from build options, NOT from
header contents, so changing a layout header silently reuses stale firmware. Any future
layout experiment must force `TT_METAL_DISABLE_PRECOMPILED_FW=1` or regenerate the bundle.

### Corrections to earlier sections

- **5s's cross-arm agreement claim is withdrawn.** It compared the aggregator's per-core
  `samples` against the host's raw entry count. Those differ: `samples` counts ACCEPTED
  deltas, so a re-armed FPU counter lands in `resets` and an implausible wall delta is
  dropped silently. The journal header's `head` (`head += behind`) is the raw count and the
  only comparable quantity; the probe now reports entries/accepted/resets separately.
  Resets turn out to be negligible (1,280 against 5.5M), so the residual cross-arm gap is
  **window misalignment** -- the host arm's window comes from `[ring-drain]` lines
  quantised to 5 s, making a nominal 45 s window 45 +/- 10 s. Per-arm loss percentages are
  sound; absolute cross-arm counts must not be quoted until the windows are timestamped.
- **[[tt-umd-tunnel-channel-pinning]]'s "the collector cannot do this itself" is wrong.**
  `Cluster::configure_active_ethernet_cores_for_mmio_device` is only a wrapper;
  `TTDevice::get_remote_communication()` and
  `RemoteCommunication::set_remote_transfer_ethernet_cores()` are public and reachable from
  a bare TTDevice, and the call site matches `RemoteChip`'s line for line. Implemented as
  `--pin-tunnel`. Measured effect is MIXED -- it zeroed one chip's drain and cost one
  chip's aggregator launch, overall 12.9% vs 11.3% unpinned -- so it is off by default. The
  Cluster also sets the MMIO chip's OWN RemoteCommunication, which this does not; that is
  the likely gap.
- **Two more shared-state defects of the same family as the log throttle**: `static` locals
  `last_period_assert_us` and `last_period_probe_us` inside the drain lambda became shared
  once it ran per-chip, so one chip of eight would have had its sampler period reasserted.
  Moved into `ChipState`.

### Unrelated finding worth keeping

UMD logs `Large transfer to remote chip without system memory setup` for anything above
`256 * 4` bytes when `SysmemManager` is null, and caps `MAX_BLOCK_SIZE` at 1024 B instead
of `ETH_ROUTING_BLOCK_SIZE` (32 KiB). The drain reads exactly 1024 B so it is unaffected,
but any bulk remote read from a monitoring process is 32x under-blocked.

### Where this leaves the feature

1. MMIO chips: **not justified**. Host-pull with a parallel drain is lossless.
2. Remote chips: justified *only* by contention immunity, at 10-33% host loss.
3. If the `MEM_UTIL_SAMPLER_SIZE` init hang is fixed, 126 slots very likely removes (2) as
   well, and with it the entire feature.

So the ordering is:

1. **Parallel drain: done.** MMIO chips are solved with zero device footprint.
2. **Decide whether 10-33% loss on remote chips is acceptable.** That is a product call, not
   an engineering one, and it should be made before any more device work.
3. If it is not acceptable, **the eth aggregator scoped to remote chips only** is the right
   mechanism -- smallest permanent footprint (none), and the hazard mitigations are known.
   Finish the pinning fix (set the MMIO chip's own RemoteCommunication too) first, since it
   attacks the contention directly and costs nothing on-device.
4. **Do not grow the sampler ring.** Not because it hangs, but because it taxes every
   workload permanently and moves an ABI boundary.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Dispatch owns idle eth cores — but **only under ETH dispatch**. `dispatch_core_manager.cpp:268` adds *every* inactive eth core to the pool, guarded by `resolve_dispatch_core_type(...) == CoreType::ETH`. Default is WORKER, where idle eth is untouched. ETH dispatch is not exotic: 8-chip WH needs it for 2 CQs | `host/agg_core_select.hpp` refuses to start under ETH dispatch (3.4). There is no claim API on WH — `ServiceCoreManager::claim()` is Blackhole/UBB-Galaxy only. Lifting this needs the RT-profiler's reserve-from-the-back pattern upstream (7.6) |
| `assert_inactive_ethernet_cores()` resets `RiscType::ALL` on idle eth cores — called unconditionally from `assert_cores()` and from init under `INIT_FABRIC` | Aggregator lives for ONE device-open epoch. Launch after init; host detects the reset via a stalled `sweep_count` and re-attaches. Do not assume permanence |
| Kernel lifetime — aggregator must be persistent across program dispatch | **RESOLVED (3.5).** `IDLE_ETH` appears nowhere in `impl/program/dispatch.cpp`; a kernel that never returns owns the core by construction. No mechanism needed |
| Watcher / inspector walk inactive eth cores (`watcher_server.cpp:523`, `watcher_device_reader.cpp:430`, `inspector/data.cpp:383`) | Verify they tolerate a non-dispatch kernel; may need an exclusion |
| Aggregator NOC traffic shares the chip NOC with the workload | ~2.6 MB/s at 10 kHz sweep; measure in run C, tune sweep rate |
| WH-only — BH eth differs (2 ERISCs, different L1 map) **and BH harvests eth/PCIe** (2.1): 2 of 14 eth channels always fused off, and exactly one of the two PCIe tiles is always harvested with the survivor varying | Scope to WH. BH p150a has no remote chips; 6U BH is all-MMIO. If a BH topology with remote chips ships: derive the landing tile from `get_cores(CoreType::PCIE)` (never constant `(2,0)`) and compute the spare-eth budget rather than asserting it |
| Tensix row harvesting differs per chip within one system, so a NOC0 core walk needs a per-chip table | Gatherer walks TRANSLATED space, where WH compacts live rows to a fixed contiguous range (2.1b); fall back to an explicit list if `noc_translation_enabled` is false |
| `RiscType::ERISC0`/`ERISC1` alias `BRISC`/`TRISC0` in UMD (`risc_type.hpp`), so reset code copy-pasted to a Tensix core silently means something else | The launch-message path (3.5) needs no reset call at all; if one is ever added, use `RiscType::ALL` |
| A dead fabric client never releases its EDM connection — the aggregator cannot call `sender.close()` because it never returns, so every exit is a dirty one. The next client on that `link_idx` starves (5f: 5 sweeps vs 73,284) | One aggregator per chip on its own `link_idx`. Raise with fabric owners (7.4). The link recovers at fabric re-init |
| Launching onto a core that already runs an aggregator corrupts the live kernel and starts nothing (5f finding 2) | `rank_aggregator_eth_cores()` gives independent callers distinct cores; `stop_aggregator()` before any relaunch; never launch blind |
| Aggregator dies silently → stale data read as live | Host checks `sweep_count` advances; falls back to per-core drain if stalled |

## 7. Open questions

1. ~~**Persistent idle-eth kernel lifetime.**~~ **RESOLVED 2026-08-29 (3.5).** A kernel
   that never returns owns the idle eth core by construction: the firmware regains
   control only on return, and `IDLE_ETH` appears nowhere in `impl/program/dispatch.cpp`.
   No dispatch-persistence mechanism is needed. The real lifetime bound is device init --
   `assert_inactive_ethernet_cores()` resets `RiscType::ALL` -- so the aggregator lives
   for one device-open epoch and the host must detect the reset and re-attach.
2. ~~**Does the collector need tt-metal?**~~ **RESOLVED 2026-08-29 (3.5): no.**
   `llrt::get_risc_binary()` is a pure ELF parse with no cluster dependency, the span
   writes go through UMD, and the dev_msgs `launch_msg_t` layout is already generated
   into `tt_metal/hw/inc`. The collector stays standalone. The remaining work is a build
   task: emit the aggregator ERISC ELF as a fixed artifact instead of JIT-compiling it.
3. ~~**Should this land with `configure_active_ethernet_cores_for_mmio_device()`?**~~
   **ANSWERED 2026-08-29 (5h): YES, and it is not a smaller win -- it is the fix for
   7.7.** Restricting the tunnel to the channel pair that actually links the MMIO chip
   to its remote chip took remote launch from 1/6 to 14/14. It applies to every
   remote-chip operation on a T3K, not just this feature, and is very likely a
   contributor to 5c. Recommend tt-metal call it at device init.
4. **PARTLY ANSWERED 2026-08-29 (5j): a separate PID CAN start the aggregator and join
   a live EDM** -- demonstrated with a UMD-only process writing a 4-byte go word. The
   connection args are derived, not allocated, so no coordination is needed. What
   remains is CONTENTION: `sender_channel = 0` is the only local-worker channel per EDM
   and holds one worker at a time, so the aggregator needs a `link_idx` the workload is
   not using. **For the fabric owners: can a low-rate telemetry client share an EDM
   sender channel, or can one be reserved?** The leave-cleanly half is unchanged -- Sharpened by 5f finding 3: a client that
   dies without `sender.close()` does not release its connection, and the next client
   on that `link_idx` starves (measured 5 sweeps vs 73,284). Since the aggregator's
   kernel never returns, it can never call `close()` -- so this is its normal exit
   path, not an edge case.
   *For the fabric team.* **Blocks M3 (mid-workload attach) and nothing else** -- M1/M2
   launch in the process that initialised fabric. The free eth channels have no peer
   (5d), so fabric is the only way off a remote chip, and the workload owns the EDM.
5. **Can `EDM_NOC_VC` be made configurable, or use VC 1 for PCIe destinations?**
   *For the fabric team.* Would unblock pushing straight to host memory (5d). Not on
   the critical path -- the MMIO-chip-L1 landing spot removes the tunnel regardless.
6. **Reserve an idle eth core from the back of the FD dispatch pool.** *For the
   dispatch owners.* Needed only under ETH dispatch, where the pool swallows every
   inactive eth core and no claim API exists on WH. The real-time profiler already
   does exactly this for a tensix, in the same function
   (`logical_dispatch_cores.back(); pop_back();` into
   `reserved_realtime_profiler_core_by_device_`), so the shape is established. Until
   then the aggregator refuses to start under ETH dispatch (3.4). **Blocks 2-CQ T3K
   only** -- every WORKER-dispatch configuration is unaffected.
7. ~~**Why does launching a kernel onto a remote chip wedge the NON_MMIO tunnel?**~~
   **RESOLVED 2026-08-29 (5h).** UMD defaults the tunnel to every active eth channel on
   the MMIO chip -- six on a T3K, four of which link to other boards entirely and are
   contended by fabric -- and `wait_for_non_mmio_flush` waits for all of them. Pinning
   to the link pair that reaches the target remote chip: 1/6 -> 14/14. The remaining
   work is upstream (see 7.3), not investigation.

## 5u. Coexistence measured: BOTH workloads keep full speed, but SHUTDOWN live-locks the tunnel -- 2026-08-31

The two arms the acquisition fix had never been tested against. Both ran with aggregators
on all 8 chips of the T3K, monitor in a separate process from the workload.

### Test A -- heavy CCL over tt-fabric + monitor: PASS

    up: 8/8   skipped: 0
    journal-fed: 8/8   ring-drain lines: 0
    workload A: {"label": "ccl-mon", "ok": true, "ccl_per_iter": 4, "num_devices": 8,
                 "elapsed_s": 420.04, "iters": 475800, "per_iter_ms": 0.8828}
    fatals A: 0        stopped: 8/8

~1.9M fabric collectives in 7 min. 0.8828 ms/iter vs 0.858 unmonitored = 2.9% slower.
This is the arm that used to HANG. It now also stops clean, 8/8.

### Test B -- Llama-3.3-70B + monitor: workload PASS, monitor SHUTDOWN FAIL

    aggregators up: 8/8   core-skips: 4
    journal-fed: 8/8   ring-drain lines: 0
    Average speed: 96.72ms @ 10.34 tok/s/user
    =========== 1 passed, 45 deselected, 4 warnings in 643.78s (0:10:43) ===========

10.34 tok/s against a 10.35 unmonitored baseline -- 0.1%, inside noise. `core-skips: 4`
is the three-condition pre-flight refusing four busy cores BEFORE writing a byte, which
is the behaviour whose absence corrupted fast dispatch in 5q.

So steady-state coexistence holds for both fabric-heavy and inference workloads.
The failure is entirely in teardown.

### The shutdown live-lock

On `kill -INT` the collector did not exit. Llama's teardown then blocked:

    UMD | Waiting for lock 'NON_MMIO_2_PCIe' which is currently held by thread
           TID: 107452, PID: 107327 (robust_mutex.cpp)
    UMD | Waiting for lock 'NON_MMIO_3_PCIe' which is currently held by thread
           TID: 107446, PID: 107327

PID 107327 is the collector. Per-thread state, 9 min after the SIGINT (utime in ticks):

    tid=107327 utime=625   wchan=futex_wait_queue
    tid=107446 utime=59367 wchan=0
    tid=107452 utime=57383 wchan=0
    tid=107453 utime=975   wchan=futex_wait_queue

`wchan=0` with ~590 s of CPU each: 107446 and 107452 are SPINNING at 100%, not blocked --
and they are exactly the two TIDs UMD names as holding the two remote-tunnel locks. This
is a live-lock, not a deadlock, and the collector's own log stopped advancing 527 s
earlier (123 lines, unchanged over a 6 s recheck), so the spin makes no progress either.

Mechanism: the journal feed issues `read_from_device(..., landing_base + 64, 8192)` per
chip per tick. On a NON_MMIO chip that 8 KiB becomes many tunnel transactions under the
interprocess lock, and the path spins without re-checking `g_stop`. Two remote chips ->
two held locks -> every other process on those tunnels starves. It is the *remote* chips
again, and the lock is interprocess, so the blast radius is the whole machine.

Recovery, in order, all verified:
  - `kill -9` on the collector -> UMD's robust mutexes were recovered by the kernel and
    pytest exited 20 s later (with a teardown backtrace, not a clean exit).
  - `--stop-aggregator` then could not run: it needs topology discovery, and discovery
    now times out on a frozen eth heartbeat left by the kernels it was trying to stop.
    Chicken-and-egg -- the 5-8 min hang from the trap list, reached from a new direction.
  - `tt-smi -r` cleared it; `--journal-probe` then reports `none found`, discovery clean.

### Consequences

1. SIGINT must exit. A signal that cannot interrupt a UMD tunnel poll is not a stop.
   The discovery-phase `_exit(130)` guard (5j) needs an equivalent for the sample loop.
2. Never hold a UMD interprocess lock across a multi-block remote read. The 8 KiB journal
   read has to be chunked with a `g_stop` check between chunks, or moved off the locked
   path entirely.
3. `--stop-aggregator` must not depend on full topology discovery, or a stuck monitor is
   unrecoverable without a board reset. This is the single highest-value fix: it converts
   every future wedge from "reset the board" into "run the stop".
4. Steady state is NOT the risk any more -- shutdown is. Both workloads ran at full speed
   with the monitor attached; the monitor just cannot let go.

### Unrelated, still open

- `compute_busy_p1000` reads ~0 across all 8 chips during Llama decode (only chip 6, at
  0.1%), while Test A's CCL arm showed 11.9-16.8% on chips 5-7. Feed counters prove the
  writes happen: `published=38400 skipped_seq=0 skipped_dwall=0` on 8/8. Batch-1 decode
  is bandwidth-bound so low is expected, but not uniformly zero. Unresolved.
- The first Test B attempt died with `Bus error (core dumped)` in BOTH the collector and
  the pytest simultaneously, on a board carrying 37,091 AER entries and an `a1:00.0`
  physical-layer RxErr storm. SIGBUS in two unrelated processes at once is a PCIe fault,
  not our code. It did not recur after the board reset.

## 5v. The shutdown fixes hold; "the TUI is hung" was three separate publisher bugs -- 2026-08-31

### Shutdown, measured under live fabric load

    up: 8/8   skips: 2        journal-fed: 8/8
    collector exited in 6s code 130   (watchdog fired: 1)
    workload ALIVE after collector exit (good)
    stopped 8 of 8 aggregator(s).

The 5u live-lock is closed. A SIGINT that used to leave two threads spinning on the
interprocess tunnel locks forever now takes the process down in 6 s, the workload behind
it survives, and the ethernet cores clear 8/8 with no reset. Three changes did it:

1. `read_chunked_stoppable` -- the journal read is issued in <=1 KiB pieces with a g_stop
   check between them, so UMD's interprocess lock is released between tunnel blocks
   instead of being held across all three.
2. A shutdown watchdog: once a stop is requested the process leaves within a grace period
   (5 s, TTNVTOP_SHUTDOWN_GRACE_S) whether its threads joined or not. A thread inside
   UMD's tunnel poll cannot be interrupted, so leaving is the only thing that releases the
   lock. The robust mutexes are then recovered by the kernel -- measured in 5u, 20 s.
3. Shutdown asks every live aggregator to retire before joining, so a normal Ctrl-C leaves
   the ethernet cores clean without a second command.

### RETRACTED: falling back to local-only

I added an automatic re-exec that dropped remote discovery when discovery timed out, and
backed it out the same session. Remote chips ARE the case this feature exists for, so
"give up on remote telemetry" is not an acceptable automatic response to a wedged tunnel.
`--local-only` survives only as an explicit user choice, never a fallback.

### "The TUI is hung" -- three bugs, none of them in the TUI

The viewer was the suspect and was innocent. What it faithfully displayed:

**a) Two collectors silently stole each other's SHM files.** `ShmPublisher::open` did
`shm_unlink` then `shm_open(O_CREAT)`, believing it was clearing a stale file. What it
actually did was let a second collector detach the inode the first was still writing to
and create a fresh one. From then on the first published into an inode no reader could
open. Observed with two live collectors -- the pid stamp differed per file and half the
grid was frozen:

    tt_device_0_util pid=115222 age= 41.26s F:nz=0    <- writer alive, publishing nowhere
    tt_device_4_util pid=115216 age=  0.08s F:nz=18

Now an flock on the SHM fd: released on close OR on process death, so a crashed
collector's file is takeable and a live one's is not. No pid liveness guessing, no unlink
race. A second collector is refused by name, with the holder's pid.

**b) Device I/O on the publish path froze every chip at once.** The publish loop called
`get_clock()` per chip inline, commented "non-blocking" -- it is a legacy ARC message on
Wormhole, and we measured it timing out (`Timed out after waiting 1000 ms for ARC to
respond`). The ported `dram_update` had the same shape on the sampler thread, iterating
all chips. Either one stalling froze `last_update_us` for every chip, which is precisely a
hung grid. Both are now per-chip telemetry threads; publishing touches no device at all.
This is the FOURTH instance of the same defect (ring drain 5t, journal feed, DRAM, AICLK):
any all-chips loop that does device I/O starves every chip behind the slowest link.
Afterwards, all 8 chips: `age=0.02-0.11s`.

**c) Two writers fought over the same two fields.** The journal feed writes
`compute_busy_p1000`/`sfpu_busy_p1000` for the pipe `counter_sel` says was measured; the
publish loop independently recomputed BOTH from its own EWMAs every tick. Whichever ran
last won, so a reading jumped between chips and between the FPU and SFPU fields, and the
unmeasured pipe got clobbered with a stale EWMA -- reported as "appearing intermittent,
and sometimes telemetry just hangs", and the likely explanation of the flat zeros in 5u.
The feed now owns those fields on any chip with `journal_active`.

### Still open

- AICLK reads 0 on some remote chips (chip 7 consistently, chip 5 intermittently) while
  its neighbours report 878-1000 MHz. Per-chip now, so it no longer harms anything else.
- The chip-summary line truncates: cores showing 0.8% render as a chip average of `F 0%`,
  which reads as "no activity" when there is some. A rounding choice in the viewer.
- The single-writer refusal is UNVERIFIED. The test's second collector exited rc=3 --
  discovery timeout, contending with the first for the tunnel -- so it never reached
  `shm_open` and the flock path did not run. Needs a retest with a longer discovery bound.
- Under a CCL-only workload F/S sit near 1%, which is plausible (fabric collectives are
  NOC and DRAM work, not FPU math: DRAM read 33 GB/s per chip, 12% of 288) but is not the
  same thing as having been validated against a known-utilization kernel.

## 5w. CALIBRATED: FPU and DRAM are true, SFPU is a DUPLICATE of FPU -- 2026-08-31

Everything before this measured whether the monitor crashed, perturbed a workload, or
stayed fresh. None of it measured whether the NUMBER is true. Under a CCL workload F and S
read ~1%, which is plausible -- fabric collectives are NOC and DRAM movement, not FPU math
-- but plausible is not measured, and "FPU and SFPU were almost 0" is not answerable
without a reference.

### Method: sweep a known duty cycle, avoid needing a peak constant

`scripts/calib_duty.py` runs back-to-back 2048x2048 bf16 matmuls for a fraction d of each
40 s phase and idles the rest, with a device sync so the idle is real, sweeping d over
{0, 25, 50, 75, 100}%. The independent variable is the HOST-MEASURED busy fraction, not the
target, so overshoot in `time.sleep` does not matter. Then:

    monitor_reading(d) must be LINEAR in d, through the origin.

Linearity and a zero intercept are both peak-free, which is the point: a wrong peak
constant makes a correct counter look broken and vice versa. The slope then calibrates the
reading, and only afterwards is its value compared against a published peak.

`scripts/shm_probe.py` samples every SHM file at 10 Hz in a separate process (the monitor
must never be in-process with what it measures); `scripts/calib_report.py` joins the two
after the fact, because the phase file only exists once the workload has exited.

### Consistency: the stalls are gone

    chip                       n   max age  mean age  stalls>1s  aiclk=0
    tt_device_0_util        2114     0.10s     0.05s          0        0
    ... identical for 1-6 ...
    tt_device_7_util        2114     0.10s     0.05s          0        0

16,912 samples, 8 chips, zero stalls, zero AICLK dropouts -- chip 7 included, which was the
chip reported as stalling. This is the host-drain path (no aggregator in this run).

### FPU: linear, and absolutely right

     host busy |   0     1     2     3     4     5     6     7
         0.0%  | 0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
         8.7%  | 1.16  0.69  0.93  1.26  1.23  1.12  1.32  1.07
        19.2%  | 3.77  3.38  3.58  3.80  4.54  3.72  4.08  4.06
        40.0%  | 9.58  8.85 10.23  9.83 10.87  9.76 10.30 10.12
        99.7%  |21.59 20.35 21.94 22.53 22.48 22.17 21.81 22.54

    slope 0.210-0.231, intercept -0.40..+0.12, R^2 0.990-0.996 on all eight chips.

Perfectly linear with a zero intercept. The slope is not a defect: at 99.7% host-busy the
FPU is issuing ~22% of cycles, and an INDEPENDENT path confirms that is the truth --

     host%   iters   ach.TFLOP/s/chip   monitor F%   implied bf16 peak
      19.2   12036             5.17         3.87              133.7
      40.0   25430            10.92         9.94              109.9
      99.7   64220            27.58        21.93              125.8

27.58 TFLOP/s achieved at full load divided by the monitor's 21.93% implies a bf16 peak of
125.8 TFLOP/s per ASIC, which is the right order for Wormhole b0. The implied peak is
roughly constant across duties (the 8.7% phase reads 181, where the signal is weakest and
host timing noise dominates), which is what a true duty-cycle fraction looks like. So the
FPU reading is trustworthy in absolute terms, and the near-zero readings under CCL were
CORRECT -- collectives really do almost no FPU math.

A hypothesis worth recording as DISPROVEN: I expected the chip average to be diluted by
inactive cores (an earlier sample showed `F:nz=14` of 64). At full duty all 64 cores are
nonzero and the average over active cores equals the chip average, 21.9%. Not dilution.

### SFPU: INVALID -- it is a copy of the FPU signal

The workload runs NO SFPU ops (`--sfpu` not passed; `ttnn.exp` never called). SFPU must
therefore read ~0. It does not:

     host busy |     0     1     2     3     4     5     6     7
         8.7%  |  1.47  1.47  1.47  1.47  1.47  1.47  1.47  1.47
        19.2%  |  3.85  3.85  3.85  3.85  3.85  3.85  3.85  3.84
        99.7%  | 21.54 21.32 21.50 21.19 21.35 20.68 20.18 20.12

It tracks the duty cycle at almost exactly the FPU's magnitude, and at the low phases it is
identical to three decimal places across all eight chips -- which is not a physical
measurement. FPU and SFPU share one counter block (counter_sel 0 = FPU, 1 = SFPU), so the
defect is in that selection or in the attribution of the result: the same count is landing
in both fields. Until this is fixed the SFPU axis must not be trusted or shown.

### DRAM: exact

     host%   iters  expect GB/s   monitor GB/s   ratio
      8.7    4648          2.9            2.9    1.00
     19.2   12036          7.6            7.6    1.00
     40.0   25430         16.0           16.0    1.00
     99.7   64220         40.4           40.5    1.00

Against the bytes a 2048 matmul must move (2 reads + 1 write, 25.2 MB), the NIU counters
agree to 1.00 at all four load levels. Nothing to fix.

### What is validated, and what is not

  - FPU%: linear, zero intercept, absolutely correct. TRUSTED.
  - DRAM GB/s: exact at four load levels. TRUSTED.
  - Publish freshness: 0 stalls in 16,912 samples across 8 chips. TRUSTED.
  - SFPU%: reads the FPU signal. BROKEN, do not display.
  - CCL/fabric: coexistence measured (5u/5v, 2.9% slowdown), but the monitor exposes no
    ethernet-link metric, so there is nothing to calibrate against the fabric goldens in
    tests/tt_metal/tt_fabric/test_infra/golden/. The fabric axis is absent, not wrong.
  - All of the above is the HOST-DRAIN path. The on-chip aggregator path has not been put
    through this ramp, and it is the path with the different arithmetic (busy/wall deltas
    folded on-chip). It needs the same treatment before the aggregator numbers are trusted.

## 5x. The frozen TUI is a VIEWER bug: it matches by path, but the inode changes -- 2026-08-31

The screenshot that "looks soooo wrong" showed every core on a chip reporting the identical
value (chip 0: all 64 at 0.8%, chip 3: all at 16.2%) with `@ 0 MHz` on chip 7, and was
byte-identical to a screenshot taken much earlier in the session. At that moment the live
SHM said something else entirely:

    tt_device_0_util age=0.04s aiclk=1000 F distinct=1 (min 0.0 max 0.0)
    ... all eight chips: fresh, all zeros, no workload running ...

Fresh files, zeros in them, and a viewer on screen showing 16.2%. The viewer was not
reading the files it appeared to be reading.

`refresh_maps()` decides an existing mapping is still valid by comparing the PATH. A
collector restart unlinks the file and creates a new inode at the same name; an unlinked
inode that someone still has mapped stays alive and frozen forever. So the viewer held a
dead orphan, rendered its last frame indefinitely, and looked perfectly healthy doing it.
Now it compares `st_dev`/`st_ino` and re-maps when the inode changes.

This is the same class of bug as 5v(a) and its exact counterpart: there, the PUBLISHER
unlinked and stole the name; here the READER failed to notice the name had been re-pointed.
Fixing only the publisher side left this half live -- a collector restart still froze the
view, which is why "restarting the viewer fixes it" was the observed workaround.

It also explains why my calibration in 5w passed while the display was wrong: the probe
reads SHM directly, so it never went through the viewer's mapping. And note what the
calibration did NOT check -- it regressed CHIP AVERAGES against a known duty cycle and
never once asserted that per-core values are LOCALIZED. A bug handing every core the same
number would have passed all of 5w. The uniform-across-64-cores appearance is still
unexplained and is now the open question; `Fnz` from 5w (41.9 of 64 nonzero at 8.7% duty)
says the host path does differentiate cores, so the uniformity may be the frozen frame
rather than a real mapping defect -- but that is a hypothesis, not a measurement.

### AICLK dropouts were self-inflicted blanking

`aiclk=0` rotated between chips run to run (5,7 then 4,6,7), which is a flaky ARC mailbox,
not a clock that is genuinely zero. The telemetry thread zeroed the field on every
transient `get_clock()` failure, so the viewer flipped to "@ 0 MHz" and back. It now keeps
the last good value, and 0 means only "never read successfully".

### BOARD DOWN -- and it was my doing

After the calibration the T3K wedged with:

    Timeout waiting for Ethernet core service remote IO request.
    Location: /project/device/common/utils.hpp:178

Every entry point hits it -- collector discovery aborts, and `tt-smi -ls` and `tt-smi -r`
both fail inside re-initialization. `tt-smi -r` does reset the PCI devices [0,1,2,3] and
then fails re-init, twice, because the wedge is on the REMOTE chips' ethernet and a PCIe
reset of the MMIO ASICs does not reach it. This needs a power cycle.

The cause was mine: I ran `pkill -9` on the collector. Killing a process mid-tunnel-
transaction is the documented way to wedge the remote-IO path, and the shutdown watchdog
added in 5v exists precisely so that SIGKILL is never necessary -- the collector now leaves
on its own in 6 s. Use SIGINT and wait. Do not use -9 on this tool.

Unrelated and still unexplained: a Llama-70B run and an earlier collector both aborted
inside UMD's `wait_arc_core_start`/`read_from_arc_csm` with 1000 ms ARC timeouts before any
of this, on a board already carrying 37k+ AER entries. The ARC path on this board is flaky
independently of the ethernet wedge.

## 5y. Why tt-mgmt and the collector fight, and the architectural fix -- 2026-08-31

Observed: `tt-mgmt smi` hangs while the collector runs, and the collector's AICLK reads
fail. Both processes hold all four MMIO devices at once:

    pid=19668 python   -> /dev/tenstorrent/0,1,2,3    (the workload)
    pid=20267 tt-mgmt  -> /dev/tenstorrent/0,1,2,3    (tt-mgmt smi, default backend)

### It is not a locking bug, it is a fairness problem

Both sides use UMD, so both DO participate in UMD's interprocess `LockManager`. There is no
corruption and no missing mutual exclusion. But a lock provides exclusion, not fairness:
this collector polls 8 chips at 300 Hz and holds those locks almost continuously, so a
1 Hz telemetry poll on the other side simply never wins, and tt-mgmt times out at its
1000 ms ARC bound. It is the same starvation that made Llama's teardown crawl in 5v -- one
greedy poller, no backoff.

The reverse direction explains a whole class of symptoms I had been attributing to flaky
hardware: `aiclk=0` rotating between chips, "Timed out after waiting 1000 ms for ARC to
respond", the abort inside `wait_arc_core_start`, and a third process never completing
topology discovery. tt-mgmt was running through those measurements.

### Three ways two processes can share one accelerator's telemetry

  1. Both poll ARC from userspace. Correct only with a shared lock domain, and even then it
     degrades to starvation under an unfair poller. This is what we were doing.
  2. Read what the KERNEL already publishes. The KMD owns the ARC conversation and
     republishes the result; readers are unlimited, take no lock, and cannot block anyone.
  3. One userspace owner plus IPC fan-out. This is what the collector already does for core
     utilization via /dev/shm.

tt-mgmt does NOT need a different architecture -- it already has (2) available as
`--backend sysfs` ("lightweight, no UMD"). Its default is `--backend auto`, which tries UMD
first, which is why the shipped default collides.

### The fix on OUR side: AICLK from sysfs

The kernel already publishes what we were taking an ARC round trip for:

    /sys/class/tenstorrent/tenstorrent!<n>/  ->  tt_aiclk tt_arcclk tt_axiclk tt_heartbeat
                                                tt_asic_id tt_serial tt_fw_bundle_ver ...
    plus hwmon: temp1_input curr1_input in0_input

Verified live: tt_aiclk tracks 500 MHz idle -> 1000 MHz under load, and tt_heartbeat
advances (19064 -> 19084 in 2 s). `aiclk_from_sysfs()` now reads it, with the ARC call kept
only for remote chips, which have no PCIe node of their own.

Result, measured WITH tt-mgmt still running: all 8 chips report 500 MHz, chip 5 included --
the chip that read 0 for 3949 of 3949 samples. The remote chips' ARC fallback also started
succeeding, because dropping four ARC users left mailbox headroom for them.

TRAP, and it nearly bit: the sysfs node index is NOT the UMD chip id and NOT BDF order.

    tenstorrent!0 -> 61:00.0     tenstorrent!1 -> e1:00.0
    tenstorrent!2 -> a1:00.0     tenstorrent!3 -> 81:00.0

while UMD numbers those 61, 81, a1, e1. Mapping by ordinal would have reported chip 1's
clock as chip 3's -- silently, plausibly, and undetectably. The join must go through
`PCIDevice::get_device_num()`, which is the /dev/tenstorrent index by construction.

### Still open from the saturation attempt

The compute+fabric saturation run did NOT meet its bar and is not evidence of anything yet:
  - Workload cost: none. Control 1.0608 ms/iter vs monitored 1.0485 ms/iter over 226k and
    458k iterations, 8/8 aggregators, 8/8 journal-fed, 0 fatals.
  - But compute was NOT saturated: F_avg ~0.5% and DRAM ~23.6 GB/s, against the 5w
    calibration markers of ~22% and ~40 GB/s for 100% duty. `--ccl 4` is fabric-heavy and
    compute-light; the all-gathers dominate wall time.
  - And LOSS WAS NOT MEASURED AT ALL: `--fidelity-probe` produced no output, because as a
    fourth process it could not finish topology discovery against workload + collector +
    tt-mgmt. The loss metric must come from the ALREADY-ATTACHED collector, which tracks
    `journal_lost_reported` and `drain_lost_samples`, not from a second process that has to
    re-discover the topology under exactly the load being tested.
