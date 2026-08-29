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
| **N300 remote** | `0,1,6,7,14,15` | `2,3,4,5,8,9,10,11,12,13` | **10** |
| T3K remote (measured config) | `0,1,6,7` | — | **12** |

Channels `2,3,4,5,8,9,10,11,12,13` on a remote chip have no PHY link routed on any
N300 configuration. They are structurally idle — no recabling can claim them.

6U Galaxy is the only WH topology with zero idle channels, and it has **32 MMIO-capable
chips and zero remote chips** — it does not need this feature. Scope the aggregator to
`is_remote` chips and the constraint never binds.

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

### 3.4 Placement and launch

- New env gate `TTNVTOP_ETH_AGGREGATOR=1`, matching the `TTNVTOP_REGISTER_PROGRAMS`
  pattern in `registrar/ttnvtop_register.hpp`. Default off.
- Launched from the same registrar seam already hooked into tt-metal, after device init.
- Core selection: request from `get_inactive_ethernet_cores()` **through the dispatch
  core manager**, preferring the never-routed set `{2,3,4,5,8,9,10,11,12,13}`. Fail
  loudly at init if unavailable — never silently share a core with a dispatch kernel.
- Kernel via `CreateKernel(program, "...aggregator.cpp", core, EthernetConfig{.eth_mode
  = Eth::IDLE, ...})`.
- Journal base: `hal::get_erisc_l1_unreserved_base()` + fixed offset, passed to the
  kernel as a compile-time arg and recomputed host-side. Avoids touching
  `dev_mem_map.h` and the cross-layer lockstep it implies (host C++, dev_msgs codegen,
  RISC-V linker scripts) — the reason the Tensix reservation was painful in Phase 2.1.a.

## 4. Phasing

> **Transport direction revised 2026-08-28 (§3.3, §5c): the aggregator PUSHES its
> journal to the MMIO chip; the host never reads a remote chip.** The phase descriptions
> below predate that and still describe a host pull. Sizing and sequencing are unchanged;
> the direction of the final hop is not.

### Phase 2.2.a — Transport only (~1 week)

Aggregator mirrors each core's 1 KiB ring verbatim into a 64 KiB eth-L1 buffer; host
does one 64 KiB block read per chip. **No change to sample-loss behavior** — this phase
only proves the transport, the core allocation, the kernel lifetime, and the host
fallback path. Host decode logic is literally unchanged.

Exit criteria: remote chip fidelity identical to today, tunnel transactions per tick
drop from 256 to 1, measured workload slowdown from telemetry falls to noise.

### Phase 2.2.b — Drain-and-accumulate (~1–2 weeks)

Replace the mirror with the sweep loop and journal of §3.2. Drop
`UTIL_SAMPLER_DEFAULT_PERIOD_CYCLES` back toward 100 µs on aggregated chips and confirm
`lost` stays at zero.

Exit criteria: 100 µs sampling with <1% structural loss on an aggregated chip, vs the
~84% loss that forced the 1 ms period.

### Phase 2.2.c — Local chips, opt-in (~3 days)

Enable on local chips too. The transport win is smaller (PCIe is cheaper than the
tunnel) but the sample-loss win is identical. Gate separately so a regression on local
chips cannot take out the common path.

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

## 6. Risks

| Risk | Mitigation |
|---|---|
| Dispatch owns idle eth cores — `dispatch_core_manager.cpp:269` adds *every* inactive eth core to the dispatch pool when dispatch core type is ETH (2 CQs on N300 needs 10) | Allocate through the dispatch core manager; fail loudly at init |
| `assert_inactive_ethernet_cores()` resets all RISCs on idle eth cores at device init | Launch after init, same lifecycle position as `util_sampler` |
| Kernel lifetime — aggregator must be persistent across program dispatch | **Open question.** Fabric EDM kernels are persistent on eth cores; follow that pattern. Resolve before 2.2.a |
| Watcher / inspector walk inactive eth cores (`watcher_server.cpp:523`, `watcher_device_reader.cpp:430`, `inspector/data.cpp:383`) | Verify they tolerate a non-dispatch kernel; may need an exclusion |
| Aggregator NOC traffic shares the chip NOC with the workload | ~2.6 MB/s at 10 kHz sweep; measure in run C, tune sweep rate |
| WH-only — BH eth differs (2 ERISCs, different L1 map) | Scope to WH. BH p150a has no remote chips; 6U BH is all-MMIO. Revisit only if a BH topology with remote chips ships |
| Aggregator dies silently → stale data read as live | Host checks `sweep_count` advances; falls back to per-core drain if stalled |

## 7. Open questions

1. **Persistent idle-eth kernel lifetime** — what is the supported mechanism for a
   kernel that outlives program dispatch? Fabric does this; confirm the pattern is
   reusable outside fabric init. *Blocks 2.2.a.*
2. **Does the collector need tt-metal, or can it stay standalone?** Today it uses
   `umd::TopologyDiscovery` and never calls `LocalChip::start_device()`
   (`collector/main.cpp:16`). Launching a kernel needs tt-metal. Either the aggregator
   launches from the workload process via the registrar seam (preferred — keeps the
   collector standalone), or the collector gains a tt-metal dependency (rejected).
3. **Should this land with `configure_active_ethernet_cores_for_mmio_device()`?**
   Pinning UMD's remote traffic away from fabric-carrying channels is a separate,
   smaller win that stacks with this. Worth measuring independently.
