# Project Plan: `ttnvtop-compute` — Real Compute Utilization Monitor

A concrete engineering plan to evolve the current dispatch-occupancy PoC into a live per-pipeline compute-busy% monitor that coexists with running tt-metal workloads.

---

## 1. Objectives & non-goals

**Objectives**
- Live per-Tensix, per-pipeline busy%: compute (MATH/FPU/SFPU), UNPACK, PACK, stall, NOC in/out.
- Updates ≥ 4 Hz in the TUI, underlying sampling ≥ 200 Hz per core.
- **Coexists** with any running tt-metal workload with no kernel changes required by the user and <1% wall-time overhead on the workload.
- Works on local and remote chips (n300 and larger meshes) through a single process.
- Single-file TUI binary; optional SHM publisher so other tools (Prometheus exporter, `tt-mgmt monitor`) can consume the same data.

**Non-goals (for v1)**
- Per-RISC-V program-counter attribution (the exalens debug-bus path is a v2 extension).
- Ethernet / DRAM / ARC telemetry (already covered by `tt-mgmt`; we surface their existing SHM if useful).
- Replay/playback or capture-to-file (add in v3 if demand).
- Support for Quasar (Wormhole + Blackhole only for v1).

---

## 2. Acceptance criteria

1. **Correctness** — On a controlled kernel that runs `fpu_nop` for N cycles then idles for N cycles in a loop, the monitor reports ~50% `fpu_busy` averaged over 1 s, to within ±5%.
2. **Non-intrusiveness** — Running `ttnvtop-compute` during a 60 s matmul throughput test changes reported GB/s by < 1%. Both with and without the monitor attached.
3. **Coexistence** — Starts and stops cleanly while `pytest` / long-running training is underway; no `CHIP_IN_USE` contention.
4. **Load balance detection** — On a deliberately skewed workload (one shard 2× the others), the monitor visibly shows the imbalance.
5. **Stall detection** — On a workload instrumented to block on a CB, the monitor distinguishes `stall%` from `compute%`. This is the flagship capability vs today's PoC.
6. **Sampling overhead** — On-chip sampler measurably < 0.2 % of AICLK cycles at default settings.
7. **Host overhead** — < 2 % of one host core at default settings (10 Hz TUI refresh, 200 Hz underlying sampling).

---

## 3. Signal catalog

Per Tensix, each sample captures:

| Signal | Source register | Meaning |
|---|---|---|
| `aiclk_ticks` | a free-running cycle counter (e.g. `RISCV_DEBUG_REG_WALL_CLOCK_L/H`) | denominator for rates |
| `fpu_busy` | `RISCV_DEBUG_REG_PERF_CNT_FPU` (FPU_COUNTER) | MATH pipeline active cycles |
| `sfpu_busy` | `PERF_CNT_FPU` subindex | SFPU active cycles |
| `unpack0_busy`, `unpack1_busy` | `PERF_CNT_FPU` / instrn thread groups | unpack throughput |
| `pack_busy` | `PERF_CNT_FPU` / `PACKER_BUSY` | pack throughput |
| `thread_stall_*` | `PERF_CNT_INSTRN_THREAD` | idle-due-to-stall |
| `l1_rd`, `l1_wr` | `PERF_CNT_L1_*` | local L1 bandwidth proxy |
| `noc0_in`, `noc0_out`, `noc1_in`, `noc1_out` | `PERF_CNT_L1_*` ring bandwidth counters | cross-core/DRAM traffic |
| `dispatched` | `mailboxes_t.go_messages[idx].signal` | today's signal, kept as overlay |
| `host_assigned_id` | `launch_msg.kernel_config.host_assigned_id` | which program/kernel is live (for labels) |

Everything except `dispatched` and `host_assigned_id` needs perf-counter free-run + mux rotation. See §5.

---

## 4. System architecture

Three cleanly separable layers so each can be developed, tested, and swapped independently.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HOST PROCESS(ES)                                                            │
│                                                                             │
│   ┌──────────────────┐   ┌─────────────────────┐   ┌────────────────────┐   │
│   │   Viewer (TUI,   │   │  Prom exporter      │   │  JSON dump / trace │   │
│   │  standalone bin) │   │  (future)           │   │  (future)          │   │
│   └───────▲──────────┘   └──────────▲──────────┘   └─────────▲──────────┘   │
│           │ mmap + read             │                        │              │
│           └───────────────┬─────────┴────────────────────────┘              │
│                           │                                                 │
│                  /dev/shm/tt_device_<asic>_util      (Layer 3: Publisher)   │
│                           ▲                                                 │
│                           │ write                                           │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ Collector (Layer 2)                                                   │ │
│   │   - opens chips via umd::TopologyDiscovery (no CHIP_IN_USE lock)      │ │
│   │   - one thread per chip, UMD block reads of L1 ring per core          │ │
│   │   - diffs successive Snapshots → rates per pipeline                   │ │
│   │   - rolling 1 s window stats                                          │ │
│   │   - publishes to SHM                                                  │ │
│   └─────────────────────────┬─────────────────────────────────────────────┘ │
│                             │                                               │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              │ PCIe TLB reads (attach-safe)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DEVICE (per Tensix worker core)                                             │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ Firmware (Layer 1: on-chip sampler)                                  │  │
│   │   brisc/ncrisc preamble (modified):                                  │  │
│   │     perf_counters_free_running_init()                                │  │
│   │     util_sampler::install(every_N_cycles)                            │  │
│   │                                                                      │  │
│   │   Sampler ISR (~30 cycles):                                          │  │
│   │     read AICLK, read counter group [current_mux]                     │  │
│   │     store Snapshot into L1 ring                                      │  │
│   │     advance head, rotate mux                                         │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                           │                                                 │
│                           ▼                                                 │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ L1 reserved region (1 KiB at fixed slot, outside allocator):         │  │
│   │   UtilRingHeader { magic, version, sample_stride, head, count }      │  │
│   │   Snapshot ring[N]    // ~64B each, ~12-16 entries                   │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

The three layers:

- **Layer 1 (device/firmware):** produces snapshots into L1 rings. Owned by tt-metal firmware. Arch-specific (WH, BH).
- **Layer 2 (collector):** stateless C++ process that pulls rings over PCIe, computes deltas and rates, publishes to SHM. Runs either standalone (one dedicated `ttnvtop-collector` binary) or as a library linked into a user's workload process.
- **Layer 3 (SHM + consumers):** the contract. Viewers, exporters, and tools read from a well-known SHM schema. This is where we get the "any number of readers, any number of tools" property.

---

## 5. Layer 1 — firmware sampler design

### 5.1 Code change locations

- `tt_metal/hw/firmware/src/brisc.cc` — insert `util_sampler::init()` in firmware preamble, before the main `while (1) { go_msg_wait; kernel_run; }` loop.
- `tt_metal/hw/firmware/src/trisc.cc` — same hook for trisc (to capture FPU/PACK/UNPACK which live on trisc).
- New header `tt_metal/hw/inc/util_sampler.h` — the sampler state, ISR body, L1 ring layout.
- `tt_metal/hw/inc/dev_msgs.h` — add a new `util_sampler_msg_t` field to `mailboxes_t` (so Factory auto-generates host-side offsets, exactly like `watcher_msg_t`).
- `tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal_tensix.cpp` + `blackhole/bh_hal_tensix.cpp` — add `HalL1MemAddrType::UTIL_SAMPLER` pointing at the new region.
- `tt_metal/tools/profiler/perf_counters.hpp` — add a `perf_counters_free_running_init()` that starts counters without harvest-on-stop behavior.

### 5.2 Sampler hook

Options for how the sampler fires:

- **Periodic ISR via RISC-V timer**: `mtime`/`mtimecmp`. Cleanest, preempts kernel code. ~30 cycles per ISR.
- **Inline polling**: kernel firmware checks a cycle counter in its hot loop. Not reliable across kernels with different structures.
- **Dedicated helper RISC-V**: pin one of the 5 Tensix RISC-Vs as a sampler. Breaks every kernel that assumes all 5 are available.

Choose **periodic timer ISR**. Precedent: tt-metal already uses RISC-V timer features elsewhere; brisc + ncrisc + trisc0 each have their own CSR space. Fire on brisc (simplest, can read all shared perf counters via mmio).

Period target: 100k AICLK cycles ≈ 100 µs (10 kHz sampling). Configurable via a mailbox field the host can tune.

Overhead budget:
- ISR cost ≈ 30 cycles × 10 kHz = 300k cycles/sec per core = 0.03% at 1 GHz. Acceptable.
- Mux rotation + counter read add ~10 cycles.

### 5.3 L1 ring layout

Fixed 1 KiB region per Tensix, addressed via new `HalL1MemAddrType::UTIL_SAMPLER` constant. Layout:

```c
struct UtilRingHeader {
    uint32_t magic;             // 'TTUT' — identifies the format
    uint16_t version;           // bump on schema change
    uint16_t sample_stride_b;   // sizeof(Snapshot) for forward compat
    uint32_t capacity;          // number of slots
    uint32_t head;              // next write slot (monotonic, wraps)
    uint32_t period_cycles;     // sampler period, for host-side sanity
    uint32_t reserved[2];
};  // 32 B header

struct Snapshot {
    uint64_t aiclk_ts;          // free-running cycles at sample
    uint32_t mux_group;         // which counter group this snapshot belongs to
    uint32_t counters[12];      // group-specific: FPU, PACK, UNPACK0/1, stalls, ...
    uint32_t seq;               // write sequence for tear detection
    uint32_t pad;
};  // 64 B per sample

// Total ring: 32 + 15 * 64 = 992 B, fits in 1 KiB.
```

Host reads the whole 1 KiB in one NOC transaction per core per tick. Writes on device are 4-byte-aligned u32 stores; no packed/unaligned pain.

### 5.4 Counter mux rotation

`PERF_CNT_MUX_CTRL` selects which 8 physical counters are exposed. Define 4 groups:

- Group 0: FPU / MATH / SFPU / AICLK
- Group 1: UNPACK0 / UNPACK1 / PACKER / stall reasons
- Group 2: L1 RD / L1 WR / NOC0 rings
- Group 3: NOC1 rings / reserved

Round-robin on each sample. Effective per-group rate = 2.5 kHz. Plenty for a 10 Hz UI.

### 5.5 Kernel-cooperation edges

- When a kernel starts and calls `StartPerfCounters`, we don't want to fight it. Two modes:
  - **monitoring-off** (default today): kernel counters behave as before. Sampler is disabled.
  - **monitoring-on** (env var `TT_METAL_UTIL_MONITOR=1` at program start): firmware flips Start/Stop into **additive mode** — counters never reset, `StopPerfCounters` captures a delta instead. Sampler is active. Existing `kernel_profiler` users may need to opt out.
- The env var lives in `rtoptions`, plumbed through firmware init via an existing mailbox slot (watch how watcher_enabled is plumbed; same pattern).

### 5.6 Arch deltas WH vs BH

- Mailbox layout change is picked up by `dev_msgs` codegen → both archs get the HAL constant automatically.
- Counter register addresses live in arch-specific `tt_metal/hw/inc/internal/{wormhole,blackhole}/perf_counters_map.h` — write once per arch.
- Timer ISR setup differs slightly (BH has different CLINT addresses).
- Two sibling `.S` or `.cc` files: `util_sampler_wh.cc`, `util_sampler_bh.cc`.

---

## 6. Layer 2 — host collector

### 6.1 Modes

One binary, two modes:

- **Standalone** (`ttnvtop-collector`): the common case. Opens chips via `umd::TopologyDiscovery` (no lock). No tt-metal Cluster. Runs forever, publishes to SHM.
- **Library** (`libttnvtop_collector.a` with `ttnvtop::start()` / `stop()`): linkable into a user's workload process. Uses the already-open MetalContext cluster directly. Preferred when the user wants zero inter-process concerns.

Both modes share the same collector core — the difference is only how they get a `TTDevice*` handle. Factor that behind a `ChipAccess` interface.

### 6.2 Collector loop per chip

One `std::thread` per chip. On each tick:

1. For each worker core (cached list from SocDescriptor):
   - `TTDevice::read_from_device(util_ring_addr, 1024)` → local buffer.
   - Cast to `UtilRingHeader + Snapshot[]`.
   - Compare `head` to last-seen head; collect new snapshots.
2. For each pair of consecutive snapshots in the same mux group:
   - `Δcounter = new - old`; `Δclock = new.aiclk - old.aiclk`.
   - `busy% = Δcounter / Δclock * 100`.
3. Update rolling 1 s exponentially-weighted averages per signal per core.
4. Write aggregated view to SHM (see §7).
5. Sleep until next tick (default 100 ms).

Failure modes to handle:
- Ring header magic mismatch → firmware not patched; surface "dispatch-only" fallback.
- `seq` tear between reads → retry that snapshot.
- Chip transient read error → mark core as stale, keep going.
- New kernel loaded (detected via `host_assigned_id` change) → counters may have discontinuities; reset rolling averages for affected cores.

### 6.3 Per-tick cost budget

- Block read per core: ~2 µs PCIe latency × 128 cores × 10 Hz = ~2.5 ms/s = 0.25% host CPU.
- Delta math, EWMA updates: negligible.
- SHM writes: memory-mapped stores, negligible.

Target: collector uses < 2% of one host core at defaults.

---

## 7. Layer 3 — SHM schema & viewer

### 7.1 SHM layout

File: `/dev/shm/tt_device_<asic_unique_id>_util`. Mirrors the existing `memory_stats_shm.hpp` pattern so `tt-mgmt` already knows where to look.

```c
struct UtilShmHeader {
    char magic[4];              // 'TTUT'
    uint16_t version;           // 1
    uint16_t struct_size;       // sizeof(PerCoreView)
    uint64_t asic_id;
    uint32_t arch_id;           // wh=1, bh=2
    uint64_t epoch_us;          // start time of this collector
    uint64_t last_update_us;    // wall-clock of last write
    uint32_t num_cores;
    uint32_t host_assigned_id;  // current program id, 0 if unknown
    uint32_t reserved[4];
};  // 64 B

struct PerCoreView {
    uint8_t noc_x, noc_y, logical_x, logical_y;
    uint8_t is_remote;
    uint8_t dispatched;         // latest 1-bit from go_msg
    uint16_t reserved;
    uint16_t compute_busy_p1000;  // per-mille (0..1000), smoothed
    uint16_t unpack_busy_p1000;
    uint16_t pack_busy_p1000;
    uint16_t stall_p1000;
    uint16_t noc0_in_mbps, noc0_out_mbps;
    uint16_t noc1_in_mbps, noc1_out_mbps;
    uint32_t samples_seen;
    uint32_t last_kernel_id;
};  // 32 B per core

// Total: 64 + N_cores * 32 ≤ 4 KiB for a Tensix chip.
```

Using per-mille (0–1000) instead of floats keeps the struct small and aligned; viewers can render as `%` with one decimal.

### 7.2 Write discipline

- Single writer (collector), many readers (viewers).
- Fixed-size record with `last_update_us` at end of each update so readers can detect staleness.
- Double-buffer the `PerCoreView[]` if we ever see tearing in practice (unlikely at this granularity; u16 stores are atomic enough on x86).

### 7.3 Viewer

Rewrite the existing `ttnvtop` binary to be a **pure SHM reader**: zero UMD dependency. Opens `/dev/shm/tt_device_*_util`, renders the TUI. Safe to run as many instances as desired on the same box. Works even when no workload is running (shows "no collector attached" per chip).

Two-column layout we already have, but each core row becomes a stacked segmented bar:

```
(1,1)  C████ U██ P█  S▓▓▓  I░░░░░░░░░░░     74%
       ^^^^ ^^ ^^   ^^^^^
       compute unpack pack  stall       idle
```

Segments are colored with ANSI 256-color sequences. Background tinted by `dispatched` bit so you can tell "stalled with no kernel" from "stalled with kernel dispatched" at a glance.

Header shows current `host_assigned_id` → program name via a small `id→name` registry the collector populates from `launch_msg`.

---

## 8. Phased delivery

### Phase 1 — Host-only prototype, no firmware change (2–3 weeks)

Goal: validate end-to-end plumbing with the counters in whatever mode they're already in. Insight into signal quality.

Tasks:
- Add `TT_METAL_UTIL_MONITOR` env var plumbing.
- Patch `ckernel_perf_unpack_pack.cc` to make Start/Stop idempotent under the env var (small surgical change).
- Host collector reads counter groups directly over PCIe, one group per tick, round-robin. Lower effective resolution but proves the pipeline.
- SHM v0 schema.
- Viewer reads SHM, renders stacked bars.
- Tests: correctness against a `fpu_nop` / `pack_nop` micro-kernel with known cycle counts.

Exit criterion: on a cooperating kernel (micro-benchmark), monitor reports sensible `compute%` matching the kernel's theoretical activity within ±5%.

### Phase 2 — On-chip sampler, L1 ring (3–4 weeks)

Goal: decouple sampling resolution from PCIe latency; make sampler independent of kernel cooperation.

Tasks:
- Add `util_sampler_msg_t` to `dev_msgs.h`; regen HAL.
- Implement brisc-side timer ISR, mux rotation, ring writes.
- WH first, BH next.
- Host collector switches from register polling to L1 ring reads; handles `seq` tear, ring wrap, kernel-change discontinuities.
- Validate: overhead measurement on MLPerf-style matmul. Must pass §2.2.

Exit criterion: `ttnvtop-compute` running during `pytest tests/my_matmul.py` changes reported GB/s by < 1%, and shows meaningful `compute%` that tracks kernel behavior.

### Phase 3 — Polish and productization (2–3 weeks)

Tasks:
- BH arch parity.
- Program ID → name labeling (read `launch_msg.kernel_config.host_assigned_id` and keep a small map the collector maintains from host context if available).
- `dispatched` overlay on the bar (background tint) — shows "stalled but dispatched" vs "idle and not dispatched".
- Multi-chip view tested on an n300 and a multi-n300 mesh.
- `ttnvtop-collector` as a systemd-optional service for always-on monitoring.
- Documentation: how to read the display, what each bar segment really measures, known caveats.

Exit criterion: meets all §2 acceptance criteria.

### Phase 4 — Extensions (opportunistic, post-v1)

- **PC-sampling mode** via debug-bus L1 sampling (the exalens mechanism) — adds compute vs stall attribution to symbol ranges in the loaded kernel ELF.
- **DRAM / ETH / ARC telemetry** — consume existing `tt-mgmt` SHM; overlay in the TUI header.
- **Perfetto / JSON dump** — replay in Chrome trace view.
- **Prometheus exporter** — simple `/metrics` endpoint scraping SHM.
- **Kernel attribution** — correlate bursts with program IDs; show a top-N "hot kernels" panel.
- **Headless CI mode** — record a run's utilization to a file; fail CI if utilization < threshold.

---

## 9. Specific code deliverables, by layer

### Firmware
- `tt_metal/hw/inc/util_sampler.h` — new
- `tt_metal/hw/firmware/src/util_sampler.cc` — new
- `tt_metal/hw/firmware/src/{brisc,trisc}.cc` — ≈ 5-line insertion each
- `tt_metal/hw/inc/hostdev/dev_msgs.h` — add `util_sampler_msg_t` to `mailboxes_t`
- `tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal_tensix.cpp` — add `HalL1MemAddrType::UTIL_SAMPLER`
- `tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp` — same for BH
- `tt_metal/tools/profiler/perf_counters.hpp` — `free_running_init()`
- `tt_metal/impl/context/rtoptions.*` — `TT_METAL_UTIL_MONITOR` env var

### Host collector
- `tt_metal/tools/ttnvtop/collector/` — new directory
  - `chip_access.hpp` — abstract interface (UMD-direct vs via-MetalContext)
  - `chip_access_umd.cpp` — `TopologyDiscovery`-based, attach-safe
  - `chip_access_metal.cpp` — for library mode
  - `sampler_reader.cpp` — L1 ring reads, delta math
  - `ewma.hpp` — rolling-average helpers
  - `shm_publisher.cpp` — mirrors `memory_stats_shm.hpp`
  - `main.cpp` — wires it up, runs the loop
- `tt_metal/tools/ttnvtop/lib/` — optional library variant (`ttnvtop::start()` / `stop()`)

### Viewer
- `tt_metal/tools/ttnvtop/viewer/` — pure SHM reader, no UMD
  - `main.cpp`, `tui.cpp`, `shm_reader.cpp`

### Build
- `tt_metal/tools/ttnvtop/CMakeLists.txt` — targets: `ttnvtop-collector`, `ttnvtop`, `libttnvtop_collector.a`

### Tests
- `tests/tt_metal/tt_metal/tools/ttnvtop/` — gtest suite
  - `test_fpu_microbench.cpp` — known-activity kernel, asserts monitor output
  - `test_overhead_matmul.cpp` — throughput diff with/without monitor
  - `test_mux_rotation.cpp` — counter-group completeness
  - `test_coexist_pytest.cpp` — shell-driven: start monitor, start workload, verify no lock contention

---

## 10. Data correctness & signal accuracy

The subtle risks that need explicit validation:

1. **Counter saturation.** Counters are 32-bit; at 1 GHz FPU running flat out, saturation in ~4.3 s. Sampling at 10 kHz → deltas are always < 100k, far from saturation. OK. Document the bound.
2. **Mux rotation artifact.** During the tick we switch mux, the counter for the *previous* group is frozen but the next group resumes counting. Deltas stay correct as long as the delta is taken within-group. Verify with a synthetic test that runs only UNPACK and confirm `pack_busy% == 0`.
3. **Kernel boundaries.** When a kernel finishes and another starts, brisc state may reset. Collector detects via `host_assigned_id` change and discards one tick's deltas.
4. **Free-running vs kernel-scoped counters.** Some existing `kernel_profiler` code uses the same counters. Under `TT_METAL_UTIL_MONITOR=1`, kernel_profiler values will look different. Document the tradeoff; make them mutually exclusive.
5. **NOC-ring counters semantics.** Those count *traversals at a router*, not end-to-end bytes. The unit shown in the TUI is "traversals/s" or "MB/s assuming 32 B packets" — be honest in the label.

---

## 11. Performance budget — where we're allowed to spend

| Component | Budget | How we stay under it |
|---|---|---|
| Sampler ISR on device | ≤ 0.1% AICLK cycles | 30 cycles × 10 kHz = 0.03% at default |
| L1 stolen per core | ≤ 1 KiB | Fixed 1 KiB region |
| PCIe reads per tick | ≤ 1 per core | 1 KiB block read per core |
| Host CPU at defaults | ≤ 2% of one core | 128 reads × 2 µs × 10 Hz |
| SHM size per chip | ≤ 4 KiB | ≈ 3 KiB at 128 cores |
| Additional launch overhead per kernel | 0 cycles | Sampler is independent of kernel boundaries |

---

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Firmware change breaks existing kernel_profiler | medium | Opt-in via env var; CI test that default-mode is byte-identical to pre-patch |
| L1 1 KiB reservation conflicts with user allocations | low | Carve from existing MEM_MAP_END reserved area (watcher/profiler already do this) |
| Counter free-run breaks a user who relied on kernel-reset behavior | medium | Gated by env var; document in release notes |
| Multi-process workload sees sampler overhead spikes | low | Sampler runs at fixed 10 kHz; deterministic cost |
| Remote chip reads over ETH are slower than mmio | certain, minor | Per-chip thread so remote chip latency doesn't block local chip; budget 2× for remote |
| BH counter register map differs more than expected | medium | Phase 2 WH-first, BH next; don't block v1 WH release on BH |
| Someone runs 4 ttnvtops simultaneously | low | Viewer is SHM-only; collector should be a singleton (advisory lock on SHM file) |

---

## 13. Work estimate

- Phase 1: ~3 weeks, 1 engineer. Host-heavy.
- Phase 2: ~4 weeks, 1 engineer who's comfortable with firmware. The timer-ISR + mux rotation are the hard parts.
- Phase 3: ~3 weeks, 1 engineer.
- **Total to v1: ~10 weeks / 1 FTE.**

Critical path is the firmware change in Phase 2 — it needs a tt-metal firmware reviewer. Start that conversation early (PR the `util_sampler.h` skeleton + RFC at the end of Phase 1).

---

## 14. Open questions before we start

1. **Who owns the firmware patch?** Does this live in the main tt-metal repo, or as a separately-built overlay? In-repo is cleaner but needs buy-in from the firmware team.
2. **Is there an existing tt-metal monitoring RFC to align with?** Don't want to duplicate effort with something already on a roadmap.
3. **Is `tt-mgmt`'s SHM layout the right base to extend, or a separate namespace (`/dev/shm/tt_device_*_util` vs adding fields to `_memory`)?** Recommend keeping them separate for clean ownership but worth confirming.
4. **Default on or default off?** Recommend default-off behind `TT_METAL_UTIL_MONITOR=1` for v1; consider default-on in a later release after field validation.
5. **Is there interest in exporting to Grafana from day one?** If yes, Phase 3 should include the Prometheus exporter; if no, defer to Phase 4.

---

## 15. Relationship to the existing PoC

The current `ttnvtop` binary at `tt_metal/tools/ttnvtop/` (dispatch-occupancy via `go_msg.signal`) is Phase 0 of this plan. It is the signal that ships as the **fallback path** when the firmware sampler is absent or disabled. The viewer code in §7.3 is a direct evolution of the current two-column TUI — the only change is which SHM it reads from and how many segments each bar has.

This means:
- Nothing in the current PoC is throwaway.
- Users who can't or don't want to enable the firmware patch still get dispatch-occupancy.
- Users who enable it get true compute%, through the same UI.
- The `umd::TopologyDiscovery` coexistence trick survives unchanged across all phases.

---

## Appendix A — Architecture diagram from the pre-plan design discussion (for reference)

```
          ── DATA FLOW PER TICK (e.g. 100 ms) ──

 T=0 ms   HW/FW sampler on each core autonomously snapshots counters
          every 100k cycles and writes 64B records into that core's
          L1 ring (runs continuously, zero host involvement).

 T=100 ms Host collector wakes. For each core:
            read_from_device(ring_addr, 1024 B)  ─┐
          128 cores × ~2 µs PCIe ≈ 260 µs total   │
                                                  ▼
          Diff consecutive snapshots → per-pipeline rates.
          EWMA smooth over last ~10 ticks.
                                                  │
                                                  ▼
          Publish to /dev/shm/tt_device_<id>_util.
          Viewer processes reading SHM redraw TUI.

 T=200 ms Repeat.
```
