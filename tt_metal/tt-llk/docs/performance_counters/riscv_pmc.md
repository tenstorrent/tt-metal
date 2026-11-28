# RISC-V performance counters

Reference for the performance monitor inside the RISC-V cores of a Tensix tile and for `TT_METAL_PROFILE_RISCV_PMC`, the tt-metal runtime option that captures it in the device profiler. It follows the structure of the Tensix counter documentation: [hardware_reference.md](hardware_reference.md) covers the Tensix MMIO counters, this file the monitor that sits inside the cores themselves; section 9 compares the two.

## 1. Support matrix

| Architecture | Core | Counters present | Captured by `TT_METAL_PROFILE_RISCV_PMC` |
|---|---|---|---|
| Blackhole | BRISC, NCRISC, TRISC0, TRISC1, TRISC2 | yes: `mcycle`, `minstret`, `mhpmcounter3/4` | yes |
| Blackhole | active ethernet tile ERISC0 | yes: same core and CSRs as the baby cores | yes |
| Quasar | TRISC0..3 of each Neo | yes: same CSR set and event layout, two extra IF events (section 6) | yes |
| Quasar | data-movement cores DM0..DM7 | yes: `mcycle`, `minstret`, four programmable counters `mhpmcounter3..6`, set-and-mask event selects (section 7) | yes, DM0 and the DMs that run a kernel (DM2..DM7) |
| Wormhole | all RISC-V cores | no: the cores implement no performance counter CSRs | no |
| Blackhole | idle ethernet tile, subordinate ERISC1, DRAM tile RISC | same core design | no, see below |

Blackhole is documented in the public ISA documentation: its baby-core CSR list contains `mcycle`,
`minstret`, `mhpmcounter3/4`, `mhpmevent3/4` and `mcountinhibit` at the addresses used here, and says
`mcountinhibit` can inhibit the HPM counters but not `mcycle` or `minstret`. The counters have been
checked against the kernel wall-clock zone on silicon.

Quasar TRISCs use the same core design as the Blackhole baby cores, with the same CSR addresses. The
Quasar data-movement cores are 64-bit cores with the standard RISC-V counter CSRs. The Quasar firmware
runs both in machine mode. The Quasar capture was run on the Quasar emulator (a known-count test: two zones that differ by exactly 10000
instructions on the four TRISCs); it has not been run on Quasar silicon.

Wormhole RISC-V cores have no `mcycle`, `minstret` or `mhpm*` CSRs, so there is nothing to capture, and
the option fails the JIT build on Wormhole instead of producing empty data.

The Blackhole ethernet cores are the same baby core as BRISC and NCRISC: the same CSR file, the same
`mhpmevent` nibble layout and the same event tables. The kernel on the active ethernet ERISC0 is captured and
its records land in the profiler buffer the host reads for the ethernet core (profiler name `ERISC`). The idle
ethernet firmware and the subordinate ERISC1 firmware have no hook: the host reads the device profiler of
active ethernet cores only, and only the first profiler buffer of each, so markers from either would never
reach the log. The DRAM tile RISC has the same core but no hook: the host never reads a DRAM tile profiler
and the DRAM firmware does not set one up.

## 2. Where the definitions live

The headers are in tt-llk next to the Tensix counter headers (`tools/include/perf_counters/riscv_pmc_*.h`, namespace `llk::perf::riscv`) and the host side is `tools/python/tt_llk_perf/`; the two READMEs there list what each file holds. The metal side is `tt_metal/hw/inc/internal/profiler/riscv_pmc.hpp`: marker ids, payload layout and emission.

A record carries the raw `mhpmevent` select, not an enumerator ordinal, because the select format is the hardware's own and a run may OR several events. The host decodes names by looking selects up in the tables, so `PmcEvent` only has to stay append only.

Adding an event: append the enumerator to `PmcEvent`, add one `{PmcEvent::NAME, select}` entry to the right table and bump the array size. The host parsers read the headers, so nothing else changes.

## 3. Using it in tt-metal

```bash
export TT_METAL_PROFILE_RISCV_PMC=0x10110111,0xA0000000   # mhpmevent3, mhpmevent4
```

One or two selects, hex or decimal, comma separated; `mhpmevent4` defaults to 0. The selects use the
nibble layout of section 5 and apply to every core except the Quasar data-movement cores, which take
their own one to four selects in `TT_METAL_PROFILE_RISCV_PMC_DM` (section 7). Left unset, the DM counters stay idle and
only `mcycle` and `minstret` are captured on those cores. Setting the option
forces the device profiler on. The selects are compile-time constants of the kernel and firmware builds and are
part of the build key, so changing them rebuilds both. Both selects zero is rejected.

Each core emits three `TS_DATA` markers per kernel launch (five on a Quasar data-movement core with counter 5 or 6 armed), payload in the `data` column of `profile_log_device.csv`: the counter deltas, the `mcycle` and `minstret` deltas and the selects. `tools/python/tt_llk_perf/riscv_pmc.py` has the layout.

On a Blackhole Tensix core the deltas cover `kernel_main()` (`run_kernel()` on a TRISC), after the go message, and a core that runs no kernel in a launch publishes nothing for it; dispatch kernels are not captured. On an ethernet core they cover the kernel call only. On Quasar DM0, which runs no kernel itself, they cover the window from the TRISC launch to the end of the dataflow-buffer drain. The selects are re-emitted every
launch so each capture says which events it counted. The room for the markers is held back from the
kernel's own zones at launch, so a kernel that fills the profiler buffer still gets its markers; only a buffer that
was already full before the launch drops them. The Tensix counter multi-pass flow keeps the
markers from pass 0 only, which is correct because the selects do not change between passes.

Only one event pair is captured per run (two pairs on a Quasar data-movement core); a sweep over several pairs needs one run per pair. Without
the option the hooks compile to nothing.

## 4. CSR map

| Address | CSR | Notes |
|---|---|---|
| `0xB00` / `0xB80` | `mcycle` / `mcycleh` | Free-running, writable, not gated by `mcountinhibit`. `0xC00` is a read/write alias. |
| `0xB02` / `0xB82` | `minstret` / `minstreth` | Counts dispatched instructions; not gated by `mcountinhibit`. `0xC02` is an alias. |
| `0xB03` / `0xB83` | `mhpmcounter3` / `3h` | Programmable counter. |
| `0xB04` / `0xB84` | `mhpmcounter4` / `4h` | Programmable counter. |
| `0xB05`, `0xB06` | `mhpmcounter5`, `mhpmcounter6` | Programmable counters of the Quasar data-movement cores only. |
| `0x323`, `0x324` | `mhpmevent3`, `mhpmevent4` | Event selects, section 5. |
| `0x325`, `0x326` | `mhpmevent5`, `mhpmevent6` | Event selects of the Quasar data-movement cores only, section 7. |
| `0x320` | `mcountinhibit` | Only bits 3 and 4 are implemented. Not written by tt-metal. |
| `0x7C0` | `cfg` (custom) | Bit 8 clear-on-read, bit 7 saturate instead of wrap. Firmware programs other bits of it at boot; the capture reads deltas and never touches it. |

CSRs that are not implemented read as zero; there is no illegal-instruction trap. `mhpmevent5..8`
exist as addresses only and read zero on the baby cores; on the Quasar data-movement cores `mhpmevent5` and `mhpmevent6` are live.

## 5. Programming model

An `mhpmevent` value is eight 4-bit nibbles, one per pipeline domain. Each nibble selects one event
index from that domain's 16-bit event bus.

| `mhpmevent` bits | Domain |
|---|---|
| `[3:0]` | BP: branch predictor |
| `[7:4]` | IF: instruction fetch |
| `[11:8]` | DE: decode stalls, hazards, memory dispatch |
| `[15:12]` | DE1: decode instruction mix |
| `[19:16]` | EX: retire and mispredict |
| `[23:20]` | LS: load/store unit |
| `[27:24]` | TTCSR: Tensix status CSRs held high |
| `[31:28]` | MEM: local memory, L1, instruction cache port |

Index 0 of every domain is a tie-off, and a zero nibble disables the domain. The selected events of
all domains are OR-ed before the increment, so a counter advances by at most 1 per cycle and cannot
tell which of several selected events fired.

Examples: `0x10110111` selects stall-any in BP, IF, DE, EX, LS and MEM, a union of every pipeline
stall. `0xA0000000` is MEM index 10, `MEM_INS_BUF_WRITE`, one Tensix instruction pushed.
`0x0000D000` is DE1 index 13, `DE1_VALID_TT_INSTR`. `0x00030000` is EX index 3, all mispredicts.
`0xE0000000` and `0xF0000000` are the instruction cache access and hit events.

## 6. Events

`riscv_pmc_blackhole.h` lists every event of the baby cores with its select and a one-line meaning; Quasar TRISCs have the same events plus the two in `trisc_extra_events` of `riscv_pmc_quasar.h` (IF 7 and 8). The select of an event is its domain index shifted into the domain's nibble.

On Blackhole silicon:

- `EX_STALL_ANY` is always zero: hazards are resolved at decode.
- The TTCSR domain is not useful: queue status is non-zero on every cycle of a busy workload, busy status within a fraction of a percent of it, and status CSRs 0 and 1 read zero. On Quasar the status CSRs 0..7 hold the Neo and TRISC index, so those events are constants.
- `MEM_DATA_RD_REQ_L1` and `MEM_DATA_RD_REQ_HIT_LOCACHE` do not track the core's own L1 reads; `MEM_DATA_L1_RD_ONGOING` does.

## 7. Quasar data-movement cores

The DM cores use the standard 64-bit counters at the same CSR addresses as above (`mcycle`, `minstret`,
`mhpmcounter3..6`, `mhpmevent3..6`), four programmable counters instead of two. Counters 3 and 4 use
markers 9092 and 9094 like every other core; counters 5 and 6 use markers 9095 and 9096, emitted only when
select 5 or 6 is non-zero, so a run that arms two counters has the three-marker format. Differences from the
baby cores:

- `mhpmevent` is `(event mask << 8) | event set`. The counter advances when any masked event of the set
  fires. Set them with `TT_METAL_PROFILE_RISCV_PMC_DM=<mhpmevent3>[,<mhpmevent4>[,<mhpmevent5>[,<mhpmevent6>]]]`.
  A missing select leaves its counter idle.
- `mcycle` does not count while the core waits in WFI.
- The HPM counters are 40 bits and are not reset: the capture zeroes them at the start of each launch.
- A counter increments one cycle after its event.

The select of one event is `((1 << bit) << 8) | set`; OR the masks of events of the same set. `dm_events` in `riscv_pmc_quasar.h` lists the events of sets 0 (commits by class), 1 (pipeline interlocks and flushes) and 2 (cache misses and releases).

Bits 3 to 5 of set 2 are TLB events, which read zero because there is no virtual memory, and are not in the table. Examples:
`0x201` counts long-latency interlocks, `0x801` instruction cache blocked cycles, `0x1001` data cache
blocked cycles, `0x202` data cache misses.

Four counters let a run read two event sets at once, for example `0x201,0x801,0x202,0x402` counts long-latency
interlocks, instruction cache blocked cycles, data cache misses and data cache releases in one launch.

Which DMs are captured: DM0 (window above) and every DM that runs a kernel. DM1 only configures the
dataflow-buffer remapper and is not captured. The Quasar dispatch firmware is not captured.

## 8. Semantics and traps

- **Increment is an OR.** Several selected events count at most 1 per cycle.
- **`mcountinhibit` on Blackhole.** The shared event enable tests bit 3 for both counters, so writing
  bit 3 freezes both counters. Bit 4 gates counter 4 only. To stop counter 3 alone, select no event
  (`mhpmevent3 = 0`). Quasar uses bit 4 for counter 4. tt-metal never writes `mcountinhibit`; its reset
  value runs both.
- **Deltas are 32 bits.** `mhpmcounter3/4` are 64 bits with a sticky overflow bit not visible from
  software, but the capture keeps the low words, so a delta is modulo 2^32 and `mcycle` wraps after about
  3 s at 1.35 GHz. That is far above a Tensix launch, but persistent ethernet kernels such as fabric routers
  live for the whole program and their capture is not meaningful.
- **Clear-on-read is per half-word** and a 64-bit read cannot be atomic. The capture takes deltas of the
  low words instead.
- **Events are sampled one cycle late and only while armed.** Ignore the first cycle after arming.
- **Clock gating.** On Blackhole a TRISC's `mcycle` can freeze while the core is blocked on a Tensix PC
  buffer read, so it can be shorter than wall time on a core that mostly waits for the Tensix; BRISC and
  NCRISC follow wall time. On Quasar a TRISC is gated only when completely idle.
- **The capture itself** reads four CSRs per launch, which registers as a few serializing stalls.

## 9. Not the same thing as the Tensix perf counters

| | RISC-V PMC (this document) | Tensix perf counters |
|---|---|---|
| Access | CSR instruction from the core itself | MMIO debug registers |
| Scope | per core, private | shared Tensix resource, banked by unit |
| Observes | fetch, decode, execute, load/store, instruction cache, L1 port of one core | FPU, packers, unpackers, instruction path, L1 |
| Option | `TT_METAL_PROFILE_RISCV_PMC` | `TT_METAL_PROFILE_PERF_COUNTERS` |

Both can be on in one run. The profiler's multi-pass flow keeps the PMC rows from the first pass. Rows
join on core coordinates.

## 10. Reading the markers on the host

```python
from tt_llk_perf import riscv_pmc

launch = riscv_pmc.unpack_launch(counts_9092, cycles_9093, events_9094)
kind = riscv_pmc.core_kind("blackhole", "TRISC_1")        # "baby"
riscv_pmc.decode_launch("blackhole", kind, launch)         # event names, mcycle, minstret, derived metrics
```

`unpack_launch` takes the two upper payloads as optional fourth and fifth arguments and `decode_launch` then also names `event5` and `event6` on data-movement cores. `riscv_metrics.compute_riscv_metrics(cycles, instret, counts)` gives `riscv_ipc_ratio`, `riscv_stall_pct`, `riscv_mem_stall_per_tti_push_ratio` (needs `MEM_STALL_ANY` and `MEM_INS_BUF_WRITE` armed in the same run) and, per armed event, `riscv_<event>_pct` (per cycle) and `riscv_<event>_per_instr_ratio`. Absent inputs read `None`; `riscv_stall_pct` needs the stall union of section 5 armed on one counter. `perf_counter_analysis.decode_riscv_pmc_markers(rows, arch)` pairs the marker streams of every core on every device (three, or five on a Quasar data-movement core with counter 5 or 6 armed) and returns a dataframe; it takes marker rows from a device log with the `PCIe slot` column as `device_id` and is not part of the ops report, which only carries the Tensix counter markers (id 9090).

## 11. Open questions

- Quasar TRISC and DM capture on silicon; only the emulator has run it.
- Idle ethernet and ERISC1 capture need the host profiler to read those cores first.
- The meaning of individual per-stage events comes from the hardware description; only the union counter, `mcycle`, `EX_INST_RET` and the TRISC-only `MEM_INS_BUF_WRITE` were validated on silicon.
