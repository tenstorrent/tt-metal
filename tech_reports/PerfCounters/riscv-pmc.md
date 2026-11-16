# RISC-V performance counters in the Tensix baby cores

Reference for the hardware performance monitor that sits inside the baby RISC-V cores of a Tensix
tile, and for `TT_METAL_PROFILE_RISCV_PMC`, the runtime option that captures it in the device
profiler. This is a different mechanism from the Tensix MMIO counters behind
`TT_METAL_PROFILE_PERF_COUNTERS` (see [perf-counters.md](perf-counters.md)); section 8 compares them.

## 1. Support matrix

| Architecture | Core | Counters present | Captured by `TT_METAL_PROFILE_RISCV_PMC` |
|---|---|---|---|
| Blackhole | BRISC, NCRISC, TRISC0, TRISC1, TRISC2 | yes: `mcycle`, `minstret`, `mhpmcounter3/4` | yes |
| Quasar | TRISC0..3 of each Neo | yes: same CSR set and event layout, two extra IF events (section 5) | yes |
| Quasar | data-movement cores | yes, but a different design (event-set plus mask selects, four counters) | no, see below |
| Wormhole | BRISC, NCRISC, TRISC0/1/2 | not established | no |
| Wormhole, Blackhole | Ethernet cores (ERISC) | not established | no |

Blackhole is documented in the public ISA documentation: its baby-core CSR list contains `mcycle`,
`minstret`, `mhpmcounter3/4`, `mhpmevent3/4` and `mcountinhibit` at the addresses used here, and says
`mcountinhibit` can inhibit the HPM counters but not `mcycle` or `minstret`. The counters have been
checked against the kernel wall-clock zone on silicon (section 10).

Quasar TRISCs use the same core design as the Blackhole baby cores, with the same CSR addresses. The
Quasar firmware already runs these cores in machine mode and programs the same custom cfg CSR
(`0x7C0`) that Blackhole firmware does. The capture builds and links for every Quasar TRISC but has not
been run on Quasar silicon or a simulator.

Quasar data-movement cores are 64-bit cores with four HPM counters (`mhpmcounter3..6`) whose
`mhpmevent` value is an event-set id plus a bit mask of events, not the nibble layout of section 4.
They need their own select format and a hook in `dm.cc`; that is not part of this option.

Wormhole: the public ISA documentation lists no CSRs for the baby cores, nothing in tt-metal touches
`mcycle` or `mhpm*` on Wormhole, and no hardware description of the Wormhole baby core was available
to confirm the block. Unimplemented CSRs read zero instead of trapping on Blackhole (section 3), so a
probe on silicon would not fail loudly: a Wormhole run counts as support only if the `mhpmcounter3/4`
deltas are non-zero and depend on the selected event. With the option set, a Wormhole build fails with
an error instead of producing empty data. For Ethernet cores no evidence was found either way.

## 2. Using it

```bash
export TT_METAL_PROFILE_RISCV_PMC=0x10110111,0xA0000000   # mhpmevent3, mhpmevent4
```

One or two selects, hex or decimal, comma separated; `mhpmevent4` defaults to 0. Setting the option
forces the device profiler on. The selects are compile-time constants of the baby-core firmware and are
part of the firmware build key, so changing them rebuilds the firmware. Both selects zero is rejected.

Each core emits three `TS_DATA` markers per kernel launch, all payload in the `data` column of
`profile_log_device.csv`:

| Marker id | `data` |
|---|---|
| 9092 | `(mhpmcounter4 delta << 32) \| mhpmcounter3 delta` |
| 9093 | `(minstret delta << 32) \| mcycle delta` |
| 9094 | `(mhpmevent4 << 32) \| mhpmevent3` |

The deltas cover the kernel invocation in the firmware (waypoint `R` to `D`). On Blackhole BRISC the
bracket also covers `wait_for_go_message()` when BRISC has no kernel. The selects are re-emitted every
launch so each capture says which events it counted. If the per-core profiler buffer is full the three
markers are dropped rather than displacing zone markers. The Tensix counter multi-pass flow keeps the
markers from pass 0 only, which is correct because the selects do not change between passes.

Firmware cost with the option on: BRISC grows by about 250 B, NCRISC by about 200 B, the Blackhole
TRISCs by about 215 B and the Quasar TRISCs by 256 to 276 B. The Blackhole BRISC region is 9472 B so
that the hook still links next to the five Tensix counter groups (9336 B worst case).

Only one event pair is captured per run; a sweep over several pairs needs one run per pair. Without
the option the hooks compile to nothing and the firmware images are byte-identical to a build without
this feature.

## 3. CSR map

| Address | CSR | Notes |
|---|---|---|
| `0xB00` / `0xB80` | `mcycle` / `mcycleh` | Free-running, writable, not gated by `mcountinhibit`. `0xC00` is a read/write alias. |
| `0xB02` / `0xB82` | `minstret` / `minstreth` | Counts dispatched instructions; not gated by `mcountinhibit`. `0xC02` is an alias. |
| `0xB03` / `0xB83` | `mhpmcounter3` / `3h` | Programmable counter. |
| `0xB04` / `0xB84` | `mhpmcounter4` / `4h` | Programmable counter. |
| `0x323`, `0x324` | `mhpmevent3`, `mhpmevent4` | Event selects, section 4. |
| `0x320` | `mcountinhibit` | Only bits 3 and 4 are implemented. Not written by tt-metal. |
| `0x7C0` | `cfg` (custom) | Bit 8 clear-on-read, bit 7 saturate instead of wrap. Firmware programs other bits of it at boot; the capture reads deltas and never touches it. |

CSRs that are not implemented read as zero; there is no illegal-instruction trap. `mhpmevent5..8`
exist as addresses only and read zero.

## 4. Programming model

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
stall. `0xA0000000` is MEM index 10, `memPmcInsBufWrite`, one Tensix instruction pushed.
`0x0000D000` is DE1 index 13, `dePmcDeValidTTInstr`. `0x00030000` is EX index 3, all mispredicts.
`0xE0000000` and `0xF0000000` are the instruction cache access and hit events.

## 5. Event catalog

Names follow the hardware's own naming. Indices are the nibble value to put in the domain.

BP, nibble `[3:0]`

| Idx | Event | Meaning |
|---|---|---|
| 1 | `bpPmcPcStallAny` | fetch PC stalled for any reason |
| 2 | `bpPmcBpStallNoIdRtr` | stalled, decode not ready |
| 3 | `bpPmcBpStallNoMemRtr` | stalled, instruction memory not ready |
| 4 | `bpPmcBpStallPcFifoFull` | stalled, PC FIFO full or nearly full |
| 5 | `bpPmcBpStallBpFifoFull` | stalled, predictor FIFO full or nearly full |
| 6 | `bpPmcBtbHit` | branch target buffer hit |
| 7, 8, 9 | `bpPmcPredTypeJalr`, `bpPmcPredTypeJal`, `bpPmcPredTypeBr` | predicted instruction type |

IF, nibble `[7:4]`

| Idx | Event | Meaning |
|---|---|---|
| 1 | `ifPmcIfStallAny` | instruction FIFO empty or not popped |
| 2 | `ifPmcIfStallNoIdRtr` | fetch has an instruction, decode not ready |
| 3 | `ifPmcIfStallLqFull` | fetch has an instruction, load queue full |
| 4 | `ifPmcIfStallLqSbFull` | fetch has an instruction, load skid buffer full |
| 5 | `ifPmcReqStallAny` | instruction request stalled |
| 6 | `ifPmcReqStallNoMemRtr` | request stalled, instruction memory not ready |
| 7 | `ifPmcIfStallNoIfRts` | Quasar only: front end starved, decode ready but nothing to give |
| 8 | `ifPmcIdStallLd` | Quasar only: decode stalled on a load-use hazard |

DE, nibble `[11:8]`

| Idx | Event | Meaning |
|---|---|---|
| 1 | `dePmcDeStallAny` | decode has an instruction but cannot issue it |
| 2 | `dePmcDeStallNoUnitRtr` | execution unit not ready |
| 3 | `dePmcDeStallIfEmpty` | nothing to decode |
| 4 | `dePmcDeStallSync` | serializing operation draining the load queue |
| 5 | `dePmcDeStallHazard` | register hazard |
| 6, 7, 8 | `dePmcHazardInt`, `dePmcHazardFp`, `dePmcHazardVec` | hazard by register file |
| 9 | `dePmcRegDataForward` | operand forwarding used |
| 10, 11 | `dePmcMemOpDispatchLd`, `dePmcMemOpDispatchSt` | scalar load, store dispatched |
| 12, 13 | `dePmcMemInstrDispatchVecLd`, `dePmcMemInstrDispatchVecSt` | vector load, store dispatched |
| 14, 15 | `dePmcExOpDispatchMul`, `dePmcExOpDispatchDiv` | multiply, divide dispatched |

DE1, nibble `[15:12]`

| Idx | Event | Meaning |
|---|---|---|
| 1 | `dePmcExOpDispatchCsr` | CSR, ecall, ebreak dispatched |
| 2, 3, 4, 5 | `dePmcExOpDispatchIExt`, `...MExt`, `...AExt`, `...ZbExt` | integer, multiply/divide, atomic, bit-manipulation dispatched |
| 6, 7 | `dePmcFpOpDispatchS`, `dePmcFpOpDispatchZfh` | single, half precision FP dispatched |
| 8..12 | `dePmcVecOpDispatchCsr`, `...Int`, `...Fixed`, `...Float`, `...Ldst` | vector instruction classes |
| 13 | `dePmcDeValidTTInstr` | Tensix instruction issued by this core |
| 14 | `dePmcDeValidVecReplay` | vector replay |
| 15 | `dePmcDeValidIllegal` | illegal instruction |

EX, nibble `[19:16]`

| Idx | Event | Meaning |
|---|---|---|
| 1 | `exPmcExStallAny` | execute back-pressure; observed always zero on Blackhole, hazards are resolved at decode |
| 2 | `exPmcExStallNoInstr` | execute starved |
| 3 | `exPmcMisPredAny` | any branch mispredict |
| 4, 5 | `exPmcMisPredNewBr`, `exPmcMisPredDir` | branch not in the target buffer, wrong direction |
| 6, 7 | `exPmcMisPredTgtCond`, `exPmcMisPredTgtUncond` | wrong target, conditional, unconditional |
| 8 | `exPmcInstRet` | instruction retired; reproduces `minstret` |

LS, nibble `[23:20]`

| Idx | Event | Meaning |
|---|---|---|
| 1 | `lsPmcLsStallAny` | memory operation valid but not accepted |
| 2 | `lsPmcLsStallNoMemRtr` | memory port not ready |
| 3 | `lsPmcLsStallLdqFull` | load queue full |
| 4 | `lsPmcLsStallStoreBufFull` | store waiting on a full write buffer |
| 5 | `lsPmcLsStallSkidBufFull` | skid buffer full |
| 6 | `lsPmcFenceOp` | fence |
| 7 | `lsPmcLsStallWaitForData` | oldest load waiting for data |
| 8 | `lsPmcOpRet` | load queue retire |
| 9 | `lsPmcStoreReqWins` | store won arbitration over a load |
| 10 | `lsPmcWrBufSTLH` | store-to-load hazard in the write buffer |
| 11 | `lsPmcWrBufSTSC` | store coalescing |
| 12, 13 | `lsPmcLoadOp`, `lsPmcStoreOp` | 32-bit load, store |
| 14, 15 | `lsPmcVecLoadOp`, `lsPmcVecStoreOp` | vector load, store |

TTCSR, nibble `[27:24]`

| Idx | Event | Meaning |
|---|---|---|
| 1..8 | `ttcsrPmcSstatus7High` ... `ttcsrPmcSstatus0High` | status CSR 7..0 non-zero (index 1 is status 7) |
| 9 | `ttcsrPmcBstatusHigh` | Tensix busy status non-zero |
| 10 | `ttcsrPmcQstatusHigh` | Tensix queue status non-zero |

On Blackhole this domain is not useful: queue status is non-zero on every cycle of a busy workload,
busy status within a fraction of a percent of it, and status CSRs 0 and 1 read zero. On Quasar the
status CSRs 0..7 hold the Neo and TRISC index, so those events are constants.

MEM, nibble `[31:28]`

| Idx | Event | Meaning |
|---|---|---|
| 1 | `memPmcStallAny` | memory request not granted |
| 2, 3 | `memPmcLocalRegRdOngoing`, `memPmcMailboxRdOngoing` | local register, mailbox read in flight |
| 4 | `memPmcDataL1RdOngoing` | L1 read in flight |
| 5, 6, 7, 8 | `memPmcCfgRegRdOngoing`, `memPmcRegRdOngoing`, `memPmcTdmaRegRdOngoing`, `memPmcPcBufRdOngoing` | Tensix config, register, TDMA register, PC buffer read in flight |
| 9 | `memPmcDataAccReqLocal` | local memory access accepted |
| 10 | `memPmcInsBufWrite` | Tensix instruction buffer write accepted: one Tensix instruction pushed |
| 11 | `memPmcInsBufWriteMultiple` | multi-word push |
| 12, 13 | `memPmcDataRdReqL1`, `memPmcDataRdReqHitLoCash` | observed not to track this core's L1 reads on Blackhole; use index 4 |
| 14, 15 | `memPmcIFaccess`, `memPmcIFaccessHit` | instruction cache access, hit |

Only TRISC2 on Blackhole is built with the vector unit, so vector events read zero on the other cores.

## 6. Semantics and traps

- **Increment is an OR.** Several selected events count at most 1 per cycle.
- **`mcountinhibit` on Blackhole.** Bit 4 does not gate counter 4: the shared event enable tests bit 3
  for both counters, so writing bit 3 freezes both and writing bit 4 alone leaves counter 4 running.
  Quasar uses bit 4 for counter 4. tt-metal never writes `mcountinhibit`; its reset value runs both.
- **Counters are wide.** `mhpmcounter3/4` are 64 bits with a sticky overflow bit not visible from
  software. The low word of `mcycle` wraps after about 3 s at 1.35 GHz, far above a launch.
- **Clear-on-read is per half-word** and a 64-bit read cannot be atomic. The capture takes deltas of the
  low words instead.
- **Events are sampled one cycle late and only while armed.** Ignore the first cycle after arming.
- **Clock gating.** On Blackhole a TRISC's `mcycle` can freeze while the core is blocked on a Tensix PC
  buffer read, so it can be shorter than wall time on a core that mostly waits for the Tensix; BRISC and
  NCRISC follow wall time. On Quasar a TRISC is gated only when completely idle.
- **The capture itself** reads four CSRs per launch, which registers as a few serializing stalls.

## 7. Telling "blocked on Tensix" from a pipeline stall

The `*StallAny` events see only the RISC-V pipeline. A core waiting for the Tensix datapath is usually
not pipeline-stalled.

- **Tensix instruction back-pressure is a store stall.** A Tensix instruction is a write to the
  instruction buffer address (`memPmcInsBufWrite`). When the Tensix cannot take it, the write is not
  granted: look at `memPmcStallAny` (MEM 1), `lsPmcLsStallStoreBufFull` (LS 4) and `lsPmcLsStallAny`
  (LS 1), not at DE or EX.
- **Semaphore and circular-buffer waits retire instructions.** A wait loop retires at full rate: high
  `minstret` per `mcycle`, high `lsPmcLoadOp`, few stalls. IPC near 1 with no stalls over a long window
  is a spinning core.
- **`dePmcDeStallSync` is a load-queue drain, not a Tensix wait.**
- **Normalize by work, not by cycles.** Per-cycle fractions of different cores share a denominator and
  correlate with each other by construction. Compare counts per Tensix instruction pushed, armed in the
  same run, and compare across configurations rather than across cores at one configuration.

A single-issue pipeline gives `stall_union + IPC ~ 1`, so the union counter adds little over IPC; the
information is in the per-stage events.

## 8. Not the same thing as the Tensix perf counters

| | RISC-V PMC (this document) | Tensix perf counters |
|---|---|---|
| Access | CSR instruction from the core itself | MMIO debug registers |
| Scope | per core, private | shared Tensix resource, banked by unit |
| Observes | fetch, decode, execute, load/store, instruction cache, L1 port of one core | FPU, packers, unpackers, instruction path, L1 |
| Option | `TT_METAL_PROFILE_RISCV_PMC` | `TT_METAL_PROFILE_PERF_COUNTERS` |

Both can be on in one run. The profiler's multi-pass flow keeps the PMC rows from the first pass. Rows
join on core coordinates.

## 9. What changed in tt-metal

| Path | Change |
|---|---|
| `tt_metal/hw/inc/internal/profiler/riscv_pmc.hpp` | CSR arm/read, deltas, marker emission |
| `tt_metal/hw/firmware/src/tt-1xx/{brisc,ncrisc,trisc}.cc` | hooks around the kernel invocation (Blackhole) |
| `tt_metal/hw/firmware/src/tt-2xx/trisc.cc` | hook around the kernel invocation (Quasar) |
| `tt_metal/llrt/rtoptions.{hpp,cpp}` | option parsing, getters, compile-hash suffix |
| `tt_metal/jit_build/build.cpp` | `-DPROFILE_RISCV_PMC{,_EV3,_EV4}`, architecture check |

Nothing was added to the memory map, `dev_msgs.h`, the HAL or the host profiler: the markers use the
existing per-RISC profiler buffer. The hash suffix is added only when the option is on, so existing JIT
caches and pre-compiled firmware are unchanged.

## 10. Measured on Blackhole

- No dropped markers: 110 cores x 5 RISCs x 3 launches gave 1650 rows of each marker.
- The `mcycle` delta agrees with the wall-clock kernel zone to within 0.03 percent on every core type.
- `exPmcInstRet` reproduces the `minstret` delta to 0.999995.
- Stall union plus IPC is 1.00 on every core type.
- `memPmcInsBufWrite` is exactly zero on BRISC and NCRISC and non-zero on the three TRISCs.

## 11. Open questions

- Whether the Wormhole baby cores carry this block, and with which encoding. A Wormhole run with the
  hooks enabled would settle it: arm a known event (for example `0x0000D000` on a compute kernel) and
  check for non-zero, event-dependent counts.
- Quasar TRISC capture on silicon or a simulator.
- Quasar data-movement cores need their own select format and hook.
- The meaning of individual per-stage events comes from the hardware description; only the union
  counter, `mcycle`, `exPmcInstRet` and the TRISC-only `memPmcInsBufWrite` were validated on silicon.
