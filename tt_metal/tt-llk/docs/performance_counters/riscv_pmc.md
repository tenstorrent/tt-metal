# RISC-V performance counters

Reference for the performance monitor inside the RISC-V cores of a Tensix tile and for `TT_METAL_PROFILE_RISCV_PMC`, the tt-metal runtime option that captures it in the device profiler. It follows the structure of the Tensix counter documentation: [hardware_reference.md](hardware_reference.md) covers the Tensix MMIO counters, this file the monitor that sits inside the cores themselves; section 10 compares the two.

## 1. Support matrix

| Architecture | Core | Counters present | Captured by `TT_METAL_PROFILE_RISCV_PMC` |
|---|---|---|---|
| Blackhole | BRISC, NCRISC, TRISC0, TRISC1, TRISC2 | yes: `mcycle`, `minstret`, `mhpmcounter3/4` | yes |
| Quasar | TRISC0..3 of each Neo | yes: same CSR set and event layout, two extra IF events (section 6) | yes |
| Quasar | data-movement cores DM0..DM7 | yes: `mcycle`, `minstret`, `mhpmcounter3..6`, set-and-mask event selects (section 7) | yes, DM0 and the DMs that run a kernel (DM2..DM7) |
| Wormhole | all RISC-V cores | no: the cores implement no performance counter CSRs | no |
| Blackhole | Ethernet and DRAM cores | probably, not captured | no |

Blackhole is documented in the public ISA documentation: its baby-core CSR list contains `mcycle`,
`minstret`, `mhpmcounter3/4`, `mhpmevent3/4` and `mcountinhibit` at the addresses used here, and says
`mcountinhibit` can inhibit the HPM counters but not `mcycle` or `minstret`. The counters have been
checked against the kernel wall-clock zone on silicon (section 12).

Quasar TRISCs use the same core design as the Blackhole baby cores, with the same CSR addresses. The
Quasar data-movement cores are 64-bit cores with the standard RISC-V counter CSRs. The Quasar firmware
runs both in machine mode. The Quasar capture builds and links for every core above but has not been run
on Quasar silicon or a simulator.

Wormhole RISC-V cores have no `mcycle`, `minstret` or `mhpm*` CSRs, so there is nothing to capture, and
the option fails the JIT build on Wormhole instead of producing empty data.

The Blackhole Ethernet cores and DRAM cores run the same `configure_csr()` as the Tensix baby cores, which
writes the custom cfg CSR `0x7C0`, so they most likely carry the same CSR block. No hook is added for
them.

## 2. Where the definitions live

The hardware description is in tt-llk, next to the Tensix counter headers (`tools/include/perf_counters/`, namespace `llk::perf::riscv`, self-contained):

| Header | Holds |
|---|---|
| `riscv_pmc_types.h` | `CoreKind` (`BABY`, `TRISC`, `DM`: how the select is encoded), `PmcEvent` (event names, append only), `Entry` = `{event, select}` |
| `riscv_pmc_blackhole.h`, `riscv_pmc_quasar.h`, `riscv_pmc_wormhole.h` | The per-core-kind select tables: `baby_events` (Blackhole), `trisc_events` and `dm_events` (Quasar), an empty `baby_events` on Wormhole, which has no counters. The Quasar tables carry `LLK_PERF_TABLE_SECTION` like `quasar.h` |
| `riscv_pmc_inventory.h` | Picks the arch header from `ARCH_*` and exposes `events_for(CoreKind)` |
| `riscv_pmc_hw.h` | The CSR addresses and the primitives `select(event3, event4)`, `zero_counters()`, `read_mcycle()`, `read_minstret()`, `read_counter3()`, `read_counter4()`. Not available on Wormhole (`#error`) |

The metal consumer is `tt_metal/hw/inc/internal/profiler/riscv_pmc.hpp`: marker ids, payload layout and emission only. The host side is `tools/python/tt_llk_perf/`: `headers.py` parses the tables (`riscv_event_tables(arch)`, `riscv_select_events(arch, kind, select)`, `riscv_event_select`), `riscv_pmc.py` decodes the marker payloads and `riscv_metrics.py` derives IPC, stall and per-instruction metrics. `tools/tracy/perf_counter_analysis.py` has `decode_riscv_pmc_markers`.

Unlike the Tensix counters, a record carries the raw `mhpmevent` select, not an enumerator ordinal, because the select format is the hardware's own and a run may OR several events. The host decodes names by looking selects up in the tables, so `PmcEvent` has no wire-format constraint beyond being append only for stable names.

The firmware hooks are in `tt_metal/hw/firmware/src/tt-1xx/{brisc,ncrisc,trisc}.cc` and `tt-2xx/{trisc,dm}.cc`; the option is parsed in `tt_metal/llrt/rtoptions.cpp` and turned into `-DPROFILE_RISCV_PMC{,_EV3,_EV4,_DM_EV3,_DM_EV4}` in `tt_metal/jit_build/build.cpp`. Nothing was added to the memory map, `dev_msgs.h` or the HAL: the markers use the existing per-RISC profiler buffer. The hash suffix is added only when the option is on.

Adding an event: append the enumerator to `PmcEvent`, add one `{PmcEvent::NAME, select}` entry to the right table and bump the array size. The host parsers read the headers, so nothing else changes.

## 3. Using it in tt-metal

```bash
export TT_METAL_PROFILE_RISCV_PMC=0x10110111,0xA0000000   # mhpmevent3, mhpmevent4
```

One or two selects, hex or decimal, comma separated; `mhpmevent4` defaults to 0. The selects use the
nibble layout of section 5 and apply to every core except the Quasar data-movement cores, which take
their own pair in `TT_METAL_PROFILE_RISCV_PMC_DM` (section 7). Left unset, the DM counters stay idle and
only `mcycle` and `minstret` are captured on those cores. Setting the option
forces the device profiler on. The selects are compile-time constants of the baby-core firmware and are
part of the firmware build key, so changing them rebuilds the firmware. Both selects zero is rejected.

Each core emits three `TS_DATA` markers per kernel launch, all payload in the `data` column of
`profile_log_device.csv`:

| Marker id | `data` |
|---|---|
| 9092 | `(mhpmcounter4 delta << 32) \| mhpmcounter3 delta` |
| 9093 | `(minstret delta << 32) \| mcycle delta` |
| 9094 | `(mhpmevent4 << 32) \| mhpmevent3` |

The deltas cover the kernel invocation in the firmware (waypoint `R` to `D`). On Quasar DM0, which runs
no kernel itself, they cover the window from the TRISC launch to the end of the dataflow-buffer drain. On Blackhole BRISC the
bracket also covers `wait_for_go_message()` when BRISC has no kernel. The selects are re-emitted every
launch so each capture says which events it counted. If the per-core profiler buffer is full the three
markers are dropped rather than displacing zone markers. The Tensix counter multi-pass flow keeps the
markers from pass 0 only, which is correct because the selects do not change between passes.

Firmware cost with the option on: BRISC grows by about 250 B, NCRISC by about 200 B, the Blackhole
TRISCs by about 215 B, the Quasar TRISCs by 256 to 276 B and the Quasar DM image by about 740 B (8048 B
of 12288 B). The Blackhole BRISC region is 9472 B so that the hook still links next to the five Tensix
counter groups (9336 B worst case).

Only one event pair is captured per run; a sweep over several pairs needs one run per pair. Without
the option the hooks compile to nothing and the firmware images are byte-identical to a build without
this feature.

## 4. CSR map

| Address | CSR | Notes |
|---|---|---|
| `0xB00` / `0xB80` | `mcycle` / `mcycleh` | Free-running, writable, not gated by `mcountinhibit`. `0xC00` is a read/write alias. |
| `0xB02` / `0xB82` | `minstret` / `minstreth` | Counts dispatched instructions; not gated by `mcountinhibit`. `0xC02` is an alias. |
| `0xB03` / `0xB83` | `mhpmcounter3` / `3h` | Programmable counter. |
| `0xB04` / `0xB84` | `mhpmcounter4` / `4h` | Programmable counter. |
| `0x323`, `0x324` | `mhpmevent3`, `mhpmevent4` | Event selects, section 5. |
| `0x320` | `mcountinhibit` | Only bits 3 and 4 are implemented. Not written by tt-metal. |
| `0x7C0` | `cfg` (custom) | Bit 8 clear-on-read, bit 7 saturate instead of wrap. Firmware programs other bits of it at boot; the capture reads deltas and never touches it. |

CSRs that are not implemented read as zero; there is no illegal-instruction trap. `mhpmevent5..8`
exist as addresses only and read zero.

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

## 6. Event catalog

Names are the `PmcEvent` enumerators in `tools/include/perf_counters/riscv_pmc_types.h`; the select is the value to OR into `mhpmevent`. Indices are the nibble value of the domain. Quasar TRISC cores have two more IF events (7 and 8).

BP, nibble `[3:0]`

| Idx | Event | Select | Meaning |
|---|---|---|---|
| 1 | `BP_PC_STALL_ANY` | `0x1` | fetch PC stalled for any reason |
| 2 | `BP_STALL_NO_ID_RTR` | `0x2` | stalled, decode not ready |
| 3 | `BP_STALL_NO_MEM_RTR` | `0x3` | stalled, instruction memory not ready |
| 4 | `BP_STALL_PC_FIFO_FULL` | `0x4` | stalled, PC FIFO full or nearly full |
| 5 | `BP_STALL_BP_FIFO_FULL` | `0x5` | stalled, predictor FIFO full or nearly full |
| 6 | `BP_BTB_HIT` | `0x6` | branch target buffer hit |
| 7 | `BP_PRED_TYPE_JALR` | `0x7` | predicted instruction type |
| 8 | `BP_PRED_TYPE_JAL` | `0x8` | predicted instruction type |
| 9 | `BP_PRED_TYPE_BR` | `0x9` | predicted instruction type |

IF, nibble `[7:4]`

| Idx | Event | Select | Meaning |
|---|---|---|---|
| 1 | `IF_STALL_ANY` | `0x10` | instruction FIFO empty or not popped |
| 2 | `IF_STALL_NO_ID_RTR` | `0x20` | fetch has an instruction, decode not ready |
| 3 | `IF_STALL_LQ_FULL` | `0x30` | fetch has an instruction, load queue full |
| 4 | `IF_STALL_LQ_SB_FULL` | `0x40` | fetch has an instruction, load skid buffer full |
| 5 | `IF_REQ_STALL_ANY` | `0x50` | instruction request stalled |
| 6 | `IF_REQ_STALL_NO_MEM_RTR` | `0x60` | request stalled, instruction memory not ready |

DE, nibble `[11:8]`

| Idx | Event | Select | Meaning |
|---|---|---|---|
| 1 | `DE_STALL_ANY` | `0x100` | decode has an instruction but cannot issue it |
| 2 | `DE_STALL_NO_UNIT_RTR` | `0x200` | execution unit not ready |
| 3 | `DE_STALL_IF_EMPTY` | `0x300` | nothing to decode |
| 4 | `DE_STALL_SYNC` | `0x400` | serializing operation draining the load queue |
| 5 | `DE_STALL_HAZARD` | `0x500` | register hazard |
| 6 | `DE_HAZARD_INT` | `0x600` | hazard by register file |
| 7 | `DE_HAZARD_FP` | `0x700` | hazard by register file |
| 8 | `DE_HAZARD_VEC` | `0x800` | hazard by register file |
| 9 | `DE_REG_DATA_FORWARD` | `0x900` | operand forwarding used |
| 10 | `DE_MEM_OP_DISPATCH_LD` | `0xA00` | scalar load, store dispatched |
| 11 | `DE_MEM_OP_DISPATCH_ST` | `0xB00` | scalar load, store dispatched |
| 12 | `DE_MEM_INSTR_DISPATCH_VEC_LD` | `0xC00` | vector load, store dispatched |
| 13 | `DE_MEM_INSTR_DISPATCH_VEC_ST` | `0xD00` | vector load, store dispatched |
| 14 | `DE_EX_OP_DISPATCH_MUL` | `0xE00` | multiply, divide dispatched |
| 15 | `DE_EX_OP_DISPATCH_DIV` | `0xF00` | multiply, divide dispatched |

DE1, nibble `[15:12]`

| Idx | Event | Select | Meaning |
|---|---|---|---|
| 1 | `DE1_EX_OP_DISPATCH_CSR` | `0x1000` | CSR, ecall, ebreak dispatched |
| 2 | `DE1_EX_OP_DISPATCH_I_EXT` | `0x2000` | integer, multiply/divide, atomic, bit-manipulation dispatched |
| 3 | `DE1_EX_OP_DISPATCH_M_EXT` | `0x3000` | integer, multiply/divide, atomic, bit-manipulation dispatched |
| 4 | `DE1_EX_OP_DISPATCH_A_EXT` | `0x4000` | integer, multiply/divide, atomic, bit-manipulation dispatched |
| 5 | `DE1_EX_OP_DISPATCH_ZB_EXT` | `0x5000` | integer, multiply/divide, atomic, bit-manipulation dispatched |
| 6 | `DE1_FP_OP_DISPATCH_S` | `0x6000` | single, half precision FP dispatched |
| 7 | `DE1_FP_OP_DISPATCH_ZFH` | `0x7000` | single, half precision FP dispatched |
| 8 | `DE1_VEC_OP_DISPATCH_CSR` | `0x8000` | vector instruction classes |
| 9 | `DE1_VEC_OP_DISPATCH_INT` | `0x9000` | vector instruction classes |
| 10 | `DE1_VEC_OP_DISPATCH_FIXED` | `0xA000` | vector instruction classes |
| 11 | `DE1_VEC_OP_DISPATCH_FLOAT` | `0xB000` | vector instruction classes |
| 12 | `DE1_VEC_OP_DISPATCH_LDST` | `0xC000` | vector instruction classes |
| 13 | `DE1_VALID_TT_INSTR` | `0xD000` | Tensix instruction issued by this core |
| 14 | `DE1_VALID_VEC_REPLAY` | `0xE000` | vector replay |
| 15 | `DE1_VALID_ILLEGAL` | `0xF000` | illegal instruction |

EX, nibble `[19:16]`

| Idx | Event | Select | Meaning |
|---|---|---|---|
| 1 | `EX_STALL_ANY` | `0x10000` | execute back-pressure; observed always zero on Blackhole, hazards are resolved at decode |
| 2 | `EX_STALL_NO_INSTR` | `0x20000` | execute starved |
| 3 | `EX_MISPRED_ANY` | `0x30000` | any branch mispredict |
| 4 | `EX_MISPRED_NEW_BR` | `0x40000` | branch not in the target buffer, wrong direction |
| 5 | `EX_MISPRED_DIR` | `0x50000` | branch not in the target buffer, wrong direction |
| 6 | `EX_MISPRED_TGT_COND` | `0x60000` | wrong target, conditional, unconditional |
| 7 | `EX_MISPRED_TGT_UNCOND` | `0x70000` | wrong target, conditional, unconditional |
| 8 | `EX_INST_RET` | `0x80000` | instruction retired; reproduces `minstret` |

LS, nibble `[23:20]`

| Idx | Event | Select | Meaning |
|---|---|---|---|
| 1 | `LS_STALL_ANY` | `0x100000` | memory operation valid but not accepted |
| 2 | `LS_STALL_NO_MEM_RTR` | `0x200000` | memory port not ready |
| 3 | `LS_STALL_LDQ_FULL` | `0x300000` | load queue full |
| 4 | `LS_STALL_STORE_BUF_FULL` | `0x400000` | store waiting on a full write buffer |
| 5 | `LS_STALL_SKID_BUF_FULL` | `0x500000` | skid buffer full |
| 6 | `LS_FENCE_OP` | `0x600000` | fence |
| 7 | `LS_STALL_WAIT_FOR_DATA` | `0x700000` | oldest load waiting for data |
| 8 | `LS_OP_RET` | `0x800000` | load queue retire |
| 9 | `LS_STORE_REQ_WINS` | `0x900000` | store won arbitration over a load |
| 10 | `LS_WR_BUF_STLH` | `0xA00000` | store-to-load hazard in the write buffer |
| 11 | `LS_WR_BUF_STSC` | `0xB00000` | store coalescing |
| 12 | `LS_LOAD_OP` | `0xC00000` | 32-bit load, store |
| 13 | `LS_STORE_OP` | `0xD00000` | 32-bit load, store |
| 14 | `LS_VEC_LOAD_OP` | `0xE00000` | vector load, store |
| 15 | `LS_VEC_STORE_OP` | `0xF00000` | vector load, store |

TTCSR, nibble `[27:24]`

| Idx | Event | Select | Meaning |
|---|---|---|---|
| 1 | `TTCSR_SSTATUS7_HIGH` | `0x1000000` | status CSR 7 non-zero |
| 2 | `TTCSR_SSTATUS6_HIGH` | `0x2000000` | status CSR 6 non-zero |
| 3 | `TTCSR_SSTATUS5_HIGH` | `0x3000000` | status CSR 5 non-zero |
| 4 | `TTCSR_SSTATUS4_HIGH` | `0x4000000` | status CSR 4 non-zero |
| 5 | `TTCSR_SSTATUS3_HIGH` | `0x5000000` | status CSR 3 non-zero |
| 6 | `TTCSR_SSTATUS2_HIGH` | `0x6000000` | status CSR 2 non-zero |
| 7 | `TTCSR_SSTATUS1_HIGH` | `0x7000000` | status CSR 1 non-zero |
| 8 | `TTCSR_SSTATUS0_HIGH` | `0x8000000` | status CSR 0 non-zero |
| 9 | `TTCSR_BSTATUS_HIGH` | `0x9000000` | Tensix busy status non-zero |
| 10 | `TTCSR_QSTATUS_HIGH` | `0xA000000` | Tensix queue status non-zero |

On Blackhole this domain is not useful: queue status is non-zero on every cycle of a busy workload, busy status within a fraction of a percent of it, and status CSRs 0 and 1 read zero. On Quasar the status CSRs 0..7 hold the Neo and TRISC index, so those events are constants.

MEM, nibble `[31:28]`

| Idx | Event | Select | Meaning |
|---|---|---|---|
| 1 | `MEM_STALL_ANY` | `0x10000000` | memory request not granted |
| 2 | `MEM_LOCAL_REG_RD_ONGOING` | `0x20000000` | local register, mailbox read in flight |
| 3 | `MEM_MAILBOX_RD_ONGOING` | `0x30000000` | local register, mailbox read in flight |
| 4 | `MEM_DATA_L1_RD_ONGOING` | `0x40000000` | L1 read in flight |
| 5 | `MEM_CFG_REG_RD_ONGOING` | `0x50000000` | Tensix config, register, TDMA register, PC buffer read in flight |
| 6 | `MEM_REG_RD_ONGOING` | `0x60000000` | Tensix config, register, TDMA register, PC buffer read in flight |
| 7 | `MEM_TDMA_REG_RD_ONGOING` | `0x70000000` | Tensix config, register, TDMA register, PC buffer read in flight |
| 8 | `MEM_PC_BUF_RD_ONGOING` | `0x80000000` | Tensix config, register, TDMA register, PC buffer read in flight |
| 9 | `MEM_DATA_ACC_REQ_LOCAL` | `0x90000000` | local memory access accepted |
| 10 | `MEM_INS_BUF_WRITE` | `0xA0000000` | Tensix instruction buffer write accepted: one Tensix instruction pushed |
| 11 | `MEM_INS_BUF_WRITE_MULTIPLE` | `0xB0000000` | multi-word push |
| 12 | `MEM_DATA_RD_REQ_L1` | `0xC0000000` | observed not to track this core's L1 reads on Blackhole; use index 4 |
| 13 | `MEM_DATA_RD_REQ_HIT_LOCACHE` | `0xD0000000` | observed not to track this core's L1 reads on Blackhole; use index 4 |
| 14 | `MEM_IF_ACCESS` | `0xE0000000` | instruction cache access, hit |
| 15 | `MEM_IF_ACCESS_HIT` | `0xF0000000` | instruction cache access, hit |

Quasar TRISC only:

| Idx | Event | Select | Meaning |
|---|---|---|---|
| IF 7 | `IF_STALL_NO_IF_RTS` | `0x70` | front end starved, decode ready but nothing to give |
| IF 8 | `IF_ID_STALL_LD` | `0x80` | decode stalled on a load-use hazard |

## 7. Quasar data-movement cores

The DM cores use the standard 64-bit counters at the same CSR addresses as above (`mcycle`, `minstret`,
`mhpmcounter3..6`, `mhpmevent3..6`). `mhpmcounter3` and `mhpmcounter4` are captured, so the markers keep
the same format. Differences from the baby cores:

- `mhpmevent` is `(event mask << 8) | event set`. The counter advances when any masked event of the set
  fires. Set it with `TT_METAL_PROFILE_RISCV_PMC_DM=<mhpmevent3>[,<mhpmevent4>]`.
- `mcycle` does not count while the core waits in WFI.
- The HPM counters are 40 bits and are not reset: the capture zeroes them at the start of each launch.
- A counter increments one cycle after its event.

Event sets (the select of one event is `((1 << bit) << 8) | set`; OR the masks of events of the same set):

| Set | Bit | Event |
|---|---|---|
| 0 | 0 | `DM_COMMIT_EXCEPTION` (`0x100`) |
| 0 | 1 | `DM_COMMIT_LOAD` (`0x200`) |
| 0 | 2 | `DM_COMMIT_STORE` (`0x400`) |
| 0 | 3 | `DM_COMMIT_AMO` (`0x800`) |
| 0 | 4 | `DM_COMMIT_SYSTEM` (`0x1000`) |
| 0 | 5 | `DM_COMMIT_ARITH` (`0x2000`) |
| 0 | 6 | `DM_COMMIT_BRANCH` (`0x4000`) |
| 0 | 7 | `DM_COMMIT_JAL` (`0x8000`) |
| 0 | 8 | `DM_COMMIT_JALR` (`0x10000`) |
| 0 | 9 | `DM_COMMIT_MUL` (`0x20000`) |
| 0 | 10 | `DM_COMMIT_DIV` (`0x40000`) |
| 1 | 0 | `DM_PIPE_LOAD_USE` (`0x101`) |
| 1 | 1 | `DM_PIPE_LONG_LATENCY` (`0x201`) |
| 1 | 2 | `DM_PIPE_CSR_INTERLOCK` (`0x401`) |
| 1 | 3 | `DM_PIPE_ICACHE_BLOCKED` (`0x801`) |
| 1 | 4 | `DM_PIPE_DCACHE_BLOCKED` (`0x1001`) |
| 1 | 5 | `DM_PIPE_BRANCH_MISPREDICT` (`0x2001`) |
| 1 | 6 | `DM_PIPE_TARGET_MISPREDICT` (`0x4001`) |
| 1 | 7 | `DM_PIPE_FLUSH` (`0x8001`) |
| 1 | 8 | `DM_PIPE_REPLAY` (`0x10001`) |
| 1 | 9 | `DM_PIPE_MULDIV_INTERLOCK` (`0x20001`) |
| 2 | 0 | `DM_MEM_ICACHE_MISS` (`0x102`) |
| 2 | 1 | `DM_MEM_DCACHE_MISS` (`0x202`) |
| 2 | 2 | `DM_MEM_DCACHE_RELEASE` (`0x402`) |

Bits 3 to 5 of set 2 are TLB events, which read zero because there is no virtual memory, and are not in the table. Examples:
`0x201` counts long-latency interlocks, `0x801` instruction cache blocked cycles, `0x1001` data cache
blocked cycles, `0x202` data cache misses.

Which DMs are captured: DM0 (window above) and every DM that runs a kernel. DM1 only configures the
dataflow-buffer remapper and is not captured. The Quasar dispatch firmware is not captured.

## 8. Semantics and traps

- **Increment is an OR.** Several selected events count at most 1 per cycle.
- **`mcountinhibit` on Blackhole.** The shared event enable tests bit 3 for both counters, so writing
  bit 3 freezes both counters. Bit 4 gates counter 4 only. To stop counter 3 alone, select no event
  (`mhpmevent3 = 0`). Quasar uses bit 4 for counter 4. tt-metal never writes `mcountinhibit`; its reset
  value runs both.
- **Counters are wide.** `mhpmcounter3/4` are 64 bits with a sticky overflow bit not visible from
  software. The low word of `mcycle` wraps after about 3 s at 1.35 GHz, far above a launch.
- **Clear-on-read is per half-word** and a 64-bit read cannot be atomic. The capture takes deltas of the
  low words instead.
- **Events are sampled one cycle late and only while armed.** Ignore the first cycle after arming.
- **Clock gating.** On Blackhole a TRISC's `mcycle` can freeze while the core is blocked on a Tensix PC
  buffer read, so it can be shorter than wall time on a core that mostly waits for the Tensix; BRISC and
  NCRISC follow wall time. On Quasar a TRISC is gated only when completely idle.
- **The capture itself** reads four CSRs per launch, which registers as a few serializing stalls.

## 9. Telling "blocked on Tensix" from a pipeline stall

The `*StallAny` events see only the RISC-V pipeline. A core waiting for the Tensix datapath is usually
not pipeline-stalled.

- **Tensix instruction back-pressure is a store stall.** A Tensix instruction is a write to the
  instruction buffer address (`MEM_INS_BUF_WRITE`). When the Tensix cannot take it, the write is not
  granted: look at `MEM_STALL_ANY`, `LS_STALL_STORE_BUF_FULL` and `LS_STALL_ANY`, not at DE or EX.
- **Semaphore and circular-buffer waits retire instructions.** A wait is a loop of L1 loads. On
  Blackhole BRISC and NCRISC it retires about 0.3 to 0.4 instructions per cycle, the same band as data
  movement work, and no event marks a wait. IPC alone cannot tell a waiting core from a working one.
- **`DE_STALL_SYNC` is a load-queue drain, not a Tensix wait.**
- **Normalize by work, not by cycles.** Per-cycle fractions of different cores share a denominator and
  correlate with each other by construction. Compare counts per Tensix instruction pushed, armed in the
  same run, and compare across configurations rather than across cores at one configuration.

A single-issue pipeline gives `stall_union + IPC ~ 1`, so the union counter adds little over IPC; the
information is in the per-stage events.

## 10. Not the same thing as the Tensix perf counters

| | RISC-V PMC (this document) | Tensix perf counters |
|---|---|---|
| Access | CSR instruction from the core itself | MMIO debug registers |
| Scope | per core, private | shared Tensix resource, banked by unit |
| Observes | fetch, decode, execute, load/store, instruction cache, L1 port of one core | FPU, packers, unpackers, instruction path, L1 |
| Option | `TT_METAL_PROFILE_RISCV_PMC` | `TT_METAL_PROFILE_PERF_COUNTERS` |

Both can be on in one run. The profiler's multi-pass flow keeps the PMC rows from the first pass. Rows
join on core coordinates.

## 11. Reading the markers on the host

```python
from tt_llk_perf import riscv_pmc

launch = riscv_pmc.unpack_launch(counts_9092, cycles_9093, events_9094)
kind = riscv_pmc.core_kind("blackhole", "TRISC_1")        # "baby"
riscv_pmc.decode_launch("blackhole", kind, launch)         # event names, mcycle, minstret, derived metrics
```

`riscv_metrics.compute_riscv_metrics(cycles, instret, counts)` gives `riscv_ipc_ratio`, `riscv_stall_pct`, `riscv_mem_stall_per_tti_push_ratio` (needs `MEM_STALL_ANY` and `MEM_INS_BUF_WRITE` armed in the same run) and, per armed event, `riscv_<event>_pct` (per cycle) and `riscv_<event>_per_instr_ratio`. Absent inputs read `None`. The metrics live apart from `metrics.py` because that module's keys are the warehouse metric catalog. `perf_counter_analysis.decode_riscv_pmc_markers(rows, arch)` pairs the three streams of every core and returns a dataframe; it takes marker rows from a device log and is not part of the ops report, which only carries the Tensix counter markers (id 9090).

## 12. Measured on Blackhole

- No dropped markers: 110 cores x 5 RISCs x 3 launches gave 1650 rows of each marker.
- The `mcycle` delta agrees with the wall-clock kernel zone to within 0.03 percent on every core type.
- `EX_INST_RET` reproduces the `minstret` delta to 0.999995.
- Stall union plus IPC is 1.00 on every core type.
- `MEM_INS_BUF_WRITE` is exactly zero on BRISC and NCRISC and non-zero on the three TRISCs.

## 13. Open questions

- Quasar TRISC and DM capture on silicon or an emulator.
- Blackhole Ethernet and DRAM cores run the same `configure_csr()` as the Tensix cores, so they most likely carry the same CSRs; no hook exists.
- The meaning of individual per-stage events comes from the hardware description; only the union counter, `mcycle`, `EX_INST_RET` and the TRISC-only `MEM_INS_BUF_WRITE` were validated on silicon.
