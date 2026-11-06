# Tensix Performance Counters: Summary for Agents

Condensed from two Confluence pages:

| Page | Version / date | Role |
| --- | --- | --- |
| [Quasar performance counters overview](https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/2876048051/Quasar+performance+counters+overview) | v12, 2026-09-20 | **Authoritative for Quasar.** Audited against the A0 hardware design and an emulator selector sweep of 23 ops. Corrects the older page in several places (listed below). |
| [Performance Counters](https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/987693362/Performance+Counters) | v24, 2026-01-29 | Hardware-team overview covering Wormhole, Blackhole and Quasar. Use it for Wormhole/Blackhole; for Quasar, prefer the overview page above where they disagree. |

Per-op sweep numbers live on a third page, [Quasar performance counters investigation](https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/2876965026/Quasar+performance+counters+investigation).

The unpack-to-dest work targets **Quasar** (Tensix NEO). If this file and the code in the repo disagree, trust the code
(see "Where this lives in the repo" at the end) and flag the difference.

---

## 1. Model

- A Quasar Tensix cluster has **4 NEOs**. Each NEO has its own complete set of counters and registers.
- Each NEO has four register-visible **banks**: `FPU`, `TDMA_UNPACK`, `TDMA_PACK`, `INSTRN_THREAD`. Quasar has **no L1
  bank**; instead each NEO has one **L1 client event counter** (section 6).
- A bank is made of **slices**. Each slice has two 1-bit inputs, called *request* and *grant*, and counts cycles each was
  1. Every bank instance also has one **reference counter** (`ref`) that counts every active cycle of the window.
- The readout gives `ref` on `OUT_L`, and either the request count or the grant count on `OUT_H`:
  - **Format bit (mode register bit 16) = 0** → `OUT_H` = request count. These are "selects 0..255".
  - **Format bit = 1** → `OUT_H` = grant count. Software calls these "select 256 + n", meaning slice `n` read as grant.
- On most slices the request and grant inputs are **two unrelated events**, not a request/grant pair. Treat them as two
  independent counters. The `request - grant` difference only means "stall cycles" where noted (section 8).
- A third per-slice counter (stall = request and not grant) exists in hardware but **cannot be read** via registers.
- An event count can never exceed `ref`, so `OUT_H / OUT_L` is a fraction (level events) or a rate (pulse events).
- Counters are 32-bit, wrap modulo 2^32, no saturation, no overflow flag.

## 2. Registers

### Address windows

| Who | Debug-block register at offset `off` | L1 client register at offset `off_l1` |
| --- | --- | --- |
| A TRISC, reaching **its own** NEO | `0x0080_0000 + off` | `0x0080_A000 + off_l1` |
| Any master over the NoC, NEO `n` (0..3) | `0x0180_0000 + n * 0x1_0000 + off` | `0x0180_0000 + n * 0x1_0000 + 0xA000 + off_l1` |

A harvested NEO reads back 0. The metal profiler uses the NoC window from DM0 to reach all four NEOs.

### Debug-block counter registers

| Offset | Register | Fields |
| --- | --- | --- |
| 0x00 | `PERF_CNT_INSTRN_THREAD0` | [31:0] reference period |
| 0x04 | `PERF_CNT_INSTRN_THREAD1` | [7:0] mode; [13:8] select (6 bits, 51 valid slices); [16] format |
| 0x08 | `PERF_CNT_INSTRN_THREAD2` | [0] start; [1] stop |
| 0x0C | `PERF_CNT_TDMA_UNPACK0` | [31:0] reference period |
| 0x10 | `PERF_CNT_TDMA_UNPACK1` | [7:0] mode; [12:8] select (5 bits, over the shared 21-entry TDMA space); [16] format |
| 0x14 | `PERF_CNT_TDMA_UNPACK2` | [0] start; [1] stop |
| 0x18 | `PERF_CNT_FPU0` | [31:0] reference period |
| 0x1C | `PERF_CNT_FPU1` | [7:0] mode; [8] select (1 bit); [16] format |
| 0x20 | `PERF_CNT_FPU2` | [0] start; [1] stop |
| 0x24 | `PERF_CNT_ALL` | [0] start all; [1] stop all (FPU and INSTRN_THREAD only) |
| 0x40 | `DBG_FEATURE_DISABLE` | |
| 0x8C | `PERF_CNT_TDMA_PACK0` | [31:0] reference period |
| 0x90 | `PERF_CNT_TDMA_PACK1` | [7:0] mode; [12:8] select (same shared 21-entry space); [16] format |
| 0x94 | `PERF_CNT_TDMA_PACK2` | [0] start; [1] stop |
| 0x98 / 0x9C | `PERF_CNT_OUT_L/H_INSTRN_THREAD` | ref / request-or-grant |
| 0xA0 / 0xA4 | `PERF_CNT_OUT_L/H_TDMA_UNPACK` | ref / request-or-grant |
| 0xA8 / 0xAC | `PERF_CNT_OUT_L/H_TDMA_PACK` | ref / request-or-grant |
| 0xB0 / 0xB4 | `PERF_CNT_OUT_L/H_FPU` | ref / request-or-grant |
| 0xCC | `WALL_CLOCK_0` | wall clock [31:0], live (reading it latches the upper word) |
| 0xD0 | `WALL_CLOCK_1` | wall clock [63:32], **live** |
| 0xD4 | `WALL_CLOCK_1_AT` | upper word latched by the last `WALL_CLOCK_0` read |
| 0xF4 | `PERF_CNT_MUX_CTRL` | Has no effect on any `OUT` register on Quasar (only steers a debug bus; L1 mux field is dead) |

Mode-register bits other than [7:0], the select field and [16] are stored but unused.

**Do not use** the legacy offsets in `tensix.h` / `tensix_debug.h` (Wormhole-era layout, commented out). Only offsets
0x00..0x20 match Quasar. The generated `t6_debug_map.h` is authoritative.

### Counting modes (mode register [7:0])

| Mode | Behaviour |
| --- | --- |
| 0 | Free running between start and stop. **Use this.** |
| 1 | Auto stop when `ref == reference_period`. The window is **period + 1** cycles; `OUT_L` reads period + 1. |
| 2 | `ref` is forced to 0 every cycle (there is no "wrap" behaviour). Unusable for fractions. |
| 3..255 | Same as mode 0. |

### Start / stop semantics

- Start and stop are **level bits, rising-edge triggered**. Write 0 before writing 1 again.
- Start clears every count in the bank (and its `ref`) and begins counting on the next cycle. It is the **only** clear.
- Stop freezes all counts until the next start. Start and stop in the same cycle resolve as start.
- `PERF_CNT_ALL` is ORed with the bank's own bits **before** edge detection, so a start-all edge is lost in any bank
  whose own start bit is still 1 (same for stop).
- **No hardware event starts or stops a counter.** Only the register bits do (see corrections, section 9).

### Programming sequence (bank counters)

```cpp
// XYZ = INSTRN_THREAD, FPU, TDMA_UNPACK, TDMA_PACK. Use the window base for the target NEO.
uint32_t fmt = 0;                         // 0 = request count, 1 = grant count (the "256 + n" selects)
PERF_CNT_XYZ1 = (fmt << 16) | (sel << 8) | 0 /*mode 0*/;
PERF_CNT_XYZ2 = 0x0;  PERF_CNT_XYZ2 = 0x1;   // rising edge on start: clear and count
// ... kernel region ...
PERF_CNT_XYZ2 = 0x0;  PERF_CNT_XYZ2 = 0x2;   // rising edge on stop: freeze
uint32_t ref = PERF_CNT_OUT_L_XYZ;
uint32_t val = PERF_CNT_OUT_H_XYZ;
// Re-aim select/format and re-read as many times as you like; the frozen counts don't change.
PERF_CNT_XYZ2 = 0x0;                          // re-arm
```

- Changing select or format only moves the readout mux; it does not disturb counting. One window serves every select
  of a bank.
- `OUT_L` and `OUT_H` are two separate bus reads of a value re-registered every clock, so on a **running** bank they
  come from different cycles. Only a **stopped** bank gives a consistent pair.

## 3. The shared TDMA readout space

`TDMA_UNPACK` and `TDMA_PACK` have **separate control** (start/stop/mode) but **one shared readout array of 21 entries**:

| Entries | Owner | Started/stopped by |
| --- | --- | --- |
| 0..10 | unpack instance | `PERF_CNT_TDMA_UNPACK2` only |
| 11..18 | pack instance | `PERF_CNT_TDMA_PACK2` only |
| 19, 20 | constant 0 (including `ref`) | |
| 21..31 | out of range, undefined | |

Either `OUT` pair can read any entry. For example, the UNPACK registers with select 11 return the pack instance's packer-0 Dest
read counter. `OUT_L` returns the `ref` of the instance that **owns** the entry. The two halves have independent windows,
and `PERF_CNT_ALL` does not reach either one.

## 4. Quasar counter catalogue

Names are the software enumerators from `tt_metal/tt-llk/tools/include/perf_counters/quasar.h` / `types.h`.
"Sweep" is how many of the 23 emulator-swept ops gave a non-zero count. "Zero on every op" means the counter is wired
up but no swept kernel exercised it, which is different from tied to 0. In the swept kernels, thread 0 does unpack,
thread 1 does math, thread 2 does pack, and thread 3 is unused.

### 4.1 `FPU` (1-bit select)

| Name | Sel | Counts | Read as | Sweep |
| --- | --- | --- | --- | --- |
| `FPU_COUNTER` | 0 | Cycles an FPU-datapath instruction is popped from the math issue stage (ALU op with sources ready, Dest load/read, MOVB2A), **plus power-ramp dummy ops**. An issue pulse, i.e. an instruction count, not busy cycles. | `/ref` = FPU issue rate | 22/23 |
| `SFPU_COUNTER` | 1 | Cycles an SFPU instruction advances out of SFPU stage s1 (≈ one per SFPU instruction) | `/ref` = SFPU utilization | 3/23 |
| `MATH_COUNTER` | 257 | Cycles with FPU issue **or** SFPU issue (at most 1 per cycle) | `/ref` = math busy. `FPU + SFPU - MATH` = cycles both issued together | 23/23 |

Select 256 is tied to 0.

### 4.2 `TDMA_UNPACK` (entries 0..10)

| Name | Sel | Counts | Read as | Sweep |
| --- | --- | --- | --- | --- |
| `MATH_SRC_DATA_READY` | 0 | An FPU ALU instruction sits in the math issue stage with SrcA and SrcB valid | **Does not** measure math waiting on unpack, because instructions only enter that stage once sources are valid. Use INSTRN_THREAD 49/50 for that. | 11/23 |
| `MATH_NOT_D2S_STALLED` | 1 | A math instruction is in the issue stage and not held by the post-MOVD2A/MOVD2B countdown | `1 - this / MATH_INSTRN_AVAILABLE` = Dest→Src move hazard loss | 22/23 |
| `MATH_INSTRN_STARTED` | 3 | Math issue stage popped (one per math instruction that starts) | `/MATH_INSTRN_AVAILABLE` = math pipeline utilization | 22/23 |
| `MATH_INSTRN_AVAILABLE` | 4 | A math instruction is present in the issue stage (issuing or stalled) | Denominator for math-side ratios | 22/23 |
| `SRCB_WRITE_REQ` | 5 | Unpacker presents a SrcB write (any of 8 lanes) | SrcB write pressure; denominator for 259/260 | 12/23 |
| `SRCA_WRITE_REQ` | 6 | Unpacker presents a SrcA write (any of 8 lanes). THCON/other writes count too. | Denominator for 261/262 | 22/23 |
| `UNPACK0_BUSY_THREAD0` | 7 | Unpacker 0 busy for thread 0 | `/ref` = unpacker 0 utilization | 22/23 |
| `UNPACK1_BUSY_THREAD0` | 8 | Unpacker 1 busy for thread 0 | `/ref` | 13/23 |
| `UNPACK2_BUSY_THREAD0` | 9 | Unpacker 2 (Quasar has 3 unpackers) busy for thread 0 | `/ref` | zero on every op |
| `UNPACK0_BUSY_THREAD1` | 10 | Unpacker 0 busy for thread 1 | Thread-1 share of unpacker 0 | zero on every op |
| `SRCB_WRITE_NOT_BLOCKED_OVR` | 259 | SrcB write presented **and** target SrcB bank may be overwritten (its dvalid is clear) | `/SRCB_WRITE_REQ` = fraction not waiting on math to consume SrcB | 12/23 |
| `SRCB_WRITE_NOT_BLOCKED_PORT` | 260 | SrcB write presented and SrcB write port free | `1 - this / SRCB_WRITE_REQ` = port-blocked rate | 12/23 |
| `SRCA_WRITE_NOT_BLOCKED_OVR` | 261 | SrcA write presented and target SrcA bank may be overwritten | `1 - this / SRCA_WRITE_REQ` = overwrite-blocked rate | 22/23 |
| `SRCA_WRITE_NOT_BLOCKED_PORT` | 262 | SrcA write presented and SrcA write port not taken by a math-side write | `/SRCA_WRITE_REQ`. Equalled select 6 on every swept op. | 22/23 |
| `SRCA_WRITE_TID_EVEN` / `_ODD` | 263 / 265 | Lane-0 SrcA write from an even / odd thread | Lane 0 of 8 only, thread-id bit 0 only | 22/23 / zero |
| `SRCB_WRITE_TID_EVEN` / `_ODD` | 264 / 266 | Same for SrcB | | 12/23 / zero |

Tied to 0: select 2 (fidelity stall), 256, 257 (fidelity half-cycle counts). Select 258 duplicates select 3.
Selects 11–18 read the pack entries (section 3).

Unpacker-busy coverage is incomplete. Only 4 of the 12 thread×unpacker pairs are observable: thread 0 with unpackers 0, 1
and 2, and thread 1 with unpacker 0.

### 4.3 `TDMA_PACK` (entries 11..18)

| Name | Sel | Counts | Read as | Sweep |
| --- | --- | --- | --- | --- |
| `PACKER0_DEST_READ_REQ` | 11 | Packer 0 requests a Dest read | `/PACKER_BUSY` = Dest-read efficiency | 23/23 |
| `PACKER_BUSY` | 18 | Any packer busy for any thread (OR over both packers, can't split) | `/ref` = packer utilization | 23/23 |
| `DEST_READ_GRANTED_0` | 267 | Packer 0 Dest read accepted (pack request FIFO not full) | `/PACKER0_DEST_READ_REQ` = acceptance rate. Real req/grant pair, so `11 - 267` = FIFO back-pressure cycles. | 23/23 |
| `MATH_NOT_STALLED_DEST_WR_PORT` | 271 | Math instruction in issue stage, not a Dest-writing op held by the Dest write-port stall | `1 - this / MATH_INSTRN_AVAILABLE` (different instance: align windows) | 22/23 |
| `MATH_NOT_SCOREBOARD_STALLED` | 272 | Math instruction in issue stage, not held by the Dest dependency (RAW) scoreboard | `1 - this / MATH_INSTRN_AVAILABLE` = scoreboard stall rate | 22/23 |

Selects 12–17, 268–270, 273, 274 are tied to 0. Packer 1's Dest reads are **not counted** anywhere.

### 4.4 `INSTRN_THREAD` (6-bit select, 51 slices, `t` = thread 0..3)

**Per-class "available" counters, selects 0..31 = class × 4 + t.** Each counts cycles where the instruction at the head
of thread `t`'s buffer is of that class **and is not blocked** by its unit or the thread. A blocked head instruction
counts in `THREAD_STALLS_t` instead.

| Class | Selects | Notes |
| --- | --- | --- |
| CFG | 0–3 | WRCFG, RDCFG, RMWCIB*, CFGSHIFTMASK, COMMIT_SHADOW, RV_WRCFG |
| SYNC | 4–7 | SEMINIT/POST/GET/WAIT, STALLWAIT, WAIT_TILES, WAIT_FREE, ATGETM/ATRELM, MOP, MOP_CFG (PUSH_TILES is PACK class, POP_TILES is UNPACK class) |
| THCON | 8–11 | GPR arithmetic, LOAD/STORE IND/REG, L1 atomics, DMANOP |
| XSEARCH | 12–15 | **Tied to 0 on Quasar**; not mapped by software |
| INSTISSUE | 16–19 | CLEARDVALID, SETRWC, INCRWC, ZEROACC, ZEROSRC |
| MATH | 20–23 | MVMUL, ELW*, GAPOOL/GMPOOL, MOV*, SHIFTXB. **Also counts INSTISSUE.** Subtract 16–19 for pure math. |
| UNPACK | 24–27 | UNPACR* (all three unpackers), including UNPACR_DEST_* |
| PACK | 28–31 | PACR* (both packers) |

**Per-thread stall and issue counters:**

| Name | Sel | Counts | Sweep |
| --- | --- | --- | --- |
| `THREAD_STALLS_t` | 32+t | Thread `t` has a head-of-buffer instruction that **doesn't issue** this cycle, for any reason: unit stall, TDMA path stall, or STALLWAIT/SEMWAIT. Tile-counter waits show up as pack/unpack unit stalls. Replay-queue stalls and halts are not counted. | t0 21/23, t1 23/23, t2 23/23, t3 zero |
| `THREAD_INSTRUCTIONS_t` | 256+t | Thread `t`'s instruction buffer dequeued: one instruction of **any** class, **including NOP, halt and invalid** pops. `/ref` = thread IPC. | t0–t2 23/23 |

Grants 260–287 all duplicate `THREAD_INSTRUCTIONS_t`: every class grant of a thread is the same any-class dequeue
signal, so they are **not** per-class issue counts. Grants 288–306 are tied to 0. Selects 51–63 and 307–319 are out of range.

**NEO-wide stall reasons, selects 36..50.** These are ORed over threads, can overlap each other in time, and do **not**
sum to `THREAD_STALLS`. Read each as `/ref`.

| Name | Sel | Counts cycles where... | Sweep |
| --- | --- | --- | --- |
| `TILE_COUNTER_STALL_PACK` | 36 | a pack instruction is held while its thread's last wait was WAIT_TILES/WAIT_FREE | 2/23 |
| `TILE_COUNTER_STALL_UNPACK` | 37 | an unpack instruction is held on a tile-counter wait | 19/23 |
| `SRCS_STALL_PACK` | 38 | packer 1 (the packer that reads SrcS) has a request but SrcS isn't readable | zero |
| `SRCS_STALL_SFPU` | 39 | an SFPU load/store to SrcS finds the bank not ready (unpacker hasn't written it) | zero |
| `SRCS_STALL_UNPACK` | 40 | an unpacker SrcS write finds the bank not writable (still unread) | zero |
| `DEST_STALL_PACK` | 41 | packer 0 has a pending Dest read that the Dest arbiter isn't granting (dvalid or arbitration loss) | 18/23 |
| `DEST_STALL_SFPU` | 42 | an SFPLOAD/SFPSTORE is stalled on Dest access | 3/23 |
| `DEST_STALL_MATH` | 43 | a math ALU / Dest-load instruction in the issue stage is stalled on Dest access | 18/23 |
| **`DEST_STALL_UNPACK`** | **44** | **an unpacker write routed to Dest is stalled by the Dest arbiter (dvalid or arbitration loss)** | **1/23** |
| `SFPU_DATA_HAZARD_STALL` | 45 | an SFPU instruction is valid at s1 but not consumed (any hazard; overlaps 42) | 3/23 |
| `FPU_DATA_HAZARD_STALL` | 46 | a math instruction is in the issue stage and doesn't start, for reasons other than power ramp (overlaps 43) | 19/23 |
| `SRCB_STALL_UNPACK` | 47 | an unpacker SrcB write is blocked because the target bank still holds unconsumed data | 4/23 |
| `SRCA_STALL_UNPACK` | 48 | an unpacker SrcA write **to SrcA (not to Dest)** is blocked because the target bank still holds unconsumed data | 10/23 |
| `DVALID_STALL_MATH` | 49 | some thread's math instruction is waiting because SrcA or SrcB is not yet valid (unpacker hasn't delivered) | 20/23 |
| `SRCA_STALL_MATH` | 50 | some thread's math instruction is waiting for SrcA valid. `49 - 50` = SrcB-only wait. | 20/23 |

### 4.5 Notes specific to unpack-to-dest

- An unpacker writing to Dest uses the **SrcA write interface** with its target switched to Dest. `DEST_STALL_UNPACK` (44)
  counts those writes being held by the Dest arbiter. `SRCA_STALL_UNPACK` (48) deliberately excludes them.
- Inference, not stated on the page: `SRCA_WRITE_REQ` (6) ORs all SrcA-interface write lanes, so unpack-to-dest writes
  probably show up there too. Verify by comparing a pure unpack-to-dest kernel against an unpack-to-SrcA kernel.
- `DEST_STALL_UNPACK` was non-zero on only 1 of the 23 swept ops, so few existing kernels exercise it.
- Unpacker busy (`TDMA_UNPACK` 7–10) and `UNPACK_INSTRN_AVAILABLE_t` / `THREAD_STALLS_t` on the unpack thread give the
  issue-side view. L1 client subports 5–24 (section 6) give the L1 read side.

## 5. Wall clock

- 64-bit free-running cycle counter on the same clock as the counter banks. Never stops, no start/stop/clear.
- **Consistent read: `WALL_CLOCK_0` then `WALL_CLOCK_1_AT`** (0xCC then 0xD4). Reading `WALL_CLOCK_0` latches the upper
  word into `WALL_CLOCK_1_AT`.
- `WALL_CLOCK_1` (0xD0) is the **live** upper word, so `WALL_CLOCK_0` + `WALL_CLOCK_1` can straddle a carry.
- It's the only common time base across banks that have independent start/stop, and it's the reference the metal profiler
  uses for the L1 client counter window.

## 6. L1 client event counter (Quasar replacement for the L1 bank)

One 32-bit counter per NEO in the l1_client register block (local offset 0xA000):

| Offset in block | Register | Field | Bits | Meaning |
| --- | --- | --- | --- | --- |
| 0xAC | `GROUP_PERF_CTRL` | `PERF_CNT_ENABLE` | [0] | enable sampling |
| 0xAC | `GROUP_PERF_CTRL` | `PERF_CNT_SUBPORT_SEL` | [9:4] | subport 0..36 (37..63 undefined) |
| 0xAC | `GROUP_PERF_CTRL` | `PERF_CNT_EVENT_SEL` | [14:12] | event 0..7 |
| 0xB0 | `GROUP_PERF_CNT` | `PERF_CNT` | [31:0] | +1 per cycle the selected event is 1; **saturates**; **cleared by any read**; writable |

So TRISC addresses are `0x0080_A0AC` (ctrl) and `0x0080_A0B0` (count).

**One subport × event pair per window per NEO.** Sweeping selections means sweeping runs. There's no reference counter,
so divide by a cycle count for the same interval taken from the wall clock (metal profiler) or the INSTRN_THREAD `ref`
(tt-llk harness).

### Subports

| Subport | Client |
| --- | --- |
| 0–3 | TRISC thread 0–3 L1 data accesses |
| 4 | THCON |
| 5–8 | unpacker 0, L1 read interface 0, 16 B lanes 0–3 |
| 9–12 | unpacker 0, L1 read interface 1, lanes 0–3 |
| 13–16 | unpacker 1, interface 0, lanes 0–3 |
| 17–20 | unpacker 1, interface 1, lanes 0–3 |
| 21–24 | unpacker 2, interface 0, lanes 0–3 (unpacker 2 has one interface) |
| 25–28 | packer 0, L1 write interface 0, lanes 0–3 |
| 29–32 | packer 0, interface 1, lanes 0–3 |
| 33–36 | packer 1, interface 0, lanes 0–3 (packer 1 has one interface) |

### Events

| Event | Software name | Counts | Scale | Scope |
| --- | --- | --- | --- | --- |
| 0 | — | always 0 | | |
| 1 | `SBANK_POP` | cycles with at least one grant from super bank (subport mod 4) | ×1 | **whole port**, super bank (subport mod 4) |
| 2 | `ISSUE_STALL_CARRY` | lanes holding a valid request the super bank didn't accept | ×4 | whole port, super bank |
| 3 | `ISSUE_WORK_CARRY` | lanes granted | ×4 | whole port, super bank |
| 4 | `FLEX_STALL_CARRY` | this subport's flex request FIFOs holding a request that isn't popped | ×4 | per subport |
| 5 | `FLEX_WORK_CARRY` | this subport's flex FIFO entries popped toward the super bank | ×4 | per subport |
| 6 | `PENDING_REQS_CARRY` | integral of outstanding requests over time | ×64, **×128 on subports 25–32** | per subport |
| 7 | `ORDER_FIFO_ACTIVE` | cycles with at least one outstanding request | ×1 | per subport |

- Events 1–3 on an unpack/pack subport describe **super bank (subport mod 4) of the whole port**, not the selected lane.
  On THCON (subport 4), events 1–3 alias TRISC subport 0.
- Software encodes selections as `256 + subport * 8 + event` (`QUASAR_L1_CLIENT_EVENT`). Validity rule
  (`l1_client_selection_is_valid`): `sel < 296 && event != 0 && !(subport == 4 && event <= 3)`, giving 256 valid selections.
- Reading: events 1 and 7 → `count / cycles` is a rate. Events 2–5 → `4 * count / cycles` is mean lanes or FIFOs per cycle.
  Event 6 → `64 * count / cycles` (128 on 25–32) is mean outstanding requests. Dividing by the event-7 rate gives mean
  occupancy while active, and combining with event 5 gives latency via Little's law.
- Sequence: write `GROUP_PERF_CTRL = 1 | subport << 4 | event << 12`, read `GROUP_PERF_CNT` once to clear, run the region,
  read `GROUP_PERF_CNT` **exactly once**. Disabling first stops new sampling, but a level event (1 or 7) that is already
  high keeps counting until it drops, so the window really ends at the read.
- Sweep coverage so far: events 1–7 sampled on subports 0, 5 and 25 only. The other selections are wired up but have not been
  read on the emulator.

## 7. Derived ratios and multi-pass caveats

- Level event: `OUT_H / OUT_L` = fraction of the window. Pulse event (instruction start, dequeue, FIFO accept): events per
  cycle. Valid in mode 0 and mode 1.
- `grant / request` is an acceptance ratio, and `request - grant` is the stall count, **only** for TDMA unpack 259–262 vs
  5/6 and pack 267 vs 11. Elsewhere the two sides are unrelated.
- **Cross-bank ratios** (for example `MATH_NOT_SCOREBOARD_STALLED / MATH_INSTRN_AVAILABLE`, or `FPU_COUNTER / MATH_INSTRN_AVAILABLE_1`)
  divide counts from separately started instances. They're only consistent if the windows opened and closed together,
  and `PERF_CNT_ALL` can't do that for the TDMA banks.
- Selects read from one frozen window are fine. Selects collected over several runs each have their own `ref`, so
  normalize per run before combining. L1 client numbers always come from different runs than the bank counters.
- tt-llk's derived-metric names (for example `dest_stall_unpack_pct`, `srca_write_eff_pct`) are implemented in
  `tt_metal/tt-llk/tools/python/tt_llk_perf/metrics.py`.

## 8. Gotchas (Quasar)

- Starting a bank resets **every** counter in it, so two code regions timing the same bank clobber each other.
- `PERF_CNT_ALL` reaches FPU and INSTRN_THREAD only, and its edges are masked while a bank's own bit is 1.
- Read `OUT_L`/`OUT_H` only on a stopped bank.
- `FPU_COUNTER` includes power-ramp dummy ops.
- `MATH_INSTRN_AVAILABLE_t` includes INSTISSUE-class instructions. `THREAD_INSTRUCTIONS_t` includes NOP, halt and invalid pops.
- Per-class grants (260–287) are **not** per-class issue counts.
- `MATH_SRC_DATA_READY` (unpack 0) doesn't show math waiting on unpack. Use INSTRN_THREAD 49/50.
- TID_EVEN/ODD (263–266) look at lane 0 and thread-id bit 0 only.
- Packer 1's Dest reads aren't counted. `PACKER_BUSY` is the OR over both packers.
- Only 4 of 12 unpacker×thread busy pairs are observable.
- Fidelity counters (unpack 2, 256, 257) are tied to 0, and 258 duplicates 3. XSEARCH (INSTRN_THREAD 12–15) is tied to 0.
- Out-of-range selects (TDMA 21–31, INSTRN_THREAD 51–63, L1 subport 37–63) are undefined. TDMA 19–20 read 0.
- `GROUP_PERF_CNT` clears on read and saturates. Event 6 scaling differs on subports 25–32.
- `WALL_CLOCK_1` is live, so use `WALL_CLOCK_1_AT` for the latched upper word.

## 9. Corrections the Quasar overview makes to the older hardware-team page

| Older page says | Actually (Quasar A0) |
| --- | --- |
| `PERF_CNT_<bank>1[16:8]` is a 9-bit `counter_sel` | Bit 16 is the **format** bit (request vs grant). The select is 1/5/6 bits wide (FPU/TDMA/INSTRN_THREAD). "Counter 256+n" = slice n read as grant. |
| Mode 1 stops after `ref_period` cycles | Window is `ref_period + 1` cycles |
| INSTRN_THREAD 256+n..284+n = "<class> instructions issued by thread n" | All of them are the same any-class dequeue count for thread n, including NOP/halt/invalid |
| INSTRN_THREAD 32+n = stalls due to STALLWAIT/SEMWAIT | Any head-of-buffer instruction that doesn't issue; STALLWAIT/SEMWAIT is just one cause |
| TDMA_UNPACK 7+n = unpack busy, thread n | 7 = U0/T0, 8 = U1/T0, 9 = U2/T0, 10 = U0/T1 |
| TDMA_UNPACK 0 unusable | Confirmed. Use INSTRN_THREAD 49/50 for math waiting on sources |
| TDMA_PACK lists 11, 267, 18, 272 | 271 (`MATH_NOT_STALLED_DEST_WR_PORT`) is also readable |
| "Tensix Events" can start/stop counters on dvalid, Dest valid, src ready, tagged TTI_NOP | **Not implemented.** Only register bits start/stop counters |
| `WALL_CLOCK_1` returns the latched upper word | `WALL_CLOCK_1` is live; latched word is `WALL_CLOCK_1_AT` (0xD4) |
| L1 counters: FIFO full, rsp buffer full, per-bank/lane pushes, bank conflicts | Superseded by the 8 L1 client events in section 6 |
| TRISC counters are RISC-V CSRs | They're in the TRISC core, outside the debug block, and not covered by the overview |

## 10. Blackhole and Wormhole (from the older page)

Same register model, but the bit-16 format-bit correction above probably applies there too. `FPU` 0/1/257, `TDMA_UNPACK`
1, 3–6, 259, 261 and `TDMA_PACK` 11, 18, 267, 272 have the same meanings, except that on BH/WH `TDMA_UNPACK` 3/4 exclude SFPU.
Differences:

- Three threads, so per-thread blocks step by 3: avail counters are CFG 0+n, SYNC 3+n, THCON 6+n, XSEARCH 9+n,
  MOVE 12+n, FPU 15+n, UNPACK 18+n, PACK 21+n. `256+n` = all instructions issued by thread `n`.
- `24+n` = thread `n` stalled by STALLWAIT/SEMWAIT/STREAMWAIT.
- `27`/`28` = waiting for srcA/srcB clear, `29`/`30` = waiting for srcA/srcB valid. These count only explicit STALLWAIT
  conditions, increment at most once per cycle for any thread, and count even when nothing is actually stalled.
- `31+n .. 55+n` (step 3) = thread `n` waiting for, in order: THCON idle, UNPACK idle, PACK idle, MATH idle, nonzero
  semaphore, non-full semaphore, MOVE (XMOV) idle, MMIO idle, SFPU idle.
- `TDMA_UNPACK` 7–10 = unpacker 0/1 busy due to thread 0/1 (7: U0/T0, 8: U1/T0, 9: U0/T1, 10: U1/T1). 9 and 10 should be ~0
  with standard kernels.
- Suggested utilization proxies: `TDMA_UNPACK[261] / [7]` ≈ unpacker 0, `TDMA_UNPACK[259] / [8]` ≈ unpacker 1.
- BH/WH have a real `L1` bank (8 counters, client group chosen by `PERF_CNT_MUX_CTRL`).
- Wall clock: `WALL_CLOCK_0` then `WALL_CLOCK_1`. The older page says `_1` is latched; on Quasar use `_1_AT`.

## 11. Where this lives in the repo

The tt-llk headers are the software source of truth for selector values and register addresses:

- `tt_metal/tt-llk/docs/performance_counters/hardware_reference.md`: prose reference, including the "Quasar windows",
  "Quasar only" and "L1 client events (Quasar)" sections.
- `tt_metal/tt-llk/tools/include/perf_counters/quasar.h`: Quasar `{name, select}` tables per bank, plus
  `l1_client_selection_is_valid(sel)`.
- `tt_metal/tt-llk/tools/include/perf_counters/types.h`: `PerfCounterType` enumerator names. This list is append-only.
- `tt_metal/tt-llk/tools/include/perf_counters/registers.h`: `bank_regs(bank, window)`, `LOCAL_REGS_WINDOW`,
  `neo_window(n)`, `l1_client_regs(window)`.
- `tt_metal/tt-llk/tools/include/perf_counters/hw.h`: `configure`, `start`, `stop`, `select`, `read_ref`,
  `read_count`, `l1_client_start/stop/read`.
- `tt_metal/tt-llk/tests/helpers/include/counters.h`: LLK test-harness wiring.
- `tt_metal/tt-llk/tools/python/tt_llk_perf/`: host-side table parsing (`headers.py`) and derived metrics (`metrics.py`).

## Dest-path experiments: RTL constraints supplied 2026-11-04

These are user-supplied RTL findings, not independently verified waveform results.
For a full 1024-element FP32 tile, transfer-only lower bounds are:

| Stage | Best cycles | Basis |
|---|---:|---|
| U0 to Dest | 32 | Two 64 B/clk ports; 128 B/clk Dest bus |
| U2 to SrcS | 64 peak, about 85 sustained with hashing on | One 64 B/clk L1 interface is the peak. U2's L1 port allows only 12 reads in flight, so with the DV hash table it sustains about 48 B/clk (see "Unpacker 2 rate limit" below) |
| SrcS to Dest SFPU copy | 64 issue cycles | 32 loads + 32 stores, shared 1 instruction/clk |
| PACK0 from Dest | 32 | L1 write 2 × 64 B = 128 B/clk sets the FP32 limit; Dest read is 256 B/clk (4 rows × 16 columns × 32 bits), so 16 reads per tile |
| PACK1 from SrcS | 64 | 64 B/clk; SrcS is its only legal source |

Only U0 writes Dest. Double banking overlaps transfers; it does not halve their
individual transfer time. U0 shares the FPU write port even across banks. SFPU
loads contend with PACK0 for Dest reads; SrcS accesses wait on dvalid rather than
port arbitration. Done bits toggle flags at completion; `SFPLOAD.done` also affects
Dest flags. SrcS has two banks with three slice slots each, not six independently
queueable input slices: input/output convention uses slots 0/2.

U0 may take about 64 cycles with hashing disabled. Enabling CTRL alone is
insufficient: FN0 MATCH must also match the addresses. The new test helper reads
CTRL, MASK, MATCH and ADDR4–11 from 0x0080A000 through 0x0080A028. It never writes
these registers. Hash mapping must agree with the host/NoC writer before buffers
are populated. A readback proves the local mapping, not the NoC mapping.

The concise suite is `quasar/perf_dest_write_path_quasar.py` (26 cases) plus
`quasar/perf_dest_preload_matmul_quasar.py` (6 cases). Matching functional checks
are in `quasar/test_dest_write_experiments_quasar.py`. Existing C++ kernels,
helpers and firmware remain unchanged; experiments use new C++ files. Historical
Cartesian sweeps are in the corresponding `*_exhaustive_quasar.py` host files.

Experimental cases cover startup, delayed input issue, delayed pack issue,
noncontiguous tiles, dependent SFPU math depth, identical exp math on both paths,
timestamp overhead, partial banks, a reduced-pack writer proxy, PACK1 output, and
FPU pressure. Delays are RISC issue gaps, not exact hardware service delays.
The writer proxy still packs one face; it is not an isolated engine measurement.
PACK1 uses the existing SrcS helper's tile auto-loop and arithmetic implementation;
it is a separate output architecture, not an alternative way to preload Dest.
The FPU test still includes U0 reconfiguration overhead; it is not an isolated FPU
write-port arbitration test. True per-stage service times and arbitration attribution
remain waveform/calibration work because U2/T3 and PACK1 have no busy counters.

Run with `--enable-perf-counters` to capture per-bank reference cycles, issue,
SrcS/Dest stalls, SFPU/FPU activity and PACK0 requests/grants. L1 selection uses
`LLK_PERF_L1_CLIENT_SEL`; collect extra selectors only for anomalous cases. Never
add overlapping stall counters to reconstruct wall time. The offline command
`python quasar/dest_write_analysis.py <stamps.jsonl>` emits lower bounds and
investigation flags for every observation. It cannot establish contention-free
RTL violations from timestamps alone. Preserve both the JSONL and counter CSVs.

No new simulator results have been collected. The old report's attribution of
64-cycle FP32 throughput solely to PACK0 is not established; do not use it to
justify removal of unpack-to-Dest without the new controlled measurements.

## Run commands

With your Quasar environment activated, switch to this checkout's test directory. The test paths below are relative
to `python_tests`, not `python_tests/quasar`, so run this `cd` even if your setup script already moved you into
`quasar/`:

```bash
cd /localdev/nmohammed/tt-metal-sem-counters/tt_metal/tt-llk/tests/python_tests
```

Run correctness first:

```bash
RUNNER_TEMP=/tmp/dwp-functional pytest --run-simulator --timeout=600 -rN \
  quasar/test_dest_write_experiments_quasar.py
```

After correctness passes, run performance with counters:

```bash
RUNNER_TEMP=/tmp/dwp-perf pytest --run-simulator --enable-perf-counters \
  --timeout=600 -rN \
  quasar/perf_dest_write_path_quasar.py \
  quasar/perf_dest_preload_matmul_quasar.py
```

Analyze the recorded timing against the supplied RTL lower bounds:

```bash
python quasar/dest_write_analysis.py \
  /tmp/dwp-perf/tt-llk-build/dest_write_path_bank_stamps.jsonl
```

The analyzer flags discrepancies for investigation; stage attribution still needs
counter evidence. Hash settings are read back, never changed by these tests.
These commands have not been run to validate the new experiments.

### Focused rerun after the Dest SFPU row-counter fix

The experimental Dest arithmetic walk now uses the existing SFPU start/done
helpers per tile. The done helper resets RWC_D, preventing the previous tile's
64-row advance from being added to the next tile's section base. This affects
exp and the light/heavy SFPU arithmetic cases on the unpack-to-Dest path.
The change has not been rerun on the simulator.

From the test directory above, compare both paths for the affected workloads:

```bash
RUNNER_TEMP=/tmp/dwp-sfpu-fix pytest --run-simulator --timeout=600 -rN \
  quasar/test_dest_write_experiments_quasar.py \
  -k 'exp_same_math or sfpu_light or sfpu_heavy'
```

### Validation completed 2026-11-04

Using the supplied Quasar environment and this checkout, the fixed experiment
suite passed all 32 correctness cases (6 focused SFPU cases, then 26 remaining)
and all 32 performance cases with counters enabled. This supersedes the earlier
notes that the new experiments had not been run. The previously failing
`exp_same_math-unpack_dest` case passed.

Logs, functional/performance timestamps, counter reports, Parquet output and
offline lower-bound triage are preserved in
[logs/validated_20261104T172013Z](logs/validated_20261104T172013Z/).
The performance run ID is `local-20261104T172013Z`.

Local hash readback matches the supplied DV default table. Baseline FP32 copy
at four tiles per bank measured 42 cycles/tile for A-hw and 83.25 for B-done
using steady bank timestamps. These include bank/software overhead and are not
isolated engine transfer latencies. A passing performance case establishes run
completion/report generation, not agreement with an RTL throughput target.


### Multi-slot SrcS validation — 2026-11-04

**All three input slots work for U2 → SrcS → SFPU → Dest → PACK0.**
Only the new slot tests were run for this validation: 9 bit-exact correctness
cases and 6 counter-enabled performance cases passed. Existing production
kernels, LLK helpers and firmware were not changed for this experiment.

The new test-only source is `dest_write_srcs_slots_quasar_test.cpp`. It uses an
FP32 16×4 descriptor (256 B per slot), increments the destination index for
intermediate slots, and sets dvalid only on the last slot. The SFPU reads offsets
0/4/8, releases SrcS only on the group's final load, and stores into Dest.
Slot 2 is input here; this is **not safe to reuse unchanged with PACK1 output in
slot 2**.

Coverage is deliberately small:

- Slots 1/2/3 with two-pass auto-loop capture, using unique FP32 bit patterns.
- Slots 1/2/3 with three tiles per Dest bank: 48 slices, no remainder.
- Slots 1/2/3 with four tiles per Dest bank: 64 slices; three slots require
  21 full groups and a final one-slot handoff. Repeated bank reuse checks for
  skipped, repeated and stale data.

The six streaming combinations also ran with counters. Results below use the
postprocessed TILE_LOOP profiler cycles per tile, including drain, synchronization
and loop configuration costs. Steady pack-side timestamp medians are separate
observations, not isolated unpack/SFPU/pack service times.

| Input slots per SrcS bank | 3 tiles: profiler cycles/tile | 4 tiles: profiler cycles/tile | 3 tiles: steady stamp cycles/tile | 4 tiles: steady stamp cycles/tile |
|---|---:|---:|---:|---:|
| 1 | 101.06 | 96.08 | 100.00 | 94.75 |
| 2 | 103.60 | 96.72 | 102.17 | 95.25 |
| 3 | 102.50 | 109.00 | 100.67 | 108.75 |

Three slots did not improve this copy. With four tiles, it was 13.45% slower
than the matched one-slot test. SFPU instruction activity fell from 66.24% to
58.42%, SFPU SrcS-stall activity rose from 20.03% to 24.13%, and PACK0 Dest-stall
activity rose from 54.56% to 60.60%. These counters overlap and must not be
summed. U2 SrcS-stall activity was zero in both four-tile cases. The three-tile,
three-slot case recorded 0.45% U2 SrcS-stall activity.

This implementation conservatively drains U2 before changing loop configuration
or resetting indices. The remainder adds a separate drain/configuration/issue
sequence. The measurements include those costs; counters do not isolate their
individual cycle contributions. These results do not prove that every possible
three-slot schedule is slower. In particular, they are not directly comparable
to the earlier B-done baseline as an isolated slot-count change.

Peak RTL floors are 4 transfer cycles per FP32 slice for U2, 64 per tile
for U2, 64 SFPU copy issue cycles per tile, and 32 PACK0 transfer cycles per tile.
U2 does not sustain its peak with hashing on: see "Unpacker 2 rate limit" below.
Stages may overlap. Observed loop throughput is above the ideal 64-cycle/tile
bottleneck floor; this deserves scheduling investigation, not a claim of an RTL
bandwidth violation. No 40-cycle UNPACR2 startup constant is assumed. The run
read back enabled hashing with MATCH=0 and the supplied DV mapping. Exact
stage latencies and the attribution of remaining bubbles still need narrower
measurement windows or waveforms.

Artifacts: [logs/srcs_slots_20261104T184627Z](logs/srcs_slots_20261104T184627Z/).
Run ID: `local-20261104T184627Z`. The archive includes successful logs, bit-exact
and performance bank stamps, raw/postprocessed CSVs and Parquet. Earlier compile
and queue-timeout logs are retained separately; they are not correctness failures
of the completed implementation.

### Run only the new multi-slot tests

Activate the supplied Quasar environment and exports first. These commands run
only the new slot tests. Simulator startup has its own queue timeout; increasing
pytest's timeout alone does not extend it. The completed performance run used a
3600-second startup allowance in a temporary runner, without changing the harness.

```bash
cd /localdev/nmohammed/tt-metal-sem-counters/tt_metal/tt-llk/tests/python_tests

RUNNER_TEMP=/tmp/dwp-slots-functional pytest --run-simulator --timeout=3900 -rN -x \
  quasar/test_dest_write_srcs_slots_quasar.py

RUNNER_TEMP=/tmp/dwp-slots-perf pytest --run-simulator --enable-perf-counters \
  --timeout=3900 -rN -x quasar/perf_dest_write_srcs_slots_quasar.py
```

### Unpacker 2 rate limit — 2026-11-04

**With hashing on, Unpacker 2 sustains about 48 B/clk, not its 64 B/clk peak. This sets the FP32 SrcS path at
about 84 cycles per tile.** The SFPU and the SrcS hand-off are not the limit.

64 B/clk is the width of U2's L1 port and of its SrcS write. It is the peak, not the sustained rate. The RTL
review found that U2's L1 port allows only 12 reads in flight. The round trip for one read is about 16 cycles,
so U2 sustains 12 × 64 B / 16 cycles = 48 B/clk. At that rate one 256 B slice takes about 5.3 cycles, not 4.

All results are writer-only (PACK0 packs one face per bank), FP32, 4 tiles per bank, hashing on:

| Test | cycles/tile | SFPU busy | SFPU waits for SrcS | U2 waits for SrcS |
|---|---:|---:|---:|---:|
| SFPU only: U2 fills SrcS once, the SFPU copies with no U2 traffic | 66.25 | 95% | 0.1% | 0% |
| U2 feeds the SFPU (configure-once stream, 1 slot) | 83.75 | 75% | 20% | 0% |

The U2 L1 client counters for the U2-fed case show the same limit:

| Counter (each of the 4 super banks) | Value |
|---|---:|
| Cycles with a grant to U2 | 18.8% |
| Cycles with a U2 request that the super bank did not accept | 1.2–2.3% |

Four super banks × 18.8% = 75% of cycles with a 64 B grant, which is 48 B/clk. L1 rarely refuses U2. U2 does
not send a request for the remaining cycles because its 12-read window is full.

Changes that do not help, because they do not change the in-flight limit:

| Change (3 tiles per bank, full pack, no per-bank U2 drain) | cycles/bank |
|---|---:|
| 1 slot per hand-off | 266 |
| 2 slots per hand-off, 2 UNPACR2 | 268 |
| 2 slots per hand-off, 1 UNPACR2 (16×8 descriptor) | 270 |
| 3 slots per hand-off, 3 UNPACR2 | 275 |

One UNPACR2 across 3 slots (16×12 descriptor) is not a legal tile shape. U2 drops the instruction and never sets
data valid, so the SFPU waits forever. The legal FP32 choices are 16×4 (1 slot) and 16×8 (2 slots).

Hashing changes the sustained rate. The earlier hash comparison measured the same writer-only B-done copy at
4 tiles per bank: **71.0 cycles/tile with hashing off and 82.1 with hashing on**. Hashing on is still the
correct setting, because U0 and PACK0 need it to reach 128 B/clk. The SrcS path pays about 11 cycles per tile
for it.

Still to confirm: the outstanding-read counter (event 6) on U2 subports 21–24 should average near 12.

Corrections to earlier sections:

- For U2 subports 21–24, the super bank is the subport's position in the port (0–3), not subport mod 4.
- `UNPACK2_BUSY` reads 0 because that counter bank does not tap Unpacker 2's busy bit on TRISC3.
- Event 6 is the mean number of outstanding reads (×64). Event 7 is the fraction of cycles with at least one
  read in flight.

Artifacts: [logs/u2_l1_sweep](logs/u2_l1_sweep/). `base_counters/` holds the SFPU-only runs; `sel_<n>/` holds
one L1 client selection each.
