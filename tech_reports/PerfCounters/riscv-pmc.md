# RISC-V performance counters inside the Tensix baby cores

Purpose: reference notes for the hardware performance monitor that lives *inside* each baby RISC-V
core (BRISC, NCRISC, TRISC0/1/2). Written so a future session can pick this up without re-reading
the RTL. Everything here was read directly out of the RTL, not inferred from documentation — there
is no vendor documentation for this block.

**Provenance.** RTL: `ws-tensix` at `/proj_sw/user_dev/nrajovic/ws-tensix`, branch
`blackhole_rtl`, tip `85fbcdf`. Software side: tt-metal at `62836841e95`. Investigated 2026-09-18.

**Scope caveat.** `ws-tensix/chips/` contains only `blackhole`, and the branch is `blackhole_rtl`,
so everything below is verified for **Blackhole only**. Whether Wormhole's older baby cores or
Quasar's cores carry the same block is unverified and needs a different RTL checkout.

---

## 1. TL;DR

Each baby core has a real RISC-V HPM block:

- `mcycle` / `minstret` — free-running 64-bit, writable, **not** gated by `mcountinhibit`. The
  cheapest instrumentation available and the natural denominator for any rate.
- `mhpmcounter3` / `mhpmcounter4` — two programmable counters, armable on any of **93 distinct
  microarchitectural events** grouped into 8 pipeline domains.
- Reached with plain `csrr` / `csrw` from the kernel running on that core. No NOC traffic, no
  debug-register arming, no host round-trip, private to that hart.
- **Unused by tt-metal before this work** (§10): unprogrammed and reading zero on every core in
  every build. Now available opt-in through `TT_METAL_PROFILE_RISCV_PMC` (§11).

This is a *different mechanism* from the Tensix MMIO perf counters at `0xFFB12000` that the
`PROFILE_PERF_COUNTERS` path uses. See §9.

**Read §7 before interpreting any stall number.** These counters measure the RISC-V pipeline. A core
waiting on Tensix is usually *not* pipeline-stalled, and the events that do reveal a Tensix wait are
not the ones with "Stall" in the name.

---

## 2. Where the RTL is

All paths relative to `/proj_sw/user_dev/nrajovic/ws-tensix/`.

| File | What it holds |
|---|---|
| `src/hardware/briscv/rtl/tt_csr.sv` | The whole counter block: CSR storage, event select decode, increment logic, read mux. Lines 171–326 are the interesting part. |
| `src/hardware/briscv/rtl/tt_briscv_pkg.vh` | **Authoritative event names and bit positions**, lines 185–312. The `csr_cfg_s` struct at 151–178 defines the custom cfg CSR. |
| `src/hardware/common/rtl/briscv_csr_regs.h` | CSR address macros. |
| `src/hardware/briscv/rtl/tt_ifetch.sv` | Drives the BP and IF event buses (631–644, 646–656). |
| `src/hardware/briscv/rtl/tt_id.sv` | Drives the DE and DE1 event buses (1534–1553, 1511–1532). |
| `src/hardware/briscv/rtl/tt_ex.sv` | Drives the EX event bus (1204–1216). |
| `src/hardware/briscv/rtl/tt_mem.sv` | Drives the LS event bus (800–815); bit 11 from `tt_store_fifo.sv:209`, bit 7 from `tt_lq.sv:272`. |
| `src/hardware/briscv/rtl/tt_local_regs_data_mem.sv` | Drives the MEM event bus (975–990). |

Integration: `tt_tensix.sv` → `tt_risc_wrapper` → `tt_brisc` → pipeline stages + `tt_csr.sv`.

Things that look authoritative but are **not**: `briscv/registers/rdl/csr_regs.rdl` declares generic
32-entry HPM arrays that do not match the implementation, and the DV testbench
`tb_uvm/tests/core_brisc_test_lib.sv` checks `mhpmcounter3`–`10` even though only 3 and 4 exist.

---

## 3. CSR map

| Address | CSR | Access | Behaviour in RTL |
|---|---|---|---|
| `0xB00` / `0xB80` | `mcycle` / `mcycleh` | R/W | Increments every cycle unless being written. Not gated by `mcountinhibit`. |
| `0xC00` / `0xC80` | `cycle` / `cycleh` | R/W | Same physical register, aliased into the unprivileged range — `rdcycle` works. |
| `0xB02` / `0xB82` | `minstret` / `minstreth` | R/W | Increments on `i_id_ex_instdisp`. RTL comment: dispatch count equals retire count on Blackhole. Not gated by `mcountinhibit`. |
| `0xC02` / `0xC82` | `instret` / `instreth` | R/W | Alias of `minstret` — `rdinstret` works. |
| `0xB03` / `0xB83` | `mhpmcounter3` / `3h` | R/W | Programmable counter. 65 bits internally; bit 64 is a sticky overflow flag, not software-visible. |
| `0xB04` / `0xB84` | `mhpmcounter4` / `4h` | R/W | Identical structure to counter 3. |
| `0x323` | `mhpmevent3` | R/W | Event select for counter 3. Eight 4-bit nibbles, one per domain. |
| `0x324` | `mhpmevent4` | R/W | Event select for counter 4. |
| `0x320` | `mcountinhibit` | R/W | Only bits 3 and 4 wired. Bits 0 and 2 (standard cycle/instret inhibit) not implemented. See the bug in §6. |
| `0x7C0` | `cfg` (custom) | R/W | Bit 8 `cfgPmcClrOnRd`, bit 7 `cfgDisPmcWrapArnd`. **Readable in RTL**, even though tt-metal only ever writes it. |
| `0x325`–`0x328` | `mhpmevent5`–`8` | — | Address macros exist, logic and read-mux entries are commented out. Reads return 0. |

Unimplemented CSR reads return 0 silently — the read mux ends in `default: rdata = '0'`
(`tt_csr.sv:733`). There is **no illegal-instruction exception**, so runtime probing for counter
width does not work.

---

## 4. Programming model

`mhpmevent3` is not a flat event ID. It is carved into eight 4-bit nibbles; each nibble selects one
event index out of a 16-bit event bus exported by one pipeline domain.

| `mhpmevent` bits | Domain | Bus | Driven in |
|---|---|---|---|
| `[3:0]` | BP — branch predictor | `i_pmc_bp_events` | `tt_ifetch.sv:631` |
| `[7:4]` | IF — instruction fetch | `i_pmc_if_events` | `tt_ifetch.sv:646` |
| `[11:8]` | DE — decode stalls / hazards / mem dispatch | `i_pmc_de_events` | `tt_id.sv:1534` |
| `[15:12]` | DE1 — decode instruction-mix | `i_pmc_de1_events` | `tt_id.sv:1511` |
| `[19:16]` | EX — retire and mispredict | `i_pmc_ex_events` | `tt_ex.sv:1204` |
| `[23:20]` | LS — load/store unit | `i_pmc_ls_events` | `tt_mem.sv:800` |
| `[27:24]` | TTCSR — Tensix status CSRs held high | `pmc_ttcsr_events` | `tt_csr.sv:769` |
| `[31:28]` | MEM — local mem / L1 / icache port | `i_pmc_mem_events` | `tt_local_regs_data_mem.sv:975` |

Index 0 in every domain is a hardwired `*PmcNone` tie-off, and the RTL treats a zero nibble as
"domain disabled" (`sel = |CSR_MHPMEVENT3[nibble]`), so leaving a nibble at 0 costs nothing.

The decode is `|(bus[15:0] & (1'b1 << nibble))` (`tt_csr.sv:292-299`). The `1'b1` is widened to 16
bits by the context of the `&`, so this behaves correctly — it is not the classic Verilog shift-width
trap it looks like.

### Worked example — icache hit rate

MEM bit 14 is `memPmcIFaccess`, MEM bit 15 is `memPmcIFaccessHit`. MEM is the top nibble, so the
values are `0xE0000000` and `0xF0000000`.

```asm
    li    t0, 0xE0000000
    csrw  0x323, t0            # mhpmevent3 <- MEM.memPmcIFaccess
    li    t0, 0xF0000000
    csrw  0x324, t0            # mhpmevent4 <- MEM.memPmcIFaccessHit
    csrwi 0xB03, 0             # zero mhpmcounter3 (low 32)
    csrwi 0xB04, 0             # zero mhpmcounter4 (low 32)
    csrwi 0xB00, 0             # zero mcycle (low 32) - the denominator
    csrwi 0x320, 0             # mcountinhibit = 0 -> counters run
    # ... region of interest ...
    csrr  a0, 0xB03            # icache accesses
    csrr  a1, 0xB04            # icache hits
    csrr  a2, 0xB00            # cycles
```

Order matters: write `mhpmevent` **before** clearing `mcountinhibit`, because `o_pmc_event_en` also
clock-gates the event pipe registers upstream (§6, last item).

`csrwi` writes a zero-extended 5-bit immediate, so it only clears the low 32 bits of a 64-bit
counter. Fine for short windows; zero the `*h` halves too if the window can overflow 2^32 cycles.

Other useful one-liners: `0x00030000` = EX `exPmcMisPredAny` (all branch mispredicts),
`0x0000D000` = DE1 `dePmcDeValidTTInstr` (Tensix TTI instructions issued), `0x00100000` = LS
`lsPmcLsStallAny`.

---

## 5. Event catalog

Names and bit positions from `tt_briscv_pkg.vh:185-312`. Bits not listed in a domain are tied to
zero by an explicit `*PmcUnused` field. 93 real events plus 8 `None` tie-offs.

### BP — branch predictor, nibble `[3:0]`

| Idx | Event | RTL expression |
|---|---|---|
| 0 | `bpPmcNone` | `1'b0` — tie-off |
| 1 | `bpPmcPcStallAny` | `bp_fetch_stall` |
| 2 | `bpPmcBpStallNoIdRtr` | `~request_stall_no_idrtr & ~i_id_rtr` |
| 3 | `bpPmcBpStallNoMemRtr` | `~request_stall_no_rtr & ~i_instrn_req_rtr` |
| 4 | `bpPmcBpStallPcFifoFull` | `i_instrn_req_rtr & (pc_deco_fifo_full \| pc_deco_fifo_becoming_full)` |
| 5 | `bpPmcBpStallBpFifoFull` | `i_instrn_req_rtr & (bp_deco_fifo_full \| bp_deco_fifo_becoming_full)` |
| 6 | `bpPmcBtbHit` | `bp_hit` |
| 7 | `bpPmcPredTypeJalr` | `pred_instrn_type == 2'b11` |
| 8 | `bpPmcPredTypeJal` | `pred_instrn_type == 2'b10` |
| 9 | `bpPmcPredTypeBr` | `pred_instrn_type == 2'b01` |

### IF — instruction fetch, nibble `[7:4]`

| Idx | Event | RTL expression |
|---|---|---|
| 0 | `ifPmcNone` | `1'b0` — tie-off |
| 1 | `ifPmcIfStallAny` | `~instrn_fifo_pop \| instrn_fifo_empty` |
| 2 | `ifPmcIfStallNoIdRtr` | `o_if_rts & ~i_id_rtr` |
| 3 | `ifPmcIfStallLqFull` | `o_if_rts & i_mem_fe_lqfull` |
| 4 | `ifPmcIfStallLqSbFull` | `o_if_rts & i_mem_fe_skidbuffull` |
| 5 | `ifPmcReqStallAny` | `request_stall` |
| 6 | `ifPmcReqStallNoMemRtr` | `~request_stall_no_rtr & ~i_instrn_req_rtr` |

### DE — decode stalls, hazards, memory dispatch, nibble `[11:8]`

| Idx | Event | RTL expression |
|---|---|---|
| 0 | `dePmcNone` | `1'b0` — tie-off |
| 1 | `dePmcDeStallAny` | `(i_if_instrn_rts \| id_replay) & (~units_rtr \| sync_stall \| raw_hazard_stall)` |
| 2 | `dePmcDeStallNoUnitRtr` | `… & ~sync_stall & ~raw_hazard_stall & ~units_rtr` |
| 3 | `dePmcDeStallIfEmpty` | `~id_replay & ~i_if_instrn_rts & ~sync_stall & ~raw_hazard_stall & units_rtr` |
| 4 | `dePmcDeStallSync` | `… & sync_stall & ~raw_hazard_stall & units_rtr` |
| 5 | `dePmcDeStallHazard` | `… & ~sync_stall & raw_hazard_stall & units_rtr` |
| 6 | `dePmcHazardInt` | `int_rs1_hazard_stall \| int_rs2_hazard_stall` |
| 7 | `dePmcHazardFp` | `fp_rs1/rs2/rs3_hazard_stall \| fp_store_hazard_stall` |
| 8 | `dePmcHazardVec` | `vec_vs1/vs2/vs3/rs1/fs1_hazard_stall` |
| 9 | `dePmcRegDataForward` | OR of int + fp hazard-forward terms |
| 10 | `dePmcMemOpDispatchLd` | `id_rts &` scalar load (`TYPE_I` / `TYPE_A` / `TYPE_IF`) |
| 11 | `dePmcMemOpDispatchSt` | `id_rts &` scalar store (`TYPE_S` / `TYPE_SF`) |
| 12 | `dePmcMemInstrDispatchVecLd` | `id_rts & v_ext & vecldst_autogen.load` |
| 13 | `dePmcMemInstrDispatchVecSt` | `id_rts & v_ext & vecldst_autogen.store` |
| 14 | `dePmcExOpDispatchMul` | `id_rts & EncType == TYPE_RM & funct7 == 'h1` |
| 15 | `dePmcExOpDispatchDiv` | `id_rts & EncType == TYPE_R & funct7 == 'h1` |

### DE1 — decode instruction-mix, nibble `[15:12]`

| Idx | Event | RTL expression |
|---|---|---|
| 0 | `dePmcNone` | `1'b0` — tie-off |
| 1 | `dePmcExOpDispatchCsr` | `id_rts & o_id_type_e` (CSR / ecall / ebreak) |
| 2 | `dePmcExOpDispatchIExt` | `id_rts & i_ext` |
| 3 | `dePmcExOpDispatchMExt` | `id_rts & (EncType == RM \| R) & funct7 == 'h1` |
| 4 | `dePmcExOpDispatchAExt` | `id_rts & a_ext` (atomics) |
| 5 | `dePmcExOpDispatchZbExt` | `id_rts & b_ext` (bit-manip) |
| 6 | `dePmcFpOpDispatchS` | `id_rts &` single-precision FP dispatch |
| 7 | `dePmcFpOpDispatchZfh` | `id_rts &` Zfh half-precision dispatch |
| 8 | `dePmcVecOpDispatchCsr` | `id_rts & vsetOp_to_ex` |
| 9 | `dePmcVecOpDispatchInt` | `id_rts & v_ext & ~ldst & ~(rnden \| saturate)` |
| 10 | `dePmcVecOpDispatchFixed` | `id_rts & v_ext & ~ldst & (rnden \| saturate)` |
| 11 | `dePmcVecOpDispatchFloat` | `id_rts & v_ext & ~ldst & funct3 in {OPFVV, OPFVF}` |
| 12 | `dePmcVecOpDispatchLdst` | `id_rts & vec_ldst_vld` |
| 13 | `dePmcDeValidTTInstr` | `id_rts & trisc_op` — **Tensix TTI instruction issued** |
| 14 | `dePmcDeValidVecReplay` | `id_rts & id_replay` (vector replay mode) |
| 15 | `dePmcDeValidIllegal` | `id_rts & illegal_op` |

### EX — retire and branch mispredict, nibble `[19:16]`

**Measured: `exPmcExStallAny` (index 1) reads identically zero** on all five cores, all 110 cores,
every sample. This is genuine, not a wiring gap: arming index 8 (`exPmcInstRet`) instead reproduces
the `minstret` delta to a ratio of 0.999995, which proves the EX event bus is live and correctly
selected. The EX stage simply never back-pressures on this pipeline — hazards and unit-readiness are
resolved at decode, which is why every `dePmcHazard*` and `de_stall_hazard` event lives in the DE
domain. Use index 2 (`exPmcExStallNoInstr`, EX starved of work) for anything EX-related.

| Idx | Event | RTL expression |
|---|---|---|
| 0 | `exPmcNone` | `1'b0` — tie-off |
| 1 | `exPmcExStallAny` | `i_id_ex_rts & (~i_mem_ex_rtr \| hazard_stall \| int_div_stall)` |
| 2 | `exPmcExStallNoInstr` | `~i_id_ex_rts & (i_mem_ex_rtr & ~hazard_stall & ~int_div_stall)` |
| 3 | `exPmcMisPredAny` | `o_ex_bp_mispredict` |
| 4 | `exPmcMisPredNewBr` | `mis_bp_ntkn_ex_tkn & ~ex_bp_predicted` (branch not in BTB) |
| 5 | `exPmcMisPredDir` | `mis_bp_ntkn_ex_tkn & ex_bp_predicted` (direction) |
| 6 | `exPmcMisPredTgtCond` | `mis_bp_addr_mismatch & ex_branch_target_rts` |
| 7 | `exPmcMisPredTgtUncond` | `mis_bp_addr_mismatch & ex_jalr_target_rts` |
| 8 | `exPmcInstRet` | `i_id_ex_instdisp` — same signal that drives `minstret` |

### LS — load/store unit, nibble `[23:20]`

| Idx | Event | RTL expression |
|---|---|---|
| 0 | `lsPmcNone` | `'h0` — tie-off |
| 1 | `lsPmcLsStallAny` | `mem_tx_valid & ~mem_ready` |
| 2 | `lsPmcLsStallNoMemRtr` | `o_data_req & load_gnt & ~i_data_req_rtr` |
| 3 | `lsPmcLsStallLdqFull` | `lq_full` |
| 4 | `lsPmcLsStallStoreBufFull` | `mem_tx_valid & mem_store & wb_full` |
| 5 | `lsPmcLsStallSkidBufFull` | `skidbuf_full` |
| 6 | `lsPmcFenceOp` | `load_gnt & mem_fence` |
| 7 | `lsPmcLsStallWaitForData` | LQ head valid & load & `~lq_rd_en` (`tt_lq.sv:272`) |
| 8 | `lsPmcOpRet` | `lq_rden` — load-queue retire |
| 9 | `lsPmcStoreReqWins` | `mem_load_valid & ~load_gnt & store_gnt` |
| 10 | `lsPmcWrBufSTLH` | `mem_tx_valid & mem_load & wb_hit` (store-to-load hazard) |
| 11 | `lsPmcWrBufSTSC` | `fifo_wr_en & fifo_merge_en` (`tt_store_fifo.sv:209`) — store coalescing |
| 12 | `lsPmcLoadOp` | `load_gnt & mem_ready & ~vec & ~fence & ~amo` (32b load) |
| 13 | `lsPmcStoreOp` | `mem_store_valid & ~mem_vecldst_vld` (32b store) |
| 14 | `lsPmcVecLoadOp` | `load_gnt & mem_ready & mem_vecldst_vld` |
| 15 | `lsPmcVecStoreOp` | `mem_store_valid & mem_vecldst_vld` |

### TTCSR — Tensix status CSRs held high, nibble `[27:24]`

| Idx | Event | RTL expression |
|---|---|---|
| 0 | `ttcsrPmcNone` | `1'b0` — tie-off |
| 1–8 | `ttcsrPmcSstatus7High` … `ttcsrPmcSstatus0High` | `\|CSR_TTCFG_S{7..0}STATUS` — **note the reversed order**: index 1 is S7, index 8 is S0 |
| 9 | `ttcsrPmcBstatusHigh` | `\|CSR_TTCFG_BSTATUS` |
| 10 | `ttcsrPmcQstatusHigh` | `\|CSR_TTCFG_QSTATUS` |

### MEM — local memory / L1 / icache port, nibble `[31:28]`

| Idx | Event | RTL expression |
|---|---|---|
| 0 | `memPmcNone` | `1'b0` — tie-off |
| 1 | `memPmcStallAny` | `i_brisc_dmem_req & ~o_dmem_brisc_gnt` |
| 2 | `memPmcLocalRegRdOngoing` | `any_local_reg_rd_ongoing` |
| 3 | `memPmcMailboxRdOngoing` | `any_mailbox_rd_ongoing` |
| 4 | `memPmcDataL1RdOngoing` | `any_risc_data_l1_rd_ongoing` |
| 5 | `memPmcCfgRegRdOngoing` | `any_cfg_reg_rd_ongoing` |
| 6 | `memPmcRegRdOngoing` | `any_reg_rd_ongoing` (Tensix reg) |
| 7 | `memPmcTdmaRegRdOngoing` | `any_tdma_reg_rd_ongoing` |
| 8 | `memPmcPcBufRdOngoing` | `any_pc_buf_rd_ongoing` |
| 9 | `memPmcDataAccReqLocal` | `brisc_local_req_accepted` (local mem rd or wr) |
| 10 | `memPmcInsBufWrite` | `req_accepted & target_tensix_instrn_buf & type_write` — TTI push |
| 11 | `memPmcInsBufWriteMultiple` | `memPmcInsBufWrite & \|be[15:4]` — multi-word push, implies TRISC-cache hit |
| 12 | `memPmcDataRdReqL1` | `req_accepted & target_l1 & type_read` |
| 13 | `memPmcDataRdReqHitLoCash` | `o_dmem_brisc_rvalid_1` — L1 read served by low cache |
| 14 | `memPmcIFaccess` | `i_brisc_icache_request` |
| 15 | `memPmcIFaccessHit` | `i_brisc_icache_hit` |

**Measured 2026-09-22: indices 12 and 13 do not observe this core's L1 reads.** `memPmcDataRdReqL1`
peaks at 1e-4 of cycles and `memPmcDataRdReqHitLoCash` is flat zero, while `memPmcDataL1RdOngoing`
(index 4) clearly shows L1 reads in the same workload. Use index 4 for L1 read activity.

---

## 6. Semantics and traps

**`mcountinhibit` bit 4 does not actually gate counter 4.** The increment is gated by
`~CSR_MCOUNTINHIBIT[4]`, but the global event enable tests bit 3 in *both* terms
(`tt_csr.sv:326`):

```verilog
assign o_pmc_event_en = ((|CSR_MHPMEVENT3[31:0] & ~CSR_MCOUNTINHIBIT[3])
                       | (|CSR_MHPMEVENT4[31:0] & ~CSR_MCOUNTINHIBIT[3]));
```

So `mcountinhibit = 0x10` leaves counter 4 running, and `mcountinhibit = 0x8` silently freezes
counter 4 as well by killing the shared enable. **Treat `mcountinhibit` as one global run bit at
bit 3.** This looks like an RTL bug rather than intent.

**The increment is an OR, never a popcount.** The addend is the boolean OR of the eight
`(sel & event)` terms (`tt_csr.sv:231-235`, `260-264`), so the maximum rate is one count per cycle
no matter how many domains fire together. A counter armed on several events cannot tell you which
one fired, and a cycle where several fire counts as 1. Totalling two event streams needs both
counters.

**65 bits wide with a sticky overflow flag.** Each counter is `reg [64:0]` (`tt_csr.sv:218`). Bit 64
is not visible through any CSR; it exists so that `cfgDisPmcWrapArnd` (cfg bit 7) can saturate
instead of wrapping once the 64-bit value overflows. Leave the bit clear and the counter wraps.

**Clear-on-read is per half-word.** With `cfgPmcClrOnRd` (cfg bit 8) set, reading the low CSR zeroes
bits 31:0 and reading the high CSR zeroes bits 63:32 plus the overflow bit (`tt_csr.sv:223-229`). A
64-bit snapshot must read high then low and **cannot be made atomic**. For short windows just read
the low word.

**`mcycle` and `minstret` are writable and ungated** (`tt_csr.sv:171-207`). Plain 64-bit up-counters
with a write port and no `mcountinhibit` term. `mcycle` is the denominator for turning any event
count into a per-cycle rate; `minstret` gives IPC for free.

**Events are sampled one cycle late, and only while armed.** Every event bus passes through a
`tt_pipe_stage` enabled by `o_pmc_event_en` (`tt_ifetch.sv:644`, `tt_mem.sv:815`, `tt_ex.sv:1216`).
The buses hold stale values until an event select is programmed and go stale again the moment it is
cleared. Expect one cycle of skew and do not trust the first cycle after arming.

**Unimplemented CSRs read zero rather than trapping** — see §3.

---

## 7. Telling "blocked on Tensix" apart from a pipeline stall

This is the single biggest interpretation trap, and it matters most on the TRISCs.

**The `*StallAny` events only see the RISC-V pipeline.** They count cycles where a stage could not
advance because of its own hazards, backpressure, or empty upstream queue. A baby core that is
waiting for the Tensix datapath is, from its own pipeline's point of view, usually running perfectly
well. So a TRISC can burn 90% of its cycles waiting on Tensix and still report near-zero
`dePmcDeStallAny`.

Three distinct mechanisms, each with a different signature:

### 7.1 TTI backpressure shows up as a *store* stall, not a decode stall

A Tensix instruction is issued by the baby core as a **write to the Tensix instruction buffer
address** — that is literally what `memPmcInsBufWrite` counts
(`req_accepted & target_tensix_instrn_buf & type_write`, MEM index 10). When Tensix cannot accept it,
the write back-pressures the load/store unit. So the observable signature of "TRISC blocked because
Tensix is full" is:

| Event | Domain / index | What it shows |
|---|---|---|
| `memPmcStallAny` | MEM 1 | `i_brisc_dmem_req & ~o_dmem_brisc_gnt` — the memory request was not granted |
| `lsPmcLsStallStoreBufFull` | LS 4 | `mem_tx_valid & mem_store & wb_full` — write buffer full behind it |
| `lsPmcLsStallAny` | LS 1 | `mem_tx_valid & ~mem_ready` |

Do **not** look for this in the DE or EX domains. It never appears there.

### 7.2 Semaphore and circular-buffer waits show up as retired instructions

A `cb_wait_front` / semaphore spin is a load-compare-branch loop. It retires instructions at full
rate. The signature is *high* `minstret` relative to `mcycle` (IPC near the loop's natural rate),
high `lsPmcLoadOp` (LS 12), and **low** stall counts everywhere. If you see a core with a big cycle
count, a healthy IPC and almost no stalls, it is spinning, not working.

This is why `mcycle` and `minstret` being free in every run matters: the IPC is what separates
"spinning" from "stalled".

### 7.3 The TTCSR domain is the only direct view of Tensix state

Nibble `[27:24]` counts cycles where a Tensix status register, mirrored live into this core's CSR
file, was non-zero. The mirroring is real hardware tracking, not something software has to poll:

```verilog
assign i_qstatus_wren = |(CSR_TTCFG_QSTATUS ^ i_tt_qstatus);
...
else if (i_qstatus_wren & ~CSR_CFG.cfgDisTTStatusReg)
    CSR_TTCFG_QSTATUS <= i_tt_qstatus;
```
(`tt_csr.sv:545-567`, same pattern for `BSTATUS` and `S0..S7STATUS`)

| Event | Index | Meaning |
|---|---|---|
| `ttcsrPmcQstatusHigh` | 10 | Tensix instruction queue status non-zero — **Tensix has queued work outstanding** |
| `ttcsrPmcBstatusHigh` | 9 | Tensix busy status non-zero — **some Tensix unit is busy** |
| `ttcsrPmcSstatus0High` … `7High` | 8 … 1 | Sync/semaphore status register N non-zero (note: index 8 is S0, index 1 is S7) |

**Measured 2026-09-18: this domain saturates and is useless as a utilisation measure.** On SDPA
prefill, `ttcsrPmcQstatusHigh` equals `mcycle - 1` on every one of the 550 core-instances, and
`ttcsrPmcBstatusHigh` is within 0.06–0.4% of `mcycle`. That is the predictable consequence of the
event being `|CSR_TTCFG_QSTATUS` — a reduction-OR over a whole 32-bit status register, so any single
non-idle bit anywhere in it pins the event high. It tells you Tensix was not completely idle, which
under a real workload is always true.

So do **not** use the TTCSR domain for "how blocked on Tensix was this core". Use §7.1 instead: the
LS and MEM store-backpressure events, which on the same capture cleanly separate the five cores.
**Measured 2026-09-22: the narrower bits do not rescue it.** `ttcsrPmcSstatus0High` and
`ttcsrPmcSstatus1High` read exactly zero on all five cores, so the whole TTCSR domain is unusable on
this silicon.

**Prerequisite:** this whole domain goes dead if `cfgDisTTStatusReg` (cfg CSR `0x7C0`, bit 29) is
set, because the shadow registers stop tracking the Tensix wires. tt-metal only ever touches bits 0,
3 and 18 of that CSR, so the domain is live in all current builds — but check this first if the
TTCSR counts come back as zero.

### 7.4 `dePmcDeStallSync` is not a Tensix wait

Worth stating explicitly because the name invites the mistake. `sync_stall` is set when a serializing
operation has to drain the load queue, and cleared on `i_mem_fe_lqempty` (`tt_id.sv:798-817`). It is
a memory-ordering serialization stall — CSR accesses and similar — and has nothing to do with Tensix.
Note that our own instrumentation reads four CSRs per launch, which will itself register here.

### 7.5 Practical decomposition

For a given core, per run, normalise everything by that run's `mcycle`:

- `pipeline_stall_frac` = union-of-stalls counter (`0x10110111`) / `mcycle`
- `tti_push_rate`       = `memPmcInsBufWrite` / `mcycle`
- `ipc`                 = `minstret` / `mcycle`
- `stall_per_tti`       = `memPmcStallAny` / `memPmcInsBufWrite`, armed in the same pass

The TTCSR-based fractions (`ttcsrPmcBstatusHigh`, `ttcsrPmcQstatusHigh`) are deliberately absent:
they saturate (§7.3). For comparing cores or configs, prefer the work-normalised `stall_per_tti`
over the per-cycle fractions (§11, "normalise by work, not by cycles").

A core that is not pipeline-stalled but has low IPC is waiting on something the RISC-V PMC cannot
see directly — go to the Tensix MMIO counters (§9) for that.

---

## 8. Which cores have it

All five baby cores instantiate the same `tt_risc_wrapper` → `tt_brisc` → `tt_csr.sv` stack, so the
block is structurally identical everywhere. **There is no parameter that removes the PMC hardware
from a core.**

| Core | Instance | Integration site | Vector events reachable |
|---|---|---|---|
| NCRISC | `noc_control` | `tt_tensix.sv:5776` | no |
| BRISC | `tensix_control` | `tt_tensix.sv:6184` | no |
| TRISC0 | `u_trisc[0]` | `tt_tensix.sv:6950` | no |
| TRISC1 | `u_trisc[1]` | `tt_tensix.sv:6972` | no |
| TRISC2 | `u_trisc[2]` | `tt_tensix.sv:6994` | **yes** |

The only per-core difference is which events can ever fire: only TRISC2 is built with
`INCL_VEC=1`, so the vector dispatch and vector load/store events read zero on the other four.

The MEM domain events are named for BRISC (`memPmcInsBufWrite` etc.) because
`tt_local_regs_data_mem.sv` is shared, but they are live on every core whenever that core drives the
matching address target.

---

## 9. Not the same thing as the Tensix perf counters

Two independent mechanisms, easy to confuse:

| | RISC-V PMC (this doc) | Tensix perf counters |
|---|---|---|
| Access | CSR instruction, from the core itself | MMIO debug regs at `0xFFB12000` |
| Width / rate | 2 counters, 65 bits, max +1 per cycle | 128-bit records: `ref_cnt` / `req_cnt` / `grant_cnt` / `stall_cnt` |
| Scope | Per-core, private to that hart | Shared Tensix resource, banked by unit |
| Observes | Fetch, decode, execute, load/store, icache, L1 port | FPU, packers, unpackers, instruction path, L1 banks |
| tt-metal support | `TT_METAL_PROFILE_RISCV_PMC` (§11) | `PROFILE_PERF_COUNTERS`, `tt_metal/hw/inc/internal/profiler/perf_counters.hpp` |

The Tensix-side counters have their own set of misnamed, aliased and dead entries; they are not
covered here.

---

## 10. Software status in tt-metal before this branch

As of `62836841e95` (main at `da966befb9c` plus the SDPA analysis branch), **nothing in tt-metal touched this hardware**. A repo-wide search for
`mhpmcounter`, `mhpmevent`, `mcountinhibit`, `mcycle` and `minstret` returned no device-code hits at
all: the counters were unprogrammed and read zero on every core in every shipping build. §11
describes the instrumentation added on this branch.

The other timing mechanisms, all still in place and untouched:

- **Device profiler timestamps come from the memory-mapped wall clock**, not `mcycle`:
  `RISCV_DEBUG_REG_WALL_CLOCK_L/H` on Wormhole and Blackhole, NEO-local wall-clock registers on
  Quasar. `tt_metal/hw/inc/api/debug/kernel_profiler.hpp`.
- **`PROFILE_PERF_COUNTERS` drives the Tensix MMIO banks**, `RISCV_DEBUG_REG_PERF_CNT_*`.
  `tt_metal/hw/inc/internal/profiler/perf_counters.hpp`.
- **The only cycle-counter instruction anywhere** is an opt-in `rdcycle` on Quasar DM cores behind
  `TT_METAL_MEASURE_DFB_INIT_TIME`, for dataflow-buffer init timing.
  `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_init.h:262`.

---

## 11. The instrumentation on this branch

Added 2026-09-18. Opt-in, off by default, and a no-op for every core and arch that is not a
Blackhole Tensix baby core.

### Enabling it

```bash
export TT_METAL_PROFILE_RISCV_PMC=0x10110111,0x0A000000   # mhpmevent3, mhpmevent4
```

One or two selects, hex or decimal, comma separated; `mhpmevent4` defaults to 0. Setting it forces
the device profiler on. The selects are compile-time constants, so they are folded into
`RunTimeOptions::get_compile_hash_string()` and therefore into the firmware build key — changing
them rebuilds the firmware instead of silently reusing the previous pass's image. The hash is
appended only when the feature is on, so every existing JIT cache and pre-compiled bundle stays
valid for anyone not using it.

### What it touches

| File | Change |
|---|---|
| `tt_metal/hw/inc/internal/profiler/riscv_pmc.hpp` | New. CSR arm/read, delta computation, marker emission. |
| `tt_metal/llrt/rtoptions.{hpp,cpp}` | `TT_METAL_PROFILE_RISCV_PMC` parsing, getters, compile hash. |
| `tt_metal/jit_build/build.cpp` | Emits `-DPROFILE_RISCV_PMC{,_EV3,_EV4}` into firmware and kernel builds. |
| `tt_metal/hw/firmware/src/tt-1xx/{brisc,ncrisc,trisc}.cc` | Include `riscv_pmc.hpp` directly, and `RiscvPmcBegin()` / `RiscvPmcEnd()` around the kernel invocation. Three files cover all five cores because `trisc.cc` is compiled three times. |

`riscv_pmc.hpp` is deliberately not included from `kernel_profiler.hpp`: the includes there sit
inside its `#if defined(PROFILE_KERNEL)` block, so on a profiler-off build the null macros would
never be defined.

The offline tooling lives on the SDPA analysis branch (`nrajovic/sdpa_runs`), not in the core
library:

| File | Change |
|---|---|
| `analysis/post_process.py` | `load_riscv_pmc()` marker loader. |
| `analysis/zone_reduce.py` | Skips the new marker ids, writes `<tag>_riscv_pmc.csv`. |
| `analysis/campaigns/riscv_pmc_table.py` | New. Decodes selects to event names, normalises by `mcycle`, emits the model-facing tables. |
| `analysis/campaigns/campaign_pmc.sh` | New. Drives the single-config multi-pass event sweep. |
| `analysis/campaigns/run_zone.sh` | Records the selects in the PROVENANCE line. |
| `analysis/campaigns/zone_decomp.py` | Folds per-RISC PMC columns into `decomp_summary.csv`. |
| `analysis/campaigns/campaign_pmc_full.sh` | New. All Tensix counters plus the full PMC sweep over a config grid. |
| `analysis/campaigns/pmc_tensix_corr.py` | New. Cross-config correlation of PMC against Tensix counters. |

Nothing was added to the memory map, `dev_msgs.h`, the HAL, or the host profiler: the readout rides
the existing per-RISC profiler buffer, which already has one slot per processor
(`profiler_msg_t::buffer[PROCESSOR_COUNT]`) and is already pushed to DRAM and serialised to
`profile_log_device.csv`.

### Wire format

Three `TS_DATA` markers per core per launch, all payload in the CSV `data` column:

| Marker id | `data` |
|---|---|
| 9092 | `(mhpmcounter4 delta << 32) \| mhpmcounter3 delta` |
| 9093 | `(minstret delta << 32) \| mcycle delta` |
| 9094 | `(mhpmevent4 << 32) \| mhpmevent3` |

Plain `TS_DATA` rather than one wider `TS_DATA_16B` packet on purpose: a `TS_DATA_16B` trailer lands
in `TTDeviceMarker::data_high`, which has **no column** in `profile_log_device.csv`. It only reaches
a reader through host-side `meta_data` enrichment, and that enrichment only runs for markers emitted
inside an open profiler zone. Keeping each value in `data` means every row survives with no
host-side change.

Re-emitting the selects every launch costs four words and makes each capture self-describing, which
is what keeps the sweep honest — a row cannot be attributed to the wrong event pair during the
offline merge.

### Design decisions worth not re-litigating

- **Deltas, not clear-on-read.** `cfgPmcClrOnRd` lives in cfg CSR `0x7C0` next to the L1-data-cache,
  gathering and memory-ordering bits that `firmware_common.h:289` programs at boot. Reading before
  and after and subtracting avoids composing read-modify-writes with those sequences entirely.
- **`mcountinhibit` is never written.** Its reset value of 0 runs both counters, so the bit-3
  aliasing bug in §6 can only bite someone who writes it.
- **No `flush_to_dram_if_full()` before the markers.** That pulls `quick_push()` into the reference
  graph, and `quick_push()` touches `noc_index`, which only `brisc.cc` defines — on NCRISC firmware
  it is dead code that LTO drops, so calling the flush breaks the NCRISC link with an undefined
  reference to `noc_index`. `timeStampedData` already checks `bufferHasRoom`, so a full buffer drops
  these markers rather than displacing zone markers, and the loader tolerates short streams.
- **BRISC has no kernel in some programs.** The hooks bracket `WAYPOINT("R")`..`WAYPOINT("D")`, which
  on BRISC spans both the kernel arm and the `wait_for_go_message()` arm. When BRISC has no kernel
  the counters therefore cover that spin, which is honest but worth knowing when reading the data.

### Verified on silicon

First capture 2026-09-18 on the p100a, SDPA prefill S=4096 q=k=128 causal, 110 cores,
`$DD=/proj_sw/user_dev/nrajovic/sdpa_campaign_p100a_pmc`, tag `pmc_stall_union`
(`mhpmevent3=0x10110111`, `mhpmevent4=0x00000001`).

- **No dropped markers.** 1650 rows of each of 9092/9093/9094 = 110 cores × 5 RISCs × 3 launches.
- **`mcycle` agrees with the wall clock to 0.03%.** Per-core `mcycle` delta against the independent
  memory-mapped-wall-clock KERNEL zone duration gives a ratio of 1.00007–1.00030 on all five cores.
  The small excess is this instrumentation's own four CSR reads sitting inside the bracket. This is
  the cross-validation that matters: two unrelated clocks agreeing means both the counters and the
  `WAYPOINT("R")`..`WAYPOINT("D")` bracketing are right.
- **`stall_union_all + IPC ≈ 1.00` on every core** (BRISC 0.598+0.402, NCRISC 0.671+0.331,
  TRISC0 0.443+0.559, TRISC1 0.783+0.238, TRISC2 0.696+0.313). Exactly what a single-issue pipeline
  should give: a cycle either retires an instruction or is stalled somewhere. Strong evidence the
  union counter measures what it claims — and a warning that it is largely redundant with IPC, so
  the information is in the per-stage decomposition, not the union.
- **Spread is real, not noise or saturation.** TRISC1 stall fraction over 220 samples
  (110 cores × 2 kept iterations): mean 0.783, std 0.042, range 0.696–0.819.
- **`exPmcInstRet` reproduces `minstret` to 0.999995**, independently confirming that an armed
  counter tracks the exact RTL signal the catalog says it does.
- **`tti_push` is exactly zero on BRISC and NCRISC** and non-zero only on the three TRISCs
  (0.056 / 0.109 / 0.097). Only TRISCs issue Tensix instructions, so the MEM domain is attributing
  correctly per core.

Full nine-pass sweep tags: `pmc_{stall_union,if_de,ex_ls,mem_tti,tensix,backpress,icache,l1,ldst}`
plus `pmc_exselftest`. Reduced into `pmc_{long,by_core,by_risc}.csv` in the same `$DD`.

### Combining with the Tensix MMIO counters

The two mechanisms compose in a single capture, verified on the `pmc_combo` tag
(`run_zone.sh pmc_combo mp` with `TT_METAL_PROFILE_RISCV_PMC` set):

- **One run yields both.** `zone_reduce.py` emits `<tag>_counters.csv` (51150 rows, all 155 Blackhole
  Tensix counters across tracy's five passes) and `<tag>_riscv_pmc.csv` (1650 rows = 110 cores × 5
  RISCs × 3 invocations) from the same `profile_log_device.csv`. No BRISC `.text` overflow.
- **Tracy's multipass merge already does the right thing, by accident.**
  `merge_perf_counter_device_logs()` takes pass 0's CSV whole and appends only marker id 9090 rows
  from later passes. So the PMC rows survive exactly once, from pass 0. That is also semantically
  correct, because the PMC event selects are constant across the Tensix passes. **Do not "fix" that
  filter to also carry 9092-9094 from later passes** — it would produce five redundant copies of
  identical data.
- **They join on `(core_x, core_y)`**, which is what makes the combination worth the run time: the
  RISC-V side says why a core stalled, the Tensix side says what the datapath achieved.
- **Cost.** `mp` mode is ~6.5 min per PMC event pair against ~1 min for `plain`, because the
  workload is replayed once per Tensix pass. A full nine-pair PMC sweep in `mp` mode is about an
  hour. Only worth it for the pairs whose correlation against Tensix counters you actually want.

### Do not correlate per-cycle fractions across cores

An earlier revision of this file reported a -0.995 cross-core correlation between TRISC_1's MEM-stall
fraction and Tensix math utilisation, read as evidence that the stall limits Tensix. **That reading
was wrong** and the method behind it does not work. Three measurements show why:

1. **The Tensix per-core utilisations are one measurement, not many.** `MATH_COUNTER`,
   `FPU_COUNTER`, `SFPU_COUNTER`, `PACKER_BUSY`, `UNPACK0_BUSY_THREAD0` and `ANY_THREAD_STALL`
   correlate with each other at 0.9958–1.0000 across the 110 cores; the first principal component
   explains **99.89%** of their variance. Correlating against any one of them is the same test.
2. **Per-core `mcycle` varies 2.09×** (1,183,872 to 2,473,190) because causal masking gives each
   core a different number of q-chunks. Every per-cycle fraction, RISC-V and Tensix alike,
   correlates with `mcycle` at |r| ≈ 0.97.
3. **Partialling out `mcycle` does not help** — the correlations stay at ±0.98 to ±1.000. That is
   the tell: this is not confounding that a control variable can remove, it is an accounting
   identity. On a core that does its work in `T` cycles, "fraction of cycles doing useful work" and
   "fraction of cycles stalled" are near-complementary by construction, the same way
   `stall_union + IPC ≈ 1` from the first sweep.

Two per-cycle fractions sharing a denominator will correlate at |r| ≈ 1 whatever the mechanism. A
correlation that strong between physically distinct quantities measured on 110 different cores is
evidence of an identity, not of a discovery.

### What does work: normalise by work, not by cycles

`pmc_combo` armed `memPmcStallAny` and `memPmcInsBufWrite` on the same run, so a stall cost per unit
of Tensix work can be formed per core without crossing captures. The dispersion of that ratio is the
discriminator:

| Core | stall fraction of cycles (cv) | stall cycles per TTI push | cv | range |
|---|---|---|---|---|
| TRISC_0 unpack | 0.009 (18.8%) | **0.156** | **1.0%** | 0.152–0.160 |
| TRISC_2 pack | 0.015 (19.2%) | **0.157** | **0.1%** | 0.157–0.157 |
| TRISC_1 math | 0.352 (34.2%) | **3.513** | **43.5%** | 0.671–4.968 |

For unpack and pack the per-push cost collapses to a constant — 0.156 cycles of memory stall per
Tensix instruction issued, identical on both and stable to 1% or better across all 110 cores. Their
cross-core variation in stall *fraction* was pure work volume. That cost is the price of issuing a
TTI and there is nothing there to optimise.

For math it does not collapse: the cost per push spans 0.67 to 4.97 cycles, a 7× spread, while the
other two threads sit at 0.157 on the same cores in the same run. So TRISC_1's stall is
**contention-dependent, not a fixed issue cost** — it is genuinely being back-pressured by something
that varies core to core. That is the real bottleneck signal, and it is invisible in the per-cycle
fraction because work volume dominates it.

This is the answer to "blocked by Tensix accepting work versus blocked by its own store path":
unpack and pack pay a constant store-path cost; math is blocked by something external and variable.

### Method note for anything downstream

- Compare **counts normalised by work** (per TTI push, per q-chunk), not fractions of cycles.
- To test a mechanism, vary a config knob and correlate **across configs**, not across cores at one
  config. Cross-core spread at a fixed config is mostly work assignment.
- Arm both events of a comparison **in the same pass**, so the ratio is formed per core without
  crossing captures that may differ in work distribution.

`pmc_tensix_correlation.csv` in `$DD` holds the raw and partial correlation table, kept as the
record of the method that does not work.

### Running a campaign with the PMC on

The PMC costs **no extra runs**. Export the selects once and every capture in the block gains
per-core PMC rows, because each core reads its own CSRs inside the same kernel launch:

```bash
export TT_METAL_PROFILE_RISCV_PMC=0x10110111,0xA0000000   # stall union, TTI push
cd $DD && $TTM/analysis/campaigns/campaign_t21.sh 2>&1 | tee $DD/t21_campaign.out
```

`run_zone.sh` records the selects in the PROVENANCE line, `zone_reduce.py` writes
`<tag>_riscv_pmc.csv` for every capture, and `zone_decomp.py` folds 33 columns per config into
`decomp_summary.csv` (`pmc_src`, `pmc_ev3`, `pmc_ev4`, then per RISC the two event fractions, IPC,
mean cycles, and the work-normalised `ev3_per_ev4` ratio with its core-to-core cv). It reads the
**zones-off** capture by preference, because zone markers perturb the stalls being measured.

Verified on the full T2.1 grid, 2026-09-18: 42 runs, zero failures, all 42 PMC captures complete at
1650 rows (110 cores × 5 RISCs × 3 invocations), 662 MB, 13:19–14:53 UTC. Per (q,k,mode) triple of
zoff + zon + zon_mp is about 6m40s, so a 14-point block is ~1h35m.

Scope caveat: `tt-metal-fresh` does not exist in this workspace, so the T2.6, T2.7 and R1e blocks
cannot run at all, PMC or not.

### T2.1 result: the metric tracks the model's error

The load-bearing numbers:

- **Stall-union cycles per TTI push correlates r = +0.924 with wall-over-model-floor** across the 14
  configs (TRISC_1; TRISC_0 +0.904, TRISC_2 +0.729). r² = 0.854.
- **It discriminates configs, not cores** — the test §"normalise by work" demanded. Between-config
  cv is 48–60%; within-config core cv is 10–12%. A 4–5× span per thread.
- **q_chunk is the driver and it is inverse** (r ≈ -0.57 on TRISC_0/1). Smaller q_chunk means each
  Tensix instruction does less work while the fixed per-push overhead stays, so cost per unit work
  balloons. k_chunk is near-irrelevant except mildly on the packer (+0.417).
- **Two regimes.** Seven configs sit at a floor of 4.16–4.84 cyc/push with a within-config core cv of
  **1.3%** — an irreducible per-instruction cost. The other seven sit at 5.01–16.82 with core cv
  **19.2%**, i.e. contention that varies core to core. Worst is causal q64 k128: 16.8 cyc/push, 4×
  the floor, IPC 0.121, 89% of cycles stalled.
- **The causal penalty is entirely a small-q effect**: 2.1× at q=64, zero by q=512.

Not established: causation. Fourteen points, and q_chunk drives both the metric and the math floor,
so part of the correlation is shared dependence. The discriminating test is the R1g hold-outs (five
head_dim 64 shapes never used to fit anything, 5 runs, ~15 min) — if a floor-plus-penalty form
predicts their wall-over-floor from their measured cyc/push, the relationship is real.

Two corrections to the RTL-derived expectations came out of this capture: the TTCSR domain saturates
(§7.3) and `exPmcExStallAny` is genuinely always zero (§5, EX). Both are recorded at the point of
use rather than only here.

Build-level checks, via `precompile_fw` (builds Blackhole firmware with no device attached):

- Host `tt_metal` builds clean; firmware links clean for all five baby cores across all 72 Blackhole
  offline configs.
- All five ELFs contain exactly two `csrw mhpmevent3/4` and eight CSR reads, with `0x10110111` and
  `0x0A000000` as literal immediates.
- Wormhole configs are byte-identical to baseline, confirming the `ARCH_BLACKHOLE` guard.
- BRISC `.text` grows 6744 → 7096 bytes (+352). Combining with `TT_METAL_PROFILE_PERF_COUNTERS=47`
  also links, so the two features coexist within the BRISC budget.
- The loader and reducer were exercised against a synthetic capture: correct unpacking, correct
  discard of invocation 0, and every select in the sweep decodes to the intended event name.

---

## 12. Open questions

- Does Wormhole's baby-core RTL have this block, and with the same event encoding? Needs a Wormhole
  RTL checkout — `ws-tensix` here is `blackhole_rtl` only.
- Same question for Quasar. tt-metal already emits `rdcycle` on Quasar DM cores, which implies at
  least `mcycle` exists there, but the `mhpmcounter` side is unverified.
- Is the `mcountinhibit[3]` asymmetry in `o_pmc_event_en` a known bug? Worth asking the Tensix RTL
  owners before building anything on top of counter 4.
- Sections 1-10 are read from RTL and have not been checked against silicon behaviour beyond what
  the first capture in §11 happens to confirm. In particular the *meaning* of individual per-stage
  events is still RTL-derived: the union counter and `mcycle` are now validated, the individual
  stall events are not.
- The one firmware-integration gotcha worth remembering: `build_Release/lib/libtt_metal.so` is
  populated by `cmake --install`, not by `cmake --build`. Building only the `tt_metal` target
  leaves `ttnn` loading a stale library, and the symptom is silent — the run succeeds and simply
  contains no PMC markers. Always `cmake --build` then `cmake --install`.
