---
name: mmio-race-audit
description: Audit LLK code for races between a RISC MMIO write to a config/GPR register and a Tensix instruction/MOP/replay that consumes it. Use after touching any raw cfg[...]=/reg_write/cfg_rmw/regfile[]= write near an UNPACR/PACR/MOP/CFGSHIFTMASK, or when adding addressing/stride/format register writes.
user_invocable: true
---

# /mmio-race-audit — MMIO write vs Tensix instruction-stream race

## The bug class (precise)
A RISC-V baby core writes a Tensix CONFIG or GPR register via **MMIO** — a direct memory-mapped store from the RISC core, NOT an instruction issued to the Tensix coprocessor. That store does **not** pass through the Tensix wait gate, so a `STALLWAIT` cannot order or hold it back. If a later **Tensix instruction / MOP run / replay execution** depends on that register, you can get:
- **write-too-late**: the consumer reads the stale value (MMIO not landed yet), or
- **write-too-early (cross-call)**: the MMIO write overwrites a register that a *prior, still-in-flight* consumer is reading.

Especially dangerous when a MOP/replay runs, an MMIO write changes a register it reads, then the MOP/replay runs again — or runs in a loop with a per-iteration MMIO change.

## UNSAFE primitives (RISC MMIO — hunt for these)
- `cfg[...] = ...` via `get_cfg_pointer()` / `get_cfg16_pointer()`
- `reg_write(addr, data)`
- `cfg_rmw(...)` / `cfg_rmw_gpr(...)`  (do `cfg_regs[addr] = ...` internally)
- `regfile[idx] = ...`  (raw GPR MMIO store — **easy to miss in greps**)
- `addr_mod_*::set(...)` / `rv_wrcfg` when implemented as a raw `volatile cfg[...] =` (true on **Quasar**; on WH/BH `addr_mod_t::set` uses the ordered `TTI_SETC16` and is SAFE)
- any raw `volatile T* p = reinterpret_cast<...>(BASE); p[i] = ...` to cfg/GPR/MOP-cfg/TDMA register space (`cfg_write()`, `xmov_*`, `mop_cfg[]=`)

## SAFE (ordered in the thread stream — not the bug)
- `cfg_reg_rmw_tensix<>` (emits `TT_RMWCIB`), `TTI_REG2FLOP`, `TTI_WRCFG`, `TTI_SETC16`, `TTI_RMWCIB*`, `TTI_SETADC*`, `TTI_SETDMAREG` — Tensix instructions, in-order through the config unit.
- `sync_regfile_write(idx)` after a `regfile[]` block — a read-back fence that retires the GPR store before dependent instructions issue. (Syncing the LAST index drains all prior regfile writes in the block.)
- A `TTI_STALLWAIT` whose **condition operand** includes `p_stall::TRISC_CFG` (ISA condition C13 = "this thread has a RISC cfg/GPR write emitted but not yet processed") placed BEFORE the consumer. NOTE: a *trailing* `TRISC_CFG` stall only orders the write before the NEXT run — it does **not** protect a *prior* in-flight consumer.
- The consuming unit is provably idle when the write lands (context-acquire / semaphore handshake), with no consumer before the function returns or a sync.

## ⚠️ Architecture difference — do not false-positive on Quasar
**Quasar has hardware Auto-TTSync.** `TRACK_GLOBAL_CFG` is ON by HW default (`tt_llk_quasar/common/inc/ckernel.h` ~L270-310; `p_ttsync` in `ckernel_instr_params.h`), which **auto-orders RISC cfg/GPR MMIO writes against the Tensix instructions that consume them** — replacing the manual `STALLWAIT(TRISC_CFG)`/`REG2FLOP`/`sync_regfile_write` discipline of WH/BH. So raw `cfg[...]=` and `addr_mod_t::set` MMIO on Quasar are the **intended idiom and are SAFE**. The comment `"turned on by default by HW, this should be removed"` means the explicit `set_ttsync_enables<TRACK_ALL>` call is redundant, NOT that the mechanism is untrustworthy. Treat WH/BH with manual-ordering rules; treat Quasar as auto-ordered. (See memory `quasar-auto-ttsync`.)
- **Residual Quasar corner** (cannot be closed without Quasar RTL): `TRACK_ALL` enables `EN_SUBDIVIDED_CFG_FOR_UNPACR`, which stops cross-sync between unpacker-0 and unpacker-1 cfg accesses — a hazard only if a TRISC write to one unpacker's reg is consumed by the *other* unpacker's UNPACR. Flag, don't assert.

## Method
1. **Enumerate** every MMIO write (across all three arches), including the easy-to-miss classes:
   ```bash
   cd tt_metal/tt-llk
   grep -rInE "\bcfg[0-9_]*\[[^]]+\]\s*=[^=]|reg_write\(|\bcfg_rmw(_gpr)?\(|\bregfile\s*\[[^]]+\]\s*=[^=]|get_cfg_pointer|get_cfg16_pointer|rv_wrcfg|\.set\(ADDR_MOD|cfg_write\(|XMOV_(L1_BASE|CMD)\s*\[" \
     tt_llk_* --include=*.h | grep -v /tests/
   ```
   Exclude non-bug-class targets: `pc_buf_base[]=`, `mailbox_base[]=` (semaphore/mailbox, not a register a Tensix instruction reads).
2. **Per site**, read the enclosing function (and callers it triggers). Determine: register written; is there a dependent Tensix instr / MOP / replay AFTER it (incl. loops & double-runs); what sync sits between; is the consuming unit idle.
3. **Confirm the dependency** — which instruction reads the register. ISA names differ slightly from `cfg_defines.h`; map by similarity + logic, and use the tt-isa-docs MCP (WH/BH) to confirm a consumer (e.g. `THCON_SEC0_REG3_Base_address` ← UNPACR addressing; `ADDR_MOD_*` ← MOP/UNPACR/PACR/MOV; `SCRATCH_SEC0_val` ← CFGSHIFTMASK).
4. **Check BOTH directions and value-invariance**:
   - within-call: is the write ordered before its consumer (e.g. a `STALLWAIT(STALL_UNPACK, TRISC_CFG)` before the MOP, with the consumer after the MOP)?
   - cross-call: with double-buffered contexts (`wait_for_next_context(2)` / `switch_config_context`), can a *prior* call's consumer still be in flight when this call's MMIO write lands? Is the written register context-banked (e.g. `REG3_Base` vs `REG3_Base_cntx1`, SAFE) or **shared** (e.g. single `SCRATCH_SEC0_val`, risky)?
   - value: even if a cross-call window exists, is the written value **invariant** across the pipelined calls (e.g. a CB `fifo_page_size` stride)? If the racing write writes the same value the reader needs, the race is **benign**. Say so explicitly.

## Verdict
- **Quasar + relies on Auto-TTSync** → SAFE (not a bug), except the `EN_SUBDIVIDED` cross-unpacker corner (flag).
- **WH/BH ordered (Tensix-instr write / sync_regfile_write / leading TRISC_CFG stall / idle handshake)** → SAFE.
- **WH/BH unordered with a reachable consumer** → RACE (real). Fix = convert the MMIO write to a Tensix-instruction write (`REG2FLOP`/`WRCFG`/`SETC16`), or add `sync_regfile_write`, or a `STALLWAIT(..., TRISC_CFG)` before the consumer.
- **Window exists but value-invariant or unit-idle in all current callers** → UNCERTAIN/latent hardening — say it is not a live bug; let the maintainer decide.

## Thoroughness (optional, for a full sweep)
Fan out one agent per file. Only if the user opts into multi-agent orchestration, run a Workflow: enumerate → per-file analyze → **adversarial verify every file with an in-thread consumer** (try to refute each SAFE verdict; default to UNCERTAIN when safety can't be proven) → completeness critic over the primitive grep (it WILL miss `regfile[` / `cfg_write(` / `xmov` unless added). Be calibrated: prove value-invariance / idle-handshake before calling a flagged site benign; surface anything unproven as UNCERTAIN with reasoning.

## Reference fixes (ground truth)
- PR #602 "Fix race between MMIO and Unpacker" — `cfg[THCON_SEC0_REG3_Base_address] = address` (MMIO) before the unpacker MOP → replaced with `TT_SETDMAREG + TTI_REG2FLOP` + `STALLWAIT(STALL_UNPACK, THCON)`.
- PR #194 "strict ordering in unpack_get_tile".
- BH `_llk_unpack_fast_untilize_bfp_block_` `SCRATCH_SEC0_val` → ordered `SETDMAREG+STALLWAIT(STALL_CFG,THCON)+WRCFG` (latent cross-call hardening; value-invariant today).

## Output
For each flagged site: `file:line`, function, register + MMIO primitive, consumer (instr/MOP/replay) + whether it loops/re-runs, sync between (and direction it covers), arch (and Auto-TTSync applicability), verdict (SAFE / RACE / UNCERTAIN-latent) with one-line reasoning, and the one-line fix. End with totals per verdict.
