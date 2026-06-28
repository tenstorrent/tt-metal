---
name: cfg-word-overlap-audit
description: Audit LLK code for cross-thread races on the backend CONFIG register file where differently-named fields written by unpack/math/pack land in the SAME 32-bit config word. Use after adding/changing any ALU_FORMAT_SPEC / ALU_ACC_CTRL / ALU_ROUNDING_MODE / STACC_RELU / THCON_SEC* write, any WRCFG_32b full-word write, or any cfg_reg_rmw_tensix on a word another thread also touches.
user_invocable: true
---

# /cfg-word-overlap-audit — Cross-thread CFG register-file word overlap

> **Ground-truth precedence:** the live ISA doc (tt-isa-docs MCP, fetched each run) outranks every rule, table, and example baked into this skill — treat those as dated illustrations. If the live ISA doc **contradicts** a baked rule here, do NOT silently proceed: surface the conflict to the user and ask whether the baked rule should be overwritten, discarded, or kept. Default to the ISA doc.
>
> **Coverage — floor, not ceiling.** The grep patterns and site lists in this skill are a **seed, not an exhaustive enumeration**. After running them, widen the search with full reasoning. The techniques here are **illustrative examples, not the allowed set** — use any approach your reasoning suggests, including ones not listed: e.g. semantic search (by behavior/effect, not just token), resolving macros / wrappers / typedefs / indirection the literal pattern can't match, following the call graph to callers and callees, and diffing the WH/BH/QSR variants to catch a site present in one arch and missing in another. If you can find a hazard, primitive, or site the encoded patterns don't cover — by any means — pursue and report it; do **not** clamp a stronger analysis to this list or to these techniques. State any residual coverage gaps explicitly (no silent caps).
>
> **Execution — parallel by default.** When enumeration yields more than a few sites/files, **fan out concurrent `Agent` calls by default** (one per file/subsystem, a fresh context each), saturating the available concurrency (~10–16 at once); go inline only for a trivial set. The per-file fan-out described under *Thoroughness* is the **default**, not an exhaustive-only option. The cross-referencing/synthesis of results stays sequential (it must follow the per-unit findings). The heavyweight **Workflow** tool still requires explicit multi-agent opt-in — it is the opt-in exhaustive tier, not the default. Don't over-spawn a tiny diff.
>
> **Persisting results — single writer, incremental.** Agents only **return** their findings; they never write a shared file (no concurrent-write clobbering). If findings are persisted to a file, the orchestrator/caller is the **sole writer** and **appends each wave's returns as they arrive** — incremental, never only-at-the-end — so an interrupt preserves every completed wave's findings.

## The bug class (precise)
The three Tensix threads (T0=unpack, T1=math, T2=pack) do **not** share GPR files, but they all write the shared **backend `Config` register file** (`Config[2][...]` at `TENSIX_CFG_BASE`). LLK addresses it by *named field* (`<REG>_ADDR32` word index + `_MASK`/`_SHAMT`). Two **differently-named** fields can occupy the **same 32-bit word** — invisible from the names alone, visible only when you resolve `_ADDR32` to a number. If different threads write the same word, you can lose one thread's field. The classic example: `STACC_RELU_*` (packer) and `ALU_ACC_CTRL_Zero_Flag_disabled_*` (math/unpack) share a word; a packer full-word write zeroes the math field.

## HW rules that decide whether an overlap is a RACE (ground these first)
From the tt-isa-docs (`RMWCIB.md`, `BackendConfiguration.md` — fetch via the tt-isa-docs MCP for WH/BH):
- **One Configuration Unit serializes all config writes** → no bit-tearing; every `WRCFG`/`RMWCIB`/`REG2FLOP`/RISC `sw` completes atomically w.r.t. other config writes.
- **`RMWCIB` is atomic per byte**: `*byte = (new & mask) | (old & ~mask)`. Two different threads RMW'ing the **same word with disjoint masks** is therefore **always safe**, regardless of order.
- **`Config` has 2 banks selected by the issuing thread's `CFG_STATE_ID`** ("any thread can access any bank"). In steady state all three threads run on bank 0 (`cfg_state_id==0`), so low words ARE physically shared. `flip_cfg_state_id`/`TTI_SETC16(CFG_STATE_ID_StateID,…)` only diverges briefly (e.g. `llk_math_fast_tilize`). Words `>= GLOBAL_CFGREG_BASE_ADDR32` are single-copy (write hits both banks).

### The ONLY two patterns that can corrupt
1. **Full-word write across a register boundary** — `TTI_WRCFG(..., p_cfg::WRCFG_32b, ADDR)` or RISC `cfg[ADDR]=`/`sw` — writes all 32 bits, so it overwrites *every other field in that word owned by another thread*. This is the sharp bug.
2. **Two threads RMW the SAME field bits** with possibly-different values → last-writer-wins. Correct value then depends entirely on inter-thread ordering.

### Safe by construction (do NOT flag)
- Disjoint-mask `cfg_reg_rmw_tensix<>` / `RMWCIB` from different threads to the same word (byte-atomic).
- A field written by only one thread.
- `t6_mutex_acquire(mutex::REG_RMW)` — **but only protects parties that take it.** ⚠️ The **math thread never takes `REG_RMW`**; unpack/pack take it in some paths only. A mutex'd unpack write is NOT protected against a concurrent math RMW of the same word — that case falls back to rules 1/2 + dataflow sync.

## Method
1. **Build the word→field map** from `cfg_defines.h` (per arch) and cross-reference thread accesses. Threads by file: `llk_unpack*`/`cunpack_common`=UNPACK, `llk_math*`/`cmath_common`=MATH, `llk_pack*`/`cpack_common`=PACK. CFG defines:
   - WH: `tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h`
   - BH: `tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h`
   - Quasar: `tt_metal/hw/inc/internal/tt-2xx/quasar/cfg_defines.h`
   Parse `#define <X>_ADDR32 <n>`, `<X>_MASK`, `<X>_SHAMT`; resolve `*_RMW` macros (first token is the `*_ADDR32`). Map every access: `cfg[SYM]=`→WR_FULL (mask 0xffffffff), `=cfg[SYM]`→RD, `cfg_reg_rmw_tensix<SYM…>`→RMW(field mask), `TTI_WRCFG(...,WRCFG_32b,SYM)`→full word, `TTI?_RMWCIB*(...,SYM)`→byte RMW. A scripted parser is the reliable way (resolve names→numeric addr; group by addr; flag addresses written by ≥2 threads). Quick grep to spot candidates by hand:
   ```bash
   cd tt_metal/tt-llk
   grep -rInE "cfg_reg_rmw_tensix<|TTI?_WRCFG\(|TTI?_RMWCIB|cfg\[[A-Za-z_]" tt_llk_* --include=*.h | grep -v /tests/
   ```
2. **Keep only words written by ≥2 distinct threads.** For each, OR together each thread's write-mask and compute pairwise bit-overlap.
3. **Classify** with the rules above:
   - any thread does a **full-word write** to the word → inspect: does it cross into another thread's field bits? (pattern 1)
   - two threads' write-masks **share bits** → same-field race (pattern 2)
   - all masks **disjoint** → SAFE (byte-atomic), note as fragile.
4. **For each non-safe word, check ordering**: is there a `mutex::REG_RMW` taken by *all* writers (rare — math doesn't)? Otherwise the only thing preventing concurrency is the dataflow handshake (`UNPACK_TO_DEST`, `MATH_PACK`, `MATH_DONE` semaphores). Decide whether a reconfig on one thread can land while another thread's op (same StateID) reads the word.

## Architecture difference — don't false-positive on Quasar
**Quasar's ALU format register uses a byte-per-operand layout** (`SrcA`/`SrcB`/`Dstacc` each a full byte — `tt_llk_quasar/common/inc/cmath_common.h`), so unpack/math/pack writes land in naturally disjoint bytes, and relu config isn't packed with `ALU_ACC_CTRL`. Combined with Quasar HW write-ordering (Auto-TTSync — HW auto-orders RISC↔Tensix cfg writes against their consumers), the WH/BH packed-word overlaps largely don't arise. Verify the address map still resolves for Quasar before concluding "none" (don't mistake a parser miss for absence).

## Verdict
- **All cross-thread writers use disjoint masked `RMWCIB`** → SAFE (byte-atomic). Note it's fragile: a future full-word write to that word breaks it.
- **Two threads RMW the same field bits**, ordered only by dataflow sync → LATENT/sync-dependent — works today, document the invariant; a misplaced reconfig races.
- **A full-word write (`WRCFG_32b`/`cfg[]=`) overwrites another thread's field in the same word** → RACE (real). Fix = make it a **masked `cfg_reg_rmw_tensix<…_MASK>`** touching only its own bits, under `mutex::REG_RMW` if a sibling path already does. (Cross-check: is the *same register* written safely elsewhere? An inconsistency between two writers of one register is the tell.)

## Thoroughness (optional, full sweep)
The word→field cross-reference is best done by a deterministic script (parse cfg_defines → resolve every access → group by addr), not an agent. Then, only if the user opts into multi-agent orchestration, run a Workflow to adversarially verify each flagged word: confirm the bit layout from `cfg_defines.h`, confirm each writer's thread + mask + mutex/sync context, and try to refute "SAFE by disjoint masks" (look for any full-word writer to that word). Report live RACE vs latent/sync-dependent vs SAFE-but-fragile separately; never upgrade "probably ordered by sync" to SAFE without naming the semaphore.

## Reference findings (ground truth — WH/BH audit)
- **RACE:** `_llk_pack_relu_config_` (`llk_lib/llk_pack_common.h`, WH ~L207 / BH ~L149) does `TTI_WRCFG(p_gpr_pack::TMP0, WRCFG_32b, STACC_RELU_ApplyRelu_ADDR32)` — an **unmasked full-word write** to the word holding `ALU_ACC_CTRL_Zero_Flag_disabled_src/dst` (bits 0-1, written by MATH `cmath_common.h` + UNPACK reduce paths). It zeroes those bits; guarded only by a per-thread `STALLWAIT(STALL_CFG, PACK)`, not a cross-thread lock. Contrast `configure_pack` (`cpack_common.h`), which writes the **same** STACC_RELU register **correctly**: masked RMW (`STACC_RELU_*_MASK`) under `mutex::REG_RMW`. Fix = mirror `configure_pack`.
- **LATENT (sync-dependent):** `ALU_FORMAT_SPEC_REG0_SrcA` / `REG1_SrcB` / `ALU_ACC_CTRL_INT8_math_enabled` (one word) are RMW'd by BOTH unpack (`cunpack_common.h` under mutex; `llk_unpack_common.h`, `enable_int8_fpu_math`) and math (`llk_math_common.h`, `llk_math_eltwise_unary_datacopy.h`, `transpose_dest`, `reduce` — none under the mutex). Same bits, last-writer-wins; safe only because the pipeline semaphores keep format reconfig from overlapping the other thread's op.
- **SAFE-fragile:** word holding `ALU_FORMAT_SPEC_REG2_Dstacc` (PACK, 25-28) + `ALU_ACC_CTRL_Fp32/SFPU_Fp32` (MATH, 29-30) + `INT8` (31); and BH `THCON_SEC0_REG1` (UNPACK `Unp_LF8_4b_exp` bit22 vs PACK `Pack_L1_Acc`/`Exp_threshold`/`Pac_LF8_4b_exp`) — all disjoint masked RMWCIB → safe by byte-atomicity.
- **Quasar:** no cross-thread shared-write words found (byte-per-operand ALU layout + HW ordering).

## Output
For each shared word: numeric `ADDR32`, the fields in it + which thread writes each (`file:line`, primitive, mask), pairwise bit-overlap, mutex/sync context, arch, verdict (SAFE-fragile / LATENT-sync-dependent / RACE) with one-line reasoning, and the one-line fix for any RACE. End with totals per verdict per arch.
