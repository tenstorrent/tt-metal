---
name: reconfig-stall-audit
description: Audit LLK reconfig/uninit/config-write functions for a MISSING stall that drains the execution unit before its config registers are rewritten (packer→PACK, unpacker→UNPACK, math→MATH|WAIT_SFPU). Use after touching cpack/cunpack/cmath, *_reconfig_*, *_uninit_, set_packer_strides, or any function that writes ALU/THCON/ADDR_MOD/stride config.
user_invocable: true
---

# /reconfig-stall-audit — Config-register reconfig without execution-unit drain

## The rule (what a correct function does)
When a flattened LLK function REWRITES config registers that a hardware execution unit reads *while running*, that unit must be **idle first** — otherwise you reprogram state out from under an in-flight op (a "reconfig escape"). The guard is a `TTI_STALLWAIT` (usually at the top of the function) whose **condition (2nd) operand** drains the matching unit:

| Function reconfigures… | condition (`wait_res`, 2nd arg) must include | because |
|---|---|---|
| **Packer** config (out/in fmt, strides, l1 offset, exp threshold, l1_acc, dest-rd-ctrl) | `p_stall::PACK` | packer reads these during PACR |
| **Unpacker** config (tile descriptor, out fmt, strides, base addr) | `p_stall::UNPACK` (or `UNPACK0` for SrcA, `UNPACK1` for SrcB) | unpacker reads these during UNPACR |
| **Math** config (ALU SrcA/SrcB fmt, INT8 enable, dest acc) | `p_stall::MATH \| p_stall::WAIT_SFPU` | the FPU **and** the SFPU share the math path — BOTH must drain |

`TTI_STALLWAIT(stall_res, wait_res)`: `stall_res` = **block mask** (which instruction classes can't issue: `STALL_CFG`=B7 blocks WRCFG/RMWCIB, `STALL_PACK`/`STALL_UNPACK`/`STALL_MATH`), `wait_res` = **condition mask** (what to wait on). The block mask just needs to block the instruction that does the config write; the **condition mask is what proves the unit is drained** — that's the bit to check.

## What to flag
A reconfig/uninit/config-writer that writes config registers with **no preceding STALLWAIT whose condition drains the matching unit**. Sub-cases:
1. **No stall at all** before the config write.
2. **Wrong condition** — e.g. `STALLWAIT(STALL_CFG, THCON)` only orders the GPR→cfg write (THCON = scalar-unit memory requests); it does **NOT** drain PACK/UNPACK/MATH. A `THCON`-only guard on packer-stride writes is the classic miss.
3. **Math reconfig missing SFPU** — has `MATH` but not `WAIT_SFPU` (or vice-versa).
4. **Self-containment gap / arch asymmetry** — the function relies on the *caller* having drained the unit, while its sibling on another arch self-guards. Wormhole functions tend to self-guard; Blackhole siblings have been caught relying on caller drains. Flag the divergence even if current callers happen to drain.

## Method
1. **Enumerate** candidates across `tt_llk_wormhole_b0`, `tt_llk_blackhole`, `tt_llk_quasar`:
   ```bash
   cd tt_metal/tt-llk
   grep -rInE "reconfig|reconfigure|_uninit_|set_packer_strides|set_packer_l1_offset|configure_(pack|unpack)|reconfigure_exp_threshold|reconfigure_packer_l1_acc" \
     tt_llk_* --include=*.h | grep -v /tests/
   ```
2. **For each**, read the function body. Identify config-register writes: `cfg_reg_rmw_tensix<>`, `TTI_WRCFG`/`TT_WRCFG`, `TTI_REG2FLOP`, `TTI_SETC16`, `TTI_RMWCIB*`, `TTI_SETADC*` to packer/unpacker/ADDR_MOD regs, `regfile[]=`+`REG2FLOP`, and helper calls (`set_packer_strides`, `set_packer_l1_offset`, `addr_mod_*::set`).
3. **Determine which unit** reads the written register (packer / unpacker / math). Use register-name semantics; confirm against the ISA docs when non-obvious (see `arch-lookup` skill / tt-isa-docs MCP for WH/BH; register names map to `tt_metal/hw/inc/internal/.../cfg_defines.h` by similarity).
4. **Check the guard**: is there a `TTI_STALLWAIT(..., <condition incl. that unit>)` before the first such write — *inside this function* (preferred) or unavoidably in the call tree? Trace helpers (`set_packer_strides` etc.) to see whether THEY stall on the unit or only on `THCON`.
5. **Compare arches**: diff the WH vs BH vs Quasar version of the same function. A unit-drain present in one and absent in another is a strong signal.

## Verdict & scope
- **Self-contained + correct condition** → SAFE.
- **Relies on caller drain (no self-guard), sibling arch self-guards** → report as a *self-containment gap* (hardening), and trace callers to say whether it's a *live* escape or only latent. To convert latent→safe, add the WH-style leading `TTI_STALLWAIT(STALL_CFG, <unit>)` or change a `THCON`-only helper stall to `<unit> | THCON`.
- **No drain and a caller can reach it with the unit busy** → live reconfig escape (real bug).

## Thoroughness (optional, for a full sweep)
For an exhaustive pass, fan out one agent per file (Explore/general-purpose), or — only if the user opts into multi-agent orchestration — a Workflow that analyzes each candidate then adversarially re-checks each SAFE verdict (try to find the un-drained path). Always report whether each finding is a *live bug* vs *self-containment hardening*, and never silently upgrade "probably safe" to "safe".

## Reference fixes (ground truth)
- PR #738 "Stall changing packer config till packer finishes" — added packer drain to packer reconfig.
- `set_packer_strides` PACK-drain fixes (BH) — `THCON` → `PACK | THCON`; and `_llk_pack_fast_tilize_uninit_` leading `STALLWAIT(STALL_CFG, PACK)` on WH vs BH.
- `_llk_math_reconfig_data_format_*` (`llk_math_common.h`) — canonical `STALLWAIT(STALL_CFG, MATH | WAIT_SFPU)`.

## Output
For each flagged site: `file:line`, function, register(s) written, the unit that consumes them, the guard present (or none) and its condition, verdict (SAFE / HARDENING-GAP / LIVE-BUG), and whether a sibling arch self-guards. End with a count and the one-line fix per flag.
