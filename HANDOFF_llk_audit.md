# Session Handoff — LLK audit + race-audit skills (resume point)

**Date:** 2026-06-26 · **Machine:** aus-wh-08 (WH) · **Repo:** tenstorrent/tt-metal
**Purpose:** resume solving the bugs found in the LLK ISA-grounded audit. Read this + `llk_audit_findings.md` to rehydrate full context.

## To resume with Claude in a new session
1. `cd /localdev/amahmud/tt-metal` (or wherever the repo lives).
2. Tell Claude: *"Read HANDOFF_llk_audit.md and llk_audit_findings.md; we're resuming the LLK bug-fix work."*
3. If on the SAME machine as today, you can instead `claude --resume` and pick this session to restore the live transcript.

## Artifacts produced today
1. **`llk_audit_findings.md`** (this repo root) — 41 findings, ISA-grounded, prioritized P0–P3, WH B0 + BH A0. THE work list. Each row has location, explanation, suggested fix, confidence.
2. **Branch `chore/llk-race-audit-skills`** (pushed to origin; tip was `4e13992d158`) — 7 reusable audit *skills* under `tt_metal/tt-llk/.claude/skills/`:
   - `mmio-race-audit`, `reconfig-stall-audit`, `cfg-word-overlap-audit`, `semaphore-handshake-audit`, `mailbox-sync-audit`, `dataflow-cb-sync-audit`, and `race-audit-all` (orchestrator that runs all six + a monotonic cross-class JOIN). A PR tracks this branch.

## Fix plan (priority order)
Start with the P1s — they are the highest-value and mostly High-confidence:
- **U1** (BH `llk_unpack_tilize.h:42-55`) — C++ variable shadowing drops a MOP body op. Clear logic bug, small fix. **Best first.**
- **C3** (`cpack_common.h set_packer_strides`) — make it self-draining (`STALLWAIT(STALL_CFG,PACK)` WH / `PACK|THCON` BH). **Highest leverage** — collapses 6 of 7 reconfig hardening-gaps.
- **C1** (`llk_pack_common.h _llk_pack_relu_config_`) — replace full-word `WRCFG_32b` with masked RMW under `mutex::REG_RMW` (mirror `configure_pack`).
- **U2** (BH `llk_unpack_untilize.h` uninit) — restore from saved GPRs instead of hardcoded constants.
- **F1** (WH `llk_math_eltwise_unary_datacopy.h:70-192`) — VERIFY on HW first (may silently drop low-16 bits of fp32/int32 broadcast); if confirmed, wrap MOVB2D(DEST_32B_LOW) in Fp32_enabled 0/1.

Then P2: **S1** (BH math config writes need `WAIT_SFPU` — systemic, ISA-confirmed), **S2** (BH SFPU `done_with_addrmod_reset` writes wrong cfg reg — delete the `SETC16(2,0)`), the Fp8_e4m3 gaps (F3/F8), int8 reconfig (F4/F5/F7), arch-divergence (N1/N3/U3/U4). See file for the rest.

## Important caveats (read before fixing)
- Findings are **candidates, not confirmed defects** — confirm each (code re-read + HW/maintainer) before landing a fix. Items marked Med/Low especially.
- Resolved NOT-bugs (do not re-chase): matmul-reuse loop (correct); register-encoding census (clean); most WH/BH SFPU divergences (correct-by-design); API-usage sample (clean). See the "Resolved / verified-correct" section of the findings file.
- Coverage gaps (not audited): Quasar; the `tt_metal/hw/ckernels/*/llk_sfpu/` SFPU op corpus (exp/gelu/sqrt/typecast/quant); full ttnn/models kernel corpus (sampled only); `llk_lib/debug` + most `experimental`.
- ISA-grounding was thin in spots (MCP index gaps) — F1 and exp-section constants rest on code-consistency + documented errata, not a primary ISA excerpt.

## Suggested working mode next time
Fix one finding at a time on a NEW branch (not the skills branch): re-read the cited code, confirm the bug, apply the suggested fix, build, and (if possible) run the relevant tt-llk test. The skills on `chore/llk-race-audit-skills` can re-verify a fix (e.g. `/cfg-word-overlap-audit`, `/reconfig-stall-audit`).
