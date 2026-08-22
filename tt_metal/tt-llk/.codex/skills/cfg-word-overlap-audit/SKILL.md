---
name: cfg-word-overlap-audit
description: >-
  Audit LLK code for races on the backend CONFIG register file where differently-named fields share the SAME 32-bit config word — both cross-thread (unpack/math/pack write the same word) and intra-thread (a full-word write clobbers a sibling field the same thread set elsewhere). Use after adding/changing any ALU_FORMAT_SPEC / ALU_ACC_CTRL / ALU_ROUNDING_MODE / STACC_RELU / THCON_SEC* write, any WRCFG_32b/cfg[]= full-word write to a multi-field word, or any cfg_reg_rmw_tensix on a word another thread also touches.
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/cfg-word-overlap-audit/SKILL.md` completely and follow its
Markdown body as the authoritative workflow. Ignore only its Claude-specific
frontmatter. Resolve its relative paths from
`.claude/skills/cfg-word-overlap-audit/` and apply the host mappings in
`AGENTS.md`.
