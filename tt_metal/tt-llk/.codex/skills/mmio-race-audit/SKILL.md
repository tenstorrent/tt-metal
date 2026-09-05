---
name: mmio-race-audit
description: >-
  Audit LLK code for races between a RISC MMIO write to a config/GPR register and a Tensix instruction/MOP/replay that consumes it. Use after touching any raw cfg[...]=/reg_write/cfg_rmw/regfile[]= write near an UNPACR/PACR/MOP/CFGSHIFTMASK, or when adding addressing/stride/format register writes.
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/mmio-race-audit/SKILL.md` completely and follow its Markdown
body as the authoritative workflow. Ignore only its Claude-specific
frontmatter. Resolve its relative paths from `.claude/skills/mmio-race-audit/`
and apply the host mappings in `AGENTS.md`.
