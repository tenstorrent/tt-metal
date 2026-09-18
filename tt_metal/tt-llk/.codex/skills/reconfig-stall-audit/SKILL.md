---
name: reconfig-stall-audit
description: >-
  Audit LLK reconfig/uninit/config-write functions for a MISSING stall that drains the execution unit before its config registers are rewritten (packer→PACK, unpacker→UNPACK, math→MATH|WAIT_SFPU). Use after touching cpack/cunpack/cmath, *_reconfig_*, *_uninit_, set_packer_strides, or any function that writes ALU/THCON/ADDR_MOD/stride config.
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/reconfig-stall-audit/SKILL.md` completely and follow its
Markdown body as the authoritative workflow. Ignore only its Claude-specific
frontmatter. Resolve its relative paths from
`.claude/skills/reconfig-stall-audit/` and apply the host mappings in
`AGENTS.md`.
