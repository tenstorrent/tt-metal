---
name: instruction-latency-audit
description: >-
  Audit hand-written Tensix/SFPU instruction sequences for missing pipeline-latency padding — where a dependent instruction consumes a multi-cycle-latency result before it is ready and a NOP (or independent-instruction spacing) is required. Use after touching any raw TTI_SFP*/TTI_* sequence, ckernel_sfpu_* kernels, or hand-assembled instruction streams. NOT a cross-thread race — an intra-thread micro-architectural hazard.
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/instruction-latency-audit/SKILL.md` completely and follow its
Markdown body as the authoritative workflow. Ignore only its Claude-specific
frontmatter. Resolve its relative paths from
`.claude/skills/instruction-latency-audit/` and apply the host mappings in
`AGENTS.md`.
