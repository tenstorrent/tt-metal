---
name: perf-optimization-audit
description: >-
  Audit Tensix/SFPU LLK compute kernels for PERFORMANCE — unfilled latency shadows/bubbles and redundant NOPs, redundant Dst/LReg store-load traffic, loop-invariant work, predication that should be branchless arithmetic (min/max/abs/setsgn), un-fused mul+add, ignored APPROXIMATION_MODE, and unroll/register-pressure mistakes. Use after touching any ckernel_sfpu_*.h, hand-written TTI_SFP*/TTI_* sequence, or the compute inner loop. This is a PERF audit (wasted cycles), NOT a correctness/race audit — pair it with instruction-latency-audit.
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/perf-optimization-audit/SKILL.md` completely and follow its
Markdown body as the authoritative workflow. Ignore only its Claude-specific
frontmatter. Resolve its relative paths from
`.claude/skills/perf-optimization-audit/` and apply the host mappings in
`AGENTS.md`.
