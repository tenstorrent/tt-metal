---
name: perturb
description: >-
  Reproduce a flaky/timing-dependent kernel failure (suspected undocumented HW race) by injecting NOPs/delays to shift inter-thread / inter-kernel / inter-core timing until the failure becomes frequent enough to isolate and minimize into a deterministic reproducer. Sweeps NOP count × injection position × actor, records the max-error scenario PER actor to a report file (rewritten after each actor finishes so you can stop early), then (on request) minimizes the test around a chosen scenario. Works for tt-metal ttnn op tests and tt-llk kernel tests.
---

# Canonical Claude Skill

From the TT-LLK repository root, read `.claude/skills/perturb/SKILL.md`
completely and follow its Markdown body as the authoritative workflow. Ignore
only its Claude-specific frontmatter. Resolve its relative paths from
`.claude/skills/perturb/` and apply the host mappings in `AGENTS.md`.
