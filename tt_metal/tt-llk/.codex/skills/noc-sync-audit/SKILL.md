---
name: noc-sync-audit
description: >-
  Audit cross-core NoC synchronization in dataflow kernels — noc_semaphore_wait/set/inc balance and direction, multicast fan-out counts, and data-before-signal NoC ordering (noc_async_write_barrier / noc_async_writes_flushed before a remote credit). The half of dataflow that dataflow-cb-sync-audit (CB credits) does not reach. Use after touching reader/writer kernels, noc_semaphore_*, noc_async_*_barrier, or any cross-core handshake not expressed as a CB.
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/noc-sync-audit/SKILL.md` completely and follow its Markdown
body as the authoritative workflow. Ignore only its Claude-specific
frontmatter. Resolve its relative paths from `.claude/skills/noc-sync-audit/`
and apply the host mappings in `AGENTS.md`.
