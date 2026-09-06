---
name: race-audit-all
description: >-
  Run all nine LLK hazard audits (mmio-race, reconfig-stall, cfg-word-overlap, semaphore-handshake, mailbox-sync, dataflow-cb-sync, srcreg-bank-sync, noc-sync, instruction-latency) across four synchronization surfaces, and add a cross-class JOIN pass that catches emergent races no single audit can see — where one audit's verdict is "safe because of an invariant owned by another audit". Use for a full hazard sweep of an LLK change, or before merging anything touching config writes, reconfig/uninit, inter-thread/cross-core sync, the SrcA/SrcB-Dst data path, or hand-written instruction sequences.
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/race-audit-all/SKILL.md` completely and follow its Markdown
body as the authoritative workflow. Ignore only its Claude-specific
frontmatter. Resolve its relative paths from `.claude/skills/race-audit-all/`
and apply the host mappings in `AGENTS.md`.
