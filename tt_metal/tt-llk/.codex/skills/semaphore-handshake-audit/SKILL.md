---
name: semaphore-handshake-audit
description: >-
  Audit LLK inter-thread synchronization (Tensix semaphores + ATGETM/ATRELM mutexes) for races/deadlock — SEMINIT correctness vs usage, post/get balance, wait-direction, RISC-MMIO-vs-Tensix ordering, and mutex acquire/release balance. Use after touching any t6_semaphore_*/semaphore_post/semaphore_get/SEMINIT/SEMWAIT/SEMPOST/SEMGET/t6_mutex_* or any math↔pack / unpack↔math handshake (MATH_PACK, UNPACK_TO_DEST, UNPACK_SYNC, MATH_DONE, FPU_SFPU).
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/semaphore-handshake-audit/SKILL.md` completely and follow its
Markdown body as the authoritative workflow. Ignore only its Claude-specific
frontmatter. Resolve its relative paths from
`.claude/skills/semaphore-handshake-audit/` and apply the host mappings in
`AGENTS.md`.
