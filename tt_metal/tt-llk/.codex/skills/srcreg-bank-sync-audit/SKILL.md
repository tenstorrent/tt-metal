---
name: srcreg-bank-sync-audit
description: >-
  Audit the shared backend DATA registers — SrcA/SrcB bank-valid (AllowedClient) + bank-flip handshake between unpacker and Matrix Unit, and the shared-once Dst/LReg overwrite hazards not already carried by the MATH_PACK semaphore or mutex::SFPU. Use after touching unpack→math dataflow, SETDVALID/CLEARDVALID, bank-flip bookkeeping, MOVD2A/MOVA2D/MOVB2D, or any cross-thread Dst/LReg access.
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/srcreg-bank-sync-audit/SKILL.md` completely and follow its
Markdown body as the authoritative workflow. Ignore only its Claude-specific
frontmatter. Resolve its relative paths from
`.claude/skills/srcreg-bank-sync-audit/` and apply the host mappings in
`AGENTS.md`.
