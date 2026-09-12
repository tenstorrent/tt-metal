---
name: mailbox-sync-audit
description: >-
  Audit LLK/compute-API use of the RISC↔RISC hardware mailboxes (mailbox_write/mailbox_read/mailbox_not_empty, TENSIX_MAILBOX*) for races/deadlock — push/pop balance per directed channel, call-count symmetry across threads, correct write-dest/read-src addressing, FIFO overflow, and the fence=nop ordering caveat. Use after touching any mailbox_write/mailbox_read, the CB tile-address/value broadcast (circular_buffer.h / cb_api.h get_tile_address/read_tile_value), unpack-to-dest dst_index passing, or the debug halt/unhalt handshake.
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/mailbox-sync-audit/SKILL.md` completely and follow its Markdown
body as the authoritative workflow. Ignore only its Claude-specific
frontmatter. Resolve its relative paths from
`.claude/skills/mailbox-sync-audit/` and apply the host mappings in
`AGENTS.md`.
