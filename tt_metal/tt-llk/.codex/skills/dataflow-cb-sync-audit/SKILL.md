---
name: dataflow-cb-sync-audit
description: >-
  Audit circular-buffer (CB) producer/consumer flow control between data-movement (reader/writer) and compute kernels — cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front credit balance, data-write-before-credit ordering (NOC flush), reserve/wait-before-access, capacity vs num_pages, single-producer/consumer, counter cache-coherency, and remote/sharded CB credits. Use after touching any cb_* call, a reader/writer/compute kernel, fifo_rd_ptr/fifo_wr_ptr/pages_received/pages_acked, or RemoteSender/ReceiverCBInterface. Scope reaches beyond tt-llk into tt_metal/hw/inc/api/dataflow and ttnn/models kernels.
---

# Canonical Claude Skill

From the TT-LLK repository root, read
`.claude/skills/dataflow-cb-sync-audit/SKILL.md` completely and follow its
Markdown body as the authoritative workflow. Ignore only its Claude-specific
frontmatter. Resolve its relative paths from
`.claude/skills/dataflow-cb-sync-audit/` and apply the host mappings in
`AGENTS.md`.
