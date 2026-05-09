# Porting an Op to Metal 2.0 — Phases 1 & 2: Port Recipe

> This is the second of two documents covering the Metal 2.0 op port workflow. **This document covers Phase 1 (mechanical translation) and Phase 2 (verification).** Phase 0 — the feasibility audit — lives in [`port_op_to_metal2_audit.md`](port_op_to_metal2_audit.md) and is a hard prerequisite to anything in this document.
>
> This document is in active construction. Phase 1 and Phase 2 are stubs and will be fleshed out in subsequent iterations.

## Read this first

**Audience**: AI agents asked to perform the actual Metal 2.0 port of a TTNN op, *after* the Phase 0 feasibility audit has cleared with GREEN status and the user has explicitly approved proceeding.

**Precondition — non-negotiable**: You may only invoke this document if:

1. The audit in [`port_op_to_metal2_audit.md`](port_op_to_metal2_audit.md) was performed for this op and produced an **overall GREEN** result (or a YELLOW with all questions resolved by the user in favor of proceeding).
2. The user has **explicitly asked you to proceed** with the port. A green audit alone is not sufficient — the user must have read the audit and given an unambiguous go-ahead.

If either condition is unmet, stop. Return to the audit document. Do not improvise.

**Operating principle**: Refusing to write code is still a successful outcome.

The audit cleared the *features* and *prereqs* known at audit time. During Phase 1 you may discover something the audit missed — a feature gate that didn't fire, a kernel pattern that doesn't translate cleanly, an interaction the audit didn't anticipate. When that happens, the correct response is the same as in Phase 0: **stop and report**. Do not improvise around it.

In particular: if you find yourself constructing a clever workaround during the port — packing data into varargs to simulate a missing field, threading a buffer address through an RTA because the binding mechanism doesn't fit, hand-rolling a synchronization primitive — **stop**. Whatever you are about to write is almost certainly wrong. Surface the problem; do not paper over it.

When you find a gap that should be added to the audit's Appendix A, flag it. The audit's gate set is improved by what Phase 1 discovers.

---

## Phase 1 — Mechanical translation

> Stub. The mechanical recipe will land here in subsequent iterations.
>
> Until then, after a green audit + explicit user go-ahead, the conceptual map in [`metal2_migration_guide.md`](metal2_migration_guide.md) is the closest available reference. Consult it as a guide for the API correspondences, but apply Phase 1's operating principle: stop and report on any unexpected gap rather than improvise.

---

## Phase 2 — Verification

> Stub. The build / test / anti-pattern self-audit checklist will land here in subsequent iterations.
>
> Anti-pattern checklist (preview — to be expanded):
> - No `tensor.buffer()->address()` or equivalent in any RTA.
> - No "lazy port" of legacy positional RTAs into varargs (every constant-indexed `get_vararg(N)` is a named arg in disguise).
> - No `set_globally_allocated_address` / non-zero `address_offset` / multi-element `format_descriptors` (these should have failed the audit; their presence here means a regression slipped through).

---

## Appendix B: Templates

> Stub. Copy-paste blocks for common port shapes (single-core reader/writer, multi-core, semaphore handshake, tensor-bound kernel) will land here in subsequent iterations.

---

## Cross-references

- Phase 0 audit and feature compatibility rules (Appendix A): [`port_op_to_metal2_audit.md`](port_op_to_metal2_audit.md)
- Conceptual API map (for human readers): [`metal2_migration_guide.md`](metal2_migration_guide.md)
