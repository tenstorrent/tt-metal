---
name: docs-writing
description: Write, trim, or review tt-llk reference documentation (docs/**/*.md and in-tree READMEs). Use when adding a doc page or section, shortening a long doc, checking a doc against the headers or the ISA, or reviewing doc changes in a PR for narrative, duplication, ISA conflicts, or oversharing.
user_invocable: true
---

# Docs Writing

## Goal

A reference doc tells a reader what the code does today and how to use it. It
is not a record of how the change was made or proven. Every sentence should
still be true and useful after the PR merges and the next build changes.

## Scope

`docs/**/*.md` and READMEs under `tools/`, `tests/` and `docs/`. Code comments
follow [code-comments](../code-comments/SKILL.md); docstrings follow
[doxygen-style.md](../../references/doxygen-style.md).

## Reference, not narrative

Keep out of reference docs:

- notes to reviewers ("nothing was added to X", "this PR moves Y");
- evidence and measurements: board names, byte or image costs, run results,
  counts that came out equal on one machine;
- analysis write-ups (how to interpret a stall, silicon studies);
- edit history ("now", "previously", "was changed to").

Put them in the PR description, an issue, or a tech report, and link it when a
reader really needs it.

## One source of truth

- Do not copy a header table into prose: event names and selects, register
  offsets, payload or L1 layouts. Link the header, or generate the table from
  it.
- Do not repeat a file-by-file list that a README already holds.
- Describe a wire or memory layout in one place (the header that defines it)
  and link from the decoder, its docstring and the doc.
- When several pages share a topic (for example, one consumer of several
  counter families), keep one authoritative section and a one-line link in
  each of the others.
- Do not explain the same rationale on several pages.

## ISA alignment

- Check hardware claims against tt-isa-documentation (WH/BH) or the Quasar
  spec the team uses.
- If the doc and the ISA disagree, name both sources and ask which one is
  correct; do not silently pick one. Flag the ISA for an update when the code
  is right.
- If the ISA is silent, mark the claim as a gap with a follow-up, not as fact.
- Do not present an ISA-defined fact (flit size, counter units, register
  addresses) as a measurement; link the ISA page.
- Units and addresses are the usual conflicts: bytes versus words, the base
  address of a register block, numbering of virtual channels or queues.

## Oversharing

tt-llk docs describe public hardware behavior and tt-llk code. Leave out:

- RTL file, module or signal names, and design-internal block details;
- firmware internals beyond the interface tt-llk uses;
- tt-metal profiler or runtime internals; they belong in tt-metal docs.

When an internal fact is the only justification, state the externally visible
behavior and cite the public source, or ask for one.

## Readability

- One topic per paragraph; split a paragraph that mixes design, measurements
  and test description.
- Prefer a short list over a long sentence chain.
- Table cells carry facts; do not repeat the same merged text across rows.
- A section that only restates the code or another page should be a link.

## Review checklist

- [ ] Describes current behavior and usage; no reviewer notes or edit history.
- [ ] No measurements, costs, or board results.
- [ ] No table or layout copied from a header; links instead.
- [ ] Each topic and rationale lives on one page; others link to it.
- [ ] Hardware claims checked against the ISA; conflicts and gaps raised as
      questions with a follow-up.
- [ ] No RTL, firmware, or metal internals.
- [ ] Paragraphs are short and single-topic.
