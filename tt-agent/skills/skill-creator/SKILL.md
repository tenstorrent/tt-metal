---
name: skill-creator
layer: meta
description: "Design and build skills for the tt-agent system — guides through rigorous design alignment before writing, then applies TT conventions. Use when creating, improving, or reviewing tt-agent skills"
---

# TT Skill Creator

## Purpose

Designs and builds high-quality skills for the tt-agent system. The most important
part of skill creation is **designing the skill well** — understanding what it should
do, exposing flaws in the spec, achieving full alignment with the developer before
writing anything.

Wraps `/skill-creator` for base mechanics (format, frontmatter, progressive load,
evals) and adds TT-specific guidance from `tt-guidelines.md`.

## When to Invoke

- Create a new tt-agent skill
- Improve or review an existing tt-agent skill
- Verify a skill follows tt-agent conventions

## Pipeline

```
design → align → write → validate
```

The design phase is where the real work happens.

## Phase 1: Design Alignment

Before writing any skill content, achieve total understanding through structured
interrogation. Ask questions **one at a time**. Do not proceed until alignment is
complete.

### What to establish:

1. **Purpose clarity** — What exactly does this skill do? What does it NOT do?
   Push until the boundary is crisp. If the developer says "it helps debug kernels",
   ask: debug what kind of failures? Hangs? Wrong output? Crashes? All three?

2. **Trigger conditions** — When should this skill fire vs another? What request
   patterns should trigger it? What adjacent skills exist and where's the boundary?

3. **Layer placement** — Orchestration, workflow, tool, or meta? Apply the decision
   rule from `tt-guidelines.md`. Challenge if the placement seems wrong.

4. **Input → output contract** — What does the skill receive? What does it produce?
   Where does output go (notes directory, code, device)?

5. **Abstraction level** — Kernel-level C++? Operator-level host C++? Model-level
   Python? Multiple levels? The skill must be explicit about this.

6. **Failure modes** — What goes wrong? How does the skill detect it failed?
   What does it do when stuck? Escalation path?

7. **Dependencies** — Which other skills does this invoke? Which invoke it?
   Does it need tt-learn for volatile info? Does it need device access?

### How to interrogate:

- Ask one question at a time. Wait for the answer.
- When the developer gives a vague answer, push back. "Helps with performance"
  is not a spec. What metric? What target? What's the iteration loop?
- Propose the worst reasonable interpretation of ambiguous answers and ask if
  that's what they mean. Forces precision.
- After each answer, state your current understanding of the skill's scope.
  Let the developer correct misunderstandings early.
- When you think you understand, summarize the full design and ask: "What did
  I get wrong? What did I miss?"

### Expose flaws:

- Look for scope creep: if the skill does 5 things, should it be 2 skills?
- Look for overlaps with existing skills in the dispatch table.
- Look for implicit assumptions about what the agent "just knows" — if the skill
  needs volatile info, that must go through tt-learn, not be assumed.
- Look for missing convergence criteria (workflow skills) or missing verification
  steps (tool skills).

## Phase 2: Write

Only after design alignment is achieved:

1. **Invoke `/skill-creator`** for base mechanics.
2. **Load `tt-guidelines.md`** and apply TT-specific constraints.
3. Write SKILL.md + sub-files.

## Phase 3: Validate

Run the self-check from `tt-guidelines.md`. Run frontmatter tests.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| TT-specific rules and constraints | `tt-guidelines.md` |
| Base skill format, frontmatter, evals | Invoke `/skill-creator` |
