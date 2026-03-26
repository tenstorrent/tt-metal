# TT-Agent Phase 1 Retrospective

Date: 2026-03-26
Commit: fd88fb7ff22
Scope: End-to-end validation — tt-skill-creator used to create tt-orchestrator

---

## What tt-skill-creator Got Right

**Guideline compliance came naturally.** Having the "point to code, not inline APIs"
rule written down (guideline 3 in tt-guidelines.md) made it easy to resist the reflex
to inline NOC API signatures or CB config details. The output naturally said "see
`knowledge/hardware/`" and "invoke `tt-learn`" instead.

**Layer placement was unambiguous.** The four-layer table with its decision rule
(does it route? → orchestration) gave a single answer with no deliberation needed.
For tt-orchestrator, the answer was obvious.

**The progressive load table pattern worked.** Requiring a table in SKILL.md pointing
to sub-files — and requiring those files to actually exist — prevented the temptation
to stuff decomposition logic into SKILL.md itself. `decomposer.md` naturally pulled
out the right content and the table made the loading intent explicit.

**Size discipline held.** SKILL.md came in well under 150 lines. decomposer.md is
~130 lines. The constraint pushed toward clarity: if it can't be said concisely,
it probably shouldn't be said in the skill at all.

**The self-check list (guideline 10)** was a clean finishing gate. Each checkbox was
verifiable without judgment calls. The `pytest` command as the final check item was
particularly good — it made "does this skill pass?" a binary question.

---

## Gaps and Improvements Needed in tt-guidelines.md

**Gap: no guidance on description trigger testing.** Guideline 5 says descriptions
should be "optimized for triggering" and references `/skill-creator` eval guidance,
but tt-guidelines.md itself has no concrete test for whether a description triggers
correctly. A few example trigger phrases alongside each skill would help.

**Gap: STATUS.md vs PLAN.md distinction is implied, not specified.** The orchestrator
skill invented a STATUS.md convention that tt-guidelines.md doesn't define. The notes
naming table (guideline 8) covers `context-`, `experiments-`, `plan-`, `profile-` but
not `status-`. Add a `status-<task>.md` row.

**Gap: "iteration budget" has no home in guidelines.** The decomposer.md created an
iteration budget table (when to escalate vs keep trying). This is a real concept for
workflow skills but tt-guidelines.md guideline 6 only covers convergence criteria for
workflow-layer skills. Orchestration skills also need an escalation concept.

**Gap: no guidance on what makes a good Skill Dispatch Table.** The dispatch table
in tt-orchestrator maps situations to skills. tt-guidelines.md doesn't mention this
pattern — it's a useful convention worth codifying for orchestration-layer skills.

**Minor: progressive load table example uses a placeholder row.** The example in
guideline 4 shows `[specific sub-task description]` as placeholder text. A real
example from an existing skill would be more instructive.

---

## Observations on "Use What You Build"

**The constraint is productive.** Being asked to use tt-skill-creator to build
tt-orchestrator (rather than just writing it directly) forced engagement with each
guideline. Guidelines that were vague became visible friction. The status-naming gap
above wasn't obvious reading tt-guidelines.md in isolation — it only surfaced when
trying to apply it to write orchestration documents.

**The test is a forcing function.** Knowing `pytest tt-agent/tests/test_skill_frontmatter.py`
had to pass at the end prevented frontmatter shortcuts. The test is simple (YAML
validity, name match) but it catches the most common mistakes cold. More test coverage
would catch more problems — e.g., a test that verifies all files referenced in the
progressive load table actually exist.

**Meta-skills need a working example to bootstrap from.** tt-skill-creator was written
before any skills existed. Using it to create tt-orchestrator confirmed the guidelines
are coherent, but also revealed that the guidelines assume the agent already knows
what a "good" skill looks like. A reference to tt-orchestrator itself (now that it
exists) would help future skill authors.

**The layer taxonomy held up under the first real case.** orchestration vs workflow vs
tool vs meta — with tt-orchestrator, there was no ambiguity. The real test will come
when a skill sits between layers (e.g., a skill that iterates AND routes).
