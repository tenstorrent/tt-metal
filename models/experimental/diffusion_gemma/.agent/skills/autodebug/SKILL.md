---
name: autodebug
description: "Run a fresh-context AutoDebug investigation and then act on the generated AUTODEBUG.md report."
---

# AutoDebug

Investigate the problem in a clean context instead of doing it in your current
one. Spawn a **fresh subagent** via the Task/Agent tool (a general-purpose
subagent) so it starts with no prior conversation state.

1. Read the brief in `scripts/AUTODEBUG_PROMPT.md` (relative to this `.agent`
   dir) and pass it as the subagent's prompt, appending the concrete `<problem>`
   and any focus path to inspect. Point the subagent at the checkout or
   subdirectory that should be inspected.
2. Instruct the subagent to write its report to `./AUTODEBUG.md`. Expect a
   serious run to take a while.

After the subagent finishes:

1. Read `AUTODEBUG.md`.
2. Check the report's headline findings against the code before trusting them.
3. Act on the report: implement the fix, ask for clarification, or explain why
   the report is inconclusive.

If the problem implies stage sequencing, run the stage slash-commands under
`commands/` in order (each stage is a `/dg-NN-...` command).
