---
name: autodebug
description: "Run a fresh-context AutoDebug investigation, verify its findings, then act only on proven root causes."
---

# AutoDebug

Investigate the problem in a clean context instead of doing it in your current
one. Spawn a **fresh subagent** via the Task/Agent tool (a general-purpose
subagent) so it starts with no prior conversation state.

1. Read
   `models/experimental/diffusion_gemma/.agent/scripts/AUTODEBUG_PROMPT.md`
   and pass it as the subagent's prompt, appending the concrete `<problem>` and
   any focus path to inspect. Point the subagent at the checkout or
   subdirectory that should be inspected.
2. Instruct the subagent to investigate **without editing implementation
   files** and to write its report to
   `models/experimental/diffusion_gemma/doc/autoreports/AUTODEBUG.md` (never the
   repo root — a report there is an out-of-folder file). Expect a serious run to
   take a while.

After the subagent finishes:

1. Read `models/experimental/diffusion_gemma/doc/autoreports/AUTODEBUG.md`.
2. **Independently verify** the headline findings against source and runtime
   evidence before trusting any of them.
3. Implement only **verified** fixes; otherwise ask for clarification or explain
   why the report is inconclusive.

If a subagent cannot be launched, perform the same read-only investigation
serially and mark the result `serial` rather than waiting for a nonexistent
tool.

For DiffusionGemma stage sequencing, read the applicable
`models/experimental/diffusion_gemma/.agent/commands/dg-NN-*.md` command and load the
`diffusion-gemma` skill first; each stage is a `/dg-NN-...` command and they run in order.
