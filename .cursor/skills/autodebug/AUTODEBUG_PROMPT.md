# Cursor AutoDebug Prompt

Read the detailed inspection heuristics in:

`models/experimental/diffusion_gemma/.agent/scripts/AUTODEBUG_PROMPT.md`

Apply them with these Cursor-specific overrides:

- use Cursor `Subagent` calls, not Claude Task/Agent or `xhigh` terminology;
- prefer multiple focused read-only subagents when investigations are
  independent;
- do not ask a subagent to mutate implementation files;
- return the report in the subagent final response; writing `AUTODEBUG.md` is
  optional;
- verify every headline claim against source and complete the causal chain;
- distinguish direct observations from interpretation;
- identify runtime checks that remain necessary when static evidence is not
  decisive.

Append the concrete problem, failing command/logs, checkout root, and focus
paths below this prompt.
