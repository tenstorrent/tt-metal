# Orientation

> This is a human-facing document. Please read this before attempting any Metal 2.0 ports!


A Metal 2.0 port is done in two steps:
 1. Audit (evaluable the op's readiness for porting)
 2. Actually port the op

The porting recipes are available on the branch `akertesz/metal2-documentation`. The recipes are designed to be AI-facing, not human-friendly.

Please familiarize yourself with Metal 2.0 by reading the introducion to `metal2_migration_guide.md`, and by browsing the (self-documenting) header files in `/tt_metal/api/tt-metalium/experimental/metal2-host-apis/`.


## Audit step
The audit step can be done by either Claude Sonnet (max effort) or Opus. The audit produces two outputs:
 - **METAL2_PORT_AUDIT.md** provides the result of the audit.
 - **METAL2_PORT_BRIEF.md** provides information required by the subsequent porting step. It's only produced if the audit was successful.


## Porting step
The porting step requires a capable AI model with **full context availability**. To ensure a good result, it is *essential* that you do the following:
 - Use **Claude Opus 1M** (with extra-high or max effort)
 - Launch the port as your **primary session**. You cannot use a subagent for the main port, because subagents cannot delegate builds to subagents.
 - Launch the port in a **fresh instance**. Do not reuse the same session that you used for the audit, or that you used for a previous port. You need to have the full context window available to ensure peek AI performance.
 - Ask the AI agent to **launch all builds and tests using subagents**. The build and test output are very context intensive.
 - If the AI agent reports that it can only port a subset of ProgramFactories for the op within its context budget, respect this and launch the remaining ones from another fresh instance.

Experiments have demonstrated that porting quality falls off significantly if you don't follow these guidelines. Please be very conscious of context window size for Metal 2.0 ports.


# Prompts

Please use the prompts below to launch the recipes.

## TODO
