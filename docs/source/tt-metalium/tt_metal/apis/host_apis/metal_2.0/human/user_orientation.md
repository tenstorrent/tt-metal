# Orientation

> This is a human-facing document.
> Humans: Please read this before attempting any Metal 2.0 ports!
> AI Porters: This document does not contain any info that you need.


A Metal 2.0 port is done in two steps:
 1. Audit (evaluate the op's readiness for porting)
 2. Actually port the op

The porting recipes are available on the branch `akertesz/op-porting-recipe` (they are not checked into main; this facilitates rapid iteration). The recipes are designed to be AI-facing, not human-friendly.

Please familiarize yourself with Metal 2.0 by reading the introduction to `ai/shared/migration_guide.md`, and by browsing the (self-documenting) header files in `tt_metal/api/tt-metalium/experimental/metal2_host_api/`.

## Google Drive MCP setup

The porting recipe relies on data stored in Google Sheets. For your Claude to access it, you must first authorize the Google Drive MCP.

 1. Go to the claude.ai website.
 2. Open Settings (click the bottom-left Tenstorrent logo).
 3. Connectors → Google Drive → Connect.

## Workspace setup

You must supply the `tt-metal` workspace for op porting. You'll launch the audit and port from the root of your checkout.

To setup your workspace:
 1. Create a branch off `akertesz/op-porting-recipe` (to get the recipes).
 2. Merge main into your branch (to ensure you have the latest fixes).

Run only one op port per workspace. Trust me, trying to run simultaneous ports within the same workspace will end in tears.

## Audit step
The audit step should be done by Claude Opus, as some audits are quite complex. The audit produces two outputs:
 - **METAL2_PREPORT_AUDIT.md** provides the result of the audit.
 - **METAL2_PORT_BRIEF.md** provides information required by the subsequent porting step. It's only produced if the audit was successful.


## Resolving audit-flagged issues

If the op audit is unsuccessful, you _must_ fix all of the flagged problems before you can port to Metal 2.0. You should check in any prereq changes as a separate branch and a separate PR, and fully test those changes before attempting a Metal 2.0 port.

A Metal 2.0 port produces _no functional changes to the op_. It should be structurally impossible for a successful Metal 2.0 port to alter an op's behavior, change an op's L1 footprint, introduce a new hang, etc. The porting recipe explicitly (and repeately) forbids the porting AI from introducing functional changes.

However, some of the pre-port audit fixups (related to illegal CB usage in particular) can introduce functional changes.

**Make sure you don't entangle the Metal 2.0 port with the pre-port fixup steps!** Doing so makes any problems virtually impossible to debug.

## Porting step
The porting step requires a capable AI model with **full context availability**. To ensure a good result, it is *essential* that you do the following:
 - Use **Claude Opus 1M** (with extra-high or max effort)
 - Launch the port as your **primary session**. You cannot use a subagent for the main port, because subagents cannot delegate builds to subagents.
 - Launch the port in a **fresh instance**. Do not reuse the same session that you used for the audit, or that you used for a previous port. You need to have the full context window available to ensure peak AI performance.
 - Ask the AI agent to **launch all builds and tests using subagents**. The build and test output are very context intensive.
 - If the AI agent reports that it can only port a subset of ProgramFactories for the op within its context budget, respect this and launch the remaining ones from another fresh instance.

Experiments have demonstrated that porting quality falls off significantly if you don't follow these guidelines. Please be very conscious of context window size for Metal 2.0 ports.


# Prompts

Please use the prompts below to launch the recipes.

## Audit

Audit the TTNN op (OP NAME HERE) for Metal 2.0 portability.

Op directory: `ttnn/cpp/ttnn/operations/...family.../...op.../`

Read and follow `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/metal2.md`.

Produce METAL2_PREPORT_AUDIT.md (and METAL2_PORT_BRIEF.md if the audit clears) in the op
directory, then stop. Do not begin the port — that runs as a separate session after I review the audit.

## Port

Port the TTNN op (OP NAME HERE) to Metal 2.0. The feasibility audit cleared GREEN and I'm asking you to proceed.

Op directory: `ttnn/cpp/ttnn/operations/...family.../...op.../`
Tests:        `tests/ttnn/unit_tests/operations/...wherever.../`
Audit report:  METAL2_PREPORT_AUDIT.md in the op directory (GREEN)
Audit brief:   METAL2_PORT_BRIEF.md in the op directory

Read and follow `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2.md`.

Delegate every build and test run to a subagent. The builds and tests are spammy; this preserves your context for the effort of the port itself.

Please commit METAL2_PORT_PLAN.md and METAL2_PORT_REPORT.md alongside the port.
