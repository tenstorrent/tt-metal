---
name: llk-test-runner-skill
description: Delegates LLK test runs to the llk-test-runner agent using @.cursor/agents/llk-test-runner.md. Use when the user asks to run tests or mentions LLK tests. Ensure test commands run to completion before reading terminal output (no polling).
---

# LLK Test Runner Skill

## Instructions

Use this skill when the user requests tests or mentions LLK tests.

1. Delegate the test run to the `llk-test-runner` agent using `@.cursor/agents/llk-test-runner.md`.
2. Provide the exact test command and any required context to the agent.
3. Do not run tests directly in the terminal.
4. Require a blocking run: the agent should run the command to completion before reading terminal output.
5. Avoid polling terminal output; rely on the agent's single final summary.

### Blocking run guidance for the agent

When running tests, use a blocking shell invocation and a sufficiently high `block_until_ms` so the command finishes before any terminal read. If a retry is needed, re-run with a higher `block_until_ms` rather than polling.

## Examples

**Example 1**
User: "Run the wormhole unit tests."
Assistant: "Delegating to llk-test-runner agent with the requested command, and requiring a blocking run."

**Example 2**
User: "Please run llk tests."
Assistant: "Delegating to llk-test-runner agent per @.cursor/agents/llk-test-runner.md."
