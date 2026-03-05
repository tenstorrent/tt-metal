# Agent Logging Mechanism

## Overview

Agent breadcrumb logging is **always enabled**. A SubagentStart hook injects logging instructions into every subagent's context automatically.

## How It Works

```
SubagentStart hook fires → injects additionalContext:
  "BREADCRUMBS ENABLED — write to {op_path}/agent_logs/{agent}_breadcrumbs.jsonl"
  → agent sees this in its context and writes breadcrumbs
```

Additionally, agent instructions and PostToolUse/PostToolUseFailure hooks unconditionally require breadcrumb logging on every test run.

## Components

### SubagentStart Hook

**Script**: `.claude/scripts/logging/inject-logging-context.sh`

**Registered in**: `.claude/settings.local.json` under `hooks.SubagentStart`

**Behavior**:
1. Fires whenever any subagent starts
2. Outputs JSON with `additionalContext` containing breadcrumb instructions

The `additionalContext` is injected directly into the agent's context by Claude Code — it is not prompt text that the agent might ignore.

### Breadcrumb Helper

**Script**: `.claude/scripts/logging/append_breadcrumb.sh`

**Usage** (by agents):
```bash
.claude/scripts/logging/append_breadcrumb.sh "{operation_path}" "{agent_name}" '{"event":"...", "details":"..."}'
```

## What Agents Receive

When logging is enabled, each agent's context includes:

```
BREADCRUMBS ENABLED — You MUST write breadcrumbs to {operation_path}/agent_logs/{agent_name}_breadcrumbs.jsonl
where {operation_path} is the operation directory from your prompt.

Use the append_breadcrumb.sh helper:
  .claude/scripts/logging/append_breadcrumb.sh "{operation_path}" "{agent_name}" '{"event":"...", "details":"..."}'

Log after each significant action: file reads, design decisions, test runs (pass/fail/hang),
debugging hypotheses, and fixes applied.
```

The agent derives `{operation_path}` from its own prompt (which always contains the operation path).

## Why This Approach

| Approach | Reliability | Problem |
|----------|-------------|---------|
| Prompt text ("enable logging") | Low | Agents sometimes ignore prompt instructions |
| Script check (`check_logging_enabled.sh`) | Low | Agents skip Step 0 or script checks wrong file |
| **SubagentStart hook + additionalContext** | **High** | **Injected by infrastructure, not parse-dependent** |

The hook fires for every subagent launch. The `additionalContext` appears in the agent's context window alongside system instructions — it cannot be skipped or ignored the way a prompt phrase can.
