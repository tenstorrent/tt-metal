---
name: LLK Investigation Agent Prompt (Coordinator)
description: Stage 2 coordinator. Splits investigation into 3 parallel sub-agents (device-side, host-side, usage patterns). This file documents the sub-agents and how to invoke them.
type: reference
---

## Overview

Investigation is split into 3 parallel sub-agents that each produce structured tables. The orchestrator launches all 3 in parallel, then consolidates their outputs.

| Sub-Agent | File | What it does |
|-----------|------|-------------|
| Device-side | `llk_investigation_device_agent.md` | Wrapper signatures, LLK/ckernel impl, init state, parameter semantics |
| Host-side | `llk_investigation_host_agent.md` | Code generation, program factory, CB layout, parameter flow, derived param opportunities |
| Usage patterns | `llk_investigation_usage_agent.md` | ALL kernel call sites, boilerplate patterns, hardware constraints, chaining patterns |

## Prerequisites

- Stage 1 (Discovery + Locate) must have run first. The device-side and host-side agents use the Stage 1 Locator Results (Phase 4 output) to avoid wasting time searching for files.
- If Stage 1 was skipped (ops list was already known), run Stage 1 Phase 4 (locate) to generate the Locator Results table before launching these agents.

## How to invoke

```python
# Stage 0.5 must have produced locator_results (a table of file paths per op)

# Launch all 3 in parallel
Agent(subagent_type="Explore", run_in_background=True,
    prompt=device_template
        .replace("{{GROUP_NAME}}", group_name)
        .replace("{{LLK_CATEGORY}}", "elementwise unary")
        .replace("{{OPS_LIST}}", ops_list)
        .replace("{{LOCATOR_RESULTS}}", locator_results))

Agent(subagent_type="Explore", run_in_background=True,
    prompt=host_template
        .replace("{{GROUP_NAME}}", group_name)
        .replace("{{LLK_CATEGORY}}", "elementwise unary")
        .replace("{{OPS_LIST}}", ops_list)
        .replace("{{CODEGEN_FILE}}", "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp")
        .replace("{{LOCATOR_RESULTS}}", locator_results))

Agent(subagent_type="Explore", run_in_background=True,
    prompt=usage_template
        .replace("{{GROUP_NAME}}", group_name)
        .replace("{{LLK_CATEGORY}}", "elementwise unary")
        .replace("{{OPS_LIST}}", ops_list))
```

## Trust-the-Output Protocol

Once the 3 sub-agents complete, the orchestrator MUST:
1. Read and consolidate their structured table outputs
2. NOT re-read the same source files the agents read
3. If an agent's output is unclear or suspicious, ask it for clarification via SendMessage rather than re-reading source

The orchestrator's job is to make JUDGMENT CALLS (grouping, design decisions) using the agent-provided data, not to re-do the agents' research.

## Output Consolidation

The orchestrator writes `{category}_investigation.md` by combining the 3 sub-agent outputs:
- Device-side tables (wrapper sigs, init state, params, ckernel functions)
- Host-side tables (codegen, program factory, param flow, derived param opportunities)
- Usage tables (call sites, boilerplate, hardware constraints, chaining)

## Placeholders

- `{{LLK_CATEGORY}}` — operation category (e.g. elementwise unary)
- `{{GROUP_NAME}}` — functional sub-group (e.g. Activations)
- `{{OPS_LIST}}` — comma-separated operation names
- `{{LOCATOR_RESULTS}}` — table from Stage 0.5 Locator agent
- `{{CODEGEN_FILE}}` — path to the op_utils file for this category

## Known groups

Groups are category-specific. The Stage 1a Catalog agent discovers groups for each category using naming patterns or `{{KNOWN_GROUPS}}` seed data from this HQ. Do not hardcode group assignments here — they belong in the catalog output and are passed as `{{KNOWN_GROUPS}}` to the catalog agent when available from prior runs.
