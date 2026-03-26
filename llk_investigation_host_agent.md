---
name: LLK Investigation Host Agent Prompt
description: "Investigation sub-agent: host-side analysis. Reads program factory, TTNN op definition, CB configuration, runtime arg flow, and host-side parameter transforms. Produces structured tables."
type: reference
---

## Usage

Invoke with `subagent_type: Explore`. Replace placeholders:
- `{{GROUP_NAME}}` — functional sub-group (e.g. Activations)
- `{{LLK_CATEGORY}}` — operation category (e.g. elementwise unary)
- `{{OPS_LIST}}` — comma-separated operation names
- `{{CODEGEN_FILE}}` — path to op_utils (e.g. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`)
- `{{LOCATOR_RESULTS}}` — the locator table from Stage 0.5

## Prompt Template

```
Analyze the HOST-SIDE configuration of these {{GROUP_NAME}} {{LLK_CATEGORY}} operations: {{OPS_LIST}}.

BREADCRUMB LOGGING — do this first:
Derive CATEGORY_SLUG from {{LLK_CATEGORY}} (lowercase, spaces → underscores).
BCRUMB="agent_logs/${CATEGORY_SLUG}_host_breadcrumbs.jsonl"
Run at start:
  mkdir -p agent_logs
  echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"host","group":"{{GROUP_NAME}}","ops":"{{OPS_LIST}}","codegen":"{{CODEGEN_FILE}}"}' >> $BCRUMB

Log granularly. Capture the reasoning chain for each op, not just conclusions.

After finding each op in codegen (record the exact generated strings):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","op":"OP","codegen_file":"path:line","generated_init":"exp_tile_init<{0}u>();","generated_compute":"exp_tile<{0}u>({1});","in_section":"parameterized/default"}' >> $BCRUMB

For each host-side transform identified (explain what it does and WHY it exists):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op":"OP","fact":"host_transform","param":"alpha","transform":"std::bit_cast<uint32_t>(alpha)","why":"kernel receives uint32 but user passes float — bit-cast on host side before sending"}' >> $BCRUMB
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","op":"OP","fact":"host_transform","param":"alpha","transform":"1.0f / alpha","why":"kernel uses reciprocal internally but user-facing API takes alpha directly"}' >> $BCRUMB

After reading program factory for each op:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"read","op":"OP","factory_file":"path","cb_layout":"c0:BF16 c1:BF16 c16:BF16","runtime_args_order":["src_addr","dst_addr","num_tiles","alpha_uint32"],"note":"e.g. alpha already bit-cast before being packed into runtime args"}' >> $BCRUMB

When identifying a derived param opportunity (explain the full chain):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"decision","op":"OP","what":"derived_param_opportunity","param":"alpha","user_api_type":"float","host_computes":"1.0f/alpha → bit_cast to uint32","kernel_receives":"uint32","helper_should":"accept float alpha, compute reciprocal and bit_cast internally","reason":"caller currently writes: op_utils adds 1/alpha transform — helper can absorb both"}' >> $BCRUMB

When discovering factory sharing (or lack thereof):
  echo '{"ts":"'"$(date -Iseconds)"'","event":"finding","fact":"factory_sharing","ops":["OP1","OP2","OP3"],"shared":true/false,"reason":"all use same unary_op_multi_core factory with identical CB layout"}' >> $BCRUMB

At completion:
  echo '{"ts":"'"$(date -Iseconds)"'","event":"complete","ops_analyzed":N,"derived_param_opportunities":M,"shared_factory":true/false}' >> $BCRUMB
Write agent_logs/${CATEGORY_SLUG}_host_execution_log.md: per-op codegen lines, exact transform chains, factory sharing status, full derived-param opportunity descriptions with user→host→kernel value flow.



Use the locator results to find each op's files — do NOT search for files yourself:
{{LOCATOR_RESULTS}}

For EACH operation:

1. CODE GENERATION — read the codegen file at the line from the locator. Extract:
   - The generated init string (exact text)
   - The generated compute string (exact text)
   - Does it appear in get_op_init_and_func_parameterized (has params) or get_op_init_and_func_default (no params)?
   - What host-side transforms are applied to parameters? (e.g., `std::bit_cast<uint32_t>(param0)`, `1.0f / param0`)

2. PROGRAM FACTORY — read the program factory file from the locator. Extract:
   - Which CBs are created (indices, data formats)
   - How runtime args are packed (which values, in what order)
   - Is there any host-side computation of derived parameters?
   - Do all ops in this group share the same program factory and CB layout?

3. HOST-SIDE PARAMETER TRANSFORMS — for each parameter:
   - What is the user-facing API? (e.g., `ttnn::elu(tensor, alpha)`)
   - What transforms happen before the kernel sees it? (bit-cast, reciprocal, scaling)
   - Could the kernel compute any of these transforms itself?

OUTPUT FORMAT (strict tables — minimal narrative):

## Host-Side Analysis: {{GROUP_NAME}}

### Code Generation

| Operation | Parameterized? | Generated Init | Generated Compute | Host Transforms |
|-----------|---------------|----------------|-------------------|-----------------|

### Program Factory

| Operation | Factory File | CB Layout (indices:formats) | Runtime Args | Shared Factory? |
|-----------|-------------|----------------------------|-------------|-----------------|

### Parameter Flow (Host → Device)

| Operation | Param | User API Value | Host Transform | Kernel Receives | Could Kernel Compute? |
|-----------|-------|---------------|----------------|-----------------|----------------------|

### Derived Parameter Opportunities

List parameters where the host pre-computes a value (reciprocal, sqrt, etc.) that the device kernel could compute internally. One line per opportunity:
`{op}.{param}: host computes {transform} — helper could absorb`
```
