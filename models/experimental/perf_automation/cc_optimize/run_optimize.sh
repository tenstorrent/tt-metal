#!/bin/bash
# Drive Claude Code headless as the CONTINUOUS optimize loop on seamless, using the perf-mcp
# deterministic tools. Capability test of the Claude-Code-native design (no FSM iterations).
set -e
cd /home/ttuser/tt-metal
CC_DIR=models/experimental/perf_automation/cc_optimize

read -r -d '' PROMPT <<'EOF' || true
You are optimizing the TTNN model at models/demos/hf_seamless_m4t_medium for device_ms, using the
perf-mcp MCP tools. This is a CAPABILITY TEST of a continuous optimize loop — demonstrate the loop
on the top 1-2 bottleneck buckets, then STOP and summarize. Do NOT run for hours.

Procedure:
1. Call mcp__perf-mcp__git_head (remember the clean sha) and mcp__perf-mcp__profile_model (see the
   per-bucket gap_ms + bound_by + roofline_target_ms).
2. For the bucket with the largest gap_ms: decide KNOB vs KERNEL from bound_by — compute+under-occupied
   grid -> occupy grid / lower math fidelity; memory -> lower dtype / shard into L1; dispatch -> fuse
   / author a ttl kernel. Read the executed source under models/demos/hf_seamless_m4t_medium first,
   then make ONE edit on the real call path (change the tensor's memory_config/dtype, not just a
   program_config kwarg).
3. Call mcp__perf-mcp__check_pcc. If status != ok -> mcp__perf-mcp__git_revert to the clean sha; note it.
4. Call mcp__perf-mcp__measure_candidate. THE IRON RULE: a real win requires check_pcc ok AND
   measure_candidate verdict 'valid' AND is_real_gain true. If so -> mcp__perf-mcp__git_commit with a
   clear message (lever + before->after). Otherwise -> mcp__perf-mcp__git_revert. A 'REJECTED'
   measurement is NEVER a win no matter how low its device_ms (it means the edit crashed/garbled the
   profile).
5. Repeat for at most 2 buckets, then STOP and report: what you tried, the pcc + measure verdicts,
   what you kept vs reverted (and why), and the cumulative device_ms vs the roofline target.

Use the perf-mcp tools for ALL measurement and git — never eyeball a timing or commit by hand.
Log your reasoning as you go.
EOF

claude -p "$PROMPT" \
  --mcp-config "$CC_DIR/mcp_config.json" \
  --strict-mcp-config \
  --allowedTools mcp__perf-mcp__profile_model mcp__perf-mcp__measure_candidate mcp__perf-mcp__check_pcc mcp__perf-mcp__git_head mcp__perf-mcp__git_commit mcp__perf-mcp__git_revert Read Edit Bash Grep Glob \
  --output-format stream-json --verbose
