#!/bin/bash
# Whole-pipeline cc_optimize: optimize EVERY mode of a MULTI-MODAL model, so "optimize the model"
# covers the full composed emit-e2e pipeline — not just one head. Each mode retargets the perf-mcp
# server (PERF_MCP_PERF_TEST/CASE/PCC injected into a per-mode mcp config) so one manifest serves all
# heads. The mode list comes from a TSV (cols: mode<TAB>perf_nodeid<TAB>case<TAB>pcc_model_rel_nodeid);
# this is what the discovery schema's `perf_tests` list feeds once auto-generated.
#
# Usage: run_pipeline.sh <model_dir> <modes.tsv>
set -u
cd /home/ttuser/tt-metal
CC=models/experimental/perf_automation/cc_optimize
BASE=$CC/mcp_config.json
MODEL_DIR=${1:?usage: run_pipeline.sh <model_dir> <modes.tsv>}
MODES_FILE=${2:?usage: run_pipeline.sh <model_dir> <modes.tsv>}

while IFS=$'\t' read -r MODE PERF CASE PCC; do
  [ -z "${MODE:-}" ] && continue
  case "$MODE" in \#*) continue;; esac
  echo "==================== OPTIMIZING MODE: $MODE ===================="
  CFG=/tmp/mcp_${MODE}.json
  # per-mode mcp config = base config + the 3 retarget overrides in the server's env block
  /home/ttuser/tt-metal/python_env/bin/python -c "
import json
c=json.load(open('$BASE')); e=c['mcpServers']['perf-mcp']['env']
e['PERF_MCP_PERF_TEST']='$PERF'; e['PERF_MCP_PERF_CASE']='$CASE'; e['PERF_MCP_PCC_TEST']='$PCC'
json.dump(c,open('$CFG','w'),indent=2)
"
  PROMPT="Optimize the '$MODE' mode of $MODEL_DIR for device_ms using the perf-mcp tools. Follow the
optimize-model procedure: git_head (checkpoint), profile_model, pick the largest gap_ms bucket,
decide knob/fuse/kernel from bound_by, edit the REAL call path, check_pcc (must be ok),
measure_candidate (IRON RULE: a real win needs check_pcc ok AND verdict valid AND is_real_gain; a
REJECTED measure is never a win regardless of device_ms), git_commit verified wins / git_revert the
rest. For the kernel rung read models/experimental/perf_automation/GUIDELINES/11_TT_LANG_KERNELS.md.
Run CONTINUOUSLY until at the roofline floor or PCC-safe ideas are exhausted for every bucket. LEAVE
CLEAN before stopping (revert any in-progress edit); end with git_head. Report start->final device_ms,
every committed win, and distance to the roofline target for this mode."
  claude -p "$PROMPT" --mcp-config "$CFG" --strict-mcp-config \
    --allowedTools mcp__perf-mcp__profile_model mcp__perf-mcp__measure_candidate mcp__perf-mcp__check_pcc mcp__perf-mcp__git_head mcp__perf-mcp__git_commit mcp__perf-mcp__git_revert Read Edit Write Bash Grep Glob \
    --output-format stream-json --verbose
  echo "==================== MODE $MODE DONE ===================="
done < "$MODES_FILE"
echo "==================== WHOLE-PIPELINE RUN COMPLETE ===================="
