#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/proj_sw/user_dev/jrock/tt-metal/codex_udpate_status_test_loop.log"
LIST_FILE="/proj_sw/user_dev/jrock/tt-metal/models/demos/deepseek_v3/tests/fused_op_unit_tests/codexapi_fused_op_unit_test_list.txt"

for i in $(seq 0 0); do
  TASK="$(grep -m1 -v '^[✅❌]' "$LIST_FILE" || true)"
  if [ -z "$TASK" ]; then
    echo "[${i}] No remaining tasks found in $LIST_FILE" | tee -a "$LOG_FILE"
    break
  fi
  echo "[${i}] $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
  set +e
  echo "Running task $TASK" | tee -a "$LOG_FILE"
  codexapi task "First read the AGENTS_GUIDE_ADD_TEST.md file for more context on what's contained in fused op unit tests and specifically pay attention to the secrtion 'Update resutls'. Run the update results process for the op $TASK." --max-iterations 3 --progress | tee -a "$LOG_FILE"
  cmd_status=${PIPESTATUS[0]}
  set -e

  if [ "$cmd_status" -eq 0 ]; then
    mark="✅"
  else
    mark="❌"
  fi

  line_num="$(grep -n -m1 -F -x "$TASK" "$LIST_FILE" | cut -d: -f1 || true)"
  if [ -n "$line_num" ]; then
    tmp_file="${LIST_FILE}.tmp"
    awk -v ln="$line_num" -v mark="$mark" 'NR==ln{$0=mark" "$0}1' "$LIST_FILE" > "$tmp_file"
    mv "$tmp_file" "$LIST_FILE"
  else
    echo "[${i}] Could not find task line to mark: $TASK" | tee -a "$LOG_FILE"
  fi

  echo "" | tee -a "$LOG_FILE"
done
