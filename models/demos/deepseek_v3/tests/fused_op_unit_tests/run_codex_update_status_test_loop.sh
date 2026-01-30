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
  codexapi task "First read the AGENTS_GUIDE_ADD_TEST.md file for more context on what's contained in fused op unit tests. Remove the previous results file in 'models/demos/deepseek_v3/tests/fused_op_unit_tests/MODULE/test_results/OP_NAME_results.csv', where MODULE is the module and OP_NAME is the op name, and remove all logs in the 'logs' subfolder for this op. Test all test funcitons and parameter configurations for the op $TASK. Since some tests take very long (especially the long sequence lengths), test each parameter combination separately; use CI=false. Create a csv summary of results in 'models/demos/deepseek_v3/tests/fused_op_unit_tests/MODULE/test_results/OP_NAME_results.csv', where MODULE is the module and OP_NAME is the op name. Before each run, reset the machine using 'tt-smi -glx_reset'. Set a 20 mins timeout, if you need a longer timeout update the prompt in run_codex_update_status_test_loop.sh with the new timeout, look at the output log before killing the run, leave it running if the log changed within the last 10 mins. Use the following structure for the csv: test_name,status,pcc,e2e perf, device perf,failure_reason (optional),comment (optional), link, timestamp. Add one line per concrete test (parameter combination) with all details filled in. In 'comment' column add any comments for potential fixes if the test is failing or any other comments of interest. Copy the log file for each test configuration into the logs sub-folder and add a link to the log file to the 'link' column in the results csv. Update the README.md with the new status of the test; see Update instructions in the README.md for more details." --max-iterations 3 --progress | tee -a "$LOG_FILE"
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
