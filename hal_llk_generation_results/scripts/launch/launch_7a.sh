#!/bin/bash
# Run 7a: harness run with the run-6 configuration and the run-6b prompt
# (PROMPT_FILE=prompt-run6b.md = prompt-run5.md + the reset / no-reset requirement), on our bh-30 board.
# Run INSIDE the bh-30 IRD container (ssh -p 47030 bh-30). Starts the harness in the background and
# prints the harness log path.
set -u
B=/proj_sw/user_dev/ctr-sbratic
H=$B/hal-harness-run5rules
D=$B/tools/run7a

echo "== container checks"
{
  echo "hostname: $(hostname)"
  echo "boards in /dev/tenstorrent: $(ls /dev/tenstorrent | wc -l)"
  echo "claude: $(~/.local/bin/claude --version)"
  echo "/proj_sw mounted: $(test -d $B && echo yes || echo NO)"
  echo "my running processes: $(ps -eo args | grep -E "[r]un_hal_gen.sh|[c]laude -p|[p]ytest" | wc -l)"
  echo "prompt md5: $(md5sum < $H/prompt-run6b.md | cut -c1-8)"
} | tee $D/container_checks.txt
grep -q "boards in /dev/tenstorrent: 1$" $D/container_checks.txt || { echo "STOP: container does not see exactly one board"; exit 2; }
grep -q "claude: 2.1.287" $D/container_checks.txt || { echo "STOP: claude version differs from run 6 (2.1.287)"; exit 2; }
grep -q "/proj_sw mounted: yes" $D/container_checks.txt || { echo "STOP: /proj_sw not mounted"; exit 2; }
grep -q "my running processes: 0$" $D/container_checks.txt || { echo "STOP: my processes already running"; exit 2; }
grep -q "prompt md5: 411819d2$" $D/container_checks.txt || { echo "STOP: prompt-run6b.md changed since 2026-10-29"; exit 2; }

echo "== launching run 7a"
source $B/hal-env-ird/.venv/bin/activate
export PATH=$PATH:$HOME/.local/bin
cd $H
LOG=$B/hal-runs/harness_$(date +%Y%m%d_%H%M%S)_B.log
MODE=B TESTS="test_eltwise_binary:add" HAL_FILTER=0 PROMPT_FILE=$H/prompt-run6b.md \
HAL_COMMIT=b9e522d39ae534f0281499dae11b425574d89d0d DOCS_COMMIT=d8c6fa7c22fb89e109178778a537e5860fae0ac6 \
DOCS_INVALID="address_counters_hal_interface.html replay_hal_interface.html gpr_ops_hal_interface.html atomic_hal_interface.html" \
MODEL=claude-opus-5-5 TIMEOUT=4h MAX_TURNS=500 MAX_BUDGET=60 ELFRAY=$B/tools/elfray \
REPO=$B/tt-metal RUNS_ROOT=$B/hal-runs RESULTS_ROOT=$B/hal-results \
  nohup ./run_hal_gen.sh > $LOG 2>&1 < /dev/null &
echo "harness pid $!, log $LOG" | tee $D/launch_out.txt
