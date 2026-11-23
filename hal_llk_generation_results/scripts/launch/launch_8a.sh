#!/bin/bash
# Run 8a: one HAL-arm run (mode B) under the run 7a conditions with the cfg-cost prompt
# (prompt-run6b-cfgcost.md), then the postcheck. Run INSIDE the bh-30 IRD container of job 151073
# (ssh -p 47030 bh-30, board 2 = the board of runs 7a-7e). Checks the container and the md5 of every
# file the run depends on, then starts driver_8a.sh with setsid nohup and returns.
# Progress: tools/run8a/progress.txt
set -u
B=/proj_sw/user_dev/ctr-sbratic
H=$B/hal-harness-run5rules
D=$B/tools/run8a

echo "== container checks"
{
  echo "hostname: $(hostname)"
  echo "boards in /dev/tenstorrent: $(ls /dev/tenstorrent | wc -l)"
  echo "board index: $(ls /dev/tenstorrent | tr '\n' ' ')"
  echo "claude: $($B/tools/claude-2.1.287/claude --version)"
  echo "claude logged in: $(timeout 30 $B/tools/claude-2.1.287/claude auth status < /dev/null 2>/dev/null | grep -q '"loggedIn": true' && echo yes || echo NO)"
  echo "/proj_sw mounted: $(test -d $B && echo yes || echo NO)"
  echo "my running processes: $(ps -eo args | grep -E "[r]un_hal_gen.sh|[c]laude -p|[p]ytest|[r]un_full_gdb|[d]river_" | wc -l)"
  echo "harness md5: $(md5sum < $H/run_hal_gen.sh | cut -c1-8)"
  echo "prompt md5: $(md5sum < $H/prompt-run6b-cfgcost.md | cut -c1-8)"
  echo "settings md5: $(md5sum < $H/hal-settings-B.json | cut -c1-8)"
  echo "metrics md5: $(md5sum < $H/metrics.py | cut -c1-8)"
  echo "postcheck md5: $(md5sum < $B/tools/run7bcde/postcheck_7x.sh | cut -c1-8)"
  echo "dyn md5: $(md5sum < $B/tools/dyn_gdb/run_full_gdb.sh | cut -c1-8)"
} | tee $D/container_checks.txt
grep -q "reservation-151073$" $D/container_checks.txt || { echo "STOP: not the container of job 151073"; exit 2; }
grep -q "boards in /dev/tenstorrent: 1$" $D/container_checks.txt || { echo "STOP: container does not see exactly one board"; exit 2; }
grep -q "board index: 2 $" $D/container_checks.txt || { echo "STOP: board is not index 2 (runs 7a-7e)"; exit 2; }
grep -q "claude: 2.1.287" $D/container_checks.txt || { echo "STOP: claude version differs from run 7a (2.1.287)"; exit 2; }
grep -q "claude logged in: yes" $D/container_checks.txt || { echo "STOP: claude is not logged in (claude auth status)"; exit 2; }
grep -q "/proj_sw mounted: yes" $D/container_checks.txt || { echo "STOP: /proj_sw not mounted"; exit 2; }
grep -q "my running processes: 0$" $D/container_checks.txt || { echo "STOP: my processes already running"; exit 2; }
grep -q "harness md5: b4afc953$" $D/container_checks.txt || { echo "STOP: run_hal_gen.sh changed since run 7a"; exit 2; }
grep -q "prompt md5: 3dbea124$" $D/container_checks.txt || { echo "STOP: prompt-run6b-cfgcost.md is not the 2026-11-19 version"; exit 2; }
grep -q "settings md5: 8f0d3a00$" $D/container_checks.txt || { echo "STOP: hal-settings-B.json changed since run 7a"; exit 2; }
grep -q "metrics md5: 9e2e47d7$" $D/container_checks.txt || { echo "STOP: metrics.py is not v2"; exit 2; }
grep -q "postcheck md5: 2514e6d9$" $D/container_checks.txt || { echo "STOP: postcheck_7x.sh changed since runs 7b-7e"; exit 2; }
grep -q "dyn md5: b29e3daa$" $D/container_checks.txt || { echo "STOP: run_full_gdb.sh changed since runs 7a-7e"; exit 2; }
[[ -e $D/progress.txt ]] && { echo "STOP: progress.txt exists (driver already started once)"; exit 2; }

echo "== launching run 8a"
setsid nohup bash $D/driver_8a.sh > $D/driver_nohup.log 2>&1 < /dev/null &
echo "driver pid $!" | tee $D/launch_out.txt
