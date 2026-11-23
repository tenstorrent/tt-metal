#!/bin/bash
# Runs 7b-7e: four repeats of run 7a (same harness, configuration and prompt as launch_7a.sh), one after
# another on our bh-30 board. Run INSIDE the bh-30 IRD container (ssh -p 47030 bh-30).
# Checks the container and that harness, prompt and settings are the files run 7a used, then starts
# driver_7bcde.sh with setsid nohup and returns. Progress: tools/run7bcde/progress.txt
set -u
B=/proj_sw/user_dev/ctr-sbratic
H=$B/hal-harness-run5rules
D=$B/tools/run7bcde

echo "== container checks"
{
  echo "hostname: $(hostname)"
  echo "boards in /dev/tenstorrent: $(ls /dev/tenstorrent | wc -l)"
  echo "claude: $(~/.local/bin/claude --version)"
  echo "/proj_sw mounted: $(test -d $B && echo yes || echo NO)"
  echo "my running processes: $(ps -eo args | grep -E "[r]un_hal_gen.sh|[c]laude -p|[p]ytest|[r]un_full_gdb|[d]river_7bcde" | wc -l)"
  echo "prompt md5: $(md5sum < $H/prompt-run6b.md | cut -c1-8)"
  echo "harness md5: $(md5sum < $H/run_hal_gen.sh | cut -c1-8)"
  echo "settings md5: $(md5sum < $H/hal-settings-B.json | cut -c1-8)"
} | tee $D/container_checks.txt
grep -q "boards in /dev/tenstorrent: 1$" $D/container_checks.txt || { echo "STOP: container does not see exactly one board"; exit 2; }
grep -q "claude: 2.1.287" $D/container_checks.txt || { echo "STOP: claude version differs from run 7a (2.1.287)"; exit 2; }
grep -q "/proj_sw mounted: yes" $D/container_checks.txt || { echo "STOP: /proj_sw not mounted"; exit 2; }
grep -q "my running processes: 0$" $D/container_checks.txt || { echo "STOP: my processes already running"; exit 2; }
grep -q "prompt md5: 411819d2$" $D/container_checks.txt || { echo "STOP: prompt-run6b.md changed since run 7a"; exit 2; }
grep -q "harness md5: b4afc953$" $D/container_checks.txt || { echo "STOP: run_hal_gen.sh changed since run 7a"; exit 2; }
grep -q "settings md5: 8f0d3a00$" $D/container_checks.txt || { echo "STOP: hal-settings-B.json changed since run 7a"; exit 2; }
[[ -e $D/progress.txt ]] && { echo "STOP: progress.txt exists (driver already started once)"; exit 2; }

echo "== launching runs 7b 7c 7d 7e"
setsid nohup bash $D/driver_7bcde.sh > $D/driver_nohup.log 2>&1 < /dev/null &
echo "driver pid $!" | tee $D/launch_out.txt
