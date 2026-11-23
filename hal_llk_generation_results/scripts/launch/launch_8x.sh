#!/bin/bash
# Starts ONE of the runs 8c, 8d, 8e in the IRD container this script runs in (each container sees exactly
# one bh-30 board). Usage: launch_8x.sh <NAME> <expected board index>
# Checks the container, that nothing with this run name exists yet, the md5 of every file the run depends
# on (= the files of run 8b, checked 2026-11-21) and of the run8cde scripts (MD5SUMS), then starts
# driver_8x.sh <NAME> with setsid nohup and returns. Progress: tools/run<NAME>/progress.txt
set -u
NAME=${1:?usage: launch_8x.sh <NAME> <board>}
BOARD=${2:?usage: launch_8x.sh <NAME> <board>}
B=/proj_sw/user_dev/ctr-sbratic
H=$B/harness/hal-harness-run5rules
S=$B/tools/run8cde
D=$B/tools/run$NAME
PC=$B/tools/perf_cycles
DG=$B/tools/dyn_gdb
[[ $NAME =~ ^8[cde]$ ]] || { echo "STOP: NAME must be 8c, 8d or 8e"; exit 2; }
[[ -e $D ]] && { echo "STOP: $D exists (run $NAME already started once)"; exit 2; }
compgen -G "$DG/status/run${NAME}_*" > /dev/null && { echo "STOP: dynamic status files run${NAME}_* exist"; exit 2; }
for x in $PC/ws/$NAME $PC/out/retro/${NAME}_r1 $PC/out/retro/p${NAME}_orig_r1 $PC/out/retro/p${NAME}_progress.txt; do
  [[ -e $x ]] && { echo "STOP: perf item $x exists"; exit 2; }
done
(cd $S && md5sum -c --quiet MD5SUMS) || { echo "STOP: run8cde scripts differ from MD5SUMS"; exit 2; }
mkdir -p $D

echo "== container checks ($NAME)"
m() { md5sum < $1 | cut -c1-8; }
{
  echo "hostname: $(hostname)"
  echo "boards in /dev/tenstorrent: $(ls /dev/tenstorrent | wc -l)"
  echo "board: $(ls /dev/tenstorrent | tr '\n' ' ')"
  echo "claude: $($B/tools/claude-2.1.287/claude --version)"
  echo "claude logged in: $(timeout 30 $B/tools/claude-2.1.287/claude auth status < /dev/null 2>/dev/null | grep -q '"loggedIn": true' && echo yes || echo NO)"
  echo "/proj_sw mounted: $(test -d $B && echo yes || echo NO)"
  echo "my running processes: $(ps -eo args | grep -E "[r]un_hal_gen.sh|[c]laude -p|[p]ytest|[r]un_full_gdb|[r]un_hal_gdb|[d]river_|[b]uild_counts|[t]t-smi" | wc -l)"
  echo "harness md5: $(m $H/run_hal_gen.sh)"
  echo "prompt md5: $(m $H/prompt-run6b-cfgcost.md)"
  echo "settings md5: $(m $H/hal-settings-B.json)"
  echo "metrics md5: $(m $H/metrics.py)"
  echo "postcheck md5: $(m $B/tools/run7bcde/postcheck_7x.sh)"
  echo "tt_tti_counts md5 (recorded only): $(m $B/tools/tt_tti_run6/tt_tti_counts.py)"
  echo "setup_ws md5: $(m $PC/setup_ws.sh)"
  echo "driver_perf md5: $(m $PC/driver_perf.sh)"
  echo "run_perf md5: $(m $PC/run_perf.sh)"
  echo "run_hal_gdb md5: $(m $DG/run_hal_gdb.sh)"
  echo "dyn_gdb_plugin md5: $(m $DG/dyn_gdb_plugin.py)"
  echo "summarize_status md5: $(m $DG/summarize_status.py)"
  echo "make_report_en md5: $(m $B/tools/report_en/make_report_en.py)"
  echo "driver_8x md5: $(m $S/driver_8x.sh)"
} | tee $D/container_checks.txt
C=$D/container_checks.txt
need() { grep -q "$1" $C || { echo "STOP: $2"; exit 2; }; }
need "boards in /dev/tenstorrent: 1$" "container does not see exactly one board"
need "board: $BOARD $" "container board is not $BOARD"
need "claude: 2.1.287" "claude version differs from runs 7a-8b (2.1.287)"
need "claude logged in: yes" "claude is not logged in (claude auth status)"
need "/proj_sw mounted: yes" "/proj_sw not mounted"
need "my running processes: 0$" "my processes already running in this container"
need "harness md5: 4cfda0c8$" "run_hal_gen.sh is not the run 8b version (4cfda0c8)"
need "prompt md5: 3dbea124$" "prompt-run6b-cfgcost.md is not the run 8a/8b version"
need "settings md5: 8f0d3a00$" "hal-settings-B.json changed since run 7a"
need "metrics md5: 7537af1d$" "metrics.py is not the run 8b version (7537af1d)"
need "postcheck md5: 2514e6d9$" "postcheck_7x.sh changed since runs 7b-7e"
need "setup_ws md5: ee8b7a94$" "perf setup_ws.sh changed"
need "driver_perf md5: 098a4024$" "perf driver_perf.sh changed"
need "run_perf md5: eb9406c1$" "perf run_perf.sh changed"
need "run_hal_gdb md5: 27364f72$" "run_hal_gdb.sh changed"
need "dyn_gdb_plugin md5: 30eed9a5$" "dyn_gdb_plugin.py changed"
need "summarize_status md5: f9c7ab9a$" "summarize_status.py changed"
need "make_report_en md5: 97f15e2a$" "make_report_en.py changed"

echo "== launching run $NAME"
setsid nohup bash $S/driver_8x.sh $NAME > $D/driver_nohup.log 2>&1 < /dev/null &
echo "driver pid $!" | tee $D/launch_out.txt
