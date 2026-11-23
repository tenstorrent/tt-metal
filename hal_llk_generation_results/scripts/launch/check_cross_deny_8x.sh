#!/bin/bash
# For runs 8c-8e, started at nearly the same time: the harness adds a Read deny for every run dir that
# ALREADY exists in results/hal-runs when it writes ws/hal-settings.json (step 5/9). Checks that
#  - all run dirs differ,
#  - every run's settings deny every other given run's dir AND the run 8b dir,
#  - the configuration lines of every run's logs/run_meta.txt equal those of run 8b.
# Prints OK / FAIL lines and the deny-rule count from each harness log.
# Usage: check_cross_deny_8x.sh <8b run dir> <NAME>...  (waits up to 30 min for step 7/9 of every run)
set -u
B=/proj_sw/user_dev/ctr-sbratic
R8B=${1:?8b run dir}; shift
KEYS="mode base_commit hal_commit docs_commit docs_dir docs_invalid docs_remove confluence confluence_page main_ref model timeout max_turns max_budget_usd op_filter tests expect_passed hal_filter prompt_file sibling_deny dry_run raw_refmat raw_isa_dir claude_version sfpi_compiler"
declare -A RD LG
for N in "$@"; do
  LG[$N]=$(grep -oP 'log \K\S+' $B/tools/run$N/progress.txt 2>/dev/null | head -1)
  for i in $(seq 60); do
    RD[$N]=$(grep -oP '^\[harness [0-9:]+\] Run: \K\S+' ${LG[$N]:-/nonexistent} 2>/dev/null | head -1)
    [[ -n "${RD[$N]}" ]] && break; sleep 5
  done
  [[ -n "${RD[$N]}" ]] || { echo "FAIL $N: no run dir in log '${LG[$N]}'"; exit 1; }
  RD[$N]=${RD[$N]%/}
  echo "$N ${RD[$N]}"
done
[[ $(printf '%s\n' "${RD[@]}" | sort -u | wc -l) == $# ]] || { echo "FAIL: two runs share a run dir"; exit 1; }
# step 7/9 starts after run_meta.txt (incl. sfpi_compiler) and ws/hal-settings.json are written
for i in $(seq 360); do
  ALL=1; for N in "$@"; do grep -q '\] 7/9 ' ${LG[$N]} || ALL=0; done
  (( ALL )) && break; sleep 5
done
(( ALL )) || { echo "FAIL: not every run reached step 7/9 within 30 min"; exit 1; }
BAD=0
for N in "$@"; do
  echo "$N harness: $(grep -oP '^\[harness\] \K[0-9]+ deny rules' ${LG[$N]})"
  for M in "$@" 8b; do
    [[ $N == $M ]] && continue
    if [[ $M == 8b ]]; then T=$R8B; else T=${RD[$M]}; fi
    if grep -qF "/$(basename $T)/**)" ${RD[$N]}/ws/hal-settings.json; then
      echo "OK $N denies $M"
    else
      echo "FAIL $N does not deny $M"; BAD=1
    fi
  done
  for k in $KEYS; do
    a=$(grep -m1 "^$k=" $R8B/logs/run_meta.txt); b=$(grep -m1 "^$k=" ${RD[$N]}/logs/run_meta.txt)
    [[ "$a" == "$b" ]] || { echo "FAIL $N run_meta differs from 8b: '$b' vs '$a'"; BAD=1; }
  done
  echo "$N run_meta: checked $(wc -w <<< "$KEYS") configuration keys against 8b"
done
(( BAD )) && exit 1
echo "ALL OK"
