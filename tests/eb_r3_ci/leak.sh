#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass: a block-pack program (main's contiguous form, #58816's form) before an unchanged
# probe program, against a per-tile-pack program before it; the probe's device time per setting, three passes.
cd /work
export EB_R3_LOG_RULE=1
T=tests/eb_r3_ci/test_eb_leak.py
A="EB_R3_BU_MIN=1 EB_R3_NO_BLOCK_PACK=1"
for i in 1 2 3; do
  echo "##### pass $i leak: per-tile pack before vs main's block pack before"; bash tests/eb_r3_ci/ab_envs.sh "$A" "EB_R3_BU_MIN=1 EB_R3_BP_MIN=1 EB_R3_BP_KIND=1" $T
  echo "##### pass $i leak: per-tile pack before vs #58816's block pack before"; bash tests/eb_r3_ci/ab_envs.sh "$A" "EB_R3_BU_MIN=1 EB_R3_BP_MIN=1 EB_R3_BP_KIND=2" $T
done
