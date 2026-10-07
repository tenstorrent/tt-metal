#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass: leak3. Is the next op's slowdown the binary placement (a pre whose only difference is
# 1024 or 4096 dead NOPs) or the block pack (a 32-tile-per-core bf16 block-pack pre)?
cd /work
export EB_R3_LOG_RULE=1 EB_LEAK3=1
T=tests/eb_r3_ci/test_eb_leak.py
A="EB_R3_BU_MIN=1 EB_R3_NO_BLOCK_PACK=1"
for i in 1 2 3; do
  echo "##### pass $i leak3: pre without padding vs pre with 1024 NOPs"; bash tests/eb_r3_ci/ab_envs.sh "$A" "$A EB_PAD_PRE=1024" $T -k pad
  echo "##### pass $i leak3: pre without padding vs pre with 4096 NOPs"; bash tests/eb_r3_ci/ab_envs.sh "$A" "$A EB_PAD_PRE=4096" $T -k pad
  echo "##### pass $i leak3: 32-tile pre per-tile pack vs kind 2"; bash tests/eb_r3_ci/ab_envs.sh "$A" "EB_R3_BU_MIN=1 EB_R3_BP_MIN=1 EB_R3_BP_KIND=2" $T -k bf16_32
done
