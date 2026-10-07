#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass: leak2. Probes whose program is main's under every setting, after a bf16 block-pack
# program, after a DRAM add, and after a block-pack program plus a DRAM add; per RISC durations as well.
cd /work
export EB_R3_LOG_RULE=1 EB_LEAK2=1
T=tests/eb_r3_ci/test_eb_leak.py
A="EB_R3_BU_MIN=1 EB_R3_NO_BLOCK_PACK=1"
for i in 1 2 3; do
  for k in 1 2; do
    echo "##### pass $i leak2: per-tile pack before vs kind $k block pack before"; bash tests/eb_r3_ci/ab_envs.sh "$A" "EB_R3_BU_MIN=1 EB_R3_BP_MIN=1 EB_R3_BP_KIND=$k" $T
    for c in "DEVICE TRISC0 KERNEL DURATION [ns]" "DEVICE TRISC1 KERNEL DURATION [ns]" "DEVICE TRISC2 KERNEL DURATION [ns]" "DEVICE BRISC KERNEL DURATION [ns]" "DEVICE KERNEL DURATION PER CORE MAX [ns]"; do
      echo "##### pass $i leak2 kind $k col $c"; python3 tests/eb_r3_ci/prof_reduce.py /tmp/ebenvs env --col "$c" 2>&1 | grep -E "probe" | cut -c1-260
    done
  done
done
