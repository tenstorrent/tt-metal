#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, seventh review: test_group_norm_DRAM's two-pass cases with the LLK asserts on and the watcher, one
# program per job (a tripped assert stops the device). usage: r18asserts2.sh <pr|main> [-k expression]
cd /work
side=$1; KEXPR=${2:-"test_group_norm_DRAM and two_pass"}
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-} TT_METAL_LLK_ASSERTS=1 TT_METAL_WATCHER=2
K=ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp
[[ $side == main ]] && sed -i '/^#define ELTWISE_BINARY_PER_TILE_HANDOFF/d' $K
echo "##### $(date -u +%T) side $side, hand-off defines in the kernel: $(grep -c ELTWISE_BINARY_PER_TILE_HANDOFF $K), -k [$KEXPR]"
export TT_METAL_CACHE=/tmp/eb_asserts_$side; rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE
timeout -s INT -k 60 1800 python3 -m pytest -p no:cacheprovider -q -rfE -x --timeout=300 tests/ttnn/unit_tests/operations/fused/test_group_norm_DRAM.py -k "$KEXPR" > /tmp/asserts.log 2>&1
echo "rc=$?"
grep -E "PASSED|FAILED|passed|failed|tripped|TT_THROW|Timeout" /tmp/asserts.log | sed -E 's/\x1b\[[0-9;]*m//g' | cut -c1-260 | tail -40
grep -B2 -A6 "tripped" generated/watcher/watcher.log 2>/dev/null | cut -c1-300 | head -40
echo "##### end $(date -u +%T)"
