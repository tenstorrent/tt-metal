#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, seventh review: test_group_norm_DRAM (two-pass statistics, welford_groupnorm.cpp) with the LLK asserts
# on and the watcher reporting them, on the PR head merged with main, and with welford_groupnorm.cpp's hand-off defines
# removed (main's program for that kernel); first failure stops each side.
cd /work
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-} TT_METAL_LLK_ASSERTS=1 TT_METAL_WATCHER=2
K=ttnn/cpp/ttnn/operations/normalization/groupnorm/device/kernels/compute/welford_groupnorm.cpp
cp $K /tmp/wgn.orig
for side in pr main; do
  cp /tmp/wgn.orig $K
  [[ $side == main ]] && sed -i '/^#define ELTWISE_BINARY_PER_TILE_HANDOFF/d' $K
  grep -c "ELTWISE_BINARY_PER_TILE_HANDOFF" $K
  export TT_METAL_CACHE=/tmp/eb_asserts_$side; rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE
  echo "##### $(date -u +%T) side $side"
  timeout -s INT -k 60 1800 python3 -m pytest -p no:cacheprovider -q -rfE -x --timeout=300 tests/ttnn/unit_tests/operations/fused/test_group_norm_DRAM.py -k "test_group_norm_DRAM and two_pass" > /tmp/asserts_$side.log 2>&1
  echo "rc=$?"
  grep -iE "assert|watcher|tripped|passed|failed|error|Timeout" /tmp/asserts_$side.log | grep -v "^\s*$" | grep -vi "hugepage" | cut -c1-400 | tail -40
  ls generated/watcher 2>/dev/null && tail -30 generated/watcher/watcher.log 2>/dev/null | cut -c1-300
done
cp /tmp/wgn.orig $K
echo "##### end $(date -u +%T)"
