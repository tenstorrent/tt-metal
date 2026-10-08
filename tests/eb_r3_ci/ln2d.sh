#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass: layernorm_pre_allgather_2d.cpp on the merged head (main changed it again,
# mid-kernel hw_startup). Its modules whole, the per-tile define removed against as committed, bit for bit; then the 2D
# core grid cases' device time, three passes.
cd /work
N=tests/ttnn/nightly/unit_tests/operations/fused
K=ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d.cpp
grep -n "ELTWISE_BINARY_PER_TILE_HANDOFF" $K
echo "##### modules: define removed against kept"
EB_SHOW_ERR=1 EB_RUN_LIMIT=3300 bash tests/eb_r3_ci/bits_strip.sh tests/eb_r3_ci/ln2d_files.txt -p eb_seed_plugin $N/test_distributed_layernorm_pre_allgather.py $N/test_distributed_rmsnorm_allgather.py
cp $K /tmp/ln2d_orig.cpp
grep -v "^#define ELTWISE_BINARY_PER_TILE_HANDOFF" /tmp/ln2d_orig.cpp > $K
printf '%s|#define ELTWISE_BINARY_PER_TILE_HANDOFF true\n' $K > /tmp/ln2d_optin.txt
for i in 1 2 3; do
  echo "##### pass $i: 2D core grid cases, main's program against the define"
  EB_SHOW_ERR=1 bash tests/eb_r3_ci/ab_set.sh /tmp/ln2d_optin.txt $N/test_distributed_layernorm_pre_allgather.py $N/test_distributed_rmsnorm_allgather.py -k "rms_norm_2d or (test_rmsnorm_2d_core_grid_single_device and True)"
done
cp /tmp/ln2d_orig.cpp $K
