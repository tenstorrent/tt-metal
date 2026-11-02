#!/bin/bash
# Iteration 35: thread attribution of ttnn's matmul compute kernel at the gate_up forward shard shape
# (1024 x 5184 x 16384, 11x10 1D mcast, 16x5 out block, 4x1 subblocks, in0_block_w 2 and 9).
# Variants rewrite the kernel source and build into a fresh TT_METAL_CACHE (the JIT cache keys on defines and
# arguments, not on the kernel text: the first attempt reused the cached binaries); the pristine kernel is restored at the end.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
R=/home/ttuser/agobeljic/tt-metal
S=/tmp/claude-1000/-home-ttuser-agobeljic-tt-metal/0cbc71c0-30d2-4e45-a0c0-f0a2063e85a7/scratchpad
K=$R/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp
cd $R/tt-train
export TT_METAL_HOME=$R TT_METAL_RUNTIME_ROOT=$R TT_METAL_KERNEL_PATH=$R  # kernel sources from the checkout, not the build's libexec copy
unset TT_MESH_GRAPH_DESC_PATH
PY=../python_env/bin/python
for v in base nomath nopack nomath_nopack base; do
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  $PY $O/iter35_attrib_patch.py $S/bmm_kernel_pristine.cpp $K $v || break
  export TT_METAL_CACHE=$S/ttcache_$v; rm -rf $TT_METAL_CACHE  # the JIT cache does not hash the kernel source: a fresh cache per variant
  echo "=== $v $(date -u +%FT%TZ)"
  timeout 1200 $PY $O/iter35_matmul_sweep1d.py --m 1024 --k 5184 --n 16384 --grids 11x10 --in0-block-w 2,9 --subs 4x1 --out-blocks 16x5 --reps 10 > $O/iter35_attrib_$v.log 2>&1; echo "rc=$?"
  grep -E "^ *[0-9.]+ ms" $O/iter35_attrib_$v.log | head -2
done
cp $S/bmm_kernel_pristine.cpp $K
cd $R && git status --short ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp && echo "kernel restored: $(git diff --quiet -- ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp && echo clean || echo DIRTY)"
echo "=== done $(date -u +%FT%TZ)"
