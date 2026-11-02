#!/bin/bash
# Iteration 35: the same kernel attribution on the 2D path (gate_up dX^T [5184, 16384] x [1024, 16384]^T and
# w2 dX^T [8256, 5120] x [1024, 5120]^T, ttnn's own 2D choice on 11x10) and the small forward shapes.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
R=/home/ttuser/agobeljic/tt-metal
S=/tmp/claude-1000/-home-ttuser-agobeljic-tt-metal/0cbc71c0-30d2-4e45-a0c0-f0a2063e85a7/scratchpad
K=$R/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp
cd $R/tt-train
export TT_METAL_HOME=$R TT_METAL_RUNTIME_ROOT=$R TT_METAL_KERNEL_PATH=$R  # kernel sources from the checkout, not the build's libexec copy
unset TT_MESH_GRAPH_DESC_PATH
PY=../python_env/bin/python
for v in base nomath nopack nomath_nopack; do
  $PY $O/iter35_attrib_patch.py $S/bmm_kernel_pristine.cpp $K $v || break
  export TT_METAL_CACHE=$S/ttcache_$v; rm -rf $TT_METAL_CACHE  # the JIT cache does not hash the kernel source: a fresh cache per variant
  echo "=== $v $(date -u +%FT%TZ)"
  for shape in "--m 5184 --k 16384 --n 1024 --transpose-b" "--m 8256 --k 5120 --n 1024 --transpose-b" "--m 1024 --k 5184 --n 1792" "--m 1024 --k 1344 --n 5120"; do
    rm -f /dev/shm/sm_segment.* 2>/dev/null
    timeout 600 $PY $O/iter35_mm_time.py $shape --reps 10 2>&1 | grep "ttnn's choice"
  done
done
cp $S/bmm_kernel_pristine.cpp $K
cd $R && echo "kernel restored: $(git diff --quiet -- ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp && echo clean || echo DIRTY)"
echo "=== done $(date -u +%FT%TZ)"
