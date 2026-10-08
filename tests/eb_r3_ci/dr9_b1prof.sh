#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724 fourth review): device time of the fused programs that run ReduceToOneB1, the PR head
# ec7714f90f8's reduce_to_one_b1.hpp and reduce_to_one_kernel.cpp (tests/eb_r3_ci/head) against the PR's, on a Blackhole Galaxy:
# the device profiler's device-side log in slow dispatch (per-core kernel start and end, no host capture), one test per process,
# three passes of both sides in alternating order, one JIT cache per side; eb9_devprof.py compares per chip and the device time.
# usage: dr9_b1prof.sh <test names: decoder mlp moe r2o>
cd /work
export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_ALLOCATOR_MODE_HYBRID=1 TT_METAL_RUNTIME_ROOT=/work TT_METAL_DEVICE_PROFILER=1
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
B1=models/demos/deepseek_v3_b1/tests/unit_tests
H=models/demos/deepseek_v3_b1/unified_kernels/reduce_to_one_b1.hpp
KR=models/demos/deepseek_v3_b1/micro_ops/reduce_to_one_b1/kernels/reduce_to_one_kernel.cpp
declare -A F K
F[decoder]=$B1/test_decoder_block.py
K[decoder]="rigged_groups1 and t8_seven_picked and sram_bspm_off and random_weights and just_decoder_mla and just_decoder_moe and not mtp and device_params0-0-32768"
F[mlp]=$B1/test_moe_mlp.py; K[mlp]="test_mlp_with_reduce and not half and not 7sram and not all-dram"
F[moe]=$B1/test_moe_mlp.py; K[moe]="test_moe_fused_with_reduce and full_groups and t8_partial"
F[r2o]=$B1/test_reduce_to_one_b1.py; K[r2o]="test_reduce_to_one_1d"
O=/tmp/eb9p; rm -rf $O; mkdir -p $O; cp $H $O/h.pr; cp $KR $O/k.pr
use() { if [[ $1 == head ]]; then cp tests/eb_r3_ci/head/reduce_to_one_b1.hpp $H; cp tests/eb_r3_ci/head/reduce_to_one_kernel.cpp $KR; else cp $O/h.pr $H; cp $O/k.pr $KR; fi; }
for p in 1 2 3; do
  if (( p % 2 )); then order="head pr"; else order="pr head"; fi
  for v in $order; do
    use $v
    for t in "$@"; do
      export TT_METAL_CACHE=$O/cache_$v TT_METAL_PROFILER_DIR=$O/prof/$t/${p}_$v; mkdir -p $TT_METAL_CACHE $TT_METAL_PROFILER_DIR
      timeout -s INT -k 60 ${EB_RUN_LIMIT:-1800} python3 -m pytest -p eb_seed_plugin -p no:cacheprovider -o timeout_method=thread --timeout=0 -q -rfE "${F[$t]}" -k "${K[$t]}" > $O/log_${t}_${p}_$v.txt 2>&1
      echo "--- pass $p $v $t rc=$?: $(grep -E 'passed|failed|error' $O/log_${t}_${p}_$v.txt | tail -1 | cut -c1-160) $(ls $TT_METAL_PROFILER_DIR/.logs/ 2>/dev/null | tr '\n' ' ')"
    done
  done
done
use pr
python3 tests/eb_r3_ci/eb9_devprof.py $O/prof head pr
echo "##### done"
