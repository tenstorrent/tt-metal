#!/usr/bin/env bash
# Scratch, not for merge. Functional groups on the tree under test. Usage: run_func.sh <set 1|2>
set +e
cd /work; export TT_METAL_HOME=/work
U=tests/ttnn/unit_tests/operations; N=tests/ttnn/nightly/unit_tests/operations
g() { name=$1; shift; T0=$(date +%s)
  timeout -s INT -k 60 ${GT:-1500} python3 -m pytest -q -rfE -p no:cacheprovider -o timeout_method=thread "$@" > /tmp/g_$name.log 2>&1
  echo "GRP $name rc=$? $(( $(date +%s) - T0 ))s: $(grep -aE '[0-9]+ (passed|failed|error|skipped)' /tmp/g_$name.log | tail -1 | cut -c1-160)"
  grep -a -E '^(FAILED|ERROR) ' /tmp/g_$name.log | cut -d' ' -f2 | sort -u | head -15 | sed "s/^/FAILID $name /"
  grep -a -E "TT_THROW|TENSIX TIMED OUT" /tmp/g_$name.log | head -2 | cut -c1-200 | sed "s/^/HANG $name /"
  grep -a -E "^E  +[A-Za-z]|error:" /tmp/g_$name.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-260 | sort | uniq -c | sort -rn | head -4 | sed "s/^/ERRL $name /"; }
echo "== $(git log -1 --format='%h %s' | cut -c1-80)"
if [[ $1 == 1 ]]; then
g dist_sharded $U/fused/test_distributed_layernorm_sharded.py
g dist_unit $U/fused/test_distributed_layernorm.py
g ln_post_nightly $N/fused/test_distributed_layernorm_post_allgather.py
g rms_post_nightly $N/fused/test_distributed_rmsnorm_allgather.py
g wan_rmsnorm $N/transformers/test_distributed_fused_rmsnorm.py
g hardswish $U/eltwise/test_unary.py $U/eltwise/test_unary_category2_bfloat16.py $U/eltwise/test_unary_category2_float32.py $N/eltwise/backward/test_backward_hardswish.py -k hardswish
elif [[ $1 == 3 ]]; then
g dit_layernorm $N/transformers/test_distributed_dit_layernorm.py
g dit_rms_unary $N/experimental/test_dit_rms_norm_unary_fused.py
elif [[ $1 == 4 ]]; then
g binary_ng $U/eltwise/test_binary_ng_bcast_fp32_dest_acc.py $U/eltwise/test_binary_ng_sharded_fp32_batch.py $U/eltwise/test_binary_ng_program_cache.py
g kda $N/experimental/kda
elif [[ $1 == 5 ]]; then
g welford_two_pass $N/reduction/test_two_pass_reduction.py
g welford_corners $N/reduction/test_reduction_op_corners.py -k "var or std"
g welford_i32 ci_ab/test_var_i32.py
else
g ln_sharded $U/fused/test_layer_norm_sharded.py
g rms_sharded $U/fused/test_rms_norm_sharded.py
g ln_rms $U/fused/test_layer_norm.py $U/fused/test_rms_norm.py
g group_norm $U/fused/test_group_norm.py
g rotary_llama $N/experimental/test_rotary_embedding_llama.py
g binary_ng $U/eltwise/test_binary_ng_bcast_fp32_dest_acc.py $U/eltwise/test_binary_ng_sharded_fp32_batch.py $U/eltwise/test_binary_ng_program_cache.py
g sampling $U/reduce/test_sampling.py
g concat_sharded $N/data_movement/test_concat_block_sharded.py $N/data_movement/test_concat_memory_configs_and_layouts.py
g kda $N/experimental/kda
fi
echo "== done"
