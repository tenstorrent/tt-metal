#/bin/bash

set -eo pipefail

run_profiling_test(){
    if [[ -z "$ARCH_NAME" ]]; then
      echo "Must provide ARCH_NAME in environment" 1>&2
      exit 1
    fi

    echo "Make sure this test runs in a build with ENABLE_PROFILER=1"

    if [ "$ARCH_NAME" == "grayskull" ]; then
        source build/python_env/bin/activate
        export PYTHONPATH=$TT_METAL_HOME

        TT_METAL_DEVICE_PROFILER=1 pytest $TT_METAL_HOME/tests/tt_metal/tools/profiler/test_device_profiler.py -vvv

        rm -rf $TT_METAL_HOME/tt_metal/tools/profiler/logs/
        rm -rf $TT_METAL_HOME/tt_metal/tools/profiler/output/
        $TT_METAL_HOME/tt_metal/tools/profiler/profile_this.py -c "pytest -svvv tests/models/bert_large_performant/unit_tests/test_bert_large_fused_qkv_matmul.py::test_bert_large_fused_qkv_matmul_test[BFLOAT8_B-in0_DRAM-in1_L1-bias_None-out_DRAM]"

        ls $TT_METAL_HOME/tt_metal/tools/profiler/output/ops/BERT_large_fused_qkv_matmul_BFLOAT8_B-in0_DRAM-in1_L1-bias_None-out_DRAM/profile_log_ops.csv
    fi
}

run_post_proc_test(){
    source build/python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    pytest $TT_METAL_HOME/tests/tt_metal/tools/profiler/test_device_logs.py -vvv
}

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

if [[ $1 == "PROFILER" ]]; then
    run_profiling_test
elif [[ $1 == "POST_PROC" ]]; then
    run_post_proc_test
else
    run_profiling_test
    run_post_proc_test
fi
