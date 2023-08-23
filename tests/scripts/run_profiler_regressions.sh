#/bin/bash

set -eo pipefail

remove_default_log_locations(){
    rm -rf $TT_METAL_HOME/tt_metal/tools/profiler/logs/ops
    rm -rf $TT_METAL_HOME/tt_metal/tools/profiler/logs/ops_device
}

run_profiling_test(){
    if [[ -z "$ARCH_NAME" ]]; then
      echo "Must provide ARCH_NAME in environment" 1>&2
      exit 1
    fi

    if [ "$ARCH_NAME" == "grayskull" ]; then
        make clean
        make build ENABLE_PROFILER=1

        remove_default_log_locations

        source build/python_env/bin/activate
        export PYTHONPATH=$TT_METAL_HOME

        TT_METAL_DEVICE_PROFILER=1 pytest $TT_METAL_HOME/tests/tt_metal/tools/profiler/test_device_profiler.py -vvv
    fi
}

run_post_proc_test(){
    source build/python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    remove_default_log_locations

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
