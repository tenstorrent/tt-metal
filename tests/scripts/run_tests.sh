#!/bin/bash

set -eo pipefail

# Default argument values
default_tt_arch="grayskull"
default_pipeline_type="post_commit"
default_dispatch_mode="fast"

assert_requested_module_matches() {
    local actual=$1
    local expected=$2

    if [[ $actual != $expected ]]; then
        echo "Requested module should be $expected, but was $actual"
        exit 1
    fi
}

# Module tests
run_llrt_module_tests() {
    local tt_arch=$1
    local module=$2
    local pipeline_type=$3

    assert_requested_module_matches "$module" "llrt"

    # Add your logic here for module-specific tests
    echo "llrt: $module with tt-arch: $tt_arch and pipeline-type: $pipeline_type"
}

run_tt_metal_module_tests() {
    local tt_arch=$1
    local module=$2
    local pipeline_type=$3

    assert_requested_module_matches "$module" "tt_metal"

    # Add your logic here for module-specific tests
    echo "tt_metal: $module with tt-arch: $tt_arch and pipeline-type: $pipeline_type"
}

run_module_tests() {
    local tt_arch=$1
    local module=$2
    local pipeline_type=$3

    if [[ $module == "llrt" ]]; then
        run_llrt_module_tests "$tt_arch" "$module" "$pipeline_type"
    elif [[ $module == "tt_metal" ]]; then
        run_tt_metal_module_tests "$tt_arch" "$module" "$pipeline_type"
    else
        echo "Unknown module: $module"
        exit 1
    fi
}

# Pipeline tests
run_post_commit_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    # Switch to modules only soon
    # run_module_tests "$tt_arch" "llrt" "$pipeline_type"
    if [[ $dispatch_mode == "slow" ]]; then
        ./tests/scripts/run_pre_post_commit_regressions_slow_dispatch.sh
    elif [[ $dispatch_mode == "fast" ]]; then
        ./tests/scripts/run_pre_post_commit_regressions_fast_dispatch.sh
    elif [[ $dispatch_mode == "fast-multi-queue-single-device" ]]; then
        TT_METAL_NUM_HW_CQS=2 ./build/test/tt_metal/unit_tests_fast_dispatch_single_chip_multi_queue --gtest_filter=MultiCommandQueueSingleDeviceFixture.*
    fi
}

run_eager_package_end_to_end_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2

    env pytest tests/end_to_end_tests --tt-arch $tt_arch -m $pipeline_type
}

run_frequent_models_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    make tests

    source build/python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    # Please put model runs in here from now on - thank you
    ./tests/scripts/run_models.sh
}

run_frequent_api_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    make tests

    source build/python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME

    if [[ $dispatch_mode == "slow" ]]; then
        TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_frequent
        echo "Running Python API unit tests in SD for frequent..."
        ./tests/scripts/run_python_api_unit_tests.sh
    else
        if [[ $tt_arch == "wormhole_b0" ]]; then
            pytest  tests/tt_eager/python_api_testing/unit_testing/misc/test_all_gather.py -k nightly
        else
            echo "API tests are not available for fast dispatch because they're already covered in post-commit"
        fi
    fi
}

# Run frequent multi device pipeline tests - these are the t3000 + 4xn300 tests
run_frequent_multi_device_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    # Switch to modules only soon
    # run_module_tests "$tt_arch" "llrt" "$pipeline_type"
    ./tests/scripts/run_frequent_regressions_multi_device.sh
}

run_models_performance() {
    local tt_arch=$1
    local pipeline_type=$2

    ./tests/scripts/run_performance.sh --pipeline-type $pipeline_type --tt-arch $tt_arch
}

run_models_performance_bare_metal_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    # BERT large via new enqueue APIs. I know this is not a unit test, but I would like to avoid BERT large breaking, so this
    # is a safe place to put it for the time being. Need to run these as separate tests to avoid segfault (TODO(agrebenisan): Investigate why)
    if [[ $dispatch_mode == "slow" ]]; then
        env pytest -svv models/demos/metal_BERT_large_11/tests/test_bert_batch_dram.py::test_bert_batch_dram[BERT_LARGE-batch_9-BFLOAT16-DRAM]
        env pytest -svv models/demos/metal_BERT_large_11/tests/test_bert_batch_dram.py::test_bert_batch_dram_with_program_cache[BERT_LARGE-batch_9-BFLOAT16-DRAM]
    else
        echo "Not running bert-large in fast-dispatch mode on bare-metal"
    fi

    run_models_performance "$tt_arch" "$pipeline_type"
}

run_models_performance_virtual_machine_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2

    run_models_performance "$tt_arch" "$pipeline_type"
}

run_stress_post_commit_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    # Run for 23.5h to allow next run to kick off
    RUNTIME_MAX=84600
    suite_duration=4500
    max_duration=$((RUNTIME_MAX-suite_duration))
    iter=1
    while [ $SECONDS -lt $max_duration ]; do
        echo "Info: [stress] Doing iteration $iter"
        if [[ $dispatch_mode == "slow" ]]; then
            ./tests/scripts/run_pre_post_commit_regressions_slow_dispatch.sh
        else
            ./tests/scripts/run_pre_post_commit_regressions_fast_dispatch.sh
        fi
        iter=$((iter+1))
    done
}

run_post_commit_multi_device_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    # Switch to modules only soon
    # run_module_tests "$tt_arch" "llrt" "$pipeline_type"
    ./tests/scripts/run_pre_post_commit_regressions_multi_device.sh
}

run_microbenchmarks_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    export TT_METAL_DEVICE_PROFILER=1

    source build/python_env/bin/activate
    ./tests/scripts/run_moreh_microbenchmark.sh
    pytest -svv tests/tt_metal/microbenchmarks
}

run_ttnn_sweeps_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/run_ttnn_sweeps.sh
}

run_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    # Add your logic here for pipeline-specific tests
    echo "Running tests for pipeline: $pipeline_type with tt-arch: $tt_arch"
    # Call the appropriate module tests based on pipeline
    if [[ $pipeline_type == "post_commit" ]]; then
        run_post_commit_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "frequent_models" ]]; then
        run_frequent_models_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "frequent_api" ]]; then
        run_frequent_api_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "eager_host_side" ]]; then
        run_eager_package_end_to_end_pipeline_tests "$tt_arch" "$pipeline_type"
    elif [[ $pipeline_type == "eager_package_silicon" ]]; then
        run_eager_package_end_to_end_pipeline_tests "$tt_arch" "$pipeline_type"
    elif [[ $pipeline_type == *"models_performance_bare_metal" || $pipeline_type == "models_device_performance_bare_metal" ]]; then
        run_models_performance_bare_metal_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "models_performance_virtual_machine" ]]; then
        run_models_performance_virtual_machine_pipeline_tests "$tt_arch" "$pipeline_type"
    elif [[ $pipeline_type == "stress_post_commit" ]]; then
        run_stress_post_commit_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "post_commit_multi_device" ]]; then
        run_post_commit_multi_device_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "frequent_multi_device" ]]; then
        run_frequent_multi_device_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "microbenchmarks" ]]; then
        run_microbenchmarks_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "ttnn_sweeps" ]]; then
        run_ttnn_sweeps_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    else
        echo "Unknown pipeline: $pipeline_type"
        exit 1
    fi
}

validate_and_set_env_vars() {
    local tt_arch=$1
    local dispatch_mode=$2

    if [[ -z "$TT_METAL_HOME" ]]; then
      echo "Must provide TT_METAL_HOME in environment" 1>&2
      exit 1
    fi

    if [[ -n "$ARCH_NAME" && "$ARCH_NAME" != $tt_arch ]]; then
      echo "ARCH_NAME does not match provided --tt-arch" 1>&2
      exit 1
    fi

    export ARCH_NAME=$tt_arch

    if [[ $dispatch_mode == "slow" ]]; then
        if [[ -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
            echo "Trying to run --dispatch-mode slow without TT_METAL_SLOW_DISPATCH_MODE in environment" 1>&2
            exit 1
        fi
    else
        if [[ ! -z "$TT_METAL_SLOW_DISPATCH_MODE" ]]; then
            echo "Trying to run --dispatch-mode fast with TT_METAL_SLOW_DISPATCH_MODE in environment" 1>&2
            exit 1
        fi
    fi

    if [[ -z "$PYTHONPATH" ]]; then
        export PYTHONPATH=$TT_METAL_HOME
    fi
}

set_up_chdir() {
    # The user might have multiple entries in their PYTHONPATH so we should try to find the right one
    IFS=':' read -ra ENTRIES <<< "$PYTHONPATH"
    for ENTRY in "${ENTRIES[@]}"; do
      if [[ $ENTRY == *"tt-metal" ]]; then
        cd $ENTRY
        return
      fi
    done
    echo "Could not find the 'tt-metal' directory in your PYTHONPATH." 1>&2
    exit 1
}

main() {
    # Parse the arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tt-arch)
                tt_arch=$2
                shift
                ;;
            --dispatch-mode)
                dispatch_mode=$2
                shift
                ;;
            --module)
                module=$2
                shift
                ;;
            --pipeline-type)
                pipeline_type=$2
                shift
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
        shift
    done

    # Set default values if arguments are not provided
    tt_arch=${tt_arch:-$default_tt_arch}
    dispatch_mode=${dispatch_mode:-$default_dispatch_mode}
    pipeline_type=${pipeline_type:-$default_pipeline_type}

    available_dispatch_modes=("fast" "slow" "fast-multi-queue-single-device")
    available_tt_archs=("grayskull" "wormhole_b0")

    # Validate arguments
    if [[ ! " ${available_tt_archs[*]} " =~ " $tt_arch " ]]; then
        echo "Invalid tt-arch argument. Must be an available tt-arch."
        exit 1
    fi
    if [[ ! " ${available_dispatch_modes[*]} " =~ " $dispatch_mode " ]]; then
        echo "Invalid dispatch_mode argument. Must be an available dispatch_mode."
        exit 1
    fi

    validate_and_set_env_vars "$tt_arch" "$dispatch_mode"
    set_up_chdir

    if [[ -n $module ]]; then
        # Module invocation
        run_module_tests "$tt_arch" "$module" "$pipeline_type"
    elif [[ -n $pipeline_type ]]; then
        run_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    else
        echo "You must have at least a module or pipeline_type specified"
        exit 1
    fi
}

main "$@"
