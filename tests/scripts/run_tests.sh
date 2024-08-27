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
    fi
}

run_frequent_api_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    if [[ $dispatch_mode == "slow" ]]; then
        TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_frequent
        echo "Running Python API unit tests in SD for frequent..."
        ./tests/scripts/run_python_api_unit_tests.sh
    else
        if [[ $tt_arch == "wormhole_b0" ]]; then
            pytest -n auto tests/ttnn/unit_tests/operations/test_all_gather.py -k nightly
        else
            echo "API tests are not available for fast dispatch because they're already covered in post-commit"
        fi
    fi
}

run_models_performance() {
    local tt_arch=$1
    local pipeline_type=$2

    ./tests/scripts/run_performance.sh --pipeline-type $pipeline_type --tt-arch $tt_arch
}

run_models_performance_multi_device() {
    local tt_arch=$1
    local pipeline_type=$2

    ./tests/scripts/run_performance.sh --pipeline-type $pipeline_type --tt-arch $tt_arch
}

run_models_performance_bare_metal_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

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
    max_duration=84600
    iter=1
    cur_duration=0
    expected_duration=0
    while [ $expected_duration -lt $max_duration ]; do
        echo "Info: [stress] Doing iteration $iter"
        start_time=$(date +%s%N) # capture nanoseconds
        if [[ $dispatch_mode == "slow" ]]; then
            ./tests/scripts/run_pre_post_commit_regressions_slow_dispatch.sh
        else
            ./tests/scripts/run_pre_post_commit_regressions_fast_dispatch.sh
        fi
        end_time=$(date +%s%N)
        elapsed=$((end_time - start_time))/1000000000
        cur_duration=$((cur_duration + elapsed))
        avg_duration=$((cur_duration / iter))
        expected_duration=$((cur_duration + avg_duration))
        iter=$((iter+1))
        echo "Info: [stress] expected elapsed time $expected_duration, elapsed time $cur_duration, avg iteration time $avg_duration"
    done
}

run_post_commit_multi_device_unstable_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/multi_chip/run_unstable_multi_device.sh
}

run_microbenchmarks_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    export TT_METAL_DEVICE_PROFILER=1

    source python_env/bin/activate
    ./tests/scripts/run_moreh_microbenchmark.sh
    pytest -svv tests/tt_metal/microbenchmarks
}

run_ttnn_sweeps_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/run_ttnn_sweeps.sh
}

run_demos_single_card_n150_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/single_card/run_demos_single_card_n150_tests.sh
}

run_demos_single_card_n300_func_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/single_card/run_demos_single_card_n300_func_tests.sh
}

run_demos_single_card_n300_perf_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/single_card/run_demos_single_card_n300_perf_tests.sh
}

##########################T3000##########################
# Run t3000 unit tests
unit_t3000_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/t3000/run_t3000_unit_tests.sh
}

# Run t3000 frequent tests
frequent_t3000_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/t3000/run_t3000_frequent_tests.sh
}

# Run t3000 demo tests
demos_t3000_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/t3000/run_t3000_demo_tests.sh
}

# Run t3000 model perf tests
model_perf_t3000_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/t3000/run_t3000_model_perf_tests.sh --pipeline-type "$pipeline_type"
}
##########################T3000##########################

##########################TG##########################
# Run tg unit tests
unit_tg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/tg/run_tg_unit_tests.sh
}

# Run tg frequent tests
frequent_tg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/tg/run_tg_frequent_tests.sh
}

# Run tg demo tests
demos_tg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/tg/run_tg_demo_tests.sh
}

# Run tg model perf tests
model_perf_tg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/tg/run_tg_model_perf_tests.sh --pipeline-type "$pipeline_type"
}
##########################TG##########################

##########################TGG##########################
# Run tgg unit tests
unit_tgg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/tgg/run_tgg_unit_tests.sh
}

# Run tgg frequent tests
frequent_tgg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/tgg/run_tgg_frequent_tests.sh
}

# Run tgg demo tests
demos_tgg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/tgg/run_tgg_demo_tests.sh
}

# Run tgg model perf tests
model_perf_tgg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/tgg/run_tgg_model_perf_tests.sh --pipeline-type "$pipeline_type"
}
##########################TGG##########################

run_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    # Add your logic here for pipeline-specific tests
    echo "Running tests for pipeline: $pipeline_type with tt-arch: $tt_arch"
    # Call the appropriate module tests based on pipeline
    if [[ $pipeline_type == "post_commit" ]]; then
        run_post_commit_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "frequent_api" ]]; then
        run_frequent_api_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == *"models_performance_bare_metal" || $pipeline_type == "models_device_performance_bare_metal" ]]; then
        run_models_performance_bare_metal_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "models_performance_virtual_machine" ]]; then
        run_models_performance_virtual_machine_pipeline_tests "$tt_arch" "$pipeline_type"
    elif [[ $pipeline_type == "stress_post_commit" ]]; then
        run_stress_post_commit_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "microbenchmarks" ]]; then
        run_microbenchmarks_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "ttnn_sweeps" ]]; then
        run_ttnn_sweeps_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "demos_single_card_n150" ]]; then
        run_demos_single_card_n150_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "demos_single_card_n300_func" ]]; then
        run_demos_single_card_n300_func_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "demos_single_card_n300_perf" ]]; then
        run_demos_single_card_n300_perf_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    # T3000 pipelines
    elif [[ $pipeline_type == "unit_t3000_device" ]]; then
        unit_t3000_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "frequent_t3000_device" ]]; then
        frequent_t3000_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "demos_t3000_device" ]]; then
        demos_t3000_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == *"model_perf_t3000_device" ]]; then
        model_perf_t3000_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
    # TG pipelines
    elif [[ $pipeline_type == "unit_tg_device" ]]; then
        unit_tg_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "frequent_tg_device" ]]; then
        frequent_tg_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "demos_tg_device" ]]; then
        demos_tg_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == *"model_perf_tg_device" ]]; then
        model_perf_tg_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
    # TGG pipelines
    elif [[ $pipeline_type == "unit_tgg_device" ]]; then
        unit_tgg_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "frequent_tgg_device" ]]; then
        frequent_tgg_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "demos_tgg_device" ]]; then
        demos_tgg_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == *"model_perf_tgg_device" ]]; then
        model_perf_tgg_device "$tt_arch" "$pipeline_type" "$dispatch_mode"
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

    available_dispatch_modes=("fast" "slow")
    available_tt_archs=("grayskull" "wormhole_b0" "blackhole")

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

    if [[ $pipeline_type != "eager_"* ]]; then
      set_up_chdir
    fi

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
