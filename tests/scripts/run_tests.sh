#!/bin/bash

set -eo pipefail

# Default argument values
default_tt_arch="grayskull"
default_pipeline_type="post_commit"
default_dispatch_mode="fast"
default_model="None"

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
        ./tests/scripts/run_cpp_fd2_tests.sh
        ./tests/scripts/run_cpp_unit_tests.sh
    elif [[ $dispatch_mode == "fast" ]]; then
        ./tests/scripts/run_python_api_unit_tests.sh
        ./tests/scripts/run_cpp_unit_tests.sh
    fi
}

run_frequent_api_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    if [[ $dispatch_mode == "slow" ]]; then
        TT_METAL_ENABLE_ERISC_IRAM=1 TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=DispatchStress.TensixRunManyTimes
        echo "Running Python API unit tests in SD for frequent..."
        ./tests/scripts/run_python_api_unit_tests.sh
    fi
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
            ./tests/scripts/run_cpp_fd2_tests.sh
            ./tests/scripts/run_cpp_unit_tests.sh
        else
            ./tests/scripts/run_python_api_unit_tests.sh
            ./tests/scripts/run_cpp_unit_tests.sh
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

run_ttnn_sweeps_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/run_ttnn_sweeps.sh
}

##########################T3000##########################
# Run t3000 unit tests
unit_t3000_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/t3000/run_t3000_unit_tests.sh
}

# Run t3000 integration tests
frequent_t3000_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    ./tests/scripts/t3000/run_t3000_integration_tests.sh
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

    ./tests/scripts/t3000/run_t3000_perf_tests.sh --pipeline-type "$pipeline_type"
}
##########################T3000##########################

##########################TG##########################
# Run tg unit tests
# DEPRECATED: Galaxy unit tests now call pytest directly from GitHub Actions workflow
# See: .github/workflows/galaxy-unit-tests-impl.yaml
unit_tg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3
    local model=$4

    echo "ERROR: unit_tg_device is deprecated."
    echo "Galaxy unit tests now run directly from GitHub Actions workflow."
    echo "See: .github/workflows/galaxy-unit-tests-impl.yaml"
    exit 1
}

# Run tg frequent tests
frequent_tg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3
    local model=$4

    ./tests/scripts/tg/run_tg_frequent_tests.sh --model "$model"
}

# Run tg demo tests
# DEPRECATED: Galaxy demo tests now call pytest directly from GitHub Actions workflow
# See: .github/workflows/galaxy-demo-tests-impl.yaml
demos_tg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3
    local model=$4

    echo "ERROR: demos_tg_device is deprecated."
    echo "Galaxy demo tests now run directly from GitHub Actions workflow."
    echo "See: .github/workflows/galaxy-demo-tests-impl.yaml"
    exit 1
}

# Run tg model perf tests
# DEPRECATED: Galaxy model perf tests now call pytest directly from GitHub Actions workflow
# See: .github/workflows/galaxy-model-perf-tests-impl.yaml
model_perf_tg_device() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3

    echo "ERROR: model_perf_tg_device is deprecated."
    echo "Galaxy model perf tests now run directly from GitHub Actions workflow."
    echo "See: .github/workflows/galaxy-model-perf-tests-impl.yaml"
    exit 1
}
##########################TG##########################

run_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2
    local dispatch_mode=$3
    local model=$4

    # Add your logic here for pipeline-specific tests
    echo "Running tests for pipeline: $pipeline_type with tt-arch: $tt_arch"
    # Call the appropriate module tests based on pipeline
    if [[ $pipeline_type == "post_commit" ]]; then
        run_post_commit_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "frequent_api" ]]; then
        run_frequent_api_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "stress_post_commit" ]]; then
        run_stress_post_commit_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
    elif [[ $pipeline_type == "ttnn_sweeps" ]]; then
        run_ttnn_sweeps_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode"
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
        unit_tg_device "$tt_arch" "$pipeline_type" "$dispatch_mode" "$model"
    elif [[ $pipeline_type == "frequent_tg_device" ]]; then
        frequent_tg_device "$tt_arch" "$pipeline_type" "$dispatch_mode" "$model"
    elif [[ $pipeline_type == "demos_tg_device" ]]; then
        demos_tg_device "$tt_arch" "$pipeline_type" "$dispatch_mode" "$model"
    elif [[ $pipeline_type == *"model_perf_tg_device" ]]; then
        model_perf_tg_device "$tt_arch" "$pipeline_type" "$dispatch_mode" "$model"
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
    for ENTRY in "${ENTRIES[@]}"; do
      if [[ -d "$ENTRY/tt_metal" ]]; then
        cd "$ENTRY"
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
            --model)
                model=$2
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
    model=${model:-$default_model}

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
        run_pipeline_tests "$tt_arch" "$pipeline_type" "$dispatch_mode" "$model"
    else
        echo "You must have at least a module or pipeline_type specified"
        exit 1
    fi
}

main "$@"
