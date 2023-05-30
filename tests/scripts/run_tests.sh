#!/bin/bash

set -eo pipefail

# Default argument values
default_tt_arch="grayskull"
default_pipeline_type="post_commit"

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

    # Switch to modules only soon
    # run_module_tests "$tt_arch" "llrt" "$pipeline_type"
    ./tests/scripts/run_pre_post_commit_regressions.sh

}

run_frequent_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2

    # Switch to modules only soon
    ./build_tt_metal.sh
    make tests

    source build/python_env/bin/activate
    export PYTHONPATH=$TT_METAL_HOME
    python -m pip install -r tests/python_api_testing/requirements.txt

    run_post_commit_pipeline_tests "$tt_arch" "$pipeline_type"

    env python tests/scripts/run_tt_metal.py

    # Tests tensor and tt_dnn op APIs
    ./tests/scripts/run_tt_lib_regressions.sh

    # Tests profiler module
    ./tests/scripts/run_profiler_regressions.sh

    # Please put model runs in here from now on - thank you
    ./tests/scripts/run_models.sh
}

run_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2

    # Add your logic here for pipeline-specific tests
    echo "Running tests for pipeline: $pipeline_type with tt-arch: $tt_arch"

    # Call the appropriate module tests based on pipeline
    if [[ $pipeline_type == "post_commit" ]]; then
        run_post_commit_pipeline_tests "$tt_arch" "$pipeline_type"
    elif [[ $pipeline_type == "frequent" ]]; then
        run_frequent_pipeline_tests "$tt_arch" "$pipeline_type"
    else
        echo "Unknown pipeline: $pipeline_type"
        exit 1
    fi
}

validate_and_set_env_vars() {
    local tt_arch=$1

    if [[ -z "$TT_METAL_HOME" ]]; then
      echo "Must provide TT_METAL_HOME in environment" 1>&2
      exit 1
    fi

    if [ "$TT_METAL_ENV" != "dev" ]; then
      echo "Must set TT_METAL_ENV as dev" 1>&2
      exit 1
    fi

    if [[ -n "$ARCH_NAME" && "$ARCH_NAME" != $tt_arch ]]; then
      echo "ARCH_NAME does not match provided --tt-arch" 1>&2
      exit 1
    fi

    export ARCH_NAME=$tt_arch

    export PYTHONPATH=$TT_METAL_HOME
}

set_up_chdir() {
    cd $TT_METAL_HOME

}

main() {
    # Parse the arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tt-arch)
                tt_arch=$2
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
    pipeline_type=${pipeline_type:-$default_pipeline_type}

    available_tt_archs=("grayskull" "wormhole_b0")

    # Validate arguments
    if [[ ! " ${available_tt_archs[*]} " =~ " $tt_arch " ]]; then
        echo "Invalid tt-arch argument. Must be an available tt-arch."
        exit 1
    fi

    validate_and_set_env_vars "$tt_arch"
    set_up_chdir

    if [[ -n $module ]]; then
        # Module invocation
        run_module_tests "$tt_arch" "$module" "$pipeline_type"
    elif [[ -n $pipeline_type ]]; then
        run_pipeline_tests "$tt_arch" "$pipeline_type"
    else
        echo "You must have at least a module or pipeline_type specified"
        exit 1
    fi
}

main "$@"
