#!/bin/bash

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

    run_module_tests "$tt_arch" "llrt" "$pipeline_type"
}

run_pipeline_tests() {
    local tt_arch=$1
    local pipeline_type=$2

    # Add your logic here for pipeline-specific tests
    echo "Running tests for pipeline: $pipeline_type with tt-arch: $tt_arch"

    # Call the appropriate module tests based on pipeline
    if [[ $pipeline_type == "post_commit" ]]; then
        run_post_commit_pipeline_tests "$tt_arch" "$pipeline_type"
    elif [[ $pipeline_type == "hyperquotidian" ]]; then
        run_post_commit_pipeline_tests "$tt_arch" "$pipeline_type"
        run_module_tests "$tt_arch" "tt_metal" "$pipeline_type"
    else
        echo "Unknown pipeline: $pipeline_type"
        exit 1
    fi
}

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

if [[ -n $module ]]; then
    # Module invocation
    run_module_tests "$tt_arch" "$module" "$pipeline_type"
elif [[ -n $pipeline_type ]]; then
    run_pipeline_tests "$tt_arch" "$pipeline_type"
else
    echo "You must have at least a module or pipeline_type specified"
    exit 1
fi
