#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

run_perf_models_llm_javelin() {
    local tt_arch=$1
    local test_marker=$2

    if [ "$tt_arch" == "wormhole_b0" ]; then
        export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
    fi

    env pytest -n auto models/demos/falcon7b_common/tests -m $test_marker

    env QWEN_DIR=/mnt/MLPerf/tt_dnn-models/qwen/Qwen2-7B-Instruct FAKE_DEVICE=N150 pytest -n auto models/demos/qwen/tests -m $test_marker

    if [ "$tt_arch" == "wormhole_b0" ]; then
        env pytest -n auto models/demos/wormhole/mamba/tests -m $test_marker
    fi
    ## Merge all the generated reports
    env python3 models/perf/merge_perf_results.py
}

run_perf_models_cnn_javelin() {
    local tt_arch=$1
    local test_marker=$2

    # Run tests
    env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/experimental/functional_unet/tests -m $test_marker
    env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/stable_diffusion/tests -m $test_marker --timeout=480

    ## Merge all the generated reports
    env python3 models/perf/merge_perf_results.py
}

main() {
    # Parse the arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --pipeline-type)
                pipeline_type=$2
                shift
                ;;
            --tt-arch)
                tt_arch=$2
                shift
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
        shift
    done

    if [[ -z "$pipeline_type" ]]; then
      echo "--pipeline-type cannot be empty" 1>&2
      exit 1
    fi

    if [[ -z "$tt_arch" ]]; then
      echo "--tt-arch cannot be empty" 1>&2
      exit 1
    fi

    if [[ "$pipeline_type" == *"_virtual_machine"* ]]; then
        test_marker="models_performance_virtual_machine"
    elif [[ "$pipeline_type" == *"device_performance_bare_metal"* ]]; then
        test_marker="models_device_performance_bare_metal"
    elif [[ "$pipeline_type" == *"_bare_metal"* ]]; then
        test_marker="models_performance_bare_metal"
    else
        echo "$pipeline_type is using an unrecognized platform (suffix, ex. bare_metal, virtual_machine)" 2>&1
        exit 1
    fi

    if [[ "$pipeline_type" == "llm_javelin_models_performance"* ]]; then
        run_perf_models_llm_javelin "$tt_arch" "$test_marker"
    elif [[ "$pipeline_type" == "cnn_javelin_models_performance"* ]]; then
        run_perf_models_cnn_javelin "$tt_arch" "$test_marker"
    else
        echo "$pipeline_type is not recoognized performance pipeline" 2>&1
        exit 1
    fi
}

main "$@"
