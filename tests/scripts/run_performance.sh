#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

run_perf_models_other() {
    local tt_arch=$1
    local test_marker=$2

    if [ "$tt_arch" == "grayskull" ]; then
        env pytest models/demos/grayskull/resnet50/tests/test_perf_e2e_resnet50.py -m $test_marker
    fi

    if [ "$tt_arch" == "wormhole_b0" ]; then
        env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/wormhole/resnet50/tests/test_perf_e2e_resnet50.py -m $test_marker
    fi

    env pytest -n auto tests/ttnn/integration_tests/bert/test_performance.py -m $test_marker

    env pytest -n auto models/demos/ttnn_falcon7b/tests -m $test_marker

    env pytest -n auto tests/ttnn/integration_tests/whisper/test_performance.py -m $test_marker

    env pytest -n auto models/demos/metal_BERT_large_11/tests -m $test_marker

    ## Merge all the generated reports
    env python models/perf/merge_perf_results.py
}

run_perf_models_llm_javelin() {
    local tt_arch=$1
    local test_marker=$2

    env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/falcon7b_common/tests -m $test_marker

    if [ "$tt_arch" == "wormhole_b0" ]; then
        env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/mamba/tests -m $test_marker
    fi

    env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/mistral7b/tests -m $test_marker
    env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/llama31_8b/tests -m $test_marker

    ## Merge all the generated reports
    env python models/perf/merge_perf_results.py
}

run_perf_models_cnn_javelin() {
    local tt_arch=$1
    local test_marker=$2

    # Run tests
    env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto tests/device_perf_tests/stable_diffusion -m $test_marker --timeout=480

    ## Merge all the generated reports
    env python models/perf/merge_perf_results.py
}

run_device_perf_models() {
    set -eo pipefail
    local test_marker=$1

    env pytest tests/device_perf_tests/stable_diffusion -m $test_marker --timeout=600

    if [ "$tt_arch" == "grayskull" ]; then
        #TODO(MO): Until #6560 is fixed, GS device profiler test are grouped with
        #Model Device perf regression tests to make sure thy run on no-soft-reset BMs
        tests/scripts/run_profiler_regressions.sh PROFILER_NO_RESET

        env pytest models/demos/grayskull/resnet50/tests -m $test_marker

        env pytest models/demos/metal_BERT_large_11/tests -m $test_marker

        env pytest models/demos/ttnn_falcon7b/tests -m $test_marker --timeout=360

        env pytest models/demos/bert/tests -m $test_marker
    fi

    if [ "$tt_arch" == "wormhole_b0" ]; then
        env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/wormhole/resnet50/tests -m $test_marker

        env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/wormhole/mamba/tests -m $test_marker

        env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/metal_BERT_large_11/tests -m $test_marker

        env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/falcon7b_common/tests -m $test_marker
    fi

    ## Merge all the generated reports
    env python models/perf/merge_device_perf_results.py
}

run_device_perf_ops() {
    local test_marker=$1

    env pytest tests/tt_eager/ops_device_perf/run_op_profiling.py -m $test_marker

    env pytest tests/device_perf_tests/matmul_stagger/test_matmul_stagger.py -m $test_marker
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

    if [[ "$pipeline_type" == *"device_performance"* ]]; then
        run_device_perf_models "$test_marker"
        run_device_perf_ops "$test_marker"
    elif [[ "$pipeline_type" == "llm_javelin_models_performance"* ]]; then
        run_perf_models_llm_javelin "$tt_arch" "$test_marker"
    elif [[ "$pipeline_type" == "cnn_javelin_models_performance"* ]]; then
        run_perf_models_cnn_javelin "$tt_arch" "$test_marker"
    elif [[ "$pipeline_type" == *"other_models_performance"* ]]; then
        run_perf_models_other "$tt_arch" "$test_marker"
    else
        echo "$pipeline_type is not recoognized performance pipeline" 2>&1
        exit 1
    fi
}

main "$@"
