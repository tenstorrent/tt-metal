#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

run_perf_models_other() {
    local tt_arch=$1
    local test_marker=$2

    if [ "$tt_arch" == "grayskull" ]; then
        env pytest "tests/ttnn/integration_tests/resnet/test_performance.py" -m $test_marker

        env pytest "tests/ttnn/integration_tests/bert/test_performance.py" -m $test_marker

        env pytest "tests/ttnn/integration_tests/bloom/test_performance.py" -m $test_marker

        env pytest "tests/ttnn/integration_tests/t5/test_performance.py" -m $test_marker

        env pytest models/demos/ttnn_falcon7b/tests -m $test_marker

        env pytest models/experimental/vgg/tests -m $test_marker

        env pytest models/experimental/vit/tests -m $test_marker

        env pytest models/experimental/roberta/tests -m $test_marker

        env pytest models/experimental/t5/tests -m $test_marker

        env pytest models/demos/resnet/tests -m $test_marker

        env pytest models/demos/metal_BERT_large_11/tests -m $test_marker

        env pytest models/experimental/deit/tests -m $test_marker

        env pytest models/experimental/stable_diffusion/tests -m $test_marker

        env pytest models/experimental/whisper/tests -m $test_marker

        env pytest models/experimental/bloom/tests -m $test_marker

        env pytest "tests/ttnn/integration_tests/whisper/test_performance.py::test_performance" -m $test_marker

        env pytest "tests/ttnn/integration_tests/roberta/test_performance.py" -m $test_marker
    else
        echo "There are no other model perf tests for Javelin yet specified. Arch $tt_arch requested"
    fi

    ## Merge all the generated reports
    env python models/perf/merge_perf_results.py
}

run_perf_models_llm_javelin() {
    local tt_arch=$1
    local test_marker=$2

    env pytest models/demos/falcon7b/tests -m $test_marker

    if [ "$tt_arch" == "wormhole_b0" ]; then
        env pytest models/demos/mistral7b/tests -m $test_marker
    fi

    ## Merge all the generated reports
    env python models/perf/merge_perf_results.py
}

run_perf_models_cnn_javelin() {
    local tt_arch=$1
    local test_marker=$2

    echo "There are no CNN tests for Javelin yet specified. Arch $tt_arch requested"

    ## Merge all the generated reports
    env python models/perf/merge_perf_results.py
}

run_device_perf_models() {
    local test_marker=$1

    #TODO(MO): Until #6560 is fixed, GS device profiler test are grouped with
    #Model Device perf regression tests to make sure thy run on no-soft-reset BMs
    tests/scripts/run_profiler_regressions.sh PROFILER

    env pytest "tests/ttnn/integration_tests/resnet/test_performance.py" -m $test_marker

    env pytest models/demos/resnet/tests -m $test_marker

    env pytest models/demos/metal_BERT_large_11/tests -m $test_marker

    env pytest models/demos/ttnn_falcon7b/tests -m $test_marker

    env pytest models/demos/bert/tests -m $test_marker

    env pytest models/demos/mistral7b/tests -m $test_marker

    ## Merge all the generated reports
    env python models/perf/merge_device_perf_results.py
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

    if [[ "$pipeline_type" == *"_virtual_machine" ]]; then
        test_marker="models_performance_virtual_machine"
    elif [[ "$pipeline_type" == *"device_performance_bare_metal" ]]; then
        test_marker="models_device_performance_bare_metal"
    elif [[ "$pipeline_type" == *"_bare_metal" ]]; then
        test_marker="models_performance_bare_metal"
    else
        echo "$pipeline_type is using an unrecognized platform (suffix, ex. bare_metal, virtual_machine)" 2>&1
        exit 1
    fi

    if [[ "$pipeline_type" == *"device_performance"* ]]; then
        run_device_perf_models "$test_marker"
    elif [[ "$pipeline_type" == "llm_javelin_models_performance"* ]]; then
        run_perf_models_llm_javelin "$tt_arch" "$test_marker"
    elif [[ "$pipeline_type" == "cnn_javelin_models_performance"* ]]; then
        run_perf_models_cnn_javelin "$tt_arch" "$test_marker"
    elif [[ "$pipeline_type" == "other_models_performance"* ]]; then
        run_perf_models_other "$tt_arch" "$test_marker"
    else
        echo "$pipeline_type is not recoognized performance pipeline" 2>&1
        exit 1
    fi
}

main "$@"
