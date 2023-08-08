#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

run_perf_models() {
    local pipeline_type=$1

    if [[ ! -z "$FAST_DISPATCH" ]]; then
        echo "Running performance models in fast dispatch mode"
    else
        export TT_METAL_SLOW_DISPATCH_MODE=1
    fi

    env pytest tests/python_api_testing/models/vgg/tests -m $pipeline_type

    env pytest tests/python_api_testing/models/vit/tests -m $pipeline_type

    env pytest tests/python_api_testing/models/llama -m $pipeline_type

    env pytest tests/python_api_testing/models/roberta -m $pipeline_type

    env pytest tests/python_api_testing/models/t5/tests -m $pipeline_type

    env pytest tests/python_api_testing/models/resnet/tests -m $pipeline_type

    env pytest tests/python_api_testing/models/bloom -m $pipeline_type

    env pytest tests/python_api_testing/models/metal_BERT_large_15 -m $pipeline_type

    env pytest tests/python_api_testing/models/deit/tests -m $pipeline_type

    env pytest tests/python_api_testing/models/stable_diffusion/tests -m $pipeline_type

    env pytest tests/python_api_testing/models/whisper -m $pipeline_type

    ## Merge all the generated reports
    env python tests/python_api_testing/models/merge_perf_results.py
}

main() {
    # Parse the arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
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

    if [[ -z "$pipeline_type" ]]; then
      echo "--pipeline-type cannot be empty" 1>&2
      exit 1
    fi

    run_perf_models "$pipeline_type"
}

main "$@"
