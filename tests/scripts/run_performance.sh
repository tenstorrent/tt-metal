#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

run_perf_models() {
    local pipeline_type=$1

    env pytest models/demos/falcon7b/tests -m $pipeline_type

    env pytest models/experimental/vgg/tests -m $pipeline_type

    env pytest models/experimental/vit/tests -m $pipeline_type

    env pytest models/experimental/llama_old/tests -m $pipeline_type

    env pytest models/experimental/roberta/tests -m $pipeline_type

    env pytest models/experimental/t5/tests -m $pipeline_type

    env pytest models/demos/resnet/tests -m $pipeline_type

    env pytest models/demos/metal_BERT_large_11/tests -m $pipeline_type

    env pytest models/experimental/deit/tests -m $pipeline_type

    env pytest models/experimental/stable_diffusion/tests -m $pipeline_type

    env pytest models/experimental/whisper/tests -m $pipeline_type

    env pytest models/experimental/bloom/tests -m $pipeline_type

    ## Merge all the generated reports
    env python models/merge_perf_results.py
}

run_device_perf_models() {
    local pipeline_type=$1

    env pytest models/demos/resnet/tests -m $pipeline_type

    env pytest models/demos/metal_BERT_large_11/tests -m $pipeline_type
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

    if [[ "$pipeline_type" == *"device_performance"* ]]; then
        run_device_perf_models "$pipeline_type"
    else
        run_perf_models "$pipeline_type"
    fi
}

main "$@"
