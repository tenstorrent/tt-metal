#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

run_perf_models() {
    local pipeline_type=$1




    env pytest tests/models/vgg/tests -m $pipeline_type

    env pytest tests/models/vit/tests -m $pipeline_type

    env pytest tests/models/llama -m $pipeline_type

    env pytest tests/models/roberta -m $pipeline_type

    env pytest tests/models/t5/tests -m $pipeline_type

    # # Bad tests, don't enable: Hanging post commit 8/24/23 debug war room session, see PR#2297, PR#2301
    # #env pytest tests/models/resnet/tests -m $pipeline_type

    env pytest tests/models/metal_BERT_large_15 -m $pipeline_type

    env pytest tests/models/deit/tests -m $pipeline_type

    env pytest tests/models/stable_diffusion/tests -m $pipeline_type

    env pytest tests/models/whisper -m $pipeline_type

    env pytest tests/models/bloom -m $pipeline_type

    env pytest tests/models/falcon/tests -m $pipeline_type
    ## Merge all the generated reports
    env python tests/models/merge_perf_results.py
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
