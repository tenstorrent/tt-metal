#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

run_perf_models() {
    local pipeline_type=$1

    env pytest tests/python_api_testing/models/vgg/tests/perf_vgg.py

    env pytest tests/python_api_testing/models/vit/tests/perf_vit.py

    env pytest tests/python_api_testing/models/llama/perf_llama.py

    env pytest tests/python_api_testing/models/roberta/perf_roberta.py

    env pytest tests/python_api_testing/models/t5/tests/perf_t5.py

    env pytest tests/python_api_testing/models/resnet/tests/perf_resnet.py

    env pytest tests/python_api_testing/models/bloom/perf_bloom.py

    env pytest tests/python_api_testing/models/metal_BERT_large_15 -m $pipeline_type

    env pytest tests/python_api_testing/models/deit/tests/perf_deit.py

    env pytest tests/python_api_testing/models/stable_diffusion/tests/perf_unbatched_stable_diffusion.py

    env pytest tests/python_api_testing/models/whisper/perf_whisper.py

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
