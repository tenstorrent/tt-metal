#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

export PYTHONPATH=$TT_METAL_HOME
export ARCH_NAME=grayskull

env pytest tests/python_api_testing/models/stable_diffusion/perf_unbatched_stable.py

env pytest tests/python_api_testing/models/vit/tests/perf_vit.py

env pytest tests/python_api_testing/models/vgg/tests/perf_vgg.py

env pytest tests/python_api_testing/models/llama/perf_llama.py

env pytest tests/python_api_testing/models/roberta/perf_roberta.py

env pytest tests/python_api_testing/models/whisper/perf_whisper.py

env pytest tests/python_api_testing/models/t5/perf_t5.py

env pytest tests/python_api_testing/models/resnet/tests/perf_resnet.py

env pytest tests/python_api_testing/models/bloom/perf_bloom.py

env pytest tests/python_api_testing/models/metal_BERT_large_15/perf_bert15.py

env pytest tests/python_api_testing/models/deit/tests/perf_deit.py

## Merge all the generated reports
env python tests/python_api_testing/models/merge_perf_results.py
