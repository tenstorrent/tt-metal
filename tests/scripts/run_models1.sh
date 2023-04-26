#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

export PYTHONPATH=$TT_METAL_HOME
export TT_KERNEL_READBACK_DISABLE=

#env pytest tests/python_api_testing/models/bert/bert_encoder.py -k bert_encoder
env pytest tests/python_api_testing/models/bert -k bert_question_and_answering -s
