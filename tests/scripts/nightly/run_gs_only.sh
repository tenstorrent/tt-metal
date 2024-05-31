#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

echo "Running model nightly tests for GS only"

env pytest models/demos/metal_BERT_large_11/tests/test_demo.py

env pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_inference[LoFi-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_20-device_params0]

env pytest models/demos/resnet/tests/test_metal_resnet50.py::test_run_resnet50_trace_inference -k "LoFi-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_20-device_params0"
