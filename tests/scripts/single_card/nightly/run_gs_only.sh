#/bin/bash

# set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi
fail=0

echo "Running model nightly tests for GS only"

env pytest -n auto models/demos/ttnn_resnet/tests/test_ttnn_resnet50_performant.py ; fail+=$?

if [[ $fail -ne 0 ]]; then
  exit 1
fi
