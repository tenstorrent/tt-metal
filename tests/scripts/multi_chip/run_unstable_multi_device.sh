#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_mlp.py
pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_attention.py
pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_decoder.py
pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_causallm.py
