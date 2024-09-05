#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

for example in ttnn/examples/usage/*.py; do
  echo "[Info] Running ttnn example: $example"
  python3 $example
done
