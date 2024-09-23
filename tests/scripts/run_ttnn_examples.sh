#!/bin/bash

set -eo pipefail

for example in ttnn/ttnn/examples/usage/*.py; do
  echo "[Info] Running ttnn example: $example"
  python3 $example
done
