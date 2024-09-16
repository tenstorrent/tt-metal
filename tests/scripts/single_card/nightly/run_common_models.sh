#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi
fail=0

echo "Running common models for archs"

env pytest -n auto tests/nightly/single_card/common_models/ ; fail+=$?

if [[ $fail -ne 0 ]]; then
  exit 1
fi
