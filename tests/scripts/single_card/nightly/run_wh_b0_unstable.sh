#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi
fail=0

echo "Running unstable nightly tests for WH B0 only"

env TT_MM_THROTTLE_PERF=5 pytest -n auto tests/nightly/single_card/wh_b0_unstable/ --timeout=600; fail+=$?

if [[ $fail -ne 0 ]]; then
    exit 1
fi
