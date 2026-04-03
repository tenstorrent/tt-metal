#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"

./build_metal.sh --build-tests

if [[ $# -ge 1 && "$1" != -* ]]; then
    match="$1"
    shift
    TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_legacy --gtest_filter="*${match}" "$@"
else
    TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_legacy "$@"
fi
