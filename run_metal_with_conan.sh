#!/usr/bin/env bash

set -o nounset
set -o errexit
set -x

conan build . --build=missing \
    -cc core.net.http:timeout=240 \
    -o 'ttnn/*:build_tests=True' \
    --profile "$1"
