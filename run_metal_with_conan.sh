#!/usr/bin/env bash

set -o errexit
set -x

conan build . --build=missing -cc core.net.http:timeout=240 -o 'tt-nn/*:build_tests=True' --profile "$1"
export OMPI_MCA_plm_rsh_agent="" # Don't try/look-for remote shell tools for local execution.
. .conan-build/Release/generators/conanrun.sh && ./.conan-build/Release/test/tt_eager/ops/test_eltwise_unary_op || true
conan create . --build=missing -cc core.net.http:timeout=240 --profile "$1"
