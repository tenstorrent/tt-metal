#!/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
fi

cd $TT_METAL_HOME

#############################################
# Multi-Process Socket Tests                #
#############################################

echo "Running multi-process socket tests..."

mpirun -n 2 --allow-run-as-root ./build/test/tt_metal/multi_host_socket_tests
