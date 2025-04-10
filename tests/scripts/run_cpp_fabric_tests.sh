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

export TT_METAL_CLEAR_L1=1

cd $TT_METAL_HOME

#############################################
# FABRIC UNIT TESTS                         #
#############################################
echo "Running fabric unit tests now...";

TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"

#############################################
# FABRIC SANITY TESTS                       #
#############################################
echo "Running fabric sanity tests now...";

TEST_FOLDER="./build/test/tt_metal/perf_microbenchmark/routing"

TESTS=(
    # Async Write
    "1 --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16"
    "2 --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 8 --num_dest_endpoints 8 --num_links 16 --benchmark"
    "3 --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --metal_fabric_init_level 1"
    "4 --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 8 --num_dest_endpoints 8 --num_links 16 --benchmark --metal_fabric_init_level 1"
     # Async Write Mcast
    "5 --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --e_depth 1"
    "6 --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --w_depth 1"
    "7 --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --e_depth 1 --metal_fabric_init_level 1"
     # TODO: Enable benchmark functionality for mcast
     # Atomic Inc
    "8 --fabric_command 64 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16"
    "9 --fabric_command 64 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --metal_fabric_init_level 1"
     # Async Write Atomic Inc
    "10 --fabric_command 65 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16"
    "11 --fabric_command 65 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --metal_fabric_init_level 1"

     # Async Write with Push Router
    "12 --fabric_command 1 --board_type n300 --data_kb_per_tx 600 --push_router"
    "13 --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 8 --num_dest_endpoints 8 --num_links 16 --benchmark --push_router"
    "14 --fabric_command 1 --board_type n300 --data_kb_per_tx 600 --metal_fabric_init_level 1 --push_router"
    "15 --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 8 --num_dest_endpoints 8 --num_links 16 --benchmark --metal_fabric_init_level 1 --push_router"
     # Async Write Mcast with Push Router
    "16 --fabric_command 1 --board_type n300 --data_kb_per_tx 600 --num_links 16 --e_depth 1 --push_router"
    "17 --fabric_command 1 --board_type n300 --data_kb_per_tx 600 --num_links 16 --w_depth 1 --push_router"
    "18 --fabric_command 1 --board_type n300 --data_kb_per_tx 600 --num_links 16 --e_depth 1 --push_router --metal_fabric_init_level 1"
     # Atomic Inc with Push Router
    "19 --fabric_command 64 --board_type n300 --data_kb_per_tx 600 --push_router"
    "20 --fabric_command 64 --board_type n300 --data_kb_per_tx 600 --metal_fabric_init_level 1 --push_router"
     # Async Write Atomic Inc with Push Router
    "21 --fabric_command 65 --board_type n300 --data_kb_per_tx 600 --push_router"
    "22 --fabric_command 65 --board_type n300 --data_kb_per_tx 600 --metal_fabric_init_level 1 --push_router"
)

for TEST in "${TESTS[@]}"; do
    # Extract test name and arguments
    read -r TEST_NUMBER TEST_ARGS <<< "$TEST"
    echo "LOG_FABRIC: Test $TEST_NUMBER: $TEST_ARGS"
    # Execute the test command with extracted arguments
    TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} $TEST_ARGS
done
