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

TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2DFixture.*"
./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2DFixture.*"

#############################################
# FABRIC SANITY TESTS                       #
#############################################
echo "Running fabric sanity tests now...";

TEST_FOLDER="./build/test/tt_metal/perf_microbenchmark/routing"

# Async Write
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 8 --num_dest_endpoints 8 --num_links 16 --benchmark
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --metal_fabric_init_level 1
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 8 --num_dest_endpoints 8 --num_links 16 --benchmark --metal_fabric_init_level 1
# Async Write Mcast
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --e_depth 1
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --w_depth 1
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 1 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --e_depth 1 --metal_fabric_init_level 1
# TODO: Enable benchmark functionality for mcast
# Atomic Inc
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 64 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 64 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --metal_fabric_init_level 1
# Async Write Atomic Inc
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 65 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 65 --board_type n300 --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --metal_fabric_init_level 1

# Async Write with Push Router
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 1 --board_type n300 --data_kb_per_tx 600 --push_router
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 1 --board_type n300 --data_kb_per_tx 600 --benchmark --push_router
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 1 --board_type n300 --data_kb_per_tx 600 --metal_fabric_init_level 1 --push_router
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 1 --board_type n300 --data_kb_per_tx 600 --benchmark --metal_fabric_init_level 1 --push_router
# Atomic Inc with Push Router
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 64 --board_type n300 --data_kb_per_tx 600 --push_router
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 64 --board_type n300 --data_kb_per_tx 600 --metal_fabric_init_level 1 --push_router
# Async Write Atomic Inc with Push Router
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 65 --board_type n300 --data_kb_per_tx 600 --push_router
TT_METAL_SLOW_DISPATCH_MODE=1 ${TEST_FOLDER}/test_tt_fabric_sanity_${ARCH_NAME} --fabric_command 65 --board_type n300 --data_kb_per_tx 600 --metal_fabric_init_level 1 --push_router
