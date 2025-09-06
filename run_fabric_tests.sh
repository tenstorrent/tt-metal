#!/bin/bash

export TT_METAL_HOME=/home/ttuser/tt-metal
export TT_MESH_ID=0
./build_Release/test/tt_metal/multi_host_fabric_tests --gtest_filter="*RandomizedIntermeshUnicastBwd*" |& tee log.txt
#./build_Release/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*UnicastRaw"
