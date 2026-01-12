#!/bin/bash
mkdir -p dispatch_test_logs
source python_env/bin/activate
HOSTS="$1"
LOG_FILE="dispatch_test_logs/dispatch_tests_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Running dispatch tests..."
echo "Using hosts: $HOSTS"
echo "Logging to: $LOG_FILE"
echo "=========================================="
echo ""

TT_METAL_HOME=$PWD tt-run --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml --mpi-args "--host $HOSTS --mca btl_tcp_if_include ens5f0np0 --tag-output" ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="\
UnitMeshCQProgramFixture.TensixTestRandomizedProgram:\
UnitMeshRandomProgramFixture.TensixTestLargeProgramInBetweenFiveSmallPrograms:\
UnitMeshRandomProgramTraceFixture.TensixTestLargeProgramInBetweenFiveSmallProgramsTrace:\
UnitMeshRandomProgramTraceFixture.TensixTestSimpleProgramsTrace:\
UnitMeshCQTraceFixture.TensixEnqueueMultiProgramTraceBenchmark:\
UnitMeshCQTraceFixture.TensixEnqueueTwoProgramTrace:\
UnitMeshCQSingleCardBufferFixture.ShardedBufferLargeL1ReadWrites:\
UnitMeshCQSingleCardBufferFixture.ShardedBufferLargeDRAMReadWrites:\
UnitMeshCQSingleCardFixture.TensixTestSubDeviceAllocations:\
UnitMeshMultiCQMultiDeviceEventFixture.*:\
UnitMeshCQSingleCardFixture.TensixTestReadWriteMultipleCoresL1" |& tee "$LOG_FILE"
