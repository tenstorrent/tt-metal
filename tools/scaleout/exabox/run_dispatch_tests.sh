#!/bin/bash
mkdir -p dispatch_test_logs
HOSTS="$1"
DOCKER_IMAGE="$2"
LOG_FILE="dispatch_test_logs/dispatch_tests_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Running dispatch tests..."
echo "Using hosts: $HOSTS"
echo "Logging to: $LOG_FILE"
echo "Using docker image: $DOCKER_IMAGE"
echo "=========================================="
echo ""

./tools/scaleout/exabox/mpi-docker --image $DOCKER_IMAGE --empty-entrypoint --host $HOSTS -x TT_MESH_ID=0 -x TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="\
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
