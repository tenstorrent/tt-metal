#!/bin/bash
source python_env/bin/activate
export TT_METAL_HOME=/home/local-syseng/tt-metal

# tt-run --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml --mpi-args "--host bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 --mca btl_tcp_if_include ens5f0np0 --tag-output" ./build/test/tt_metal/unit_tests_eth --gtest_filter=UnitMeshCQMultiDeviceProgramFixture.ActiveEthKernelsSendInterleavedBufferAllConnectedChips

tt-run --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml --mpi-args "--host bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 --mca btl_tcp_if_include ens5f0np0 --tag-output" ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="\
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
UnitMeshCQSingleCardFixture.TensixTestReadWriteMultipleCoresL1"
