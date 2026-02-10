# Source files for tt_metal data_movement tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_DATA_MOVEMENT_SRC
    dm_common.cpp
    dram_unary/test_unary_dram.cpp
    one_to_all/test_one_to_all.cpp
    one_to_all/test_multicast_schemes.cpp
    loopback/test_loopback.cpp
    reshard_hardcoded/test_reshard_hardcoded.cpp
    deinterleave_hardcoded/test_deinterleave_hardcoded.cpp
    conv_hardcoded/test_conv_hardcoded.cpp
    one_to_one/test_one_to_one.cpp
    one_from_one/test_one_from_one.cpp
    core_bidirectional/test_core_bidirectional.cpp
    one_from_all/test_one_from_all.cpp
    all_to_all/test_all_to_all.cpp
    all_from_all/test_all_from_all.cpp
    interleaved/test_interleaved.cpp
    one_packet/test_one_packet.cpp
    interleaved_to_sharded_hardcoded/test_interleaved_to_sharded_hardcoded.cpp
    multi_interleaved/test_multi_interleaved.cpp
    dram_sharded/test_dram_sharded.cpp
    transaction_id/test_transaction_id.cpp
    direct_write/test_direct_write.cpp
    pcie_read_bw/test_pcie_read_bw.cpp
    atomics/test_atomic_semaphore_bandwidth.cpp
    tests/tt_metal/tt_metal/data_movement/CMakeLists.txt
    noc_api_latency/test_noc_api_latency.cpp
)
