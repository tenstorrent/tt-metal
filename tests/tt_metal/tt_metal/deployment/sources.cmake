# Source files for tt_metal deployment tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_DEPLOYMENT_SRC
    deployment_common.cpp
    dram/dram_base.cpp
    dram/test_dram.cpp
    eth/test_eth_link_up.cpp
    eth/test_eth_bandwidth.cpp
    eth/test_eth_bandwidth_bidir.cpp
    eth/test_eth_data_integrity_dram_bidir.cpp
    eth/test_eth_data_integrity_dram.cpp
    eth/test_eth_stress_test.cpp
)
