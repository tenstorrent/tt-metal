# Source files for tt_metal Mimir DRAM and TL1 tests
# Module owners should update this file when adding/removing/renaming source files

set(MIMIR_TESTS_SRC
    mimir_hw_state.cpp
    mimir_fw_loader.cpp
    test_mimir_host_dram.cpp
    test_mimir_host_tl1.cpp
    test_mimir_fw_load.cpp
    test_mimir_kernel_load.cpp
    test_mimir_local_dram.cpp
    test_mimir_tl1.cpp
    test_mimir_remote_dram.cpp
)
