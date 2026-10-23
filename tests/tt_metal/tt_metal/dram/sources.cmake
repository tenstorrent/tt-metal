# Source files for tt_metal Mimir DRAM and TL1 tests
# Module owners should update this file when adding/removing/renaming source files

set(MIMIR_TESTS_SRC
    mimir_hw_state.cpp
    mimir_fw_loader.cpp
    mimir_snoop_ctl.cpp
    test_mimir_host_dram.cpp
    test_mimir_host_tl1.cpp
    test_mimir_fw_load.cpp
    test_mimir_kernel_load.cpp
    test_mimir_local_dram.cpp
    test_mimir_tl1.cpp
    test_mimir_remote_dram.cpp
    test_mimir_prefetcher.cpp
    test_mimir_dm_idma.cpp
    test_mimir_dm_cache.cpp
    test_mimir_dm_addrgen.cpp
    test_mimir_dm_mesh_buffer.cpp
    test_mimir_dm_dram_unary.cpp
    test_mimir_dm_scoped_lock.cpp
    test_mimir_atomics.cpp
    test_mimir_trid.cpp
)
