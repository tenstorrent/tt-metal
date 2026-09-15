# Source files for tt_metal context tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_CONTEXT_SMOKE_SOURCES
    test_metal_env_api.cpp
    test_metal_context_api.cpp
    test_integration.cpp
)

if(TT_UMD_BUILD_GRENDEL_JTAG)
    list(APPEND UNIT_TESTS_CONTEXT_SMOKE_SOURCES test_mimir_emu.cpp)
endif()
