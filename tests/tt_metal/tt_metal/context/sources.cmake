# Source files for tt_metal context tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_CONTEXT_SMOKE_SOURCES
    test_metal_env_api.cpp
    test_metal_context_api.cpp
    test_integration.cpp
    test_mesh_device_properties.cpp
    test_simulator_termination.cpp
)

if(TT_UMD_BUILD_GRENDEL_JTAG)
    set(UNIT_TESTS_GRENDEL_EMU_SOURCES
        grendel_emu_gate.cpp
        test_mimir_emu.cpp
    )
endif()
