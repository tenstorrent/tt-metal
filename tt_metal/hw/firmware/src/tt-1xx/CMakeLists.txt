add_library(metalium_firmware INTERFACE)
add_library(TT::Metalium::Firmware ALIAS metalium_firmware)

# These source files are for the device, not host; will require cross compiling to lint them (future work).
set_target_properties(
    metalium_firmware
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

include(sources.cmake)

target_sources(
    metalium_firmware
    INTERFACE
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${FIRMWARE_JIT_API_FILES}
)

install(
    TARGETS
        metalium_firmware
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/hw/firmware/src/tt-1xx # FIXME: fix the include paths for jit_build
        COMPONENT metalium-runtime
)
