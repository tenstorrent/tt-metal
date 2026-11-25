add_library(metalium_firmware_2xx INTERFACE)
add_library(TT::Metalium::Firmware2xx ALIAS metalium_firmware_2xx)

# These source files are for the device, not host; will require cross compiling to lint them (future work).
set_target_properties(
    metalium_firmware_2xx
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

include(sources.cmake)

target_sources(
    metalium_firmware_2xx
    INTERFACE
        FILE_SET jit_api
        TYPE HEADERS
        FILES ${FIRMWARE_JIT_API_FILES}
)

install(
    TARGETS
        metalium_firmware_2xx
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/hw/firmware/src/tt-2xx # FIXME: fix the include paths for jit_build
        COMPONENT metalium-runtime
)
