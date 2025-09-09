add_library(metalium_firmware INTERFACE)
add_library(TT::Metalium::Firmware ALIAS metalium_firmware)

# These source files are for the device, not host; will require cross compiling to lint them (future work).
set_target_properties(
    metalium_firmware
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)
target_sources(
    metalium_firmware
    INTERFACE
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            blackhole/noc.c
            wormhole/noc.c
            active_erisc.cc
            active_erisck.cc
            brisc.cc
            brisck.cc
            erisc.cc
            erisc-crt0.cc
            erisck.cc
            idle_erisc.cc
            idle_erisck.cc
            ncrisc.cc
            ncrisck.cc
            subordinate_idle_erisc.cc
            tdma_xmov.c
            trisc.cc
            trisck.cc
            tt_eth_api.cpp
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
