add_library(metalium_firmware INTERFACE)
add_library(TT::Metalium::Firmware ALIAS metalium_firmware)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    # These headers are for the device, not host; will require cross compiling to verify.
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
                src/blackhole/noc.c
                src/wormhole/noc.c
                src/active_erisc.cc
                src/active_erisck.cc
                src/brisc.cc
                src/brisck.cc
                src/erisc.cc
                src/erisc-crt0.cc
                src/erisck.cc
                src/idle_erisc.cc
                src/idle_erisck.cc
                src/ncrisc.cc
                src/ncrisck.cc
                src/subordinate_idle_erisc.cc
                src/tdma_xmov.c
                src/trisc.cc
                src/trisck.cc
                src/tt_eth_api.cpp
    )

    install(
        TARGETS
            metalium_firmware
        FILE_SET
        jit_api
            DESTINATION
                ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/hw/firmware # FIXME: fix the include paths for jit_build
            COMPONENT metalium-runtime
    )
endif()
