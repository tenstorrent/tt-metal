add_library(metalium_firmware INTERFACE)
add_library(TT::Metalium::Firmware ALIAS metalium_firmware)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    target_sources(
        metalium_firmware
        PUBLIC
            FILE_SET jit_api
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
            FILES src/blackhole/noc.c src/active_erisc.cc
    )
endif()

install(
    TARGETS
        metalium_firmware
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/hw/firmware # FIXME: fix the include paths for jit_build
        COMPONENT metalium
)
