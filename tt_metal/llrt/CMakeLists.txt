# Provide unique architecture specific translation units and include paths for each ARCH

set(LLRT_SRC
    llrt.cpp
    core_descriptor.cpp
    rtoptions.cpp
    tlb_config.cpp
    tt_cluster.cpp
    tt_elffile.cpp
    tt_memory.cpp
    hal.cpp
    tunnels_from_mmio_device.cpp
)

add_library(llrt OBJECT ${LLRT_SRC})
add_library(Metalium::Metal::LLRT ALIAS llrt)
add_subdirectory(hal)
add_dependencies(llrt hal_generate_dev_msgs_header)

target_link_libraries(
    llrt
    PUBLIC
        umd::device
        TT::Metalium::HostDevCommon
    PRIVATE
        HAL::1xx
        HAL::2xx
        Tracy::TracyClient
        nlohmann_json::nlohmann_json
        Reflect::Reflect
        yaml-cpp::yaml-cpp
        enchantum::enchantum
        TT::Metalium::Common
        tt-logger::tt-logger
)
target_include_directories(llrt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(llrt PRIVATE -Wno-int-to-pointer-cast)

# Provide default install-time root for Debian-like installs without hardcoding
# Use GNUInstallDirs for a canonical libexec location; allow override via TT_METAL_INSTALL_ROOT
include(GNUInstallDirs)

set(TT_METAL_DEFAULT_INSTALL_ROOT "${CMAKE_INSTALL_FULL_LIBEXECDIR}/tt-metalium")
target_compile_definitions(llrt PRIVATE TT_METAL_INSTALL_ROOT="${TT_METAL_DEFAULT_INSTALL_ROOT}")
