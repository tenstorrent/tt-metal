include(sources.cmake)

add_library(common OBJECT)
add_library(TT::Metalium::Common ALIAS common)

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(common)
target_precompile_headers(common REUSE_FROM metal_common_pch)

target_sources(common PRIVATE ${COMMON_SOURCES})

target_include_directories(
    common
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}
        # Metalium's internal headers spell the public API unscoped, e.g.
        # impl/kernels/kernel.hpp includes "hal_types.hpp" rather than
        # <tt-metalium/hal_types.hpp>, so every internal consumer needs this root.
        # FIXME: Re-home those includes under tt-metalium/ and remove this.
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
)

target_link_libraries(
    common
    PUBLIC
        nlohmann_json::nlohmann_json
        enchantum::enchantum
        fmt::fmt-header-only
        TT::STL
        umd::tt-umd
        simde::simde
        Taskflow::Taskflow
    PRIVATE
        BLAKE3::blake3
        Metalium::Metal::Hardware
        Tracy::TracyClient
        TT::Metalium::HostDevCommon
        yaml-cpp::yaml-cpp
        Boost::asio
        tt-logger::tt-logger
)
