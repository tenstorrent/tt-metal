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
)

target_link_libraries(
    common
    PUBLIC
        # Public fabric API headers transitively include topology_solver (CaDiCaL); OBJECT libs that link common must
        # see cadical.hpp and link libcadical in the final shared library.
        TT::ThirdParty::cadical
        nlohmann_json::nlohmann_json
        enchantum::enchantum
        fmt::fmt-header-only
        TT::STL
        umd::Firmware
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
