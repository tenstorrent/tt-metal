add_library(common OBJECT)
add_library(TT::Metalium::Common ALIAS common)

target_sources(
    common
    PRIVATE
        core_assignment.cpp
        core_coord.cpp
        mesh_coord.cpp
        metal_soc_descriptor.cpp
        shape2d.cpp
        shape_base.cpp
        tt_backend_api_types.cpp
        utils.cpp
        work_split.cpp
        thread_pool.cpp
)

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
        nlohmann_json::nlohmann_json
        magic_enum::magic_enum
        fmt::fmt-header-only
        span
        small_vector
        Metalium::Metal::STL
        umd::Firmware
        umd::device
    PRIVATE
        Tracy::TracyClient
        TT::Metalium::HostDevCommon
        yaml-cpp::yaml-cpp
        Boost::asio
)
