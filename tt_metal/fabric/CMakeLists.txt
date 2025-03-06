set(FABRIC_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/control_plane.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/routing_table_generator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mesh_graph.cpp
)

add_library(fabric OBJECT ${FABRIC_SRC})

target_include_directories(fabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    fabric
    PRIVATE
        Metalium::Metal::LLRT
        umd::device
        metal_common_libs
        magic_enum::magic_enum
        fmt::fmt-header-only
        yaml-cpp::yaml-cpp
        Metalium::Metal::Impl
        TT::Metalium::HostDevCommon
)

target_precompile_headers(
    fabric
    PRIVATE
        <functional>
        <map>
        <memory>
        <unordered_map>
        <variant>
        <vector>
)

target_compile_options(fabric PRIVATE -Wno-int-to-pointer-cast)
