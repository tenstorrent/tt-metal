add_library(tt_fabric)
add_library(TT::Fabric ALIAS tt_fabric)

target_sources(
    tt_fabric
    PRIVATE
        control_plane.cpp
        routing_table_generator.cpp
        mesh_graph.cpp
)

target_link_libraries(
    tt_fabric
    PRIVATE
        Metalium::Metal
        umd::device
        metal_common_libs
        magic_enum
        fmt::fmt-header-only
        yaml-cpp::yaml-cpp
)

target_precompile_headers(
    tt_fabric
    PRIVATE
        <functional>
        <map>
        <memory>
        <unordered_map>
        <variant>
        <vector>
)

target_compile_options(tt_fabric PRIVATE -Wno-int-to-pointer-cast)

set_target_properties(
    tt_fabric
    PROPERTIES
        INSTALL_RPATH
            "${PROJECT_BINARY_DIR}/lib"
)
