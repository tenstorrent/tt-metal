add_library(fabric)

target_sources(
    fabric
    PRIVATE
        control_plane.cpp
        routing_table_generator.cpp
        mesh_graph.cpp
)

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

#set_target_properties(
#    fabric
#    PROPERTIES
#        INSTALL_RPATH
#            "${PROJECT_BINARY_DIR}/lib"
#)
