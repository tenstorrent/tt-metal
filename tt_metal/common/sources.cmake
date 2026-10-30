set(COMMON_SOURCES
    core_assignment.cpp
    core_coord.cpp
    filesystem_utils.cpp
    shape.cpp
    shape2d.cpp
    stable_hash.cpp
    tt_backend_api_types.cpp
    work_split.cpp
    host_buffer.cpp
    memory_pin.cpp
)

# Runtime-free value types shared with TT::FabricTopologyUtils: compiled into TT::Metalium::Types
# (declared in tt_metal/CMakeLists.txt), which libtt_metal links.
set(COMMON_TYPES_SOURCES
    mesh_coord.cpp
    shape_base.cpp
)
