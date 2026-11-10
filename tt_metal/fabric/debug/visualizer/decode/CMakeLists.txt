# Fabric decode: reads one rank's manifest and capture, and turns the bytes capture read into typed values.
#
# fabric_decode_core is everything decode does short of the tool. It reads its inputs through capture's readers
# (read_manifest_file, load_capture), so it links fabric_capture_core, and through it tt_metal for the manifest's model,
# nlohmann_json and TT::STL. It opens no chip and starts none of the runtime.
add_library(fabric_decode_core STATIC)
target_sources(
    fabric_decode_core
    PRIVATE
        fabric_decode_activity.cpp
        fabric_decode_findings.cpp
        fabric_decode_inputs.cpp
        fabric_decode_interpret.cpp
        fabric_decode_rank.cpp
        fabric_decode_region_json.cpp
        fabric_decode_router_analysis.cpp
        fabric_decode_routing.cpp
        fabric_decode_routing_table.cpp
        fabric_decode_snapshot.cpp
        fabric_decode_values.cpp
)
target_include_directories(fabric_decode_core PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(fabric_decode_core PUBLIC fabric_capture_core)

# fabric_decode: the tool, one process per rank's capture. It sits beside fabric_capture in build/tools. Only the tool
# links the build identity, which it hands the core for the decoded file's provenance.
add_executable(fabric_decode fabric_decode_main.cpp)
target_link_libraries(
    fabric_decode
    PRIVATE
        fabric_decode_core
        fabric_visualizer_build_identity
        cxxopts::cxxopts
)
set_target_properties(
    fabric_decode
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)
