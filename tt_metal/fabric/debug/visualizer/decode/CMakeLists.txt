# Fabric decode: reads one rank's manifest and capture, and turns the bytes capture read into typed values.
#
# fabric_decode_core is everything decode does short of the tool. It reads its inputs through capture's readers
# (read_manifest_file, load_capture), so it links fabric_capture_core, and through it tt_metal for the manifest's model,
# nlohmann_json and TT::STL. It opens no chip and starts none of the runtime.
add_library(fabric_decode_core STATIC)
target_sources(
    fabric_decode_core
    PRIVATE
        fabric_decode_inputs.cpp
        fabric_decode_snapshot.cpp
)
target_include_directories(fabric_decode_core PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(fabric_decode_core PUBLIC fabric_capture_core)
