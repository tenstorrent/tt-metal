# Fabric capture: reads a fabric run's router cores, where its manifest says they are, into a capture.
#
# fabric_capture_core is everything that needs no particular device: reading the manifest, matching it to the chips a
# reader found, reading router cores through a DeviceReader, and hashing. It links no tt_metal, so it builds without
# the runtime, and nothing in it can reach a MetalContext.
add_library(fabric_capture_core STATIC)
target_sources(
    fabric_capture_core
    PRIVATE
        fabric_capture_files.cpp
        fabric_capture_json.cpp
        fabric_capture_manifest.cpp
        fabric_capture_reads.cpp
        fabric_capture_record.cpp
        fabric_capture_targets.cpp
        fabric_capture_utils.cpp
)
target_include_directories(fabric_capture_core PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(
    fabric_capture_core
    PUBLIC
        nlohmann_json::nlohmann_json
        TT::STL
    PRIVATE
        BLAKE3::blake3
)
