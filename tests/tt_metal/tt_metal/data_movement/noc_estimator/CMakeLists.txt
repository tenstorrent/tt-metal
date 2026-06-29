# Offline tool to generate YAML latency data from CSVs
add_executable(
    generate_noc_latencies
    ${CMAKE_CURRENT_SOURCE_DIR}/offline/csv_reader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/offline/data_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/offline/data_extractor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/offline/yaml_serializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_latencies.cpp
)

target_include_directories(generate_noc_latencies PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    generate_noc_latencies
    PRIVATE
        yaml-cpp::yaml-cpp
        TT::Metalium
)

set_target_properties(
    generate_noc_latencies
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Test estimator
add_executable(test_noc_estimator ${CMAKE_CURRENT_SOURCE_DIR}/test_noc_estimator.cpp)

target_link_libraries(test_noc_estimator PRIVATE TT::Metalium)

set_target_properties(
    test_noc_estimator
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# CLI wrapper around the estimator: query the measured NoC model from the shell, emit JSON.
# Backs the perf-roofline-dm skill. Located via .claude/skills/perf-roofline-dm/noc_estimate.sh.
add_executable(noc_estimate ${CMAKE_CURRENT_SOURCE_DIR}/tools/noc_estimate.cpp)

target_link_libraries(
    noc_estimate
    PRIVATE
        TT::Metalium
        nlohmann_json::nlohmann_json
)

set_target_properties(
    noc_estimate
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
