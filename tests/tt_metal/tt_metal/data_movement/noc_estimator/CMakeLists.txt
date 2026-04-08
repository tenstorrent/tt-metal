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

target_link_libraries(generate_noc_latencies PRIVATE yaml-cpp)

set_target_properties(
    generate_noc_latencies
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Test estimator
add_executable(
    test_noc_estimator
    ${CMAKE_CURRENT_SOURCE_DIR}/common/interpolation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loader/yaml_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/noc_estimator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_noc_estimator.cpp
)

target_include_directories(test_noc_estimator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(test_noc_estimator PRIVATE yaml-cpp)

set_target_properties(
    test_noc_estimator
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
