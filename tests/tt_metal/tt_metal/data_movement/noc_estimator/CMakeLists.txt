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

# yaml-cpp is a static library that is already inside libtt_metal.so / libtt-umd.so and exported from them. Linking it
# again would embed a second copy in this executable, which ASan reports as an ODR violation, so take only its headers.
target_include_directories(
    generate_noc_latencies
    SYSTEM
    PRIVATE
        $<TARGET_PROPERTY:yaml-cpp::yaml-cpp,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(generate_noc_latencies PRIVATE TT::Metalium)

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
