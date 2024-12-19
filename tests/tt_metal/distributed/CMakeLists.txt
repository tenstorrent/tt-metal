set(UNIT_TESTS_DISTRIBUTED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed.cpp)

add_executable(distributed_unit_tests ${UNIT_TESTS_DISTRIBUTED_SRC})
target_link_libraries(
    distributed_unit_tests
    PRIVATE
        tt_metal
        test_common_libs
)

target_include_directories(distributed_unit_tests PRIVATE "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>")

set_target_properties(
    distributed_unit_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/distributed
)

# Dont do this for now
# When the test is probed something is constructed that tries to access a device
# Build machine might not have a device
# We don't use ctest in this project so we shouldn't need this yet
#gtest_discover_tests(distributed_unit_tests)
