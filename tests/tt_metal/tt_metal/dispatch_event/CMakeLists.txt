set(UNIT_TESTS_DISPATCH_EVENT_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_EnqueueWaitForEvent.cpp)

add_executable(unit_tests_dispatch_event ${UNIT_TESTS_DISPATCH_EVENT_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_dispatch_event)

target_link_libraries(unit_tests_dispatch_event PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_dispatch_event
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_dispatch_event
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
