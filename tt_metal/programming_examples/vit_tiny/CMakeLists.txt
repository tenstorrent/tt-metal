add_executable(metal_example_vit_tiny)
target_sources(metal_example_vit_tiny PRIVATE vit_tiny.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_include_directories(metal_example_vit_tiny PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    metal_example_vit_tiny
    PRIVATE
        TT::Metalium
)

list(APPEND PROGRAMMING_EXAMPLES_TEST_TARGETS metal_example_vit_tiny)
set(PROGRAMMING_EXAMPLES_TEST_TARGETS "${PROGRAMMING_EXAMPLES_TEST_TARGETS}" PARENT_SCOPE)
