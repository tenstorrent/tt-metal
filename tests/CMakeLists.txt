enable_testing()
include(GoogleTest)
add_library(test_common_libs INTERFACE)
target_include_directories(test_common_libs INTERFACE "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>")
target_link_libraries(
    test_common_libs
    INTERFACE
        pthread
        gmock_main
        nlohmann_json::nlohmann_json
        magic_enum
        fmt::fmt-header-only
        span
)

if(TT_METAL_BUILD_TESTS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal)
endif(TT_METAL_BUILD_TESTS)

if(TTNN_BUILD_TESTS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_eager) # this should go away and be replaced with link to ttnn
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ttnn)
endif(TTNN_BUILD_TESTS)
