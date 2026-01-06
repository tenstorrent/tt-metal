# 4x BH Quietbox System Health Test

add_executable(test_4x_bh_sanity)

target_sources(test_4x_bh_sanity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_sanity.cpp)

target_link_libraries(
    test_4x_bh_sanity
    PRIVATE
        tt_metal
        test_common_libs
        TT::ScaleoutTools
)

target_include_directories(
    test_4x_bh_sanity
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
)

set_target_properties(
    test_4x_bh_sanity
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/scale_out/4x_bh_quietbox
)

target_link_libraries(test_4x_bh_sanity PRIVATE TT::ScaleoutTools)
