add_executable(test_wall_clock_tear ${CMAKE_CURRENT_SOURCE_DIR}/test_wall_clock_tear.cpp)

target_link_libraries(
    test_wall_clock_tear
    PRIVATE
        tt_metal
        pthread
)

target_include_directories(
    test_wall_clock_tear
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/impl
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${PROJECT_SOURCE_DIR}/tt_metal/llrt
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
)
