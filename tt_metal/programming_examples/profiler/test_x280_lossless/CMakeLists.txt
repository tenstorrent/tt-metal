# X280 per-lane lossless step-0 harness. Needs internal Cluster + x280_driver headers, so it uses a
# custom CMakeLists (not the public-API-only CREATE_PGM_EXAMPLES_EXE macro), same as the other test_x280_*.
add_executable(test_x280_lossless ${CMAKE_CURRENT_SOURCE_DIR}/test_x280_lossless.cpp)

target_link_libraries(
    test_x280_lossless
    PRIVATE
        tt_metal
        pthread
        $<$<TARGET_EXISTS:TracyClient>:TracyClient>
)

target_include_directories(
    test_x280_lossless
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_BINARY_DIR}/tt_metal/api
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/x280_bm/include # prof_packet.h (shared 8 B format)
)
