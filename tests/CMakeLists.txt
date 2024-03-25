
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/tt_metal)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_eager)

add_custom_target(tests DEPENDS metal_tests eager_tests)
