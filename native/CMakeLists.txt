set(NATIVE_TESTS_SRC
${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(native ${NATIVE_TESTS_SRC})
target_link_libraries(native PUBLIC ttnn tt_metal)
target_include_directories(native PRIVATE
${UMD_HOME}
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/tt_metal
${PROJECT_SOURCE_DIR}/tests
${CMAKE_CURRENT_SOURCE_DIR})
