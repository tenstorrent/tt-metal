# Built as outlined in Tracy documentation (pg.12)
set(TRACY_HOME ${PROJECT_SOURCE_DIR}/tt_metal/third_party/tracy)

add_subdirectory(${TRACY_HOME})

set_target_properties(TracyClient PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    POSITION_INDEPENDENT_CODE ON    # this is equivalent to adding -fPIC
    ADDITIONAL_CLEAN_FILES "${PROJECT_BINARY_DIR}/tools"
    OUTPUT_NAME "tracy"
)

include(ExternalProject)
ExternalProject_Add(
    tracy-capture
    SOURCE_DIR ${TRACY_HOME}/capture
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DNO_FILESELECTOR=ON -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17
    BINARY_DIR ${PROJECT_BINARY_DIR}/tools/profiler/capture
    STAMP_DIR ${PROJECT_BINARY_DIR}/tmp/tracy_stamp
    TMP_DIR ${PROJECT_BINARY_DIR}/tmp/tracy_tmp
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy tracy-capture ${PROJECT_BINARY_DIR}/tools/profiler/bin/capture-release
)
ExternalProject_Add(
    tracy-csvexport
    SOURCE_DIR ${TRACY_HOME}/csvexport
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Debug -DNO_FILESELECTOR=ON -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17
    BINARY_DIR ${PROJECT_BINARY_DIR}/tools/profiler/csvexport
    STAMP_DIR ${PROJECT_BINARY_DIR}/tmp/tracy_stamp
    TMP_DIR ${PROJECT_BINARY_DIR}/tmp/tracy_tmp
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy tracy-csvexport ${PROJECT_BINARY_DIR}/tools/profiler/bin/csvexport-release
)
add_custom_target(tracy_tools ALL DEPENDS tracy-capture tracy-csvexport)
