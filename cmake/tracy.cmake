# Built as outlined in Tracy documentation (pg.12)
set(TRACY_HOME ${PROJECT_SOURCE_DIR}/tt_metal/third_party/tracy)

add_subdirectory(${TRACY_HOME})
set_target_properties(
    TracyClient
    PROPERTIES
        EXCLUDE_FROM_ALL
            TRUE
)

set_target_properties(
    TracyClient
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        POSITION_INDEPENDENT_CODE
            ON # this is equivalent to adding -fPIC
        OUTPUT_NAME
            "tracy"
)

# Our current fork of tracy does not have CMake support for these subdirectories
# Once we update, we can change this
include(ExternalProject)
ExternalProject_Add(
    tracy_csv_tools
    PREFIX ${TRACY_HOME}/csvexport/build/unix
    SOURCE_DIR ${TRACY_HOME}/csvexport/build/unix
    BINARY_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    INSTALL_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    STAMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_stamp"
    TMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_tmp"
    DOWNLOAD_COMMAND
        ""
    CONFIGURE_COMMAND
        ""
    INSTALL_COMMAND
        cp ${TRACY_HOME}/csvexport/build/unix/csvexport-release .
    BUILD_COMMAND
        cd ${TRACY_HOME}/csvexport/build/unix && CXX=g++ TRACY_NO_LTO=1 make -f
        ${TRACY_HOME}/csvexport/build/unix/Makefile
)
ExternalProject_Add(
    tracy_capture_tools
    PREFIX ${TRACY_HOME}/capture/build/unix
    SOURCE_DIR ${TRACY_HOME}/capture/build/unix
    BINARY_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    INSTALL_DIR ${PROJECT_BINARY_DIR}/tools/profiler/bin
    STAMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_stamp"
    TMP_DIR "${PROJECT_BINARY_DIR}/tmp/tracy_tmp"
    DOWNLOAD_COMMAND
        ""
    CONFIGURE_COMMAND
        ""
    INSTALL_COMMAND
        cp ${TRACY_HOME}/capture/build/unix/capture-release .
    BUILD_COMMAND
        cd ${TRACY_HOME}/capture/build/unix && CXX=g++ TRACY_NO_LTO=1 make -f ${TRACY_HOME}/capture/build/unix/Makefile
)
add_custom_target(
    tracy_tools
    ALL
    DEPENDS
        tracy_csv_tools
        tracy_capture_tools
)
