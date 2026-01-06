# CMake helper for setting relocatable RPATH on test executables
#
# This function sets both BUILD_RPATH (for build-time execution) and
# INSTALL_RPATH (for tar artifact/installed binaries) automatically.
#
# Usage:
#   tt_set_runtime_rpath(target_name)              # Links to build/lib/ only
#   tt_set_runtime_rpath(target_name TTNN)         # Links to build/lib/ and ttnn/ttnn/
#
# Requirements:
#   - Target must have RUNTIME_OUTPUT_DIRECTORY set before calling this function
#
# Why both BUILD_RPATH and INSTALL_RPATH?
#   - BUILD_RPATH: Absolute paths for build-time execution (e.g., gtest_discover_tests)
#   - INSTALL_RPATH: Relative $ORIGIN paths for tar artifacts and installed binaries
#
# Example:
#   add_executable(my_test test.cpp)
#   set_target_properties(my_test PROPERTIES
#       RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/my_tests
#   )
#   tt_set_runtime_rpath(my_test)  # Automatically calculates paths
#
function(tt_set_runtime_rpath TARGET)
    cmake_parse_arguments(ARG "TTNN" "" "" ${ARGN})

    # Get the target's output directory
    get_target_property(OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY)
    if(NOT OUTPUT_DIR)
        message(
            FATAL_ERROR
            "Target '${TARGET}' must have RUNTIME_OUTPUT_DIRECTORY set before calling tt_set_runtime_rpath(). "
            "Set it with set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ...)"
        )
    endif()

    # === BUILD_RPATH: Absolute paths for build-time AND tar artifact execution ===
    # Use CMAKE_BINARY_DIR (top-level build dir) not PROJECT_BINARY_DIR
    # because subprojects with their own project() would have different PROJECT_BINARY_DIR
    #
    # We include BOTH build-tree and source-tree paths because:
    # - During build: _ttnncpp.so is at build/ttnn/
    # - In tar artifact: _ttnncpp.so is at ttnn/ttnn/ (copied from source tree)
    # The linker searches all RPATH entries and uses whichever exists.
    #
    # Note: The tar artifact is created WITHOUT cmake --install, so BUILD_RPATH
    # (not INSTALL_RPATH) is what's embedded in the binaries.
    set(BUILD_RPATH_ENTRIES "${CMAKE_BINARY_DIR}/lib")
    if(ARG_TTNN)
        # Build tree location (for gtest_discover_tests during build)
        list(APPEND BUILD_RPATH_ENTRIES "${CMAKE_BINARY_DIR}/ttnn")
        # Source tree location (for tar artifact runtime)
        list(APPEND BUILD_RPATH_ENTRIES "${CMAKE_SOURCE_DIR}/ttnn/ttnn")
    endif()

    # === INSTALL_RPATH: Relative paths for proper cmake --install ===
    # (Currently unused since tar artifact doesn't run cmake --install,
    # but kept for future FHS package installs)
    file(RELATIVE_PATH LIB_REL_PATH "${OUTPUT_DIR}" "${CMAKE_BINARY_DIR}/lib")
    set(INSTALL_RPATH_ENTRIES "$ORIGIN/${LIB_REL_PATH}")
    if(ARG_TTNN)
        file(RELATIVE_PATH TTNN_REL_PATH "${OUTPUT_DIR}" "${CMAKE_SOURCE_DIR}/ttnn/ttnn")
        list(APPEND INSTALL_RPATH_ENTRIES "$ORIGIN/${TTNN_REL_PATH}")
    endif()

    # Convert lists to semicolon-separated strings
    list(JOIN BUILD_RPATH_ENTRIES ";" BUILD_RPATH_STRING)
    list(JOIN INSTALL_RPATH_ENTRIES ";" INSTALL_RPATH_STRING)

    # Set properties on target
    set_target_properties(
        ${TARGET}
        PROPERTIES
            BUILD_RPATH
                "${BUILD_RPATH_STRING}"
            INSTALL_RPATH
                "${INSTALL_RPATH_STRING}"
    )

    # Debug message (only shown with cmake --log-level=DEBUG)
    message(DEBUG "tt_set_runtime_rpath(${TARGET}): BUILD_RPATH=${BUILD_RPATH_STRING}")
    message(DEBUG "tt_set_runtime_rpath(${TARGET}): INSTALL_RPATH=${INSTALL_RPATH_STRING}")
endfunction()
