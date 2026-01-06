# CMake helper for setting relocatable RPATH on test executables
#
# This function sets BUILD_WITH_INSTALL_RPATH=TRUE and calculates the correct
# $ORIGIN-based INSTALL_RPATH automatically based on the target's output directory.
#
# Usage:
#   tt_set_runtime_rpath(target_name)              # Links to build/lib/ only
#   tt_set_runtime_rpath(target_name TTNN)         # Links to build/lib/ and ttnn/ttnn/
#
# Requirements:
#   - Target must have RUNTIME_OUTPUT_DIRECTORY set before calling this function
#
# Example:
#   add_executable(my_test test.cpp)
#   set_target_properties(my_test PROPERTIES
#       RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/my_tests
#   )
#   tt_set_runtime_rpath(my_test)  # Automatically calculates $ORIGIN/../../../lib
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

    # Calculate relative path from output dir to build/lib/
    # Use CMAKE_BINARY_DIR (top-level build dir) not PROJECT_BINARY_DIR
    # because subprojects with their own project() would have different PROJECT_BINARY_DIR
    file(RELATIVE_PATH LIB_REL_PATH "${OUTPUT_DIR}" "${CMAKE_BINARY_DIR}/lib")
    set(RPATH_ENTRIES "$ORIGIN/${LIB_REL_PATH}")

    # If TTNN flag is set, also add path to ttnn/ttnn/ (for _ttnncpp.so, _ttnn.so)
    # Use CMAKE_SOURCE_DIR (top-level source dir) for the same reason
    if(ARG_TTNN)
        file(RELATIVE_PATH TTNN_REL_PATH "${OUTPUT_DIR}" "${CMAKE_SOURCE_DIR}/ttnn/ttnn")
        list(APPEND RPATH_ENTRIES "$ORIGIN/${TTNN_REL_PATH}")
    endif()

    # Convert list to semicolon-separated string for INSTALL_RPATH
    list(JOIN RPATH_ENTRIES ";" RPATH_STRING)

    # Set properties on target
    set_target_properties(
        ${TARGET}
        PROPERTIES
            BUILD_WITH_INSTALL_RPATH
                TRUE
            INSTALL_RPATH
                "${RPATH_STRING}"
    )

    # Debug message (only shown with cmake --log-level=DEBUG)
    message(DEBUG "tt_set_runtime_rpath(${TARGET}): INSTALL_RPATH=${RPATH_STRING}")
endfunction()
