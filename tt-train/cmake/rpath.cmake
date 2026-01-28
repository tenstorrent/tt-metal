# RPATH helper for tt-train executables
#
# This function consolidates RPATH handling using rpath.cmake infrastructure.
# Use tt_train_set_executable_rpath() for all tt-train executables instead of
# manual CMAKE_BUILD_RPATH/CMAKE_INSTALL_RPATH settings.
#
# When built as part of the main project (BUILD_TT_TRAIN=ON), executables need to find:
#   - build/lib/libtt_metal.so
#   - ttnn/ttnn/_ttnncpp.so (in the SOURCE tree, not build tree)
#
# Usage:
#   add_executable(my_app main.cpp)
#   target_link_libraries(my_app PRIVATE ttml)
#   tt_train_set_executable_rpath(my_app TTNN)
#
# Options:
#   TTNN - Add TTNN library paths to RPATH (use when linking against ttnn)
#

function(tt_train_set_executable_rpath TARGET)
    cmake_parse_arguments(ARG "TTNN" "" "" ${ARGN})

    # Set output directory if not already set (required for tt_set_runtime_rpath)
    get_target_property(_output_dir ${TARGET} RUNTIME_OUTPUT_DIRECTORY)
    if(NOT _output_dir)
        set_target_properties(
            ${TARGET}
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY
                    ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()

    # Use rpath.cmake infrastructure when available (built as part of main project)
    if(COMMAND tt_set_runtime_rpath)
        if(ARG_TTNN)
            tt_set_runtime_rpath(${TARGET} TTNN)
        else()
            tt_set_runtime_rpath(${TARGET})
        endif()
    else()
        # Fallback for standalone builds: set RPATH manually
        # Get the output directory for relative path calculation
        get_target_property(_output_dir ${TARGET} RUNTIME_OUTPUT_DIRECTORY)

        set(_build_rpath
            "${CMAKE_BINARY_DIR}/lib"
            "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
        )
        if(ARG_TTNN)
            list(APPEND _build_rpath "${CMAKE_SOURCE_DIR}/ttnn/ttnn")
        endif()

        # Calculate relative paths for INSTALL_RPATH
        file(RELATIVE_PATH _rel_to_lib "${_output_dir}" "${CMAKE_BINARY_DIR}/lib")

        set(_install_rpath
            "$ORIGIN/${_rel_to_lib}"
            "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
        )
        if(ARG_TTNN)
            file(RELATIVE_PATH _rel_to_ttnn "${_output_dir}" "${CMAKE_SOURCE_DIR}/ttnn/ttnn")
            list(APPEND _install_rpath "$ORIGIN/${_rel_to_ttnn}")
        endif()

        set_target_properties(
            ${TARGET}
            PROPERTIES
                BUILD_WITH_INSTALL_RPATH
                    FALSE
                BUILD_RPATH
                    "${_build_rpath}"
                INSTALL_RPATH
                    "${_install_rpath}"
        )
    endif()
endfunction()
