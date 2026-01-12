# =============================================================================
# RPATH Configuration Helpers
# =============================================================================
#
# Problem This Solves
# -------------------
# Executables need to find shared libraries at runtime. The dynamic linker uses
# RPATH/RUNPATH embedded in the binary to locate dependencies. This project must
# support:
#
# 1. Developer builds (run tests from build directory without installing)
# 2. CI tar artifacts (specific directory layout for test infrastructure)
# 3. FHS packages (DEB/RPM with libraries in /usr/lib64)
#
# Additional complexity:
# - Ubuntu uses RPATH (searched BEFORE LD_LIBRARY_PATH)
# - Fedora uses RUNPATH (searched AFTER LD_LIBRARY_PATH)
# - Fedora's brp-check-rpaths requires $ORIGIN to be FIRST in RPATH
# - Same CMake configuration must work for both distros
#
# Solution Architecture
# ---------------------
# This module provides helper functions for EXECUTABLES. Core libraries (tt_metal,
# tt_stl) use a different approach - see their CMakeLists.txt files and the
# documentation in cmake/packaging.cmake for details.
#
# For executables:
# - BUILD_RPATH: Absolute paths + $ORIGIN-relative paths for build-time execution
# - INSTALL_RPATH: $ORIGIN-relative paths for installed/packaged binaries
# - BUILD_WITH_INSTALL_RPATH=FALSE: Use BUILD_RPATH during development
#
# Ensure GNUInstallDirs is available for CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# =============================================================================
# Function Reference
# =============================================================================
#
# tt_set_runtime_rpath(target [TTNN])
#   For: Test executables and tools
#   Use when: Target is an executable that needs to find libraries at runtime
#   Examples: unit_tests_*, test_*, lightmetal_runner, ttml_tests
#   Requires: RUNTIME_OUTPUT_DIRECTORY must be set before calling
#   Options: TTNN - Add TTNN library paths to RPATH
#
# tt_set_library_rpath(target)
#   For: Internal shared libraries (not installed via CPack)
#   Use when: Library only needs build-time dependency resolution
#   Sets: BUILD_RPATH only, BUILD_WITH_INSTALL_RPATH=FALSE
#   Note: Core libraries (tt_metal, tt_stl) use custom RPATH handling instead
#
# tt_set_installable_library_rpath(target)
#   For: Libraries that support multiple installation layouts
#   Use when: Library needs wheel, tar artifact, and FHS package RPATH support
#   Sets: BUILD_WITH_INSTALL_RPATH=TRUE with multi-layout INSTALL_RPATH
#   Note: Not used by tt_metal/tt_stl due to CMake install() bug workaround
#
# Wrapper Functions
# -----------------
# Subdirectories can create wrapper functions for consistent RPATH handling:
#   - tt-train/cmake/rpath.cmake: tt_train_set_executable_rpath()
#
# =============================================================================
#
# Example for executable:
#   add_executable(my_test test.cpp)
#   set_target_properties(my_test PROPERTIES
#       RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/my_tests
#   )
#   tt_set_runtime_rpath(my_test)  # Automatically calculates paths
#
# Example for library (internal, not installed):
#   add_library(my_lib SHARED lib.cpp)
#   tt_set_library_rpath(my_lib)
#
# -----------------------------------------------------------------------------
# tt_set_runtime_rpath - Configure RPATH for test executables and tools
# -----------------------------------------------------------------------------
# Sets both BUILD_RPATH (for development) and INSTALL_RPATH (for packages).
# Uses BUILD_WITH_INSTALL_RPATH=FALSE so executables use BUILD_RPATH during
# development and INSTALL_RPATH only after cmake --install.
#
# BUILD_RPATH includes:
#   - $ORIGIN-relative paths to build/lib (works from any working directory)
#   - Absolute paths to build directories (fast path for common cases)
#   - Target file directories for tt_metal and optionally TTNN
#
# INSTALL_RPATH includes:
#   - $ORIGIN-relative paths only (portable for installed packages)
#
function(tt_set_runtime_rpath TARGET)
    cmake_parse_arguments(ARG "TTNN" "" "" ${ARGN})

    get_target_property(OUTPUT_DIR ${TARGET} RUNTIME_OUTPUT_DIRECTORY)
    if(NOT OUTPUT_DIR)
        message(
            FATAL_ERROR
            "Target '${TARGET}' must have RUNTIME_OUTPUT_DIRECTORY set before calling tt_set_runtime_rpath(). "
            "Set it with set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ...)"
        )
    endif()

    # BUILD_RPATH: Multiple path types for maximum compatibility
    # - $ORIGIN-relative: Works regardless of current working directory
    # - Absolute paths: Fast path for common development scenarios
    # - Target directories: Follows actual library locations
    set(BUILD_RPATH_ENTRIES "")

    # $ORIGIN-relative paths to build/lib and build/${CMAKE_INSTALL_LIBDIR}
    file(RELATIVE_PATH _rel_to_build_lib "${OUTPUT_DIR}" "${CMAKE_BINARY_DIR}/lib")
    if(_rel_to_build_lib STREQUAL "")
        list(APPEND BUILD_RPATH_ENTRIES "$ORIGIN")
    else()
        list(APPEND BUILD_RPATH_ENTRIES "$ORIGIN/${_rel_to_build_lib}")
    endif()
    if(NOT "${CMAKE_INSTALL_LIBDIR}" STREQUAL "lib")
        file(RELATIVE_PATH _rel_to_build_libdir "${OUTPUT_DIR}" "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
        if(_rel_to_build_libdir STREQUAL "")
            list(APPEND BUILD_RPATH_ENTRIES "$ORIGIN")
        else()
            list(APPEND BUILD_RPATH_ENTRIES "$ORIGIN/${_rel_to_build_libdir}")
        endif()
    endif()

    if(TARGET tt_metal)
        list(APPEND BUILD_RPATH_ENTRIES "$<TARGET_FILE_DIR:tt_metal>")
    endif()
    if(ARG_TTNN AND TARGET TTNN::TTNN)
        list(APPEND BUILD_RPATH_ENTRIES "$<TARGET_FILE_DIR:TTNN::TTNN>")
    endif()
    # Conventional aggregate dirs (used by CI/tar layout, and some deps)
    list(APPEND BUILD_RPATH_ENTRIES "${CMAKE_BINARY_DIR}/lib")
    if(NOT "${CMAKE_INSTALL_LIBDIR}" STREQUAL "lib")
        list(APPEND BUILD_RPATH_ENTRIES "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    endif()
    if(ARG_TTNN)
        # Build tree location (for gtest_discover_tests during build)
        list(APPEND BUILD_RPATH_ENTRIES "${CMAKE_BINARY_DIR}/ttnn")
        # Source tree location (for tar artifact runtime)
        list(APPEND BUILD_RPATH_ENTRIES "${CMAKE_SOURCE_DIR}/ttnn/ttnn")
    endif()

    # === INSTALL_RPATH: Relative paths for proper cmake --install ===
    # (Currently unused since tar artifact doesn't run cmake --install,
    # but kept for future FHS package installs)
    # Prefer build/lib for install-time relative rpath, but include lib64 variant if applicable.
    file(RELATIVE_PATH LIB_REL_PATH "${OUTPUT_DIR}" "${CMAKE_BINARY_DIR}/lib")
    set(INSTALL_RPATH_ENTRIES "$ORIGIN/${LIB_REL_PATH}")
    if(NOT "${CMAKE_INSTALL_LIBDIR}" STREQUAL "lib")
        file(RELATIVE_PATH LIB64_REL_PATH "${OUTPUT_DIR}" "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
        list(APPEND INSTALL_RPATH_ENTRIES "$ORIGIN/${LIB64_REL_PATH}")
    endif()
    if(ARG_TTNN)
        file(RELATIVE_PATH TTNN_REL_PATH "${OUTPUT_DIR}" "${CMAKE_SOURCE_DIR}/ttnn/ttnn")
        list(APPEND INSTALL_RPATH_ENTRIES "$ORIGIN/${TTNN_REL_PATH}")
    endif()

    # Convert lists to semicolon-separated strings
    list(JOIN BUILD_RPATH_ENTRIES ";" BUILD_RPATH_STRING)
    list(JOIN INSTALL_RPATH_ENTRIES ";" INSTALL_RPATH_STRING)

    # Set properties on target
    # BUILD_WITH_INSTALL_RPATH=FALSE ensures BUILD_RPATH is used during build
    # (not INSTALL_RPATH which uses $ORIGIN-relative paths that won't work
    # until the libraries are in their final locations)
    set_target_properties(
        ${TARGET}
        PROPERTIES
            BUILD_WITH_INSTALL_RPATH
                FALSE
            BUILD_RPATH
                "${BUILD_RPATH_STRING}"
            INSTALL_RPATH
                "${INSTALL_RPATH_STRING}"
    )

    # Debug message (only shown with cmake --log-level=DEBUG)
    message(DEBUG "tt_set_runtime_rpath(${TARGET}): BUILD_RPATH=${BUILD_RPATH_STRING}")
    message(DEBUG "tt_set_runtime_rpath(${TARGET}): INSTALL_RPATH=${INSTALL_RPATH_STRING}")
endfunction()

# -----------------------------------------------------------------------------
# tt_set_library_rpath - Configure BUILD_RPATH for internal shared libraries
# -----------------------------------------------------------------------------
# Sets BUILD_RPATH so libraries can find dependencies during build-time linking.
# Uses BUILD_WITH_INSTALL_RPATH=FALSE (BUILD_RPATH used during development).
#
# Note: Core libraries (tt_metal, tt_stl) use custom RPATH handling with
# BUILD_WITH_INSTALL_RPATH=TRUE and install(FILES) workaround. See their
# CMakeLists.txt files for details.
#
# Why needed?
#   - Fedora uses RUNPATH (searched after LD_LIBRARY_PATH)
#   - Without BUILD_RPATH, libraries can't find each other during build
#   - BUILD_RPATH ensures consistent behavior across Ubuntu and Fedora
#
function(tt_set_library_rpath TARGET)
    # Set BUILD_RPATH to build/lib (CI/tar layout) and also build/${CMAKE_INSTALL_LIBDIR} when it differs.
    set(_build_rpath_entries "${CMAKE_BINARY_DIR}/lib")
    if(NOT "${CMAKE_INSTALL_LIBDIR}" STREQUAL "lib")
        list(APPEND _build_rpath_entries "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    endif()
    list(JOIN _build_rpath_entries ";" _build_rpath_string)

    set_target_properties(
        ${TARGET}
        PROPERTIES
            BUILD_RPATH
                "${_build_rpath_string}"
            BUILD_WITH_INSTALL_RPATH
                FALSE
    )

    # Debug message (only shown with cmake --log-level=DEBUG)
    message(DEBUG "tt_set_library_rpath(${TARGET}): BUILD_RPATH=${_build_rpath_string}")
endfunction()

# -----------------------------------------------------------------------------
# tt_set_installable_library_rpath - Multi-layout RPATH for installable libraries
# -----------------------------------------------------------------------------
# Sets INSTALL_RPATH to support multiple installation layouts (wheel, tar, FHS).
# Uses BUILD_WITH_INSTALL_RPATH=TRUE so INSTALL_RPATH is embedded at build time.
#
# Supported layouts:
#   - Python wheel: $ORIGIN/build/lib
#   - CI tar artifact: $ORIGIN/../../build/lib
#   - FHS packages (DEB/RPM): $ORIGIN (libs co-located in /usr/lib64)
#
# Note: Core libraries (tt_metal, tt_stl) do NOT use this function because
# they need the install(FILES) workaround for a CMake bug. They manually
# configure BUILD_WITH_INSTALL_RPATH=TRUE with simpler INSTALL_RPATH.
# See cmake/packaging.cmake for detailed explanation of the CMake bug.
#
# When to use this function:
#   - Libraries that need to work in wheel, tar, AND FHS layouts
#   - Libraries that can use standard install(TARGETS ... LIBRARY)
#
function(tt_set_installable_library_rpath TARGET)
    # Set BUILD_RPATH as backup for build-time linking (important for Fedora RUNPATH)
    # Prefer build/lib (CI/tar layout) but also include build/${CMAKE_INSTALL_LIBDIR} when different.
    set(_build_rpath_entries "${CMAKE_BINARY_DIR}/lib")
    if(NOT "${CMAKE_INSTALL_LIBDIR}" STREQUAL "lib")
        list(APPEND _build_rpath_entries "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    endif()
    list(JOIN _build_rpath_entries ";" _build_rpath_string)

    set_target_properties(
        ${TARGET}
        PROPERTIES
            BUILD_RPATH
                "${_build_rpath_string}"
            # Use INSTALL_RPATH during build for multi-layout support
            BUILD_WITH_INSTALL_RPATH
                TRUE
            # INSTALL_RPATH for multiple installation layouts (searched in order):
            # - $ORIGIN/build/lib      = wheel layout (wheels always use 'lib', not CMAKE_INSTALL_LIBDIR)
            # - $ORIGIN/../../build/lib = tar artifact layout (tar artifacts always use 'lib')
            # - $ORIGIN                 = FHS packages (all libs co-located in same dir, uses CMAKE_INSTALL_LIBDIR for install but RPATH is just $ORIGIN)
            # The linker searches all paths in order and uses the first one that exists.
            # Note: $ORIGIN is placed last (not first) because it's the fallback for FHS packages where all libs
            # are co-located. For wheel and tar layouts, the more specific paths ($ORIGIN/build/lib, etc.) are
            # checked first, and $ORIGIN only applies when those don't exist (i.e., in FHS package installations).
            INSTALL_RPATH
                "$ORIGIN/build/lib;$ORIGIN/../../build/lib;$ORIGIN"
    )

    # Debug message (only shown with cmake --log-level=DEBUG)
    message(
        DEBUG
        "tt_set_installable_library_rpath(${TARGET}): BUILD_RPATH=${_build_rpath_string}, INSTALL_RPATH=$ORIGIN/build/lib;$ORIGIN/../../build/lib;$ORIGIN"
    )
endfunction()

# =============================================================================
# Install-time diagnostic: List all shared libraries in the build directory
# =============================================================================
# This runs at the very start of the install step to show where all .so files are
install(
    CODE
        "
        message(STATUS \"=== Listing all shared libraries in build directory ===\")
        execute_process(
            COMMAND bash -c \"pwd ; find -name '*.so' | sort -h | xargs ls -lh\"
            WORKING_DIRECTORY \"${CMAKE_BINARY_DIR}\"
            RESULT_VARIABLE _find_result
        )
        message(STATUS \"=== End of shared library listing ===\")
    "
)
