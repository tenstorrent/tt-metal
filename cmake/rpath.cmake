# CMake helper for setting relocatable RPATH on test executables and libraries
#
# Ensure GNUInstallDirs is available for CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# =============================================================================
# RPATH Application Patterns - When to Use Which Function
# =============================================================================
#
# This module provides three functions for different use cases:
#
# 1. tt_set_runtime_rpath() - For test executables and tools
#    Use when: Target is an executable that needs to find libraries at runtime
#    - Test executables (unit_tests_*, test_*)
#    - Tools (lightmetal_runner, mem_bench, watcher_dump)
#    - Requires: RUNTIME_OUTPUT_DIRECTORY must be set before calling
#    - Sets: BUILD_RPATH (absolute paths) and INSTALL_RPATH (relative $ORIGIN paths)
#
# 2. tt_set_library_rpath() - For simple shared libraries (build-time only)
#    Use when: Target is a shared library that only needs build-time linking
#    - Libraries that don't get installed (internal libraries)
#    - Sets: BUILD_RPATH only (for build-time dependency resolution)
#
# 3. tt_set_installable_library_rpath() - For libraries that get installed
#    Use when: Target is a shared library that will be installed (via CPack or install())
#    - Main libraries (tt_metal, tt_stl, ttnncpp, ttnn)
#    - Sets: Both BUILD_RPATH and INSTALL_RPATH (supports multiple installation layouts)
#
# Alternative: Global CMAKE_BUILD_RPATH / CMAKE_INSTALL_RPATH
#    Use when: All targets in a subdirectory need the same RPATH
#    - Example: tt-train/CMakeLists.txt sets global RPATH for all tt-train targets
#    - Individual targets can still override with per-target functions
#
# Why different RPATH approaches?
#   - BUILD_RPATH: Absolute paths for build-time execution and linking (preferred for simple cases)
#   - INSTALL_RPATH: Relative $ORIGIN paths for tar artifacts and installed binaries
#   - BUILD_WITH_INSTALL_RPATH=TRUE: Use INSTALL_RPATH during build (for complex multi-layout cases)
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
# Example for simple library:
#   add_library(my_lib SHARED lib.cpp)
#   tt_set_library_rpath(my_lib)
#
# Example for installable library:
#   add_library(my_installable_lib SHARED lib.cpp)
#   tt_set_installable_library_rpath(my_installable_lib)  # Sets BUILD_RPATH for build-time linking
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
    # Build-time runpath for executables.
    #
    # Important: different toolchains/packaging flows may place shared libs in different build-tree
    # locations (e.g. build/lib, build/lib64, or a target's binary dir). To be robust, include:
    # - The actual output dir of tt_metal / TTNN targets (via generator expressions)
    # - Conventional aggregate dirs build/lib and build/${CMAKE_INSTALL_LIBDIR}
    set(BUILD_RPATH_ENTRIES "")
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

# CMake helper for setting BUILD_RPATH on shared libraries
#
# This function sets BUILD_RPATH for shared libraries to ensure they can find
# their dependencies during the build process. This is especially important
# on systems that use RUNPATH by default (like Fedora) instead of RPATH.
#
# Usage:
#   tt_set_library_rpath(target_name)
#
# Requirements:
#   - Target must be a shared library
#
# Why BUILD_RPATH for libraries?
#   - Libraries need to find other libraries during build-time linking
#   - Fedora uses RUNPATH by default, which searches LD_LIBRARY_PATH first
#   - BUILD_RPATH ensures libraries can find each other in build directories
#
# Example:
#   add_library(my_lib SHARED lib.cpp)
#   tt_set_library_rpath(my_lib)
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

# CMake helper for setting RPATH on installable shared libraries
#
# This function sets both BUILD_RPATH (for build-time linking) and INSTALL_RPATH
# (for multiple installation layouts). It uses BUILD_WITH_INSTALL_RPATH=TRUE so
# that INSTALL_RPATH is used during build time as well.
#
# This is for libraries that need complex multi-layout RPATH support like:
# - Wheel layout: $ORIGIN/build/lib (always 'lib', not CMAKE_INSTALL_LIBDIR, because wheels always use 'lib')
# - Tar artifact layout: $ORIGIN/../../build/lib (always 'lib' for tar artifacts)
# - FHS packages: $ORIGIN (all libs co-located, no subdirectory needed)
#
# Usage:
#   tt_set_installable_library_rpath(target_name)
#
# Requirements:
#   - Target must be a shared library that gets installed
#
# Why BUILD_WITH_INSTALL_RPATH=TRUE?
#   - Allows using INSTALL_RPATH during build for complex multi-layout scenarios
#   - Still sets BUILD_RPATH as backup for systems that need it
#
# Example:
#   add_library(my_installable_lib SHARED lib.cpp)
#   tt_set_installable_library_rpath(my_installable_lib)
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
