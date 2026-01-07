# CMake helper for setting relocatable RPATH on test executables and libraries
#
# Ensure GNUInstallDirs is available for CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# This file provides three functions:
#   tt_set_runtime_rpath() - For executables (requires RUNTIME_OUTPUT_DIRECTORY)
#   tt_set_library_rpath() - For shared libraries (BUILD_RPATH only)
#   tt_set_installable_library_rpath() - For installable shared libraries (BUILD_RPATH + INSTALL_RPATH)
#
# Usage:
#   tt_set_runtime_rpath(target_name)              # Links to build/${CMAKE_INSTALL_LIBDIR}/ only
#   tt_set_runtime_rpath(target_name TTNN)         # Links to build/${CMAKE_INSTALL_LIBDIR}/ and ttnn/ttnn/
#   tt_set_library_rpath(target_name)              # Sets BUILD_RPATH for build-time linking
#   tt_set_installable_library_rpath(target_name)  # Sets both BUILD_RPATH and INSTALL_RPATH
#
# Requirements for tt_set_runtime_rpath:
#   - Target must have RUNTIME_OUTPUT_DIRECTORY set before calling this function
#
# Why different RPATH approaches?
#   - BUILD_RPATH: Absolute paths for build-time execution and linking (preferred for simple cases)
#   - INSTALL_RPATH: Relative $ORIGIN paths for tar artifacts and installed binaries
#   - BUILD_WITH_INSTALL_RPATH=TRUE: Use INSTALL_RPATH during build (for complex multi-layout cases)
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
    set(BUILD_RPATH_ENTRIES "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    if(ARG_TTNN)
        # Build tree location (for gtest_discover_tests during build)
        list(APPEND BUILD_RPATH_ENTRIES "${CMAKE_BINARY_DIR}/ttnn")
        # Source tree location (for tar artifact runtime)
        list(APPEND BUILD_RPATH_ENTRIES "${CMAKE_SOURCE_DIR}/ttnn/ttnn")
    endif()

    # === INSTALL_RPATH: Relative paths for proper cmake --install ===
    # (Currently unused since tar artifact doesn't run cmake --install,
    # but kept for future FHS package installs)
    file(RELATIVE_PATH LIB_REL_PATH "${OUTPUT_DIR}" "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    set(INSTALL_RPATH_ENTRIES "$ORIGIN/${LIB_REL_PATH}")
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
    # Set BUILD_RPATH to the build lib directory
    # This ensures the library can find its dependencies during build-time linking
    set_target_properties(
        ${TARGET}
        PROPERTIES
            BUILD_RPATH
                "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
            BUILD_WITH_INSTALL_RPATH
                FALSE
    )

    # Debug message (only shown with cmake --log-level=DEBUG)
    message(DEBUG "tt_set_library_rpath(${TARGET}): BUILD_RPATH=${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
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
    set_target_properties(
        ${TARGET}
        PROPERTIES
            BUILD_RPATH
                "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
            # Use INSTALL_RPATH during build for multi-layout support
            BUILD_WITH_INSTALL_RPATH
                TRUE
            # INSTALL_RPATH for multiple installation layouts:
            # - $ORIGIN/build/lib      = wheel layout (wheels always use 'lib', not CMAKE_INSTALL_LIBDIR)
            # - $ORIGIN/../../build/lib = tar artifact layout (tar artifacts always use 'lib')
            # - $ORIGIN                 = FHS packages (all libs co-located in same dir, uses CMAKE_INSTALL_LIBDIR for install but RPATH is just $ORIGIN)
            INSTALL_RPATH
                "$ORIGIN/build/lib;$ORIGIN/../../build/lib;$ORIGIN"
    )

    # Debug message (only shown with cmake --log-level=DEBUG)
    message(
        DEBUG
        "tt_set_installable_library_rpath(${TARGET}): BUILD_RPATH=${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}, INSTALL_RPATH=$ORIGIN/build/lib;$ORIGIN/../../build/lib;$ORIGIN"
    )
endfunction()
