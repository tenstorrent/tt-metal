# Compiler selection and validation for tt-train.
#
# When building standalone (not as a tt-metal subproject), tt-train needs
# a compiler compatible with the one used to build tt-metal, since it links
# against tt-metal libraries (libtt_metal.so, _ttnncpp.so).
#
# Compiler selection (standalone top-level builds):
#   - By default, the compiler is inherited from tt-metal's CMakeCache.txt
#     to ensure ABI compatibility (TT_TRAIN_INHERIT_COMPILER=ON).
#   - To use a different compiler, disable inheritance and set explicitly:
#     cmake -DTT_TRAIN_INHERIT_COMPILER=OFF -DCMAKE_C_COMPILER=gcc-12 ...
#   - If no CMakeCache.txt is found, falls back to CMake default detection.
#   - CHECK_COMPILERS() validates the result (Clang 17+ or GCC 12+).
#
# When building as a tt-metal subproject (add_subdirectory), the parent
# project's compiler is used and inheritance is skipped.
#
# Restored after deletion in a7618b9282 ("TT-Train: bump clang version
# from 17 to 20 #36568").
#
# See: https://github.com/tenstorrent/tt-metal/issues/36993

# Read the compiler used by tt-metal from its CMakeCache.txt and set it
# as CMAKE_C_COMPILER/CMAKE_CXX_COMPILER so tt-train uses the same one.
function(INHERIT_COMPILER_FROM_TT_METAL)
    # Determine TT_METAL_HOME
    if(DEFINED ENV{TT_METAL_HOME})
        set(_tt_metal_home "$ENV{TT_METAL_HOME}")
    else()
        # Infer from directory structure: tt-metal/tt-train/ -> tt-metal/
        set(_tt_metal_home "${CMAKE_CURRENT_SOURCE_DIR}/..")
        if(NOT EXISTS "${_tt_metal_home}/tt_metal/CMakeLists.txt")
            message(STATUS "Cannot infer tt-metal location; using CMake default compiler")
            return()
        endif()
    endif()

    # Search for CMakeCache.txt in known build directories
    set(_cache_file "")
    foreach(_build_dir "build" "build_Debug" "build_Release")
        if(EXISTS "${_tt_metal_home}/${_build_dir}/CMakeCache.txt")
            set(_cache_file "${_tt_metal_home}/${_build_dir}/CMakeCache.txt")
            break()
        endif()
    endforeach()

    if(NOT _cache_file)
        message(STATUS "No tt-metal build found; using CMake default compiler")
        return()
    endif()

    # Parse CMAKE_C_COMPILER and CMAKE_CXX_COMPILER from CMakeCache.txt
    file(STRINGS "${_cache_file}" _c_compiler_line REGEX "^CMAKE_C_COMPILER:.*=")
    file(STRINGS "${_cache_file}" _cxx_compiler_line REGEX "^CMAKE_CXX_COMPILER:.*=")

    if(_c_compiler_line AND _cxx_compiler_line)
        string(REGEX REPLACE "^CMAKE_C_COMPILER:[^=]*=(.*)" "\\1" _c_compiler "${_c_compiler_line}")
        string(REGEX REPLACE "^CMAKE_CXX_COMPILER:[^=]*=(.*)" "\\1" _cxx_compiler "${_cxx_compiler_line}")

        if(EXISTS "${_c_compiler}" AND EXISTS "${_cxx_compiler}")
            message(STATUS "Inheriting compiler from tt-metal build (${_cache_file}):")
            message(STATUS "  C compiler:   ${_c_compiler}")
            message(STATUS "  C++ compiler: ${_cxx_compiler}")
            set(CMAKE_C_COMPILER "${_c_compiler}" PARENT_SCOPE)
            set(CMAKE_CXX_COMPILER "${_cxx_compiler}" PARENT_SCOPE)
        else()
            message(STATUS "Compiler paths from tt-metal CMakeCache not found on disk; using CMake default")
        endif()
    else()
        message(STATUS "Could not parse compiler from ${_cache_file}; using CMake default")
    endif()
endfunction()

function(CHECK_COMPILERS)
    message(STATUS "Checking compilers: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # Any Clang version is accepted (same as tt-metal)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0.0")
            message(FATAL_ERROR "GCC-12 or higher is required")
        endif()
    else()
        message(WARNING "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID} ! Only Clang and GCC are supported")
    endif()
endfunction()
