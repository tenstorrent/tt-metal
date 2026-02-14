# Compiler selection and validation for tt-train.
#
# tt-train links against tt-metal libraries (libtt_metal.so, _ttnncpp.so)
# which are compiled with clang-20 + libstdc++.  Using a different compiler
# risks ABI incompatibilities, so clang-20 is required.
#
# Restored after accidental deletion in a7618b9282 ("TT-Train: bump clang
# version from 17 to 20 #36568").  Simplified to clang-20-only (the original
# also carried GCC fallback paths that are no longer meaningful).
#
# See: https://github.com/tenstorrent/tt-metal/issues/36993

function(FIND_AND_SET_CLANG20)
    find_program(CLANGPP_20 clang++-20)
    find_program(CLANG_20 clang-20)

    if(NOT CLANGPP_20 OR NOT CLANG_20)
        message(
            FATAL_ERROR
            "clang-20 not found.\n"
            "tt-train requires clang-20 to match the compiler used by tt-metal.\n"
            "Install it with:  sudo apt install clang-20   (Ubuntu/Debian)\n"
            "                  sudo dnf install clang      (Fedora)\n"
            "Or set CC/CXX environment variables to point to a clang-20 installation."
        )
    endif()

    set(CMAKE_CXX_COMPILER "${CLANGPP_20}" PARENT_SCOPE)
    set(CMAKE_C_COMPILER "${CLANG_20}" PARENT_SCOPE)
endfunction()

function(CHECK_COMPILERS)
    message(STATUS "Checking compilers: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "17.0.0")
            message(WARNING "Clang 17 or higher is recommended; found ${CMAKE_CXX_COMPILER_VERSION}")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(
            FATAL_ERROR
            "GCC is not supported for tt-train standalone builds.\n"
            "tt-train links against tt-metal libraries compiled with clang-20.\n"
            "Mixing compilers risks ABI incompatibilities.\n"
            "Please use clang-20:  CC=clang-20 CXX=clang++-20 cmake ..."
        )
    else()
        message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}. Only Clang is supported.")
    endif()
endfunction()

function(ADJUST_COMPILER_WARNINGS)
    target_compile_options(
        compiler_warnings
        INTERFACE
            -Wsometimes-uninitialized
            -Wno-c++11-narrowing
            -Wno-error=local-type-template-args
            -Wno-delete-non-abstract-non-virtual-dtor
            -Wno-c99-designator
            -Wno-shift-op-parentheses
            -Wno-non-c-typedef-for-linkage
            -Wno-deprecated-this-capture
            -Wno-deprecated-volatile
            -Wno-deprecated-builtins
            -Wno-deprecated-declarations
    )
endfunction()
