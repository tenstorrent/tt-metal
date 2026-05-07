function(CHECK_COMPILERS)
    message(STATUS "Checking compilers")

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(ENABLE_LIBCXX)
            find_library(LIBC++ c++)
            find_library(LIBC++ABI c++abi)
            if(NOT LIBC++ OR NOT LIBC++ABI)
                message(
                    FATAL_ERROR
                    "libc++ or libc++abi not found. Make sure you have libc++ and libc++abi installed and in your PATH"
                )
            endif()
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12.0.0")
            message(FATAL_ERROR "GCC-12 or higher is required")
        endif()
    else()
        message(WARNING "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID} ! Only Clang and GCC are supported")
    endif()

    # Clang on Linux often leaves CMAKE_(C|CXX)_COMPILER_(AR|RANLIB) unset (no llvm-ar
    # beside the driver). CMake still prefers those for language-specific static archives,
    # which yields literal *-NOTFOUND in Ninja rules; Tracy also prefixes link/archive
    # steps with ccache (RULE_LAUNCH_LINK), so the failure surfaces as ccache not finding
    # the "compiler". Fall back to the global archiver from the initial toolchain probe.
    foreach(_lang IN ITEMS C CXX ASM)
        if(CMAKE_${_lang}_COMPILER_AR MATCHES "-NOTFOUND$")
            set(CMAKE_${_lang}_COMPILER_AR "${CMAKE_AR}" CACHE FILEPATH "Archiver for ${_lang}" FORCE)
            # Normal var was set from CMake*Compiler.cmake at project() time; it shadows CACHE until overwritten.
            set(CMAKE_${_lang}_COMPILER_AR "${CMAKE_AR}")
        endif()
        if(CMAKE_${_lang}_COMPILER_RANLIB MATCHES "-NOTFOUND$")
            set(CMAKE_${_lang}_COMPILER_RANLIB "${CMAKE_RANLIB}" CACHE FILEPATH "Ranlib for ${_lang}" FORCE)
            set(CMAKE_${_lang}_COMPILER_RANLIB "${CMAKE_RANLIB}")
        endif()
    endforeach()

    # project() writes CMake{C,CXX}Compiler.cmake before this runs. Those files keep
    # CMAKE_*_COMPILER_AR=-NOTFOUND for Clang without llvm-ar and override CACHE on the next
    # configure pass, so Ninja rules still embed the bogus path until we sync the files.
    if(CMAKE_AR AND CMAKE_RANLIB)
        file(GLOB _tt_c_compiler_info "${CMAKE_BINARY_DIR}/CMakeFiles/*/CMakeCCompiler.cmake")
        foreach(_f IN LISTS _tt_c_compiler_info)
            file(READ "${_f}" _txt)
            string(
                REPLACE
                [[set(CMAKE_C_COMPILER_AR "CMAKE_C_COMPILER_AR-NOTFOUND")]]
                "set(CMAKE_C_COMPILER_AR \"${CMAKE_AR}\")"
                _txt
                "${_txt}"
            )
            string(
                REPLACE
                [[set(CMAKE_C_COMPILER_RANLIB "CMAKE_C_COMPILER_RANLIB-NOTFOUND")]]
                "set(CMAKE_C_COMPILER_RANLIB \"${CMAKE_RANLIB}\")"
                _txt
                "${_txt}"
            )
            file(WRITE "${_f}" "${_txt}")
        endforeach()
        file(GLOB _tt_cxx_compiler_info "${CMAKE_BINARY_DIR}/CMakeFiles/*/CMakeCXXCompiler.cmake")
        foreach(_f IN LISTS _tt_cxx_compiler_info)
            file(READ "${_f}" _txt)
            string(
                REPLACE
                [[set(CMAKE_CXX_COMPILER_AR "CMAKE_CXX_COMPILER_AR-NOTFOUND")]]
                "set(CMAKE_CXX_COMPILER_AR \"${CMAKE_AR}\")"
                _txt
                "${_txt}"
            )
            string(
                REPLACE
                [[set(CMAKE_CXX_COMPILER_RANLIB "CMAKE_CXX_COMPILER_RANLIB-NOTFOUND")]]
                "set(CMAKE_CXX_COMPILER_RANLIB \"${CMAKE_RANLIB}\")"
                _txt
                "${_txt}"
            )
            file(WRITE "${_f}" "${_txt}")
        endforeach()
    endif()
endfunction()
