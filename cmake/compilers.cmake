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
    # which yields literal *-NOTFOUND in Ninja rules. Tracy used to set RULE_LAUNCH_LINK to
    # ccache (see tt_metal/third_party/tracy/cmake/config.cmake); that also prefixes archive
    # (ar) steps and breaks when COMPILER_AR is wrong. Fall back to the global archiver from
    # the initial toolchain probe; Clang toolchain files also seed CMAKE_AR/CMAKE_RANLIB early.
    #
    # CMake can also leave CMAKE_AR itself empty while still writing CMAKE_C_COMPILER_AR to
    # the sentinel CMAKE_C_COMPILER_AR-NOTFOUND, so the old logic (which required CMAKE_AR to
    # already be set) never repaired the cache or CMake*Compiler.cmake on those toolchains.
    if(NOT WIN32)
        if(NOT CMAKE_AR OR "${CMAKE_AR}" MATCHES "NOTFOUND")
            find_program(
                _tt_resolved_ar
                NAMES
                    ar
                    llvm-ar
                    gcc-ar
                    NO_CACHE
            )
            if(_tt_resolved_ar)
                set(CMAKE_AR "${_tt_resolved_ar}" CACHE FILEPATH "Path to archiver for static libraries" FORCE)
                message(STATUS "CMAKE_AR was unset/invalid; using ${CMAKE_AR}")
            endif()
        endif()
        if(NOT CMAKE_RANLIB OR "${CMAKE_RANLIB}" MATCHES "NOTFOUND")
            find_program(
                _tt_resolved_ranlib
                NAMES
                    ranlib
                    llvm-ranlib
                    gcc-ranlib
                    NO_CACHE
            )
            if(_tt_resolved_ranlib)
                set(CMAKE_RANLIB "${_tt_resolved_ranlib}" CACHE FILEPATH "Path to ranlib for static libraries" FORCE)
                message(STATUS "CMAKE_RANLIB was unset/invalid; using ${CMAKE_RANLIB}")
            endif()
        endif()
    endif()

    foreach(_lang IN ITEMS C CXX ASM)
        set(_tt_lang_ar "${CMAKE_${_lang}_COMPILER_AR}")
        if(
            CMAKE_AR
            AND NOT "${CMAKE_AR}"
                MATCHES
                "NOTFOUND"
            AND (
                NOT _tt_lang_ar
                OR "${_tt_lang_ar}"
                    MATCHES
                    "NOTFOUND"
            )
        )
            set(CMAKE_${_lang}_COMPILER_AR "${CMAKE_AR}" CACHE FILEPATH "Archiver for ${_lang}" FORCE)
            set(CMAKE_${_lang}_COMPILER_AR "${CMAKE_AR}")
        endif()
        set(_tt_lang_ranlib "${CMAKE_${_lang}_COMPILER_RANLIB}")
        if(
            CMAKE_RANLIB
            AND NOT "${CMAKE_RANLIB}"
                MATCHES
                "NOTFOUND"
            AND (
                NOT _tt_lang_ranlib
                OR "${_tt_lang_ranlib}"
                    MATCHES
                    "NOTFOUND"
            )
        )
            set(CMAKE_${_lang}_COMPILER_RANLIB "${CMAKE_RANLIB}" CACHE FILEPATH "Ranlib for ${_lang}" FORCE)
            set(CMAKE_${_lang}_COMPILER_RANLIB "${CMAKE_RANLIB}")
        endif()
        unset(_tt_lang_ar)
        unset(_tt_lang_ranlib)
    endforeach()

    # project() writes CMake{C,CXX}Compiler.cmake before this runs. Those files keep
    # CMAKE_*_COMPILER_AR=-NOTFOUND for Clang without llvm-ar and override CACHE on the next
    # configure pass, so Ninja rules still embed the bogus path until we sync the files.
    file(GLOB _tt_c_compiler_info "${CMAKE_BINARY_DIR}/CMakeFiles/*/CMakeCCompiler.cmake")
    foreach(_f IN LISTS _tt_c_compiler_info)
        file(READ "${_f}" _txt)
        if(CMAKE_AR AND NOT "${CMAKE_AR}" MATCHES "NOTFOUND")
            string(
                REPLACE
                [[set(CMAKE_C_COMPILER_AR "CMAKE_C_COMPILER_AR-NOTFOUND")]]
                "set(CMAKE_C_COMPILER_AR \"${CMAKE_AR}\")"
                _txt
                "${_txt}"
            )
            string(
                REPLACE
                [[set(CMAKE_C_COMPILER_AR CMAKE_C_COMPILER_AR-NOTFOUND)]]
                "set(CMAKE_C_COMPILER_AR \"${CMAKE_AR}\")"
                _txt
                "${_txt}"
            )
        endif()
        if(CMAKE_RANLIB AND NOT "${CMAKE_RANLIB}" MATCHES "NOTFOUND")
            string(
                REPLACE
                [[set(CMAKE_C_COMPILER_RANLIB "CMAKE_C_COMPILER_RANLIB-NOTFOUND")]]
                "set(CMAKE_C_COMPILER_RANLIB \"${CMAKE_RANLIB}\")"
                _txt
                "${_txt}"
            )
            string(
                REPLACE
                [[set(CMAKE_C_COMPILER_RANLIB CMAKE_C_COMPILER_RANLIB-NOTFOUND)]]
                "set(CMAKE_C_COMPILER_RANLIB \"${CMAKE_RANLIB}\")"
                _txt
                "${_txt}"
            )
        endif()
        file(WRITE "${_f}" "${_txt}")
    endforeach()
    file(GLOB _tt_cxx_compiler_info "${CMAKE_BINARY_DIR}/CMakeFiles/*/CMakeCXXCompiler.cmake")
    foreach(_f IN LISTS _tt_cxx_compiler_info)
        file(READ "${_f}" _txt)
        if(CMAKE_AR AND NOT "${CMAKE_AR}" MATCHES "NOTFOUND")
            string(
                REPLACE
                [[set(CMAKE_CXX_COMPILER_AR "CMAKE_CXX_COMPILER_AR-NOTFOUND")]]
                "set(CMAKE_CXX_COMPILER_AR \"${CMAKE_AR}\")"
                _txt
                "${_txt}"
            )
            string(
                REPLACE
                [[set(CMAKE_CXX_COMPILER_AR CMAKE_CXX_COMPILER_AR-NOTFOUND)]]
                "set(CMAKE_CXX_COMPILER_AR \"${CMAKE_AR}\")"
                _txt
                "${_txt}"
            )
        endif()
        if(CMAKE_RANLIB AND NOT "${CMAKE_RANLIB}" MATCHES "NOTFOUND")
            string(
                REPLACE
                [[set(CMAKE_CXX_COMPILER_RANLIB "CMAKE_CXX_COMPILER_RANLIB-NOTFOUND")]]
                "set(CMAKE_CXX_COMPILER_RANLIB \"${CMAKE_RANLIB}\")"
                _txt
                "${_txt}"
            )
            string(
                REPLACE
                [[set(CMAKE_CXX_COMPILER_RANLIB CMAKE_CXX_COMPILER_RANLIB-NOTFOUND)]]
                "set(CMAKE_CXX_COMPILER_RANLIB \"${CMAKE_RANLIB}\")"
                _txt
                "${_txt}"
            )
        endif()
        file(WRITE "${_f}" "${_txt}")
    endforeach()
endfunction()
