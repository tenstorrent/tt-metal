set(TT_LTO_ENABLED OFF)

if(TT_ENABLE_LTO)
    cmake_policy(SET CMP0069 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)

    # LTO can be requested but not supported, so handle that case
    if(result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "LTO/IPO is supported.")
        set(TT_LTO_ENABLED ON)

        # Enable one-definition-rule (ODR) warnings.
        # Only works when LTO is enabled.
        # Forcing odr to warn until current warnings are addressed.
        add_compile_options(
            -Wodr
            -Wno-error=odr
        )

        # set up lto cache for faster incremental builds and a nominal 1G limit to keep the size
        # from getting too large over time. A single everthing-enabled build results in an
        # lto-cache of ~250M.
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-cache-dir=${CMAKE_BINARY_DIR}/lto-cache>)
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-cache-policy=cache_size_bytes=1G>)
    else()
        message(WARNING "LTO/IPO is not supported: ${output}")
    endif()
endif()

if(NOT DEFINED CMAKE_LINKER_TYPE)
    find_program(MOLD ld.mold)
    if(MOLD)
        set(CMAKE_LINKER_TYPE MOLD)
        message(STATUS "Using mold linker.")
    endif()
endif()

if(TT_LTO_ENABLED AND MOLD)
    message(STATUS "LTO using Mold.")
    add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-jobs=all>)
endif()
