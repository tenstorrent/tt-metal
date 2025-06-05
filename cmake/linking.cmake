# Compressing the debug sections cuts the size of ttnn.so by half for builds with debug info.
add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,--compress-debug-sections=zstd>)

# Use mold by default if it is available.
if(NOT DEFINED CMAKE_LINKER_TYPE)
    find_program(MOLD ld.mold)
    if(MOLD)
        message(STATUS "Linker not specified. Using mold linker: ${MOLD}")
        set(CMAKE_LINKER_TYPE MOLD)
    endif()
endif()

set(TT_LTO_ENABLED OFF)

if(TT_ENABLE_LTO)
    cmake_policy(SET CMP0069 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)

    # LTO can be requested but not supported, so handle that case
    if(result)
        message(STATUS "LTO/IPO is supported. Enabling for Release/RelWithDebInfo builds.")

        # Do it this way to play nicer with ninja multi-config builds
        # This enables clang's thinLTO by default.
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)

        set(TT_LTO_ENABLED ON)

        # Enable one-definition-rule (ODR) warnings.
        # Only works when LTO is enabled.
        add_compile_options(
            -Wodr
            -Wno-error=odr # TODO(21850): Relaxing odr to warn until current errors are addressed.
        )

        # Make sure all thinLTO jobs are passed to the linker in bulk.
        # Mold won't properly parallelize thinLTO without it.
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-jobs=all>)

        # Set up lto cache for faster incremental builds and a nominal limits to keep the size
        # from getting too large over time. A single everthing-enabled Release build results
        # in an lto-cache of ~250M. ~1.7G using RelWithDebInfo.
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-cache-dir=${CMAKE_BINARY_DIR}/lto-cache>)

        # Limit size of lto cache to the least of 2GB, 10% disk space, or 10000 files.
        # By default pruning happens in 20 minute intervals. Add prune_interval to change.
        add_link_options(
            $<$<CXX_COMPILER_ID:Clang>:-Wl,--thinlto-cache-policy=cache_size_bytes=2g:cache_size=10%:cache_size_files=10000>
        )

        # Mozilla rec for non-pgo builds. Default is 100. pgo builds can get away with 5-10.
        # Reduces some excessive inlining bloat.
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-Wl,-plugin-opt=-import-instr-limit=40>)

        # Enable virtual constant propagation and more aggressive vtable optimizations.
        # Adds some link time but still faster than flto=full
        # FIXME: -fforce-emit-vtables doesn't play nice with AnyIterator for some reason.
        #add_compile_options($<$<CXX_COMPILER_ID:Clang>:-fforce-emit-vtables>)
        add_link_options($<$<CXX_COMPILER_ID:Clang>:-fwhole-program-vtables>)
    else()
        message(WARNING "LTO/IPO is not supported: ${output}")
    endif()
endif()
