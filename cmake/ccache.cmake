function(useCcache)
    # Only manipulate build settings if WE are the top-level
    if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        message(STATUS "ccache disabled -- not top level")
        return()
    endif()

    find_program(CCACHE_EXECUTABLE ccache)
    if(NOT CCACHE_EXECUTABLE)
        message(WARNING "ccache not found -- disabling")
        return()
    endif()

    if(NOT CMAKE_DISABLE_PRECOMPILE_HEADERS)
        message(STATUS "Overriding CCACHE_SLOPPINESS to work with PCH.")
        set(CCACHE_ENV "CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime")
    endif()

    # Default to zstd level-3 compression if the caller has not already chosen a compression setting.
    # This reduces remote storage (Redis) entry size by ~3-5x with negligible impact on build times.
    if(NOT DEFINED ENV{CCACHE_COMPRESS})
        list(APPEND CCACHE_ENV "CCACHE_COMPRESS=true")
        list(APPEND CCACHE_ENV "CCACHE_COMPRESSLEVEL=3")
        message(STATUS "ccache compression: defaulting to zstd level 3 (override with CCACHE_COMPRESS env var)")
    endif()

    # Opt-in distributed compilation via Icecream, for dev environments with spare
    # compute nodes (e.g. exabox). Off everywhere else by default -- including CI --
    # since it only activates when TT_ICECC_SCHEDULER_HOST is set, which we expect
    # dev environments to export globally (e.g. from the exabox image), never
    # something this repo defaults on its own.
    set(TT_ICECC_SCHEDULER_HOST
        $ENV{TT_ICECC_SCHEDULER_HOST}
        CACHE STRING "Icecream scheduler host; set to enable distributed ccache builds"
    )
    if(TT_ICECC_SCHEDULER_HOST)
        find_program(ICECC_EXECUTABLE icecc)
        if(NOT ICECC_EXECUTABLE)
            message(WARNING "TT_ICECC_SCHEDULER_HOST is set but icecc was not found -- falling back to plain ccache")
        else()
            # Icecream ships the compiler itself to remote nodes, but the bundle is built
            # per-toolchain -- and per-ABI for clang's libc++/libstdc++ split -- via
            # icecc-create-env. Map the selected toolchain to its pre-built env tarball
            # rather than relying on icecc's slower/less predictable auto-environment-creation.
            if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ENABLE_LIBCXX)
                set(_tt_icecc_env_name "clang-20-libcxx")
            elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
                set(_tt_icecc_env_name "clang-20-libstdcxx")
            elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                set(_tt_icecc_env_name "gcc-${CMAKE_CXX_COMPILER_VERSION}")
            endif()

            set(TT_ICECC_VERSION
                "/opt/tenstorrent/icecc-envs/${_tt_icecc_env_name}.tar.gz"
                CACHE STRING "Icecream toolchain env tarball for the selected compiler"
            )
            if(NOT EXISTS "${TT_ICECC_VERSION}")
                message(WARNING "Icecream env tarball not found at ${TT_ICECC_VERSION} -- falling back to plain ccache")
            else()
                list(APPEND CCACHE_ENV
                    "CCACHE_PREFIX=${ICECC_EXECUTABLE}"
                    "ICECC_VERSION=${TT_ICECC_VERSION}"
                    "ICECC_SCHEDULER_HOST=${TT_ICECC_SCHEDULER_HOST}"
                )
                message(STATUS "Icecream distributed compilation enabled (scheduler: ${TT_ICECC_SCHEDULER_HOST}, env: ${_tt_icecc_env_name})")
            endif()
        endif()
    endif()

    if(CMAKE_GENERATOR MATCHES "Ninja")
        foreach(lang IN ITEMS C CXX)
            set(CMAKE_${lang}_COMPILER_LAUNCHER
                ${CMAKE_COMMAND}
                -E
                env
                ${CCACHE_ENV}
                ${CCACHE_EXECUTABLE}
                PARENT_SCOPE
            )
        endforeach()
        message(STATUS "ccache enabled")
    endif()
endfunction()
useCcache()

# Clang's PCH binary embeds metadata (file sizes) of every header that was
# included when the PCH was built.  When a translation unit loads the PCH,
# Clang verifies the on-disk headers still match that metadata.
#
# ccache's preprocessor-mode cache key is the hash of the *preprocessed*
# output, which strips comments.  A comment-only edit to a header therefore
# produces the same hash, so ccache returns the old .pch — whose embedded
# sizes no longer match the on-disk files — and every consumer fails with:
#
#   fatal error: file 'X.hpp' has been modified since the precompiled
#   header was built: size changed
#
# Skipping ccache for the (tiny, dedicated) PCH provider targets is the
# simplest reliable fix: the .pch is always built fresh, so its metadata
# always matches.  All other translation units still benefit from ccache.
#
# This problem is Clang-specific.  GCC uses a different PCH validation
# mechanism (-fpch-preprocess) that does not embed per-header file sizes,
# so the stale-metadata issue does not arise.  Disabling ccache for GCC PCH
# providers is therefore counterproductive: GCC's .gch binary embeds a
# compilation timestamp, making each fresh build produce a different binary.
# That causes every consumer translation unit to get a different input hash
# and miss the cache.  With ccache enabled on the provider, the first worker
# stores the .gch and all subsequent workers (same run or future runs with
# identical inputs) retrieve the same binary, giving consumers consistent
# input hashes and high hit rates.
function(tt_disable_ccache_for_pch target)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        return()
    endif()
    set_target_properties(
        ${target}
        PROPERTIES
            C_COMPILER_LAUNCHER
                ""
            CXX_COMPILER_LAUNCHER
                ""
    )
endfunction()
