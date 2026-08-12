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
        # TT_ICECC_SCHEDULER_HOST being set is a deliberate ask for distributed builds.
        # If we can't actually deliver that, fail the configure loudly rather than quietly
        # falling back to plain ccache -- a silent downgrade here is easy to miss and leaves
        # a dev wondering for a long time why their build never distributes. Note this is
        # independent of the scheduler/compute nodes themselves being flaky at *build* time:
        # icecc already falls back to local compilation per-job when the scheduler or a
        # daemon is unreachable, which is the flakiness this doesn't need to handle.
        find_program(ICECC_EXECUTABLE icecc)
        if(NOT ICECC_EXECUTABLE)
            message(FATAL_ERROR "TT_ICECC_SCHEDULER_HOST is set but icecc is not installed -- install it or unset TT_ICECC_SCHEDULER_HOST to build with plain ccache")
        endif()

        # Icecream ships the compiler itself to remote nodes, but the bundle is built
        # per-toolchain -- and per-ABI for clang's libc++/libstdc++ split, since compiling
        # (not linking) against libc++ vs libstdc++ resolves a different header tree --
        # via icecc-create-env. Map the selected toolchain to its env name.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ENABLE_LIBCXX)
            set(_tt_icecc_env_name "clang-20-libcxx")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(_tt_icecc_env_name "clang-20-libstdcxx")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(_tt_icecc_env_name "gcc-${CMAKE_CXX_COMPILER_VERSION}")
        else()
            message(FATAL_ERROR "TT_ICECC_SCHEDULER_HOST is set but ${CMAKE_CXX_COMPILER_ID} has no known Icecream env mapping")
        endif()

        # Cache generated envs per-user (survives build-dir wipes, shared across checkouts).
        # Generation only runs once per toolchain; delete the tarball to force a rebuild
        # after a compiler upgrade.
        set(TT_ICECC_ENV_CACHE_DIR
            "$ENV{HOME}/.cache/tt-icecc-envs"
            CACHE PATH "Where generated Icecream toolchain env tarballs are cached"
        )
        set(_tt_icecc_tarball "${TT_ICECC_ENV_CACHE_DIR}/${_tt_icecc_env_name}.tar.gz")

        if(NOT EXISTS "${_tt_icecc_tarball}")
            find_program(ICECC_CREATE_ENV_EXECUTABLE icecc-create-env)
            if(NOT ICECC_CREATE_ENV_EXECUTABLE)
                message(FATAL_ERROR "TT_ICECC_SCHEDULER_HOST is set but icecc-create-env is not installed -- can't generate the ${_tt_icecc_env_name} toolchain env")
            endif()

            message(STATUS "Generating Icecream env for ${_tt_icecc_env_name} (first use only, cached at ${_tt_icecc_tarball})")
            set(_tt_icecc_scratch "${TT_ICECC_ENV_CACHE_DIR}/.scratch-${_tt_icecc_env_name}")
            file(REMOVE_RECURSE "${_tt_icecc_scratch}")
            file(MAKE_DIRECTORY "${_tt_icecc_scratch}")

            if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
                set(_tt_icecc_create_env_args --clang "${CMAKE_C_COMPILER}" "${CMAKE_CXX_COMPILER}")
            else()
                set(_tt_icecc_create_env_args --gcc "${CMAKE_C_COMPILER}" "${CMAKE_CXX_COMPILER}")
            endif()

            execute_process(
                COMMAND ${ICECC_CREATE_ENV_EXECUTABLE} ${_tt_icecc_create_env_args}
                WORKING_DIRECTORY "${_tt_icecc_scratch}"
                RESULT_VARIABLE _tt_icecc_create_env_result
                OUTPUT_VARIABLE _tt_icecc_create_env_output
                ERROR_VARIABLE _tt_icecc_create_env_error
            )

            file(GLOB _tt_icecc_generated "${_tt_icecc_scratch}/*.tar.gz")
            list(LENGTH _tt_icecc_generated _tt_icecc_generated_count)

            if(NOT _tt_icecc_create_env_result EQUAL 0 OR NOT _tt_icecc_generated_count EQUAL 1)
                message(FATAL_ERROR
                    "TT_ICECC_SCHEDULER_HOST is set but generating the Icecream env for "
                    "${_tt_icecc_env_name} failed (icecc-create-env exit ${_tt_icecc_create_env_result}, "
                    "produced ${_tt_icecc_generated_count} tarball(s)):\n"
                    "${_tt_icecc_create_env_output}\n${_tt_icecc_create_env_error}"
                )
            endif()

            file(RENAME "${_tt_icecc_generated}" "${_tt_icecc_tarball}")
            file(REMOVE_RECURSE "${_tt_icecc_scratch}")
            message(STATUS "Icecream env for ${_tt_icecc_env_name} generated: ${_tt_icecc_tarball}")
        endif()

        list(APPEND CCACHE_ENV
            "CCACHE_PREFIX=${ICECC_EXECUTABLE}"
            "ICECC_VERSION=${_tt_icecc_tarball}"
            "ICECC_SCHEDULER_HOST=${TT_ICECC_SCHEDULER_HOST}"
        )
        message(STATUS "Icecream distributed compilation enabled (scheduler: ${TT_ICECC_SCHEDULER_HOST}, env: ${_tt_icecc_env_name})")
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
