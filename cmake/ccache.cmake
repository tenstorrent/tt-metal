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

# A compiler's PCH binary embeds byte-exact metadata about every header that
# was included when the PCH was built, and that metadata is consulted when a
# translation unit later loads the PCH:
#
#  * Clang embeds per-header file sizes and hard-errors when an on-disk
#    header no longer matches:
#      fatal error: file 'X.hpp' has been modified since the precompiled
#      header was built: size changed
#  * GCC embeds a {file size, MD5-of-contents, once_only} record for every
#    header stacked into the .gch (the "pchf" table in libcpp/files.cc,
#    _cpp_save_file_entries).  It does not hard-error; instead, when a TU
#    includes a header whose on-disk bytes no longer match the recorded
#    (size, MD5), the '#pragma once' / include dedup silently fails, the
#    header is textually re-included on top of the PCH snapshot, and the TU
#    dies with hundreds of "error: redefinition of ..." diagnostics.
#
# ccache's preprocessor-mode cache key is the hash of the *preprocessed*
# output (-E), which is blind to comment text and to macro definitions that
# are never expanded inside the PCH's include closure.  A comment-only or
# dead-macro-only edit to any header in the closure therefore produces the
# same key, and ccache serves a stale PCH built from the pre-edit bytes.
# The embedded metadata then mismatches the on-disk headers, producing the
# failures above.
#
# Per-compiler remedy, applied via tt_configure_ccache_for_pch below:
#
#  * Clang: skip ccache for the (tiny, dedicated) PCH provider targets.
#    The .pch is always built fresh, so its metadata always matches.  All
#    other translation units still benefit from ccache.
#  * GCC: keep ccache but force *depend mode* (CCACHE_DEPEND=1) for the PCH
#    provider targets.  Depend mode never uses the preprocessor-mode key;
#    a cache hit requires the byte-exact content hash of every file listed
#    in the compiler-generated depfile to match the on-disk files — which
#    is exactly the data GCC's pchf table validates at PCH-use time, so a
#    served .gch can never be stale.  Disabling ccache outright for GCC
#    would be counterproductive: GCC's .gch binary is not reproducible
#    (it embeds a timestamp), so a freshly built .gch differs every run,
#    every consumer translation unit gets a different input hash, and the
#    whole build misses the cache.  With depend mode, identical inputs
#    across runs retrieve the identical cached .gch, keeping consumer hit
#    rates high.  (Depend mode requires the compile to emit a depfile via
#    -MD/-MMD; the Ninja generator — the only generator for which
#    useCcache() installs the launcher — always does.)
function(tt_configure_ccache_for_pch target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set_target_properties(
            ${target}
            PROPERTIES
                C_COMPILER_LAUNCHER
                    ""
                CXX_COMPILER_LAUNCHER
                    ""
        )
        return()
    endif()
    # GCC (and others): force ccache depend mode on the provider target by
    # injecting CCACHE_DEPEND=1 into the ccache launcher installed by
    # useCcache().  If ccache is not in use, leave the target alone.
    #
    # The STATUS messages below are deliberate: this protection fails
    # *silently* if the launcher shape ever stops matching (e.g. useCcache()
    # is restructured), and the resulting stale-PCH breakage only shows up
    # later as baffling redefinition errors in CI.  Keep the injected/skipped
    # outcome visible in the configure log.
    foreach(lang IN ITEMS C CXX)
        set(launcher "${CMAKE_${lang}_COMPILER_LAUNCHER}")
        if(NOT launcher MATCHES "ccache")
            message(
                STATUS
                "tt_configure_ccache_for_pch: ${target} (${lang}) has no ccache launcher; skipping depend-mode injection"
            )
            continue()
        endif()
        list(FIND launcher "env" env_index)
        if(env_index GREATER -1)
            math(EXPR value_index "${env_index} + 1")
            list(INSERT launcher ${value_index} "CCACHE_DEPEND=1")
        else()
            list(
                PREPEND
                launcher
                ${CMAKE_COMMAND}
                -E
                env
                CCACHE_DEPEND=1
            )
        endif()
        set_target_properties(
            ${target}
            PROPERTIES
                ${lang}_COMPILER_LAUNCHER
                    "${launcher}"
        )
        message(STATUS "tt_configure_ccache_for_pch: forced CCACHE_DEPEND=1 for ${target} (${lang})")
    endforeach()
endfunction()

# Deprecated name kept for backward compatibility (callers use
# `if(COMMAND tt_disable_ccache_for_pch)` guards, which would silently
# no-op if this alias were removed).  On GCC this never actually disabled
# ccache; see tt_configure_ccache_for_pch for the real behavior.
function(tt_disable_ccache_for_pch target)
    tt_configure_ccache_for_pch(${target})
endfunction()
