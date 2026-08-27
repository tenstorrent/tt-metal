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
# failures above.  (Observed in CI: merge-queue run 33091300710 for PR
# #54500, which deleted unused trailing #define lines from hal.hpp — a
# preprocessor-invisible change — and got a stale metal_test_pch .gch from
# the shared remote cache, failing all cache-miss test TUs with
# "redefinition of ... dev_msgs ..." errors.)
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
    foreach(lang IN ITEMS C CXX)
        set(launcher "${CMAKE_${lang}_COMPILER_LAUNCHER}")
        if(NOT launcher MATCHES "ccache")
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
    endforeach()
endfunction()

# Deprecated name kept for backward compatibility (callers use
# `if(COMMAND tt_disable_ccache_for_pch)` guards, which would silently
# no-op if this alias were removed).  On GCC this never actually disabled
# ccache; see tt_configure_ccache_for_pch for the real behavior.
function(tt_disable_ccache_for_pch target)
    tt_configure_ccache_for_pch(${target})
endfunction()
