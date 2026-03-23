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
function(tt_disable_ccache_for_pch target)
    set_target_properties(
        ${target}
        PROPERTIES
            C_COMPILER_LAUNCHER
                ""
            CXX_COMPILER_LAUNCHER
                ""
    )
endfunction()
