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

    # FIXME: Not (yet) coexisting Precompiled headers and ccache
    #        Extra ccache args (sloppiness) are required for PCH, and we should only be sloppy
    #        on the files that _use_ PCH.  For now treat the two features as mutually exclusive.
    if(NOT CMAKE_DISABLE_PRECOMPILE_HEADERS)
        # Be noisy to not mislead people, and also to draw attention to where to come fix it.
        message(FATAL_ERROR "Ccache is not configured to handle precompiled headers. Don't enable ccache, or disable PCH with CMAKE_DISABLE_PRECOMPILE_HEADERS. Or update the build to handle both together.")
    endif()

    if(CMAKE_GENERATOR MATCHES "Ninja")
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE} PARENT_SCOPE)
        message(STATUS "ccache enabled")
    endif()
endfunction()
useCcache()
