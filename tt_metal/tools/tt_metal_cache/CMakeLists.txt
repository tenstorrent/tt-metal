include(sources.cmake)

# Named tt-metal-cache rather than tt_metal_cache: this is a user-facing command, and it is
# the stable entry point provisioning scripts are meant to call instead of rm -rf on a path
# that is an implementation detail.
add_executable(tt-metal-cache ${TT_METAL_CACHE_SOURCES})

target_link_libraries(
    tt-metal-cache
    PRIVATE
        Metalium::Metal
        TT::STL
)

set_target_properties(
    tt-metal-cache
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

tt_reuse_precompile_headers(tt-metal-cache TT::CommonPCH)

# Unlike the debug tools alongside it, this one has to exist in a packaged install: the public
# documentation and two API docstrings name `tt-metal-cache` as the supported way to reclaim
# cache space, having replaced the rm -rf instructions they used to give.
if(TT_INSTALL)
    install(
        TARGETS
            tt-metal-cache
        RUNTIME
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            COMPONENT metalium-runtime
    )
endif()
