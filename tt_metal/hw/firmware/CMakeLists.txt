add_subdirectory(src/tt-1xx)

# The shared precompiled-header umbrella is architecture neutral, so it sits beside the per-arch
# source directories and is installed to match the layout jit_build expects at runtime.
install(
    FILES
        src/pch.h
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/hw/firmware/src
    COMPONENT metalium-runtime
)
