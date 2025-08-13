# Built as outlined in Tracy documentation (pg.12)
set(TRACY_HOME ${PROJECT_SOURCE_DIR}/tt_metal/third_party/tracy)

# Propagate ENABLE_TRACY to TRACY_ENABLE (Tracy build option)
# CMake options are propagated as PUBLIC compile definitions to tracy build
if(ENABLE_TRACY)
    set(TRACY_ENABLE ON)
else()
    set(TRACY_ENABLE OFF)
endif()

# Tracy will always be built with fallback timing enabled
# avoiding hard failures on systems without invariant TSC and using OS clocks when needed.
set(TRACY_TIMER_FALLBACK ON)

set(DEFAULT_COMPONENT_NAME ${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME})
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tracy)
set(_saved_clang_tidy "${CMAKE_CXX_CLANG_TIDY}")
set(CMAKE_CXX_CLANG_TIDY "")
add_subdirectory(${TRACY_HOME})
set(CMAKE_CXX_CLANG_TIDY "${_saved_clang_tidy}")
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ${DEFAULT_COMPONENT_NAME})

set_target_properties(
    TracyClient
    PROPERTIES
        EXCLUDE_FROM_ALL
            TRUE
        LIBRARY_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY
            "${PROJECT_BINARY_DIR}/lib"
        OUTPUT_NAME
            "tracy"
)

if(ENABLE_TRACY)
    # This setup ensures that when someone links against TracyClient:
    # Their code preserves frame pointers (-fno-omit-frame-pointer) → backtraces work.
    # Their executable exports symbols (-rdynamic) → Tracy can map addresses to names.
    target_compile_options(
        TracyClient
        PUBLIC
            -fno-omit-frame-pointer
        PRIVATE
            "$<$<CXX_COMPILER_ID:Clang>:-Wno-conditional-uninitialized>" # FIXME: Fix this upstream
    )
    target_link_options(TracyClient PUBLIC -rdynamic)
endif()

# Our current fork of tracy does not have CMake support for these subdirectories
# Once we update, we can change this

# Build Tracy tools (csvexport and capture) using CMake
add_subdirectory(${TRACY_HOME}/csvexport)
add_subdirectory(${TRACY_HOME}/capture)
