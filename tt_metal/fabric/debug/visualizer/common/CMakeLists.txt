# The commit tt-metal was built from and its build type, read from the version header that tt_metal_version_header
# refreshes on every build. The manifest records the run's and capture records its own, so both link this. It's one
# file in a library of its own, so a new commit recompiles only that file, and it links no tt_metal. Position
# independent, since fabric's objects go into libtt_metal.so. The header takes the build type from TT_METAL_BUILD_TYPE,
# as ttnn_core sets it.
add_library(fabric_visualizer_build_identity STATIC)
target_sources(fabric_visualizer_build_identity PRIVATE fabric_visualizer_build_identity.cpp)
target_include_directories(
    fabric_visualizer_build_identity
    PUBLIC
        ${PROJECT_SOURCE_DIR}
    PRIVATE
        ${PROJECT_BINARY_DIR}/generated
)
target_compile_definitions(fabric_visualizer_build_identity PRIVATE "TT_METAL_BUILD_TYPE=\"$<CONFIG>\"")
set_target_properties(
    fabric_visualizer_build_identity
    PROPERTIES
        POSITION_INDEPENDENT_CODE
            ON
)
if(TARGET tt_metal_version_header)
    add_dependencies(fabric_visualizer_build_identity tt_metal_version_header)
endif()
