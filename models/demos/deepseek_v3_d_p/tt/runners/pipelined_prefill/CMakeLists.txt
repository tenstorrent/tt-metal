# Standalone `_layer_completion` nanobind extension for the deepseek_v3_d_p pipelined-prefill
# layer-completion feature (host-local ring + reorder buffer + MPI router + the test-only consumer).
# Deliberately NOT part of ttnn: it links only TT::Metalium and is imported via
# models.demos.deepseek_v3_d_p.tt.runners.pipelined_prefill — zero ttnn API/_ttnn footprint.
#
# Added from the root CMakeLists (after ttnn, gated on WITH_PYTHON_BINDINGS) so the nanobind tooling
# (TT::Nanobind, nanobind_* helpers) and TT::Metalium are in scope. Like _ttnn, the .so is emitted to
# the BUILD tree and symlinked into this source dir (covered by the repo-root `*.so` gitignore).

include(${CMAKE_CURRENT_SOURCE_DIR}/sources.cmake) # LAYER_COMPLETION_IMPL_SRCS
add_library(layer_completion_ext SHARED)
target_sources(
    layer_completion_ext
    PRIVATE
        # shared impls (queue/reorder/router) + this module's own binding entry point + test consumer
        ${LAYER_COMPLETION_IMPL_SRCS}
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_completion_bindings.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/layer_completion_consumer.cpp
)
target_link_libraries(
    layer_completion_ext
    PRIVATE
        TT::Nanobind
        TT::Metalium
)
# tt_metal/distributed is needed for named_shm.hpp (a distributed-internal header used by the queue).
target_include_directories(layer_completion_ext PRIVATE ${PROJECT_SOURCE_DIR}/tt_metal/distributed)
nanobind_compile_options(layer_completion_ext)
nanobind_link_options(layer_completion_ext)
set_target_properties(
    layer_completion_ext
    PROPERTIES
        OUTPUT_NAME
            layer_completion
        PREFIX
            "_"
        CXX_VISIBILITY_PRESET
            "default"
        LIBRARY_OUTPUT_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}
        BUILD_RPATH
            "${PROJECT_BINARY_DIR}/lib;${PROJECT_BINARY_DIR}/tt_metal;${PROJECT_BINARY_DIR}/ttnn"
        # No INSTALL_RPATH: not installed (no install(TARGETS) rule); loaded from the build tree via the
        # source-dir symlink. Add an $ORIGIN-based RPATH (like _ttnn) only if an install rule lands.
        ADDITIONAL_CLEAN_FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/_layer_completion.so"
)
# Symlink the build-tree .so into this source dir so it imports as
# ...pipelined_prefill._layer_completion (mirrors ttnn/ttnn/_ttnn.so -> build).
add_custom_command(
    TARGET layer_completion_ext
    POST_BUILD
    COMMAND
        ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:layer_completion_ext>"
        "${CMAKE_CURRENT_SOURCE_DIR}/_layer_completion.so"
    VERBATIM
)
