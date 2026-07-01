add_library(multi_host_layer_completion OBJECT)
add_library(TT::Metalium::Test::Distributed::MultihostLayerCompletion ALIAS multi_host_layer_completion)
TT_ENABLE_UNITY_BUILD(multi_host_layer_completion)

# Shared layer-completion impl source list (LAYER_COMPLETION_IMPL_SRCS / LAYER_COMPLETION_DIR).
include(${PROJECT_SOURCE_DIR}/models/demos/deepseek_v3_d_p/tt/runners/pipelined_prefill/sources.cmake)
target_sources(
    multi_host_layer_completion
    PRIVATE
        main.cpp
        test_layer_completion_routing.cpp
        # layer-completion impls live with the pipelined-prefill feature; compile them in so this
        # MPI routing test links without the feature's standalone .so.
        ${LAYER_COMPLETION_IMPL_SRCS}
)

target_include_directories(
    multi_host_layer_completion
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
        ${PROJECT_SOURCE_DIR}/tt_metal/distributed # named_shm.hpp (used by the queue impl)
        ${LAYER_COMPLETION_DIR} # layer-completion feature headers
)

target_link_libraries(multi_host_layer_completion PRIVATE test_metal_common_libs)

add_executable(multi_host_layer_completion_tests)
set_target_properties(
    multi_host_layer_completion_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(multi_host_layer_completion_tests PRIVATE multi_host_layer_completion)
