include(sources.cmake)

add_executable(kernel_prewarm ${KERNEL_PREWARM_SOURCES})
target_link_libraries(kernel_prewarm PRIVATE tt_metal)

target_include_directories(
    kernel_prewarm
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_BINARY_DIR}/tt_metal/impl
)

set_target_properties(
    kernel_prewarm
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)
