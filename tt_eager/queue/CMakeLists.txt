
set(QUEUE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp

    CACHE INTERNAL "Queue sources to reuse in ttnn build"
)

# add_library(queue OBJECT ${QUEUE_SRCS})
# target_link_libraries(queue PUBLIC metal_header_directories compiler_flags)
# target_include_directories(queue PUBLIC
#     ${UMD_HOME}
#     ${PROJECT_SOURCE_DIR}
#     ${PROJECT_SOURCE_DIR}/tt_metal
#     ${PROJECT_SOURCE_DIR}/tt_eager
#     ${CMAKE_CURRENT_SOURCE_DIR}
#     ${PROJECT_SOURCE_DIR}/tt_metal/third_party/fmt
# )
