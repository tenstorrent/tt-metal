
set(QUEUE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/queue.cpp
)

add_library(queue STATIC ${QUEUE_SRCS})
target_link_libraries(queue PUBLIC tt_metal_libs)
target_include_directories(queue PUBLIC
                            ${UMD_HOME}
                            ${CMAKE_SOURCE_DIR}
                            ${CMAKE_SOURCE_DIR}/tt_metal
                            ${CMAKE_SOURCE_DIR}/tt_eager
                            ${CMAKE_CURRENT_SOURCE_DIR}
)
