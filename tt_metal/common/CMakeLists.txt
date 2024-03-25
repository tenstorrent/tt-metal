
set(COMMON_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/metal_soc_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_backend_api_types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

set(TRACY_SRCS ${CMAKE_SOURCE_DIR}/tt_metal/third_party/tracy/public/TracyClient.cpp)

# note: tracy .so not built here, haven't tested tracy build yet, might need to change something
add_library(tracy_o OBJECT ${TRACY_SRCS})
target_link_libraries(tracy_o PUBLIC ${COMMOND_LIBS} compiler_flags metal_header_directories)
target_include_directories(tracy_o PUBLIC ${CMAKE_SOURCE_DIR}/tt_metal/third_party/tracy/public/tracy)
target_compile_options(tracy_o PUBLIC -DTRACY_NO_CONTEXT_SWITCH)

add_library(common OBJECT ${COMMON_SRCS})
target_link_libraries(common PUBLIC tracy_o compiler_flags metal_common_libs metal_header_directories) # adding compiler_flags here is redundant, but idk if tracy should be here so I'm leaving it

target_include_directories(common PUBLIC
    ${UMD_HOME}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tt_metal
    ${CMAKE_SOURCE_DIR}/tt_metal/third_party/fmt
)
