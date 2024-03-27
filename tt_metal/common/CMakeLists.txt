
set(COMMON_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/metal_soc_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_backend_api_types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

set(TRACY_SRCS ${CMAKE_SOURCE_DIR}/tt_metal/third_party/tracy/public/TracyClient.cpp)

# note: tracy .so not built here, haven't tested tracy build yet, might need to change something
add_library(tracy OBJECT ${TRACY_SRCS})
target_link_libraries(tracy PUBLIC ${COMMOND_LIBS} compiler_flags)
target_include_directories(tracy PUBLIC ${CMAKE_SOURCE_DIR}/tt_metal/third_party/tracy/public/tracy)
target_compile_options(tracy PUBLIC -DTRACY_NO_CONTEXT_SWITCH)

add_library(common OBJECT ${COMMON_SRCS})
target_link_libraries(common PUBLIC umd_lib tracy ${COMMON_LIBS} compiler_flags) # adding compiler_flags here is redundant

target_include_directories(common PUBLIC
                                    ${UMD_HOME}
                                    ${CMAKE_SOURCE_DIR}
                                    ${CMAKE_SOURCE_DIR}/tt_metal
                                    ${CMAKE_SOURCE_DIR}/tt_metal/third_party/fmt)
