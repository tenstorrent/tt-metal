
set(COMMON_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/metal_soc_descriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_backend_api_types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

set(TRACY_SRCS ${CMAKE_SOURCE_DIR}/tt_metal/third_party/tracy/public/TracyClient.cpp)

add_library(tracy OBJECT ${TRACY_SRCS})
target_link_libraries(tracy PUBLIC ${COMMOND_LIBS})
# target_link_options(tracy PUBLIC -rdynamic)
target_include_directories(tracy PUBLIC ${CMAKE_SOURCE_DIR}/tt_metal/third_party/tracy/public/tracy)
target_compile_options(tracy PUBLIC ${METAL_COMPILE_OPTIONS} -DTRACY_NO_CONTEXT_SWITCH)  # should be public?

# add_library(tracy_lib SHARED $<TARGET_OBJECTS:tracy>)
# target_link_libraries(tracy_lib PUBLIC ${COMMOND_LIBS})
# target_include_directories(tracy_lib PUBLIC ${CMAKE_SOURCE_DIR}/tt_metal/third_party/tracy/public/tracy)

add_library(common OBJECT ${COMMON_SRCS})
target_link_libraries(common PUBLIC umd_lib tracy ${COMMON_LIBS})    # need to add tracy as dependancy here !!!

target_include_directories(common PUBLIC ${UMD_HOME})
target_include_directories(common PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(common PUBLIC ${CMAKE_SOURCE_DIR}/tt_metal)
target_include_directories(common PUBLIC ${CMAKE_SOURCE_DIR}/tt_metal/third_party/fmt)

target_compile_options(common PUBLIC ${METAL_COMPILE_OPTIONS})
