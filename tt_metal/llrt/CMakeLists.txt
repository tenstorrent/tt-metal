# add CFLAG  -Wno-int-to-pointer-cast

set(LLRT_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/llrt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rtoptions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_cluster.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_hexfile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tt_memory.cpp)

add_library(llrt OBJECT ${LLRT_SRC})
target_link_libraries(llrt PUBLIC common)   # device dependency should be included in umd
target_compile_options(llrt PRIVATE -Wno-int-to-pointer-cast)
target_compile_options(llrt PUBLIC ${METAL_COMPILE_OPTIONS})
