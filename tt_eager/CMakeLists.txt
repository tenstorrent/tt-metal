
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tensor)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/queue)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_dnn)

set(TT_EAGER_OBJECTS
    $<TARGET_OBJECTS:tt_dnn>
    $<TARGET_OBJECTS:tensor>
    $<TARGET_OBJECTS:queue>
)
add_library(tt_eager SHARED ${TT_EAGER_OBJECTS})
target_link_libraries(tt_eager PUBLIC compiler_flags tt_metal)
target_include_directories(tt_eager PUBLIC
    ${UMD_HOME}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tt_metal
    ${CMAKE_SOURCE_DIR}/tt_eager
)

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_lib)
