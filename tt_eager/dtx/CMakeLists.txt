
set(DTX_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/dtx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dtx_evaluate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/parallelize_generic_tensor_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_collapse_transformations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_convert_tensor_layout_3d_conv_act_to_2Dmatrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_generate_transfer_addresses.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_reverse_transformations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_transpose_xy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_transpose_yz.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_row_major_memory_store.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_row_major_memory_store_blocks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_convert_abstract_tensor_to_channels_last_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_tilize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_pad_2d_matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_block_2d_matrix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_block_2d_with_duplicate_blocks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_generate_groups_outermost_dim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_simple_high_level_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pass_conv_transform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/util_vector_of_ints.cpp
)

add_library(dtx STATIC ${DTX_SRCS})

target_link_libraries(dtx PUBLIC tt_metal_libs)
target_include_directories(dtx PUBLIC
                            ${UMD_HOME}
                            ${CMAKE_SOURCE_DIR}
                            ${CMAKE_SOURCE_DIR}/tt_metal
                            ${CMAKE_SOURCE_DIR}/tt_eager
                            ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_options(dtx PUBLIC ${METAL_COMPILE_OPTIONS} -Wno-int-to-pointer-cast)
