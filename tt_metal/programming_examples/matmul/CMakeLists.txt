cmake_minimum_required(VERSION 3.22...3.30)
project(metal_examples_matmul)

# Create the matmul_common library
add_library(matmul_common INTERFACE)
add_library(Matmul::Common ALIAS matmul_common)

# Add the headers to the library
target_sources(
    matmul_common
    INTERFACE
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            matmul_common/bmm_op.hpp
            matmul_common/bmm_op_utils.hpp
            matmul_common/bmm_op_utils_untilize.hpp
            matmul_common/bmm_op_utils_untilize_reduce.hpp
            matmul_common/bmm_op_utils_untilize_reduce_mcast.hpp
            matmul_common/bmm_op_utils_untilize_reduce_mcast_2d.hpp
            matmul_common/bmm_op_utils_untilize_reduce_mcast_2d_optimized.hpp
)

# Install the library so it can be found by subdirectories
install(TARGETS matmul_common EXPORT MatmulCommon FILE_SET api DESTINATION include/matmul)

# Install the export file
install(EXPORT MatmulCommon NAMESPACE Matmul:: DESTINATION lib/cmake/matmul-common)
