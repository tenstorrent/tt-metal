add_library(matmul_common INTERFACE)
add_library(Matmul::Common ALIAS matmul_common)

target_include_directories(matmul_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
